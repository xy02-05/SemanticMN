"""
video_sampler.py

Ego4D Video Feature Sampler - 高效加权采样器

核心功能：
1. 加载过滤后的ego4d video/fused features和预计算的topK采样候选
2. 根据bridge指令的task_index，高效并行采样ego4d video features
3. 返回采样的video features和对应的text相似度权重(用作soft InfoNCE的label)
4. 支持全局随机负样本采样，提供真正的远距离负样本

设计原理：
- 预计算TopK候选，训练时只需从K个候选中采样n个
- 全部操作在GPU上完成，采样延迟 < 1ms
- 支持batch并行采样：B个query各采样n个，然后在batch内去重为共享 sampled anchors
- 返回 dense 的 query-to-anchor 相似度矩阵，使同一个 sampled anchor 可以同时对多个query为正
- 全局随机负样本：从全部filtered pool中均匀随机采样，补充"远距离"负样本

使用方式：
    sampler = VideoFeatureSampler(
        ego4d_path="ego4d_filtered_t0.5.npz",
        topk_path="ego4d_topk1024_t0.5.npz",
        device="cuda:0",
        n_random_neg=16,  # 每个batch额外采样16个全局随机负样本
    )
    # 训练时调用
    video_feats, sim_weights, pool_indices = sampler.sample(task_indices, n_per_query=4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class VideoFeatureSampler(nn.Module):
    """
    Ego4D视频特征加权采样器
    
    初始化时加载数据到GPU，训练时通过task_index高效采样。
    所有采样操作在GPU上并行完成，零CPU开销。
    """
    
    def __init__(
        self,
        ego4d_path: str,          # 过滤后的ego4d特征文件 (ego4d_filtered_t0.5.npz)
        topk_path: str,            # 预计算的topK采样候选 (ego4d_topk1024_t0.5.npz)
        n_per_query: int = 4,      # 每个query采样的video数量
        softmax_temperature: float = 0.1,  # softmax温度，控制采样集中度
        n_random_neg: int = 0,     # 每个batch额外采样的全局随机负样本数（0=不启用）
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.n_per_query = n_per_query
        self.softmax_temperature = softmax_temperature
        self.n_random_neg = n_random_neg  # 全局随机负样本数
        self.device_str = device
        self.dtype = dtype
        
        # ==================== 加载ego4d video features ====================
        ego4d_data = np.load(ego4d_path)
        video_embeds = ego4d_data['video_embeds']  # (N_filtered, 4096)
        self.n_ego4d = video_embeds.shape[0]
        self.video_dim = video_embeds.shape[1]
        
        # persistent=False：不写入state_dict，避免checkpoint臃肿（GB级数据）
        # 但仍随模型.to(device)移动
        video_tensor = torch.from_numpy(video_embeds.astype(np.float32))
        video_tensor = F.normalize(video_tensor, p=2, dim=-1)  # L2归一化
        self.register_buffer('video_embeds', video_tensor.to(dtype), persistent=False)
        
        # ==================== 加载topK采样候选 ====================
        topk_data = np.load(topk_path)
        topk_indices = topk_data['topk_indices']  # (N_bridge, K) int32
        topk_sims = topk_data['topk_sims']        # (N_bridge, K) float16
        self.n_bridge = topk_indices.shape[0]
        self.topk = topk_indices.shape[1]
        
        self.register_buffer('topk_indices',
                             torch.from_numpy(topk_indices.astype(np.int64)),
                             persistent=False)
        # clamp到[0,1]：float16精度可能导致cosine sim略>1.0
        self.register_buffer('topk_sims',
                             torch.from_numpy(topk_sims.astype(np.float32)).clamp(0.0, 1.0),
                             persistent=False)
        
        # 预计算 softmax 权重（避免每次训练 step 重复计算）
        # shape: (N_bridge, K)，每行是一个概率分布
        # 重要：build_video_pool 用 topk_indices=-1 / topk_sims=0 做填充，
        # softmax(0/τ) 仍会给这些填充位非零概率，最终通过 video_embeds[-1]
        # 静默拿到错误特征。这里在 softmax 之前把这些位置的 logit 拉成 -inf，
        # 保证 multinomial 永远不会采样到 -1 填充位。
        masked_logits = (self.topk_sims / softmax_temperature).clone()
        invalid_mask = (self.topk_indices < 0)
        masked_logits[invalid_mask] = float('-inf')
        softmax_weights = F.softmax(masked_logits, dim=-1)
        # 兜底：某行全是 -inf（该 task 一个候选都没有）→ softmax = NaN，置为 0
        softmax_weights = torch.nan_to_num(softmax_weights, nan=0.0)
        self.register_buffer('sampling_probs', softmax_weights, persistent=False)
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        
        # 显式移动到目标设备（numpy转来的tensor默认在CPU）
        self.to(device)
        
        print(f"✅ VideoFeatureSampler 初始化完成:")
        print(f"   Ego4D视频特征: {self.n_ego4d} × {self.video_dim}")
        print(f"   Bridge指令数: {self.n_bridge}, TopK: {self.topk}")
        print(f"   每次采样: {n_per_query} per query, 全局随机负样本: {n_random_neg}, device: {device}")

    def _build_dense_sim_weights(
        self,
        task_indices: torch.Tensor,
        pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        为当前 batch 构建 dense 的 query-to-anchor text 相似度矩阵。

        输入：
        - task_indices: [B]，当前 batch 的 task 索引
        - pool_indices: [M]，去重后的 sampled anchor（在 filtered pool 中的索引）

        输出：
        - dense_sims: [B, M]

        说明：
        1. 这里只复用预计算好的 topk_indices / topk_sims，不重新算文本 embedding。
        2. 如果某个 sampled anchor 不在 query 的原始 top-K 中，则该位置相似度记为 0。
        3. 因此同一个 sampled anchor 可以同时对多个 query 为正样本。
        """
        device = self.video_embeds.device
        batch_size = task_indices.shape[0]
        num_pool = pool_indices.shape[0]

        dense_sims = torch.zeros(batch_size, num_pool, device=device, dtype=torch.float32)
        topk_idx_for_batch = self.topk_indices[task_indices]   # [B, K]
        topk_sims_for_batch = self.topk_sims[task_indices]     # [B, K]

        # 逐行构建更省显存；M 通常远小于全量 filtered pool。
        for i in range(batch_size):
            matches = topk_idx_for_batch[i].unsqueeze(1) == pool_indices.unsqueeze(0)  # [K, M]
            if matches.any():
                weighted_sims = matches.to(torch.float32) * topk_sims_for_batch[i].unsqueeze(1)
                dense_sims[i] = weighted_sims.max(dim=0).values

        return dense_sims
    
    @torch.no_grad()
    def sample(
        self,
        task_indices: torch.Tensor,  # [B] bridge指令的task_index
        n_per_query: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据 task_index 采样共享 video anchors，并返回 dense 权重矩阵
        
        采样流程：
        1. 查表获取每个query的topK采样概率分布
        2. torch.multinomial并行采样每个query的局部候选
        3. 把所有sampled视频去重，形成batch共享anchor集合
        4. 为每个query计算其对所有sampled anchors的text相似度
        
        Args:
            task_indices: [B] 当前batch的bridge task_index
            n_per_query: 每个query采样数，默认使用初始化时的设定
            
        Returns:
            video_features: [M, D] 去重后的shared anchors特征（已L2归一化）
            sim_weights: [B, M] dense text 相似度矩阵
            pool_indices: [M] sampled anchors在过滤数组中的索引
        """
        if n_per_query is None:
            n_per_query = self.n_per_query
        
        B = task_indices.shape[0]
        device = self.video_embeds.device
        
        # 保留原始 task_indices 以便对无效行清零；采样本身仍复用原有逻辑
        raw_task_indices = task_indices.to(device)
        valid_task_mask = (raw_task_indices != -1)
        task_indices = raw_task_indices.clamp(0, self.n_bridge - 1)
        
        # ===== Step 1: 查表获取采样概率 =====
        # probs: [B, K]
        probs = self.sampling_probs[task_indices]
        
        # ===== Step 2: 并行采样（GPU上的multinomial，B个分布同时采样）=====
        # sampled_topk_pos: [B, n] 在topK中的位置索引(0~K-1)
        # 用 replacement=True：当某个 task 的有效候选数 < n_per_query 时，
        # multinomial(replacement=False) 在 PyTorch 里会回填零概率位（即 -1 填充位），
        # 导致后面 video_embeds[-1] 静默错位。with_replacement 的副作用只是
        # 同一 anchor 可能被采到多次，下游 torch.unique 自然去重，dense sim 不变。
        sampled_topk_pos = torch.multinomial(probs, n_per_query, replacement=True)
        
        # 强制每个有效 query 至少带上自己的 top1 候选。
        # 这样做的目的是保证后续构造共享 sampled pool 时，
        # 每个 case 都至少有一个来自自身检索结果的 anchor，
        # 避免纯随机采样把该 query 的正样本完全漏掉。
        if n_per_query > 0 and valid_task_mask.any():
            sampled_topk_pos[valid_task_mask, 0] = 0
        
        # ===== Step 3: 获取实际ego4d索引 =====
        # topk_idx_for_batch: [B, K]
        topk_idx_for_batch = self.topk_indices[task_indices]
        # sampled_ego4d_idx: [B, n] 在过滤后ego4d数组中的实际索引
        sampled_ego4d_idx = torch.gather(topk_idx_for_batch, 1, sampled_topk_pos)
        
        # ===== Step 4: 先汇总所有有效 query 的局部 sampled indices =====
        # 无效 task_index=-1 的行虽然在上面为了复用旧逻辑做了 clamp，
        # 但不应把这些行采到的 anchors 混入共享 sampled pool。
        sampled_pool_indices = sampled_ego4d_idx[valid_task_mask].reshape(-1)  # [N_valid*n]

        # ===== Step 5: 全局随机负样本 =====
        # 从全部filtered pool中均匀随机采样，提供"远距离"负样本
        # 这些样本和所有query的text sim都很低(~0.29)，是天然的hard negative
        if self.n_random_neg > 0:
            random_idx = torch.randint(0, self.n_ego4d, (self.n_random_neg,), device=device)
            sampled_pool_indices = torch.cat([sampled_pool_indices, random_idx], dim=0)

        # ===== Step 6: 在 batch 内做去重，得到共享 sampled anchors =====
        # 顺序不影响loss，后续 dense sim 矩阵会按 pool_indices 当前顺序重建。
        pool_indices = torch.unique(sampled_pool_indices, sorted=True)

        # ===== Step 7: 取 shared anchors 的 video 特征 =====
        video_features = self.video_embeds[pool_indices]  # [M, D]

        # ===== Step 8: 为每个 query 构建 dense 的 query-to-anchor 相似度矩阵 =====
        sim_weights = self._build_dense_sim_weights(task_indices, pool_indices)

        # 原始 task_index = -1 的无效样本整行清零，避免伪正样本参与监督
        if (~valid_task_mask).any():
            sim_weights[~valid_task_mask] = 0.0

        return video_features, sim_weights, pool_indices


def create_video_sampler(config: dict, device: str, dtype: torch.dtype) -> Optional[VideoFeatureSampler]:
    """
    工厂函数：从配置字典创建VideoFeatureSampler
    
    config中需要包含:
    - ego4d_filtered_path: 过滤后的ego4d特征文件路径
    - ego4d_topk_path: 预计算的topK采样候选文件路径
    - n_per_query: 每个query采样数(默认4)
    - softmax_temperature: softmax温度(默认0.1)
    - n_random_neg: 每batch全局随机负样本数(默认0，不启用)
    """
    ego4d_path = config.get('ego4d_filtered_path')
    topk_path = config.get('ego4d_topk_path')
    
    if not ego4d_path or not topk_path:
        return None
    
    return VideoFeatureSampler(
        ego4d_path=ego4d_path,
        topk_path=topk_path,
        n_per_query=config['n_per_query'],
        softmax_temperature=config['softmax_temperature'],
        n_random_neg=config['n_random_neg'],
        device=device,
        dtype=dtype,
    )
