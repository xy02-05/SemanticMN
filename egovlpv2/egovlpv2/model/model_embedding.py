"""
model_embedding.py

预计算Embedding模型封装，用于fine-grained alignment训练
支持加载text-embedding-3-large等预计算的文本特征

核心功能：
1. 加载预计算的embedding特征（.npz格式）
2. 加载文本索引（.json格式）
3. 提供compute_text方法，通过文本查找对应的预计算特征
4. 与EgoVLPv2/EgoHOD/Qwen3保持一致的接口设计

使用示例：
    model = EmbeddingModel(
        embeddings_path="/path/to/robotwin_embeddings_1024.npz",
        index_path="/path/to/robotwin_index_1024.json",
        device="cuda:0"
    )
    text_embeds = model.compute_text({"text": ["sample text"]})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional

class EmbeddingModel(nn.Module):
    def __init__(
        self,
        embeddings_path: str,
        index_path: str,
        device: str = "cuda:0", # ⚠️注意：外部调用时一定要传对 device！
        dtype: torch.dtype = torch.float32,
        normalize_embeddings: bool = True,
        preload_to_gpu: bool = True,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        print(f"🔧 Loading Embeddings on {device}...")
        
        # 1. Load Numpy
        data = np.load(embeddings_path, allow_pickle=True)
        if 'sentence_embeddings' in data.files:
            embeddings_np = data['sentence_embeddings']
        else:
            embeddings_np = data['embeddings']
        
        # 2. Convert to Tensor (CPU first)
        embeddings_tensor = torch.from_numpy(embeddings_np.astype(np.float32))
        if normalize_embeddings:
            embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=-1)

        self.embed_layer = nn.Embedding.from_pretrained(
            embeddings_tensor, 
            freeze=True 
        )
        
        # 4. 显式移动到指定 GPU，防止挤占 GPU 0
        # ⚠️ 关键：.to() 不是 inplace 操作，需要赋值回去或使用 inplace 版本
        if preload_to_gpu:
            self.embed_layer = self.embed_layer.to(device=self.device, dtype=self.dtype)
        
        # 5. 显式禁用梯度（freeze=True 应该已经处理了，但这里双保险）
        for param in self.embed_layer.parameters():
            param.requires_grad = False
        self.eval()  # 设置为 eval 模式
        
        # 6. 设置 output_dim 属性（兼容旧代码）
        self.output_dim = self.embed_layer.embedding_dim

        # 7. 原子级对齐 embedding（可选，独立于 task 级 embedding）
        self.atomic_embed_layer = None
        self.atomic_output_dim = None

        # 8. chunk-level video embedding（可选）
        # 与 atomic 同样以 nn.Embedding 形式 frozen 存放，索引通过
        # chunk_video_idx = chunk_video_offsets[ep_idx] + frame_idx // chunk_video_size
        # 在 transforms 里离线算好后随 batch 透传进 alignment 链路。
        self.chunk_video_embed_layer = None
        self.chunk_video_dim = None
        self.chunk_video_offsets = None
        self.chunk_video_size = None
        self.chunk_video_task_indices = None

        # 可选：加载 token 级预计算特征与 mask。
        # 这里不做额外归一化，保持 Qwen3/EgoHOD 原始 token hidden states 语义。
        self.token_embeddings = None
        self.token_attention_mask = None
        self.input_ids = None
        self.has_token_features = False
        self.token_output_dim = None

        if 'token_embeddings' in data.files and 'attention_mask' in data.files:
            token_embeddings_tensor = torch.from_numpy(data['token_embeddings'].astype(np.float32))
            token_attention_mask_tensor = torch.from_numpy(data['attention_mask'].astype(np.int64))

            input_ids_tensor = None
            if 'input_ids' in data.files:
                input_ids_tensor = torch.from_numpy(data['input_ids'].astype(np.int64))

            # 自动检测 Qwen3 chat template 并 mask 掉非用户文本 token
            # Qwen3 的 token 序列包含 system prompt + user text + assistant turn，
            # 其中只有 user text 部分对 token 级对齐有意义。
            # 由于 causal attention，system prompt token 在所有句子中表征完全相同（虚假一致），
            # assistant turn token 也是固定模板，必须排除。
            if input_ids_tensor is not None:
                token_attention_mask_tensor = self._mask_chat_template_tokens(
                    input_ids_tensor, token_attention_mask_tensor
                )

            self.token_embeddings = token_embeddings_tensor
            self.token_attention_mask = token_attention_mask_tensor
            self.token_output_dim = token_embeddings_tensor.shape[-1]
            self.has_token_features = True
            self.input_ids = input_ids_tensor

            if preload_to_gpu:
                self.token_embeddings = self.token_embeddings.to(device=self.device, dtype=self.dtype)
                self.token_attention_mask = self.token_attention_mask.to(device=self.device)
                if self.input_ids is not None:
                    self.input_ids = self.input_ids.to(device=self.device)
        
        # Load Index - 构建多种形式的文本索引，提高匹配成功率
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.text_to_idx = {}
        for item in index_data:
            original_text = item['text']
            idx = item['idx']
            
            # 存储多种形式的文本作为键，提高匹配率
            self.text_to_idx[original_text] = idx                    # 原始文本
            self.text_to_idx[original_text.strip()] = idx            # 去除首尾空格
            self.text_to_idx[original_text.lower()] = idx            # 小写
            self.text_to_idx[original_text.strip().lower()] = idx    # 去除空格+小写

    def compute_text(self, text_data: Dict) -> torch.Tensor:
        """通过文本字符串查找预计算embedding（兼容旧接口）"""
        texts = text_data['text']
        
        # 1. 查找索引 (CPU 操作) - 字典已包含多种文本变体
        indices = []
        for text in texts:
            idx = self.text_to_idx.get(text)  # 先尝试精确匹配
            if idx is None:
                idx = self._find_text_index(text)  # 尝试其他变体
            indices.append(idx)
        
        current_device = self.embed_layer.weight.device
        
        # 2. 转 Tensor 并移到当前设备
        indices_tensor = torch.tensor(indices, device=current_device, dtype=torch.long)
        
        # 3. 查表 (nn.Embedding 内部处理极其高效)
        with torch.no_grad():
            return self.embed_layer(indices_tensor).detach()

    def compute_text_by_index(self, task_index: torch.Tensor) -> torch.Tensor:
        """
        通过task_index直接查表获取embedding（高效模式，跳过字符串匹配）
        
        前提：npz中的embeddings按task_index排序，即 embeddings[i] 对应 task_index=i 的文本
        
        Args:
            task_index: [B] 任务索引张量（来自dataset batch中的task_index字段）
        
        Returns:
            [B, D] 文本embedding
        """
        current_device = self.embed_layer.weight.device
        # 将-1等无效索引clamp到0（与compute_text的fallback逻辑一致）
        safe_index = task_index.clamp(min=0).to(current_device)
        with torch.no_grad():
            return self.embed_layer(safe_index).detach()

    def compute_text_tokens(self, text_data: Dict):
        texts = text_data['text']
        indices = []
        for text in texts:
            idx = self.text_to_idx.get(text)
            if idx is None:
                idx = self._find_text_index(text)
            indices.append(idx)

        indices_tensor = torch.tensor(indices, device=self.embed_layer.weight.device, dtype=torch.long)
        return self.compute_text_tokens_by_index(indices_tensor)

    def compute_text_tokens_by_index(self, task_index: torch.Tensor):
        if not self.has_token_features or self.token_embeddings is None or self.token_attention_mask is None:
            raise ValueError("Token embeddings are not available in the loaded npz file.")

        current_device = self.token_embeddings.device
        safe_index = task_index.clamp(min=0).to(current_device)
        with torch.no_grad():
            token_embeddings = self.token_embeddings.index_select(0, safe_index).detach()
            attention_mask = self.token_attention_mask.index_select(0, safe_index).detach()
        return token_embeddings, attention_mask

    def compute_text_bundle_by_index(self, task_index: torch.Tensor):
        sentence_embeddings = self.compute_text_by_index(task_index)
        token_embeddings, attention_mask = self.compute_text_tokens_by_index(task_index)
        bundle = {
            'sentence_embeddings': sentence_embeddings,
            'cls_embeddings': sentence_embeddings,
            'token_embeddings': token_embeddings,
            'attention_mask': attention_mask,
        }
        if self.input_ids is not None:
            safe_index = task_index.clamp(min=0).to(self.input_ids.device)
            bundle['input_ids'] = self.input_ids.index_select(0, safe_index).detach()
        return bundle
    
    @staticmethod
    def _mask_chat_template_tokens(
        input_ids: torch.Tensor,       # [N, T]
        attention_mask: torch.Tensor,   # [N, T]
    ) -> torch.Tensor:
        """
        检测 Qwen3 chat template 格式，将非用户文本 token 的 mask 设为 0。

        Qwen3 token 序列结构：
          [<|im_start|>] system \\n ... <|im_end|> \\n   ← system prompt
          [<|im_start|>] user \\n                        ← user turn header
          <用户文本 token>                                ← 只保留这部分
          <|im_end|> \\n [<|im_start|>] assistant \\n <eos> ← assistant turn

        检测依据：第一个 token 是否为 <|im_start|> (151644)。
        如果不是 Qwen3 格式（如 CLIP），原样返回不做任何修改。
        """
        QWEN3_IM_START = 151644
        QWEN3_IM_END = 151645

        # 不是 Qwen3 格式，直接返回
        if input_ids[0, 0].item() != QWEN3_IM_START:
            return attention_mask

        N, T = input_ids.shape
        new_mask = attention_mask.clone()
        masked_count = 0

        for i in range(N):
            ids = input_ids[i]
            # 找第二个 <|im_start|>（user turn），用户文本从它之后第3个位置开始
            im_start_positions = (ids == QWEN3_IM_START).nonzero(as_tuple=True)[0]
            if len(im_start_positions) < 2:
                continue
            # 用户文本起始 = 第二个 im_start + 3（跳过 <|im_start|> user \n）
            text_start = im_start_positions[1].item() + 3
            # 用户文本结束 = 第二个 im_start 之后的第一个 <|im_end|>
            remaining = ids[im_start_positions[1].item() + 1:]
            im_end_in_remaining = (remaining == QWEN3_IM_END).nonzero(as_tuple=True)[0]
            if len(im_end_in_remaining) == 0:
                continue
            text_end = im_start_positions[1].item() + 1 + im_end_in_remaining[0].item()

            # mask 掉 [0, text_start) 和 [text_end, T)
            new_mask[i, :text_start] = 0
            new_mask[i, text_end:] = 0
            masked_count += 1

        if masked_count > 0:
            # 统计保留的平均 token 数
            avg_kept = new_mask.sum().item() / N
            avg_total = attention_mask.sum().item() / N
            print(f"✅ Qwen3 chat template 检测成功: mask 掉 system prompt + assistant turn, "
                  f"平均保留 {avg_kept:.1f}/{avg_total:.1f} token")

        return new_mask

    def _find_text_index(self, text: str) -> int:
        """
        查找文本对应的embedding索引
        
        由于初始化时已经存储了多种形式的文本（原始、小写、strip等），
        这里只需要尝试几种常见变体即可
        
        Args:
            text: 输入文本
        
        Returns:
            idx: embedding索引
        """
        # 尝试几种常见的文本变体（字典中已预存储）
        for variant in [text, text.strip(), text.lower(), text.strip().lower()]:
            idx = self.text_to_idx.get(variant)
            if idx is not None:
                return idx
        
        # 如果都找不到，返回索引0（避免训练中断）
        print(f"⚠️ Warning: Text not found '{text[:50]}...', using index 0")
        return 0
    
    # ==================== 原子级对齐 ====================

    def load_atomic_embeddings(self, embeddings_path: str, index_path: str = None):
        """
        加载原子级 embedding（独立的 embedding 表，格式与 task embedding 一致）
        调用时机：init_alignment_model_components 中根据配置加载
        """
        data = np.load(embeddings_path, allow_pickle=True)
        if 'sentence_embeddings' in data.files:
            emb_np = data['sentence_embeddings']
        else:
            emb_np = data['embeddings']

        emb_tensor = torch.from_numpy(emb_np.astype(np.float32))
        emb_tensor = F.normalize(emb_tensor, p=2, dim=-1)

        self.atomic_embed_layer = nn.Embedding.from_pretrained(emb_tensor, freeze=True)
        self.atomic_output_dim = emb_tensor.shape[-1]

        # 移到与主 embedding 相同的设备
        device = self.embed_layer.weight.device
        self.atomic_embed_layer = self.atomic_embed_layer.to(device=device, dtype=self.dtype)
        for param in self.atomic_embed_layer.parameters():
            param.requires_grad = False

        n = emb_tensor.shape[0]
        print(f"✅ Atomic embeddings loaded: {n} labels, dim={self.atomic_output_dim}, device={device}")

    def compute_atomic_text_by_index(self, atomic_label_idx: torch.Tensor) -> torch.Tensor:
        """
        通过 atomic_label_idx 查表获取原子级 text embedding
        Args:
            atomic_label_idx: [B] 原子标签索引
        Returns:
            [B, D] 原子标签 embedding
        """
        device = self.atomic_embed_layer.weight.device
        safe_idx = atomic_label_idx.clamp(min=0).to(device)
        with torch.no_grad():
            return self.atomic_embed_layer(safe_idx).detach()

    # ==================== chunk-level video lookup ====================

    def load_chunk_video_features(self, chunk_npz_path: str):
        """
        加载 chunk-level video features（由 chunk_video_extract.py 生成）。
        npz 字段：
          embeddings           [N_chunks, D]
          episode_offsets      [N_eps+1]
          chunk_local_indices  [N_chunks]
          task_indices         [N_chunks]
          chunk_size, frames_per_chunk
        """
        data = np.load(chunk_npz_path)
        emb_np = data["embeddings"].astype(np.float32)
        emb_tensor = torch.from_numpy(emb_np)
        # video features 已在生成阶段做 L2 归一化，这里再做一次更稳
        emb_tensor = F.normalize(emb_tensor, p=2, dim=-1)
        self.chunk_video_embed_layer = nn.Embedding.from_pretrained(emb_tensor, freeze=True)
        self.chunk_video_dim = emb_tensor.shape[-1]

        device = self.embed_layer.weight.device
        self.chunk_video_embed_layer = self.chunk_video_embed_layer.to(
            device=device, dtype=self.dtype)
        for p in self.chunk_video_embed_layer.parameters():
            p.requires_grad = False

        # 元信息：transforms 侧需要这两个数组才能把 (ep, frame) → chunk_video_idx
        self.chunk_video_offsets = data["episode_offsets"].astype(np.int64)
        self.chunk_video_size = int(data["chunk_size"])
        self.chunk_video_task_indices = data["task_indices"].astype(np.int64)

        n = emb_tensor.shape[0]
        print(f"✅ Chunk video embeddings loaded: {n} chunks, dim={self.chunk_video_dim}, "
              f"chunk_size={self.chunk_video_size}, device={device}")

    def compute_chunk_video_by_index(self, chunk_video_idx: torch.Tensor) -> torch.Tensor:
        """
        通过 chunk_video_idx 查表获取 chunk-level video embedding
        Args:
            chunk_video_idx: [B] 全局 chunk 索引
        Returns:
            [B, D] frozen video embedding
        """
        device = self.chunk_video_embed_layer.weight.device
        safe = chunk_video_idx.clamp(min=0).to(device)
        with torch.no_grad():
            return self.chunk_video_embed_layer(safe).detach()

    def forward(self, text_data: Dict) -> torch.Tensor:
        """前向传播（调用compute_text）"""
        return self.compute_text(text_data)
    
    def save_checkpoint(self, save_path, **kwargs):
        """
        保存Embedding模型checkpoint
        
        注意：Embedding模型使用预计算特征，没有可训练参数，
        checkpoint实际只保存配置信息，不保存权重
        
        参数：
            save_path: 保存路径（.pth文件）
        """
        checkpoint = {
            'arch': 'EmbeddingModel',
            'output_dim': self.output_dim,
            'message': 'Embedding model uses precomputed features, no trainable parameters'
        }
        torch.save(checkpoint, save_path)
        print(f"✓ Embedding模型配置已保存到: {save_path} (无需保存权重，使用预计算特征)")
    
    def load_checkpoint(self, checkpoint_path, **kwargs):
        """
        加载Embedding模型checkpoint
        
        注意：Embedding模型使用预计算特征，加载checkpoint只是验证配置
        
        参数：
            checkpoint_path: checkpoint文件路径（.pth文件）
        """
        print(f"=> 加载Embedding checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Embedding模型配置验证完成 (无需加载权重，使用预计算特征)")


def create_embedding_model(
    embeddings_path: str,
    index_path: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
    normalize_embeddings: bool = True,
    preload_to_gpu: bool = True,  # 新增：是否预加载到GPU（默认True）
) -> EmbeddingModel:
    """
    创建EmbeddingModel实例的工厂函数
    
    Args:
        embeddings_path: 嵌入特征文件路径
        index_path: 文本索引文件路径
        device: 设备
        dtype: 数据类型
        normalize_embeddings: 是否L2归一化
        preload_to_gpu: 是否预加载到GPU（推荐True，显存使用更稳定）
    
    Returns:
        EmbeddingModel实例
    """
    return EmbeddingModel(
        embeddings_path=embeddings_path,
        index_path=index_path,
        device=device,
        dtype=dtype,
        normalize_embeddings=normalize_embeddings,
        preload_to_gpu=preload_to_gpu
    )
