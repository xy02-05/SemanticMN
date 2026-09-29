"""
Alignment Evaluation - 对齐评测模块

用于评估VLA-VLM对齐效果的检索准确率:
1. 收集action和text特征
2. 计算最近邻检索 (action->text, text->action)
3. 计算top-k准确率 (top1, top5, top10)
4. 保存检索结果

核心指标: 对于每个action，找最近的text，检查是否为同一个样本
"""

import torch
import torch.nn.functional as F
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_retrieval_topk(
    action_features: torch.Tensor,  # [N, D] 投影后的action特征
    text_features: torch.Tensor,    # [N, D] 投影后的text特征
    k_values: List[int] = [1, 5, 10, 20],
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    计算检索top-k准确率
    
    原理:
    - action和text来自同一个样本，对角线为正样本
    - action[i]的ground truth text是text[i]
    - 计算cosine相似度，检查top-k是否包含正样本
    
    Args:
        action_features: [N, D] action嵌入
        text_features: [N, D] text嵌入
        k_values: 要计算的k值列表
        
    Returns:
        metrics: {f'a2t_top{k}': acc, f't2a_top{k}': acc}
        a2t_indices: [N, max_k] 每个action的topk text索引
        t2a_indices: [N, max_k] 每个text的topk action索引
    """
    N = action_features.shape[0]
    max_k = max(k_values)
    
    # L2归一化
    action_norm = F.normalize(action_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)
    
    # 相似度矩阵 [N, N]
    sim_a2t = action_norm @ text_norm.T  # action到text的相似度
    sim_t2a = sim_a2t.T  # text到action的相似度
    
    # 获取top-k索引
    _, a2t_indices = sim_a2t.topk(max_k, dim=1)  # [N, max_k]
    _, t2a_indices = sim_t2a.topk(max_k, dim=1)  # [N, max_k]
    
    # 正确标签 = 对角线
    labels = torch.arange(N, device=action_features.device)
    
    metrics = {}
    for k in k_values:
        # action->text: 检查topk中是否包含正确的text
        a2t_correct = (a2t_indices[:, :k] == labels.unsqueeze(1)).any(dim=1)
        a2t_acc = a2t_correct.float().mean().item()
        metrics[f'a2t_top{k}'] = a2t_acc
        
        # text->action: 检查topk中是否包含正确的action
        t2a_correct = (t2a_indices[:, :k] == labels.unsqueeze(1)).any(dim=1)
        t2a_acc = t2a_correct.float().mean().item()
        metrics[f't2a_top{k}'] = t2a_acc
    
    return metrics, a2t_indices[:, :20], t2a_indices[:, :20]


def format_retrieval_results(
    a2t_indices: torch.Tensor,       # [N, K] action的topk text索引
    t2a_indices: torch.Tensor,       # [N, K] text的topk action索引
    prompts: List[str],              # [N] 每个样本的prompt
    timesteps: Optional[List[int]] = None,  # [N] 每个样本的timestep
    progresses: Optional[List[float]] = None,  # [N] 任务进度(0-1)占位信息
) -> Dict:
    """
    格式化检索结果用于保存
    
    Args:
        a2t_indices: [N, K] 每个action的top-k text索引
        t2a_indices: [N, K] 每个text的top-k action索引
        prompts: 每个样本的prompt文本
        timesteps: 每个样本在轨迹中的timestep
        
    Returns:
        result dict，便于json保存
    """
    N = len(prompts)
    
    # action -> text 检索结果
    a2t_results = []
    for i in range(N):
        item = {
            'action_idx': i,
            'action_prompt': prompts[i],
            'action_timestep': timesteps[i] if timesteps else None,
            'action_progress': progresses[i] if progresses else None,
            'topk_texts': [
                {
                    'rank': r + 1,
                    'text_idx': int(idx),
                    'text_prompt': prompts[idx],
                    'is_correct': int(idx) == i,
                }
                for r, idx in enumerate(a2t_indices[i].tolist())
            ]
        }
        a2t_results.append(item)
    
    # text -> action 检索结果
    t2a_results = []
    for i in range(N):
        item = {
            'text_idx': i,
            'text_prompt': prompts[i],
            'topk_actions': [
                {
                    'rank': r + 1,
                    'action_idx': int(idx),
                    'action_prompt': prompts[idx],
                    'action_timestep': timesteps[idx] if timesteps else None,
                    'action_progress': progresses[idx] if progresses else None,
                    'is_correct': int(idx) == i,
                }
                for r, idx in enumerate(t2a_indices[i].tolist())
            ]
        }
        t2a_results.append(item)
    
    return {
        'num_samples': N,
        'action_to_text': a2t_results,
        'text_to_action': t2a_results,
    }


def save_retrieval_results(
    results: Dict,
    save_path: str,
    layer_idx: int,
    pooling_type: str,
    step: int,
):
    """
    保存检索结果到JSON文件
    
    Args:
        results: format_retrieval_results的输出
        save_path: 保存目录
        layer_idx: 层索引
        pooling_type: pooling类型
        step: 训练步数
    """
    os.makedirs(save_path, exist_ok=True)
    filename = f'retrieval_layer{layer_idx}_{pooling_type}_step{step}.json'
    filepath = os.path.join(save_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return filepath


class EvalFeatureCollector:
    """
    评测特征收集器
    
    逐batch收集action/text特征，用于后续计算检索指标
    """
    
    def __init__(self, num_layers: int, pooling_types: List[str]):
        """
        Args:
            num_layers: 层数
            pooling_types: pooling类型列表
        """
        self.num_layers = num_layers
        self.pooling_types = pooling_types
        self.reset()
    
    def reset(self):
        """重置收集器"""
        # 每层每种pooling方式分别存储
        # features[layer_idx][pooling_type] = {'action': [], 'text': []}
        self.features = defaultdict(lambda: defaultdict(lambda: {'action': [], 'text': []}))
        self.prompts = []
        self.timesteps = []
        self.progresses = []
    
    def add_batch(
        self,
        action_proj_dict: Dict[str, torch.Tensor],  # {f'{layer}_{pool}': [B, D]}
        text_proj_dict: Dict[str, torch.Tensor],    # {f'{layer}_{pool}': [B, D]}
        prompts: List[str],
        timesteps: Optional[List[int]] = None,
        progresses: Optional[List[float]] = None,
    ):
        """
        添加一个batch的特征
        
        Args:
            action_proj_dict: 投影后的action特征字典
            text_proj_dict: 投影后的text特征字典
            prompts: batch的prompt列表
            timesteps: batch的timestep列表
        """
        self.prompts.extend(prompts)
        if timesteps:
            self.timesteps.extend(timesteps)
        if progresses:
            self.progresses.extend(progresses)
        
        for key in action_proj_dict:
            # key格式: f'{layer_idx}_{pooling_type}'
            parts = key.split('_', 1)
            if len(parts) == 2:
                layer_idx, pool_type = int(parts[0]), parts[1]
            else:
                continue
            
            # 存储detach后的CPU tensor
            self.features[layer_idx][pool_type]['action'].append(
                action_proj_dict[key].detach().cpu()
            )
            self.features[layer_idx][pool_type]['text'].append(
                text_proj_dict[key].detach().cpu()
            )
    
    def compute_all_metrics(
        self,
        k_values: List[int] = [1, 5, 10, 20],
        save_dir: Optional[str] = None,
        step: int = 0,
    ) -> Dict[str, float]:
        """
        计算所有层和pooling方式的检索指标
        
        Returns:
            metrics: {f'layer{i}_{pool}_a2t_top{k}': acc, ...}
        """
        all_metrics = {}
        
        for layer_idx in sorted(self.features.keys()):
            for pool_type in self.features[layer_idx]:
                # 拼接所有batch
                action_feats = torch.cat(
                    self.features[layer_idx][pool_type]['action'], dim=0
                )
                text_feats = torch.cat(
                    self.features[layer_idx][pool_type]['text'], dim=0
                )
                
                # 计算检索指标
                metrics, a2t_indices, t2a_indices = compute_retrieval_topk(
                    action_feats, text_feats, k_values
                )
                
                # 添加前缀
                for k, v in metrics.items():
                    all_metrics[f'layer{layer_idx}_{pool_type}_{k}'] = v
                
                # 保存检索结果
                if save_dir:
                    results = format_retrieval_results(
                        a2t_indices,
                        t2a_indices,
                        self.prompts[:len(action_feats)],
                        self.timesteps[:len(action_feats)] if self.timesteps else None,
                        self.progresses[:len(action_feats)] if self.progresses else None,
                    )
                    save_retrieval_results(
                        results, save_dir, layer_idx, pool_type, step
                    )
        
        return all_metrics
    
    @property
    def num_samples(self) -> int:
        """当前收集的样本数"""
        return len(self.prompts)
