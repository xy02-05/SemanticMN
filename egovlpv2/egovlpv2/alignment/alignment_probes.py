"""
Alignment Probes - 对齐探测模型

用于训练简单的对齐层验证VLA-VLM跨域迁移能力。
冻结EgoHOD和SpatialVLA，只训练投影层。

核心思想：
- 如果简单的线性/MLP投影就能对齐 → 说明天然对齐存在
- 训练这些probes可以量化对齐的难度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .action_pooler import ActionPooler
from .cka_utils import linear_cka


class AlignmentProbe(nn.Module):
    """
    单层对齐探测模型
    
    功能:
    1. Action pooling (mean/attn)
    2. 投影到公共空间 (FC/MLP)
    3. 对比学习loss计算
    4. CKA指标计算
    """
    
    def __init__(
        self,
        action_dim: int,
        text_dim: int,
        projection_dim: int = 512,
        pooling_type: str = 'mean_pool',
        projection_type: str = 'fc',  # 'fc' or 'mlp_2'
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        """
        Args:
            action_dim: VLA action token维度
            text_dim: EgoHOD text特征维度
            projection_dim: 公共投影空间维度
            pooling_type: action pooling方式
            projection_type: 投影方式 ('fc' 或 'mlp_2')
            temperature: 对比学习温度
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.text_dim = text_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Action pooler
        self.pooler = ActionPooler(
            hidden_dim=action_dim,
            pooling_type=pooling_type,
            dropout=dropout,
        )
        
        # Action投影层
        if projection_type == 'fc':
            self.action_proj = nn.Linear(action_dim, projection_dim)
        else:  # mlp_2
            self.action_proj = nn.Sequential(
                nn.Linear(action_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
            )
        
        # Text投影层
        if projection_type == 'fc':
            self.text_proj = nn.Linear(text_dim, projection_dim)
        else:  # mlp_2
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
            )
    
    def forward(
        self,
        action_features: torch.Tensor,
        text_features: torch.Tensor,
        compute_cka: bool = True,
        return_features: bool = False,  # 是否返回投影后特征（用于评测检索）
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            action_features: [B, A, D_action] action token序列
            text_features: [B, D_text] text features
            compute_cka: 是否计算CKA指标
            return_features: 是否返回投影后特征（用于eval时计算检索指标）
            
        Returns:
            Dict包含:
            - loss: 对比学习loss
            - cka_before: 投影前CKA（如果compute_cka=True）
            - cka_after: 投影后CKA（如果compute_cka=True）
            - action_proj: [B, D] 投影后action（如果return_features=True）
            - text_proj: [B, D] 投影后text（如果return_features=True）
        """
        B = action_features.shape[0]
        
        # 1. Action pooling: [B, A, D] -> [B, D]
        pooled_action = self.pooler(action_features)
        
        # 2. 投影到公共空间
        action_proj = self.action_proj(pooled_action)  # [B, proj_dim]
        text_proj = self.text_proj(text_features)      # [B, proj_dim]
        
        # 3. L2归一化
        action_norm = F.normalize(action_proj, dim=-1)
        text_norm = F.normalize(text_proj, dim=-1)
        
        # 4. 计算对比学习loss (InfoNCE)
        # 相似度矩阵: [B, B]
        sim_matrix = action_norm @ text_norm.T / self.temperature
        
        # 对角线为正样本
        labels = torch.arange(B, device=sim_matrix.device)
        loss_a2t = F.cross_entropy(sim_matrix, labels)
        loss_t2a = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_a2t + loss_t2a) / 2
        
        result = {'loss': loss}
        
        # 5. 计算CKA（可选）
        if compute_cka:
            with torch.no_grad():
                # 投影前CKA
                cka_before = linear_cka(pooled_action, text_features)
                # 投影后CKA
                cka_after = linear_cka(action_proj, text_proj)
                
            result['cka_before'] = cka_before
            result['cka_after'] = cka_after
        
        # 6. 返回投影后特征（用于评测）
        if return_features:
            result['action_proj'] = action_proj
            result['text_proj'] = text_proj
        
        return result


class MultiLayerAlignmentProbe(nn.Module):
    """
    多层对齐探测模型
    
    同时对VLA的多层进行alignment探测，
    一次实验获得所有层的对齐效果。
    """
    
    def __init__(
        self,
        action_dim: int,
        text_dim: int,
        num_layers: int,
        projection_dim: int = 512,
        pooling_types: list = None,  # 每层要尝试的pooling类型（每层都用同一组）
        projection_type: str = 'fc',
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_layers: 要探测的层数
            pooling_types: 每层要尝试的pooling类型列表，如 ['mean_pool', 'attn_pool_1', ...]
                           如果为None，默认只用mean_pool
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.pooling_types = pooling_types or ['mean_pool']
        
        # 为每层、每种pooling创建独立的probe（层数 × 方法数）
        self.probes = nn.ModuleList()
        for _ in range(num_layers):
            layer_probes = nn.ModuleList()
            for pooling_type in self.pooling_types:
                layer_probes.append(
                    AlignmentProbe(
                        action_dim=action_dim,
                        text_dim=text_dim,
                        projection_dim=projection_dim,
                        pooling_type=pooling_type,
                        projection_type=projection_type,
                        temperature=temperature,
                        dropout=dropout,
                    )
                )
            self.probes.append(layer_probes)
    
    def forward(
        self,
        vla_features: torch.Tensor,
        text_features: torch.Tensor,
        layer_indices: list = None,
        compute_cka: bool = True,
        return_features: bool = False,  # 是否返回投影后特征（用于评测检索）
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vla_features: [B, L, A, D] 多层VLA特征
            text_features: [B, D_text] text特征
            layer_indices: 层索引（用于命名）
            return_features: 是否返回投影后特征（用于eval时计算检索指标）
            
        Returns:
            Dict包含每层的loss和CKA指标
            如果return_features=True，还包含:
            - action_proj_dict: {f'{layer}_{pool}': [B, D]}
            - text_proj_dict: {f'{layer}_{pool}': [B, D]}
        """
        B, L, A, D = vla_features.shape
        
        if layer_indices is None:
            layer_indices = list(range(L))
        
        # 初始化loss为tensor，避免空循环导致backward失败
        total_loss = torch.tensor(0.0, device=vla_features.device)
        total_count = 0
        results = {}
        action_proj_dict = {}  # 用于eval
        text_proj_dict = {}    # 用于eval
        
        for layer_idx in range(min(L, self.num_layers)):
            # 提取当前层特征
            layer_feat = vla_features[:, layer_idx, :, :]  # [B, A, D]
            
            actual_idx = layer_indices[layer_idx] if layer_idx < len(layer_indices) else layer_idx
            
            # 每层对所有pooling方式逐个计算
            for pool_idx, probe in enumerate(self.probes[layer_idx]):
                pooling_type = self.pooling_types[pool_idx]
                layer_result = probe(
                    action_features=layer_feat,
                    text_features=text_features,
                    compute_cka=compute_cka,
                    return_features=return_features,
                )
                
                # 累加loss
                total_loss = total_loss + layer_result['loss']
                total_count += 1
                
                # 记录每层+pooling指标
                results[f'loss_layer_{actual_idx}_{pooling_type}'] = layer_result['loss']
                
                if compute_cka:
                    results[f'cka_before_layer_{actual_idx}_{pooling_type}'] = layer_result['cka_before']
                    results[f'cka_after_layer_{actual_idx}_{pooling_type}'] = layer_result['cka_after']
                
                # 收集投影后特征（用于eval检索）
                if return_features:
                    key = f'{actual_idx}_{pooling_type}'
                    action_proj_dict[key] = layer_result['action_proj']
                    text_proj_dict[key] = layer_result['text_proj']
        
        # 平均loss（考虑层数 × 方法数）
        results['loss'] = total_loss / max(1, total_count)
        
        # 返回投影后特征字典
        if return_features:
            results['action_proj_dict'] = action_proj_dict
            results['text_proj_dict'] = text_proj_dict
        
        return results
