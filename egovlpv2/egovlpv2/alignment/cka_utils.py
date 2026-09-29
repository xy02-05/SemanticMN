"""
CKA (Centered Kernel Alignment) 计算工具

用于验证VLA和VLM之间的天然对齐性。
CKA无需训练即可测量两个表征空间的相似度。

核心公式:
CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

其中K = X @ X.T, L = Y @ Y.T
"""

import torch
from typing import Dict, List, Optional


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    计算Linear CKA相似度
    
    Args:
        X: [N, D1] 第一组特征（如action features）
        Y: [N, D2] 第二组特征（如text features）
        
    Returns:
        CKA score (标量, 范围[0, 1])
    
    说明:
    - CKA对线性变换不变，即CKA(X, AX) ≈ 1
    - 值越接近1，两个表征空间越相似
    - 适合验证"天然对齐"假设
    """
    # 中心化: 减去均值使特征零均值
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # 使用Frobenius范数技巧计算HSIC（避免大矩阵）
    # HSIC(X,Y) ∝ ||Y^T X||_F^2
    XTX = X.T @ X  # [D1, D1]
    YTY = Y.T @ Y  # [D2, D2]
    YTX = Y.T @ X  # [D2, D1]
    
    hsic_xy = (YTX ** 2).sum()
    hsic_xx = (XTX ** 2).sum()
    hsic_yy = (YTY ** 2).sum()
    
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)
    
    return cka


def compute_cka_per_layer(
    vla_features: torch.Tensor,
    text_features: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    逐层计算VLA与text features的CKA
    
    Args:
        vla_features: [B, L, A, D_vla] VLA隐藏状态，L是层数，A是action token数
        text_features: [B, D_text] 文本特征
        layer_indices: 实际的层索引列表（用于命名）
        
    Returns:
        Dict: 每层的CKA score, 如 {'cka_layer_5': 0.73, 'cka_layer_10': 0.81}
    
    说明:
    - 自动对action tokens做mean pooling
    - 不需要梯度（用于分析）
    """
    B, num_layers, num_actions, D_vla = vla_features.shape
    
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    
    cka_scores = {}
    
    with torch.no_grad():
        for layer_idx in range(num_layers):
            # 提取当前层特征并mean pool
            layer_feat = vla_features[:, layer_idx, :, :]  # [B, A, D]
            global_action = layer_feat.mean(dim=1)  # [B, D]
            
            # 计算CKA
            cka_score = linear_cka(global_action, text_features)
            
            # 使用实际层索引命名
            actual_idx = layer_indices[layer_idx] if layer_idx < len(layer_indices) else layer_idx
            cka_scores[f'cka_layer_{actual_idx}'] = cka_score.item()
    
    return cka_scores


def compute_cka_with_pooling(
    vla_features: torch.Tensor,
    text_features: torch.Tensor,
    pooler: 'ActionPooler',
    layer_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    使用指定的pooler计算逐层CKA
    
    Args:
        vla_features: [B, L, A, D]
        text_features: [B, D_text]
        pooler: ActionPooler实例
        layer_indices: 层索引
    """
    B, num_layers, num_actions, D = vla_features.shape
    
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    
    cka_scores = {}
    
    with torch.no_grad():
        for layer_idx in range(num_layers):
            layer_feat = vla_features[:, layer_idx, :, :]  # [B, A, D]
            global_action = pooler(layer_feat)  # [B, D]
            
            cka_score = linear_cka(global_action, text_features)
            
            actual_idx = layer_indices[layer_idx] if layer_idx < len(layer_indices) else layer_idx
            cka_scores[f'cka_layer_{actual_idx}'] = cka_score.item()
    
    return cka_scores
