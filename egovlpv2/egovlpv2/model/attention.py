"""
attention.py

Common attention modules used in alignment models.
This module provides reusable attention blocks for feature alignment:
1. CrossAttentionBlock: Standard cross-attention with residual connections
2. AlignedCrossAttentionBlock: Cross-attention variant without residual on attention output
3. SelfAttentionBlock: Self-attention for CLS token + action features processing
"""

import torch
import torch.nn as nn
from typing import Optional


class CrossAttentionBlock(nn.Module):
    """
    A single cross-attention block with full transformer components.
    Query: EgoVLPv2 features [B, 1, D]
    Key/Value: OpenVLA features [B, A, D]
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,  # [B, 1, D]
        key_value: torch.Tensor,  # [B, A, D]
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of cross-attention block."""
        
        # Cross-attention with residual connection
        attn_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask
        )
        query = self.norm1(query + self.dropout(attn_output))
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output  # [B, 1, D]


class AlignedCrossAttentionBlock(nn.Module):
    """
    A single cross-attention block with full transformer components.
    Query: EgoVLPv2 features [B, 1, D]
    Key/Value: OpenVLA features [B, A, D]
    
    与 CrossAttentionBlock 的区别:
    - 这个版本在 attention 输出上不使用残差连接
    - 只在 FFN 输出上使用残差连接
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,  # [B, 1, D]
        key_value: torch.Tensor,  # [B, A, D]
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of cross-attention block."""
        
        # Cross-attention with residual connection
        attn_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask
        )
        aligned_vla_feature = self.norm1(self.dropout(attn_output))
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(aligned_vla_feature)
        output = self.norm2(aligned_vla_feature + ffn_output)
        
        return output  # [B, 1, D]


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for CLS token + action features processing.
    
    输入: concat后的序列 [B, 1+A, D] 其中第一个token是CLS，其余是action features
    输出: 相同维度的序列 [B, 1+A, D] 其中CLS token包含了从action features聚合的信息
    
    与CrossAttentionBlock的区别:
    - CrossAttentionBlock: query从text来，key/value从action来 (跨模态注意力)
    - SelfAttentionBlock: 所有token(CLS+Action)之间相互注意 (自注意力)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        # 注意：这里是self-attention，所以query、key、value都来自同一个输入序列
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network - 与CrossAttentionBlock保持一致的结构
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization - 标准Transformer架构的norm层
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        sequence: torch.Tensor,  # [B, 1+A, D] CLS + action features
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Self-attention forward pass.
        
        Args:
            sequence: 输入序列 [B, 1+A, D]，第一个位置是CLS token，其余是action features
            attn_mask: 可选的注意力mask
            
        Returns:
            output: 输出序列 [B, 1+A, D]，CLS token(第一个位置)包含聚合后的信息
        """
        
        # Self-attention with residual connection
        # 在self-attention中，query、key、value都是同一个输入sequence
        attn_output, _ = self.self_attention(
            query=sequence,      # [B, 1+A, D] 
            key=sequence,        # [B, 1+A, D]
            value=sequence,      # [B, 1+A, D]
            attn_mask=attn_mask
        )
        # 残差连接 + LayerNorm (Post-Norm模式，与CrossAttentionBlock保持一致)
        sequence = self.norm1(sequence + self.dropout(attn_output))
        
        # Feed-forward network with residual connection  
        ffn_output = self.ffn(sequence)
        output = self.norm2(sequence + ffn_output)
        
        return output  # [B, 1+A, D]，CLS token在第一个位置














