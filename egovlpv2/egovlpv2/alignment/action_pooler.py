"""
Action Token Pooling模块

将action token序列 [B, A, D] 池化为全局action特征 [B, D]

支持三种pooling方式:
- mean_pool: 简单均值池化（无可学习参数）
- attn_pool_1: 1层cross-attention + learnable query
- attn_pool_2: 2层cross-attention
"""

import torch
import torch.nn as nn


class ActionPooler(nn.Module):
    """
    Action Token Pooling模块
    
    原理:
    - mean_pool: 直接对所有token取平均，最简单
    - attn_pool: 使用learnable query通过attention加权聚合token
      能学习哪些token更重要
    """
    
    def __init__(
        self,
        hidden_dim: int,
        pooling_type: str = 'mean_pool',
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: action token维度
            pooling_type: 'mean_pool' | 'attn_pool_1' | 'attn_pool_2'
            num_heads: attention头数
            dropout: dropout率
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        
        if pooling_type == 'attn_pool_1':
            # 1层attention: learnable query + cross-attention
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads,
                dropout=dropout, 
                batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
            
        elif pooling_type == 'attn_pool_2':
            # 2层attention
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.attn1 = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads,
                dropout=dropout, 
                batch_first=True
            )
            self.attn2 = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads,
                dropout=dropout, 
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.norm3 = nn.LayerNorm(hidden_dim)  # FFN的Post-LN
            
            # FFN层增加表达能力
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
    
    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_tokens: [B, A, D] action token序列
            
        Returns:
            [B, D] 池化后的全局特征
        """
        B = action_tokens.shape[0]
        
        if self.pooling_type == 'mean_pool':
            return action_tokens.mean(dim=1)
        
        elif self.pooling_type == 'attn_pool_1':
            query = self.query.expand(B, -1, -1)  # [B, 1, D]
            attn_out, _ = self.attn(query, action_tokens, action_tokens)
            out = self.norm(query + attn_out)
            return out.squeeze(1)
        
        elif self.pooling_type == 'attn_pool_2':
            query = self.query.expand(B, -1, -1)
            
            # 第1层attention (Post-LN)
            attn_out1, _ = self.attn1(query, action_tokens, action_tokens)
            query = self.norm1(query + attn_out1)
            
            # 第2层attention (Post-LN)
            attn_out2, _ = self.attn2(query, action_tokens, action_tokens)
            out = self.norm2(query + attn_out2)
            
            # FFN (Post-LN)
            out = self.norm3(out + self.ffn(out))
            
            return out.squeeze(1)
        
        else:
            # 默认mean pool
            return action_tokens.mean(dim=1)


def create_pooler(pooling_type: str, hidden_dim: int, **kwargs) -> ActionPooler:
    """工厂函数创建pooler"""
    return ActionPooler(
        hidden_dim=hidden_dim,
        pooling_type=pooling_type,
        **kwargs
    )
