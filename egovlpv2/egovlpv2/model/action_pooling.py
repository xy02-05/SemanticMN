"""
action_pooling.py — Action Token Pooling: [B, A, D] → [B, D]

模式: mean | mean_mlp | learnable_query | target_attention
"""

import torch
import torch.nn as nn
from typing import Optional


class ActionPooler(nn.Module):

    def __init__(
        self,
        mode: str = 'mean',
        action_dim: int = 4096,
        target_dim: int = None,
        num_heads: int = 8,
        num_layers: int = 1,
        mlp_hidden_dim: int = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.mode = mode

        if mode == 'mean':
            pass

        elif mode == 'mean_mlp':
            hidden = mlp_hidden_dim or action_dim
            self.mlp = nn.Sequential(
                nn.Linear(action_dim, hidden),
                nn.LayerNorm(hidden, eps=layer_norm_eps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, action_dim),
            )

        elif mode == 'learnable_query':
            self.query = nn.Parameter(torch.randn(1, 1, action_dim) * 0.02)
            hidden = mlp_hidden_dim or action_dim
            self.attn_layers = nn.ModuleList()
            self.attn_norms = nn.ModuleList()
            self.ffn_layers = nn.ModuleList()
            self.ffn_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.attn_layers.append(
                    nn.MultiheadAttention(
                        action_dim, num_heads,
                        dropout=dropout, batch_first=True
                    )
                )
                self.attn_norms.append(nn.LayerNorm(action_dim, eps=layer_norm_eps))
                self.ffn_layers.append(nn.Sequential(
                    nn.Linear(action_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, action_dim),
                    nn.Dropout(dropout),
                ))
                self.ffn_norms.append(nn.LayerNorm(action_dim, eps=layer_norm_eps))

        elif mode == 'target_attention':
            _target_dim = target_dim or action_dim
            if _target_dim != action_dim:
                self.target_proj = nn.Linear(_target_dim, action_dim)
            else:
                self.target_proj = nn.Identity()
            self.attn = nn.MultiheadAttention(
                action_dim, num_heads,
                dropout=dropout, batch_first=True
            )

        else:
            raise ValueError(f"不支持的 action_pool_mode: {mode}")

    def forward(
        self,
        local_action_features: torch.Tensor,       # [B, A, D_action]
        target_features: Optional[torch.Tensor] = None,  # [B, D_target]
    ) -> torch.Tensor:                              # [B, D_action]

        if self.mode == 'mean':
            return local_action_features.mean(dim=1)

        elif self.mode == 'mean_mlp':
            return self.mlp(local_action_features.mean(dim=1))

        elif self.mode == 'learnable_query':
            B = local_action_features.shape[0]
            query = self.query.expand(B, -1, -1)
            for attn, attn_norm, ffn, ffn_norm in zip(
                self.attn_layers, self.attn_norms, self.ffn_layers, self.ffn_norms
            ):
                attn_out, _ = attn(query, local_action_features, local_action_features)
                query = attn_norm(query + attn_out)
                query = ffn_norm(query + ffn(query))
            return query.squeeze(1)

        elif self.mode == 'target_attention':
            query = self.target_proj(target_features).unsqueeze(1)  # [B, 1, D_action]
            output, _ = self.attn(query, local_action_features, local_action_features)
            return output.squeeze(1)  # [B, D_action]


def create_action_pooler(
    mode: str = 'mean',
    action_dim: int = 4096,
    target_dim: int = None,
    config: dict = None,
) -> ActionPooler:
    cfg = config or {}
    return ActionPooler(
        mode=mode,
        action_dim=action_dim,
        target_dim=target_dim,
        num_heads=cfg.get('num_heads', 8),
        num_layers=cfg.get('num_layers', 1),
        mlp_hidden_dim=cfg.get('mlp_hidden_dim', None),
        dropout=cfg.get('dropout', 0.1),
        layer_norm_eps=cfg.get('layer_norm_eps', 1e-5),
    )
