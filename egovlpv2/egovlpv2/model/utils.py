"""
utils.py - Alignment Model Utilities Mixin

AlignmentModel utility functions and helper methods:
- Weight initialization
- Temperature parameter access

Simplified version: removed attention blocks and learnable temperature support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AlignmentUtilsMixin:
    """
    Alignment Model utilities Mixin

    Provides the following features:
    1. Weight initialization (_init_weights)
    2. Get temperature parameter (_get_temperature)
    """

    def _init_weights(self, module):
        """
        Initialize weights with a more sophisticated strategy, inspired by common practices in
        Transformer-based models like BERT and ViT.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias, 0.)
            if module.out_proj.weight is not None:
                nn.init.trunc_normal_(module.out_proj.weight, std=.01)
            if module.out_proj.bias is not None:
                nn.init.constant_(module.out_proj.bias, 0.)

    def _get_temperature(self) -> torch.Tensor:
        """
        获取温度参数（支持可学习温度）
        
        Returns:
            temperature: 温度值（torch.Tensor类型，保持类型一致性）
        """
        if self.learnable_temperature:
            # 温度标量始终用float32计算，避免在混精下退化成bf16/fp16导致更新不稳定。
            return torch.exp(self.log_temperature.float())
        else:
            # 直接返回预创建的buffer，避免每次调用都创建新Tensor
            return self._temperature_buffer.float()
