"""
progress_encoding.py

任务进度编码模块 - Task Progress Encoding

为action sequence添加可学习的进度编码，让模型感知任务执行阶段（初期/中期/后期）

核心思想：参考Transformer的Position Encoding，为每个action位置添加可学习的embedding
- action[0]: 任务起点 (progress=0.0)
- action[1]: 任务进行中 (progress=0.33)
- action[2]: 任务进行中 (progress=0.67)
- action[3]: 任务终点 (progress=1.0)

使用方式：
    progress_enc = ProgressEmbedding(num_positions=4, hidden_dim=512)
    openvla_layer = openvla_layer + progress_enc(batch_size, device)
"""

import torch
import torch.nn as nn


class ProgressEmbedding(nn.Module):
    """
    可学习的任务进度编码（Progress Encoding）
    
    为action sequence中的每个位置（时间步）学习独立的embedding，
    让模型能够区分任务的不同执行阶段。
    
    设计特点：
    1. **可学习**：不使用固定的sin/cos编码，而是让模型自适应学习最优表示
    2. **轻量级**：每层仅增加 num_positions * hidden_dim 个参数（如4*512=2K）
    3. **Residual融合**：通过加法融合到action特征，初始影响很小，逐步学习
    
    Args:
        num_positions (int): action sequence的长度（如4，表示当前+3个未来action）
        hidden_dim (int): 特征维度（需要与action特征维度一致）
        dropout (float): dropout率（默认0.0，训练初期建议不使用dropout避免影响收敛）
    """
    
    def __init__(
        self,
        num_positions: int,
        hidden_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_positions = num_positions
        self.hidden_dim = hidden_dim
        
        # 可学习的position embedding table
        # 每个position有独立的embedding向量
        self.embedding = nn.Embedding(num_positions, hidden_dim)
        
        # 可选的dropout层（默认不使用）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 初始化策略：使用小值，避免对原始特征影响过大
        # 参考BERT/GPT的实践，std=0.02确保初始阶段编码贡献较小
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        生成进度编码
        
        Args:
            batch_size: 当前batch的大小
            device: 目标设备（'cuda' or 'cpu'）
        
        Returns:
            progress_embeds: [1, num_positions, hidden_dim]
                           可通过广播机制加到 [B, num_positions, hidden_dim]
        """
        # 创建position索引 [0, 1, 2, ..., num_positions-1]
        positions = torch.arange(self.num_positions, device=device)  # [A]
        
        # 查表得到embedding
        embeds = self.embedding(positions)  # [A, D]
        
        # 可选的dropout（训练时生效）
        if self.dropout is not None:
            embeds = self.dropout(embeds)
        
        # 添加batch维度，方便广播
        # [A, D] -> [1, A, D]，可自动广播到 [B, A, D]
        return embeds.unsqueeze(0)
    
    def extra_repr(self) -> str:
        """
        打印模块信息（用于调试）
        """
        return f'num_positions={self.num_positions}, hidden_dim={self.hidden_dim}'

