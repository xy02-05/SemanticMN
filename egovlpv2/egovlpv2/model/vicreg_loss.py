"""
VICReg 跨模态对齐损失。

公式与默认系数复用 Meta 官方实现：
https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

在本项目中，两路增强视图被替换为动作表征与指令表征：
1. invariance：diagonal 模式拉近同一样本投影，ID 模式拉近同 ID 的全部正配对；
2. variance：分别保证两侧每个维度有足够方差，避免整体坍缩；
3. covariance：分别去除两侧维度间冗余，提升表征容量。

该损失不构造负样本，因此不会把上下文相关或物理上相近的机器人动作
错误地当作负样本推开。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class FullGatherLayer(torch.autograd.Function):
    """VICReg 官方的可微 all-gather，反向时汇总各进程梯度。"""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(output, tensor)
        return tuple(output)

    @staticmethod
    def backward(ctx, *gradients):
        all_gradients = torch.stack(gradients)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_batch(tensor: torch.Tensor) -> torch.Tensor:
    """沿 batch 维合并全部进程，保持与官方 VICReg 相同的梯度语义。"""
    return torch.cat(FullGatherLayer.apply(tensor), dim=0)


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """返回方阵的全部非对角元素，与 VICReg 官方实现一致。"""
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"covariance matrix must be square, got {matrix.shape}")
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """计算 action/text 两个表征分支之间的 VICReg 损失。"""

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        variance_target: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.variance_target = variance_target
        self.eps = eps

    def forward(
        self,
        action_features: torch.Tensor,
        text_features: torch.Tensor,
        valid_mask: torch.Tensor = None,
        gather_distributed: bool = True,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if action_features.shape != text_features.shape:
            raise ValueError(
                "VICReg requires paired action/text features with identical shape, "
                f"got {action_features.shape} and {text_features.shape}"
            )
        if action_features.ndim != 2:
            raise ValueError(
                f"VICReg expects [batch, dim] features, got {action_features.shape}"
            )
        if valid_mask is None:
            valid_mask = torch.ones(
                action_features.shape[0],
                device=action_features.device,
                dtype=torch.bool,
            )
        if valid_mask.shape != (action_features.shape[0],):
            raise ValueError(
                f"valid_mask must have shape [{action_features.shape[0]}], "
                f"got {valid_mask.shape}"
            )
        if sample_ids is not None and sample_ids.shape != (action_features.shape[0],):
            raise ValueError(
                f"sample_ids must have shape [{action_features.shape[0]}], "
                f"got {sample_ids.shape}"
            )

        # 统计量统一用 float32，避免 bf16 下方差和协方差精度不足。
        action = action_features.float()
        text = text_features.float()
        valid_mask = valid_mask.bool()
        if gather_distributed and dist.is_available() and dist.is_initialized():
            action = gather_batch(action)
            text = gather_batch(text)
            valid_mask = gather_batch(valid_mask.to(action_features.dtype)).bool()
            if sample_ids is not None:
                sample_ids = gather_batch(sample_ids)

        action = action[valid_mask]
        text = text[valid_mask]
        if sample_ids is not None:
            sample_ids = sample_ids[valid_mask]
        if action.shape[0] < 2:
            raise ValueError("VICReg requires at least two valid paired samples")

        batch_size, feature_dim = action.shape
        if sample_ids is None:
            # diagonal 模式保持 VICReg 官方的逐行配对 MSE。
            invariance_loss = F.mse_loss(action, text)
            positive_pair_count = batch_size
        else:
            # task_id/task_index 模式与 InfoNCE 对齐：同 ID 的全部跨模态组合都是正配对。
            positive_mask = sample_ids[:, None].eq(sample_ids[None, :])
            # 直接计算欧氏距离，避免平方范数展开在两侧都有大公共偏置时发生消减误差。
            pairwise_mse = torch.cdist(
                action,
                text,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            ).square() / feature_dim
            positives_per_anchor = positive_mask.sum(dim=1)
            invariance_loss = (
                (pairwise_mse * positive_mask).sum(dim=1) / positives_per_anchor
            ).mean()
            positive_pair_count = positive_mask.sum()

        action_centered = action - action.mean(dim=0)
        text_centered = text - text.mean(dim=0)
        action_std = torch.sqrt(action.var(dim=0) + self.eps)
        text_std = torch.sqrt(text.var(dim=0) + self.eps)
        variance_loss = (
            F.relu(self.variance_target - action_std).mean()
            + F.relu(self.variance_target - text_std).mean()
        ) / 2

        action_cov = action_centered.T @ action_centered / (batch_size - 1)
        text_cov = text_centered.T @ text_centered / (batch_size - 1)
        covariance_loss = (
            off_diagonal(action_cov).pow(2).sum()
            + off_diagonal(text_cov).pow(2).sum()
        ) / feature_dim

        total_loss = (
            self.sim_coeff * invariance_loss
            + self.std_coeff * variance_loss
            + self.cov_coeff * covariance_loss
        )
        stats = {
            "vicreg_invariance_loss": invariance_loss.detach(),
            "vicreg_variance_loss": variance_loss.detach(),
            "vicreg_covariance_loss": covariance_loss.detach(),
            "vicreg_action_std": action_std.mean().detach(),
            "vicreg_text_std": text_std.mean().detach(),
            "vicreg_valid_pairs": torch.tensor(
                float(batch_size), device=total_loss.device
            ),
            "vicreg_positive_pairs": torch.as_tensor(
                positive_pair_count,
                device=total_loss.device,
                dtype=torch.float32,
            ).detach(),
        }
        return total_loss, stats
