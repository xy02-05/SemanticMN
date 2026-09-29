"""
disentangle.py — 共享/特有表征分离模块

基于 DSN (Domain Separation Networks, Bousmalis et al., NeurIPS 2016) 的实现。
损失函数 DiffLoss / MSE / SIMSE 直接从 DSN 原始代码复制:
  https://github.com/fungtion/DSN/blob/master/functions.py

架构:
  action [B, D] ─→ shared_encoder ─→ z_s [B, D] → 送入投影层做对齐
                  ├→ private_encoder ─→ z_p [B, D] → 不参与对齐
                  └→ z_s + z_p ─→ decoder ─→ x_hat → 重建约束（DSN用加法而非拼接）

独立性约束（diff_mode 可选）:
  1. "dsn":          原始 DiffLoss ||X^T Y||_F^2 — B << D 时失效
  2. "cosine":       样本级 mean(cos²(z_s, z_p)) — 不依赖 B/D（推荐）
  3. "cosine_hinge": DLF 风格 CosineEmbeddingLoss(target=-1) — cos≤0 时停止梯度

重建约束: MSE(x_hat, x) + SIMSE(x_hat, x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


# ============================================================
#  DSN 原始损失函数（来源: github.com/fungtion/DSN/blob/master/functions.py）
# ============================================================

class DiffLoss(nn.Module):
    """
    DSN 原始差异损失: ||X_norm^T Y_norm||_F^2

    ⚠ 已知问题: 计算 [D,D] 矩阵，当 B << D 时（如 B=16, D=1024），
    矩阵被采样噪声淹没，loss ≈ B/D² ≈ 1.5e-5，无法区分正交和相关。
    仅在 B ≈ D 时有效（如 pre_pool 模式 B*A ≈ 800 接近 D=1024）。
    """

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CosineSimDiffLoss(nn.Module):
    """
    样本级余弦相似度独立性约束（替代 DSN DiffLoss）

    原理: 对每个样本独立计算 z_shared 和 z_private 的余弦相似度平方，
    最小化使两者趋向正交。不依赖 batch size，任何 B/D 比例下都有效。

    值域 [0, 1]: 0=完美正交，1=完全共线
    计算复杂度: O(B*D)，远快于 DiffLoss 的 O(D²)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        cos_sim = F.cosine_similarity(input1, input2, dim=-1)  # [B]
        return (cos_sim ** 2).mean()


class CosineHingeDiffLoss(nn.Module):
    """
    DLF 风格正交约束: CosineEmbeddingLoss(target=-1)

    参考: DLF (Disentangled-Language-Focused, ACL 2025)
    代码: github.com/pwang322/DLF — trains/singleTask/DLF.py L104-114

    与 CosineSimDiffLoss 的区别:
    - cos²:  始终施加梯度，即使已接近正交（平滑，收敛更精确）
    - hinge: cos ≤ 0 时停止梯度（宽容，允许 shared/private 有少量相关）

    值域 [0, 1]: 0=正交或反向，>0=有正相关
    """

    def __init__(self):
        super().__init__()
        # target=-1: 鼓励 cos(input1, input2) → -1（即反向/正交），
        # 等价于 max(0, cos(x1, x2))
        self._loss_fn = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

    def forward(self, input1, input2):
        target = torch.ones(input1.size(0), device=input1.device) * -1
        return self._loss_fn(input1, input2, target)


class MSE(nn.Module):
    """DSN 重建损失之一: 均方误差（原始实现完全复制）"""

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):
    """
    DSN 重建损失之二: Scale-Invariant MSE（原始实现完全复制）

    与MSE的区别: SIMSE先求和再平方，对全局偏移（bias）更鲁棒
    """

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


# ============================================================
#  DisentangleHead: 表征分离网络
# ============================================================

class DisentangleHead(nn.Module):
    """
    表征分离头: 将action特征分解为shared和private两部分

    与DSN对齐的关键设计:
    - 重建用加法 z_shared + z_private（DSN原始: union_code = private_code + shared_code）
    - 不用拼接（之前的实现用cat，与DSN不一致）

    encoder_type:
    - "mlp":    2层MLP（默认）
    - "linear": 单层Linear（最轻量baseline）
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        encoder_type: str = "mlp",
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.encoder_type = encoder_type

        if encoder_type == "linear":
            self.shared_encoder = nn.Linear(input_dim, input_dim)
            self.private_encoder = nn.Linear(input_dim, input_dim)
            # DSN用加法后直接进decoder，decoder输入维度 = input_dim
            self.decoder = nn.Linear(input_dim, input_dim)
        else:
            self.shared_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
            )
            self.private_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
            )
            # DSN用加法: decoder输入维度 = input_dim（不是 2*input_dim）
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, D] pooling后的global_action_features
        Returns:
            z_shared [B, D], z_private [B, D], x_recon [B, D]
        """
        z_shared = self.shared_encoder(x)
        z_private = self.private_encoder(x)
        # DSN原始: union_code = private_code + shared_code（加法，不是拼接）
        x_recon = self.decoder(z_shared + z_private)
        return z_shared, z_private, x_recon


# ============================================================
#  损失计算
# ============================================================

# 全局实例化，避免每次forward重复创建
_diff_loss_fn = DiffLoss()
_cosine_diff_loss_fn = CosineSimDiffLoss()
_cosine_hinge_diff_loss_fn = CosineHingeDiffLoss()
_mse_loss_fn = MSE()
_simse_loss_fn = SIMSE()


def compute_disentangle_losses(
    z_shared: torch.Tensor,
    z_private: torch.Tensor,
    x_recon: torch.Tensor,
    x_orig: torch.Tensor,
    diff_weight: float = 0.075,
    recon_weight: float = 0.01,
    diff_mode: str = 'cosine',
    detach_recon_target: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算分离损失

    diff_mode:
      "dsn":          DSN 原始 DiffLoss ||X^T Y||_F^2（B << D 时失效）
      "cosine":       样本级 mean(cos²(z_s, z_p))（推荐）
      "cosine_hinge": DLF 风格 max(0, cos(z_s, z_p))（cos≤0 时停止梯度）

    Args:
        z_shared:     [N, D] 共享语义表征（N=B 或 N=B*A）
        z_private:    [N, D] 模态特有表征
        x_recon:      [N, D] 重建特征
        x_orig:       [N, D] 原始输入特征
        diff_weight:  差异损失权重（DSN默认 beta=0.075）
        recon_weight: 重建损失权重（DSN默认 alpha=0.01）
        diff_mode:    独立性约束模式 "dsn" | "cosine" | "cosine_hinge"
        detach_recon_target: 是否阻断 reconstruction target 分支的梯度
    """
    # 独立性约束: 根据 diff_mode 选择算法
    if diff_mode == 'dsn':
        l_diff = _diff_loss_fn(z_private, z_shared)
    elif diff_mode == 'cosine_hinge':
        l_diff = _cosine_hinge_diff_loss_fn(z_private, z_shared)
    else:
        l_diff = _cosine_diff_loss_fn(z_private, z_shared)

    # 重建损失: MSE + SIMSE（与DSN一致）
    recon_target = x_orig.detach() if detach_recon_target else x_orig
    l_mse = _mse_loss_fn(x_recon, recon_target)
    l_simse = _simse_loss_fn(x_recon, recon_target)

    total = diff_weight * l_diff + recon_weight * (l_mse + l_simse)

    stats = {
        'disentangle_diff_loss': l_diff.item(),
        'disentangle_mse_loss': l_mse.item(),
        'disentangle_simse_loss': l_simse.item(),
        'disentangle_total_loss': total.item(),
    }
    return total, stats
