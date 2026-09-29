"""
info_theory.py — 信息论诊断量（HSIC / InfoNCE-MI 下界 / Gaussian NLL / 自由能）

关联到论文 §3.x "Information-Theoretic Interpretation"：把 DSN 三件套
解读为 Multi-View Information Bottleneck (Federici NeurIPS 2020)：
    max  I(z_s; e)  -  β I(z_s; z_p)  +  γ E[log p(h | z_s, z_p)]
其中：
    I(z_s; e)         由 InfoNCE 下界估计 (van den Oord 2018, Poole 2019)
    I(z_s; z_p)       由线性核 HSIC 上界代理 (Gretton 2005)
    log p(h | s)      高斯似然 → -MSE / (2 σ²) + const

本模块**仅做 logging**：所有计算都用 .detach()，零梯度污染。
唯一目的：把这四个量写进 loss_dict，让 SwanLab 自动 plot 信息平面。

参考实现：
- HSIC linear:  Gretton et al. 2005 ALT, eq. 4
- InfoNCE bound: van den Oord 2018 (CPC), Poole 2019 (variational MI)
- VFE:          Friston 2010 Nat Rev Neurosci
"""
import math
import torch
import torch.nn.functional as F


# ============================================================
#  线性核 HSIC：I(z_s; z_p) 的上界代理
# ============================================================

def hsic_linear(z_s, z_p, eps=1e-8):
    """线性核 HSIC 经验估计（Gretton 2005, eq. 4）。

    HSIC_lin(X, Y) = (n-1)^-2 * tr(K_x H K_y H)
                   = (n-1)^-2 * ||X̃^T Ỹ||_F^2
    其中 X̃ = HX 是 row-centered 后的样本矩阵。

    与 DSN 原 diff loss ||Z_s^T Z_p||_F^2 仅差 (n-1)^2 归一化常数。

    Args:
        z_s, z_p: [N, D] tensor，N 是 batch（或 batch * tokens），D 维度
        eps: 数值稳定 floor
    Returns:
        scalar tensor (HSIC 估计值，越接近 0 越独立)
    """
    n = z_s.shape[0]
    if n < 2:
        return torch.zeros((), device=z_s.device, dtype=z_s.dtype)
    # row-center：每列减去均值
    zs_c = z_s - z_s.mean(dim=0, keepdim=True)
    zp_c = z_p - z_p.mean(dim=0, keepdim=True)
    # cross-covariance Frobenius squared
    cross = zs_c.t() @ zp_c                    # [D_s, D_p]
    fro_sq = (cross * cross).sum()
    return fro_sq / max((n - 1) ** 2, eps)


def hsic_normalized(z_s, z_p, eps=1e-8):
    """归一化 HSIC（CKA 风格，0~1 间），便于跨 batch/scale 比较。

    nHSIC = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    """
    h_xy = hsic_linear(z_s, z_p, eps)
    h_xx = hsic_linear(z_s, z_s, eps)
    h_yy = hsic_linear(z_p, z_p, eps)
    return h_xy / torch.sqrt(h_xx * h_yy + eps)


# ============================================================
#  InfoNCE 下界：I(z_s; e) 的可观测下界
# ============================================================

def infonce_mi_lower_bound(loss_align, batch_size):
    """I(z_s; e) ≥ log B - L_NCE （van den Oord 2018, Poole 2019 eq. 6）。

    bound 上限是 log B，所以 batch size 是 IT bound 的硬天花板。

    Args:
        loss_align: scalar tensor or float, InfoNCE 损失值
        batch_size: int, 当前 batch 内有效正负样本对数
    Returns:
        scalar tensor，I_NCE 下界估计
    """
    log_B = math.log(max(batch_size, 1))
    if isinstance(loss_align, torch.Tensor):
        return torch.tensor(log_B, device=loss_align.device, dtype=loss_align.dtype) - loss_align
    return log_B - float(loss_align)


# ============================================================
#  Gaussian NLL：-log p(h | z_s, z_p) 的代理
# ============================================================

def recon_nll_gaussian(x, x_hat, sigma2=1.0, eps=1e-8):
    """高斯似然下 -log p(x | ĥ)。

    在 σ² 固定的各向同性高斯下：
        -log p(x|ĥ) = (1/2σ²) ||x - ĥ||² + const
    返回 batch 内平均 NLL（去掉 const）。
    """
    n = x.shape[0]
    if n == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    sq_err = ((x - x_hat) ** 2).sum() / max(n, 1)
    return sq_err / (2.0 * max(sigma2, eps))


# ============================================================
#  Variational Free Energy：DSN 三件套合一的 IT 量
# ============================================================

def variational_free_energy(loss_align, hsic_zs_zp, recon_nll, beta=1.0, gamma=1.0):
    """近似 Friston 2010 自由能 F = L_align + β·I(z_s; z_p) + γ·NLL。

    在 β=γ=1 时数值上等于 -ELBO；β=DSN diff_weight (0.075) 时偏弱
    compression。本函数仅做 logging，不进 backward。

    Args:
        loss_align: InfoNCE 损失（scalar tensor or float）
        hsic_zs_zp: HSIC(z_s, z_p) 估计值（scalar tensor）
        recon_nll: 重建 NLL（scalar tensor）
        beta, gamma: 权重（默认与 FEP 同 = 1.0）
    Returns:
        scalar tensor，自由能近似值
    """
    if not isinstance(loss_align, torch.Tensor):
        loss_align = torch.tensor(float(loss_align), device=hsic_zs_zp.device,
                                   dtype=hsic_zs_zp.dtype)
    return loss_align + beta * hsic_zs_zp + gamma * recon_nll


# ============================================================
#  一站式 logging：把 4 个 IT 量写进 loss_dict
# ============================================================

@torch.no_grad()
def log_information_plane_step(
    loss_dict: dict,
    disentangle_pairs: list,        # list of (z_s, z_p, x_recon, x_orig)，已 detach
    mode_losses: dict = None,       # fg_alignment_model 中的 mode_losses
    batch_size: int = 0,
    align_mode_key: str = 'as2ts',  # 取哪个对齐模式估计 InfoNCE bound
):
    """把 HSIC / InfoNCE bound / recon NLL / free energy 写进 loss_dict。

    所有键以 `it_` 前缀，避免与既有 swanlab 面板冲突。

    Args:
        loss_dict: 训练 loop 的 loss dict（in-place 修改）
        disentangle_pairs: 多层 DSN 的 (z_s, z_p, x_recon, x_orig) 列表
        mode_losses: fg_alignment_model 中的 mode_losses dict，用来取 L_align
        batch_size: 用于 InfoNCE bound 的 log B
        align_mode_key: 取哪个 mode 的 loss 当 L_align（默认 as2ts，sentence-level）
    """
    if not disentangle_pairs:
        return
    # 多层 DSN 取平均
    hsic_vals = []
    nhsic_vals = []
    recon_vals = []
    for (z_s, z_p, x_hat, x_orig) in disentangle_pairs:
        # 数值稳定：用 fp32 算 HSIC，避免 bf16 cross-cov 下溢
        z_s_f = z_s.float()
        z_p_f = z_p.float()
        x_f = x_orig.float()
        x_hat_f = x_hat.float()
        hsic_vals.append(hsic_linear(z_s_f, z_p_f).item())
        nhsic_vals.append(hsic_normalized(z_s_f, z_p_f).item())
        recon_vals.append(recon_nll_gaussian(x_f, x_hat_f).item())

    hsic_avg = sum(hsic_vals) / len(hsic_vals)
    nhsic_avg = sum(nhsic_vals) / len(nhsic_vals)
    recon_avg = sum(recon_vals) / len(recon_vals)
    loss_dict['it_hsic_zs_zp'] = hsic_avg
    loss_dict['it_hsic_zs_zp_normalized'] = nhsic_avg
    loss_dict['it_recon_nll'] = recon_avg

    # InfoNCE → I(z_s; e) 下界
    if mode_losses is not None and align_mode_key in mode_losses and mode_losses[align_mode_key]:
        # 取该 mode 多层平均 loss
        align_losses = mode_losses[align_mode_key]
        if isinstance(align_losses, list) and align_losses:
            l_align = torch.stack([l.detach() for l in align_losses]).mean().item()
        else:
            l_align = float(align_losses)
        if batch_size > 0:
            mi_lb = math.log(batch_size) - l_align
            loss_dict['it_mi_lower_zs_e'] = mi_lb
            loss_dict['it_log_batch_size'] = math.log(batch_size)
            # 自由能（β=γ=1 形式，纯诊断）
            loss_dict['it_free_energy'] = l_align + hsic_avg + recon_avg
