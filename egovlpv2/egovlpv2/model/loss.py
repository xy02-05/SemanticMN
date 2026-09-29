# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import torch
import torch.nn.functional as F
from torch import nn
import pickle
from typing import Dict, Tuple, Optional

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j, self.temperature

class EgoNCE(nn.Module):
    def __init__(self, temperature=0.05, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature

    def forward(self, x, mask_v, mask_n):
        mask_diag = torch.eye(x.shape[0]).cuda()
        if self.noun and self.verb:
            mask = mask_v * mask_n + mask_diag
        elif self.noun and not self.verb:
            mask = mask_n + mask_diag
        elif self.verb and not self.noun:
            mask = mask_v + mask_diag
        else:
            mask = mask_diag

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = F.softmax(x/self.temperature, dim=1)
        j_sm = F.softmax(x.t()/self.temperature, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1) )
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1) )
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j, mask_bool, self.temperature
        #return - loss_i - loss_j


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.2, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()

class AdaptiveMaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.4, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = weight.unsqueeze(1)
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(  w1 * self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin =  F.relu( w1_ * self.margin - (x1_ - x2_))

        return max_margin.mean()

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)


def create_alignment_masks(
    ids: torch.Tensor,
    alignment_mode: str = 'diagonal'
) -> tuple:
    """
    统一创建对齐所需的所有mask（合并positive_mask和valid_mask）
    
    注意：如果使用feature bank，ids应该是 [B+K] 尺寸（包含feature bank的id）
    
    功能：
    1. 生成positive_mask：标记正样本对
    2. 生成valid_mask：标记有效样本（task_id != -1）
    3. 生成valid_mask_2d：标记有效的样本对
    4. 额外检测：如果某行所有列都无效，将该行的valid_mask设为0（避免NaN）
    
    Args:
        ids: [B+K] 每个样本的id（task_id或task_index），-1表示无效/空指令
             如果使用feature bank，K是bank中的样本数
        alignment_mode: 对齐模式 ('diagonal', 'task_id', 'task_index')
        
    Returns:
        tuple: (positive_mask, valid_mask, valid_mask_2d)
            - positive_mask: [B+K, B+K] 正样本矩阵
            - valid_mask: [B+K] 有效样本mask（已处理"全无效行"的情况）
            - valid_mask_2d: [B+K, B+K] 有效样本对mask
    """
    B = ids.shape[0]
    device = ids.device
    
    # Step 1: 生成valid_mask（基于task_id != -1）
    valid_mask = (ids != -1).float()  # [B]
    
    # Step 2: 生成valid_mask_2d
    valid_mask_2d = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(0)  # [B, B]
    
    # Step 3: 生成positive_mask
    if alignment_mode == 'diagonal':
        positive_mask = torch.eye(B, device=device)
    else:
        # task_id或task_index模式：相同id为正样本对
        ids_row = ids.unsqueeze(1)  # [B, 1]
        ids_col = ids.unsqueeze(0)  # [1, B]
        positive_mask = (ids_row == ids_col).float()  # [B, B]
    # id为-1的样本不参与正样本匹配
    positive_mask = positive_mask * valid_mask_2d
    
    # Step 4: 检测"某行没有正样本"的情况
    row_has_valid = (positive_mask.sum(dim=1) > 0).float()
    valid_mask = valid_mask * row_has_valid
    
    return positive_mask, valid_mask, valid_mask_2d


# 保留向后兼容的单独函数
def create_positive_mask(ids: torch.Tensor, alignment_mode: str = 'diagonal') -> torch.Tensor:
    """向后兼容：只返回positive_mask"""
    positive_mask, _, _ = create_alignment_masks(ids, alignment_mode)
    return positive_mask


def create_valid_sample_mask(task_ids: torch.Tensor) -> torch.Tensor:
    """向后兼容：只返回valid_mask"""
    _, valid_mask, _ = create_alignment_masks(task_ids, 'diagonal')
    return valid_mask


def compute_anchor_valid_mask_by_ratio(
    valid_mask_2d: torch.Tensor,  # [B, N] 有效pair mask
    base_anchor_mask: torch.Tensor,  # [B] 原始的anchor有效mask
    min_valid_ratio: float = 0.0,  # 最小有效样本比例阈值（0~1）
) -> torch.Tensor:
    """
    根据每行有效样本比例计算新的anchor valid mask
    
    逻辑：如果某anchor的有效配对数/总样本数 < min_valid_ratio，则该anchor的loss不计入
    但该样本仍可作为其他anchor的正/负样本（因为我们只修改anchor_mask，不修改valid_mask）
    
    Args:
        valid_mask_2d: [B, N] 每个(anchor, key)对是否有效
        base_anchor_mask: [B] 原始的anchor有效mask
        min_valid_ratio: 最小有效样本比例阈值，0表示不过滤
        
    Returns:
        anchor_valid_mask: [B] 更新后的anchor有效mask
    """
    if min_valid_ratio <= 0.0:
        return base_anchor_mask
    
    N = valid_mask_2d.shape[1]
    # 计算每个anchor的有效配对数
    num_valid_per_row = valid_mask_2d.sum(dim=1)  # [B]
    # 计算比例
    ratio = num_valid_per_row / N
    # 生成比例mask
    ratio_mask = (ratio >= min_valid_ratio).float()
    
    return base_anchor_mask * ratio_mask


# ============================================================
# 单向对比学习loss计算函数（可被Mixin和AT2TT复用）
# ============================================================

def compute_single_direction_infonce(
    logits: torch.Tensor,           # [B, N] 已缩放的logits (logits = similarity / temperature)
    positive_mask: torch.Tensor,    # [B, N] 正样本mask
    valid_col_mask: torch.Tensor,   # [1, N] 有效列mask
    valid_anchor_mask: torch.Tensor,  # [B] 有效anchor mask
) -> torch.Tensor:
    """
    计算单向InfoNCE loss
    
    核心公式: L = -sum(soft_label * log_softmax(logits))
    
    Args:
        logits: [B, N] 已除以temperature的logits
        positive_mask: [B, N] 正样本mask（1=正样本，0=负样本）
        valid_col_mask: [1, N] 有效列mask（无效列不参与softmax）
        valid_anchor_mask: [B] 有效anchor mask（无效anchor的loss为0）
        
    Returns:
        loss: 标量，该方向的平均loss
    """
    # 将无效列的logits设为-1e9（不参与softmax）
    invalid_col_mask = (valid_col_mask == 0)
    logits = logits.masked_fill(invalid_col_mask, -1e9)
    
    # 正样本mask与valid_col_mask相乘，去掉无效列的正样本
    valid_positive_mask = positive_mask * valid_col_mask
    
    # 归一化为soft label
    pos_sum = valid_positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
    soft_labels = valid_positive_mask / pos_sum
    
    # 计算loss: -sum(soft_label * log_softmax(logits))
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample_loss = -(soft_labels * log_probs).sum(dim=-1)  # [B]
    
    # 应用valid_anchor_mask
    per_sample_loss = per_sample_loss * valid_anchor_mask
    num_valid = valid_anchor_mask.sum().clamp(min=1)
    
    return per_sample_loss.sum() / num_valid


def compute_single_direction_sigmoid(
    logits: torch.Tensor,           # [B, N] 已缩放的logits (logits = similarity / temperature - bias)
    positive_mask: torch.Tensor,    # [B, N] 正样本mask
    valid_2d_mask: torch.Tensor,    # [B, N] 有效pair mask
    valid_anchor_mask: torch.Tensor,  # [B] 有效anchor mask
) -> torch.Tensor:
    """
    计算单向Sigmoid loss
    
    核心公式: L = -log(sigmoid(z * logits))，其中z是+1(正样本)或-1(负样本)
    
    Args:
        logits: [B, N] 已应用temperature和bias的logits
        positive_mask: [B, N] 正样本mask（1=正样本，0=负样本）
        valid_2d_mask: [B, N] 有效pair mask（无效pair的loss为0）
        valid_anchor_mask: [B] 有效anchor mask（无效anchor的loss为0）
        
    Returns:
        loss: 标量，该方向的平均loss
    """
    # 转为+1/-1的labels
    labels = positive_mask * 2 - 1
    
    # 逐元素计算loss
    elem_loss = -F.logsigmoid(labels * logits)
    
    # 应用2D valid_mask
    elem_loss = elem_loss * valid_2d_mask
    
    # 每个样本的loss（分母是有效pair数量）
    num_valid_per_row = valid_2d_mask.sum(dim=1).clamp(min=1)
    per_sample_loss = elem_loss.sum(dim=1) / num_valid_per_row
    
    # 应用valid_anchor_mask
    per_sample_loss = per_sample_loss * valid_anchor_mask
    num_valid = valid_anchor_mask.sum().clamp(min=1)
    
    return per_sample_loss.sum() / num_valid


class ContrastiveLossMixin:
    """
    提供对比学习相关的通用loss工具函数：
    - 单正样本模式：对角线为正样本（默认）
    - 多正样本模式：根据task_id/task_index确定正样本对
    - Sigmoid双向对比学习
    - InfoNCE双向对比学习
    
    该Mixin假设宿主类上存在以下属性：
        self.loss_type: str, 'infonce' 或 'sigmoid'
        self.sigmoid_bias: 标量偏置（固定buffer或可学习参数）
        self._get_temperature(): 返回用于对比学习的温度（支持可学习）
    """

    def _compute_bidirectional_contrastive_loss(
        self,
        similarity: torch.Tensor,  # [B+K, B+K] 完整相似度矩阵
        batch_size: int,  # 当前batch大小
        device: Optional[torch.device] = None,
        positive_mask: Optional[torch.Tensor] = None,  # [B, B] 多正样本mask
        valid_mask: Optional[torch.Tensor] = None,  # [B] 有效样本mask
    ) -> Tuple[torch.Tensor, Dict]:
        """
        通用的双向对比学习loss计算函数
        
        支持两种模式：
        1. 单正样本模式（默认）：对角线为正样本
        2. 多正样本模式：根据positive_mask确定正样本对
        
        Args:
            similarity: [B+K, B+K] 相似度矩阵（InfoNCE已除以temperature，Sigmoid未除）
            batch_size: 当前batch大小B
            device: 设备
            positive_mask: [B, B] 可选，多正样本mask矩阵（1=正样本对，0=负样本对）
            valid_mask: [B] 可选，有效样本mask（1=有效，0=无效，无效样本loss为0）
            
        Returns:
            loss: 双向平均损失
            logit_stats: 包含正负样本logit统计信息
        """
        if device is None:
            device = similarity.device

        if self.loss_type == 'sigmoid':
            return self._compute_sigmoid_loss(similarity, batch_size, device, positive_mask, valid_mask)
        else:
            return self._compute_infonce_loss(similarity, batch_size, device, positive_mask, valid_mask)
    
    def _compute_sigmoid_loss(
        self,
        similarity: torch.Tensor,  # [B+K, B+K] 相似度矩阵（已除以temperature）
        batch_size: int,
        device: torch.device,
        positive_mask: Optional[torch.Tensor] = None,  # [B+K, B+K] 多正样本mask
        valid_mask: Optional[torch.Tensor] = None,  # [B+K] 有效样本mask
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算双向Sigmoid Loss（调用单向函数两次）
        
        注意：温度已在调用方统一通过除法应用到similarity中
        """
        total_size = similarity.shape[0]
        
        if positive_mask is None:
            positive_mask = torch.eye(total_size, device=device)
        if valid_mask is None:
            valid_mask = torch.ones(total_size, device=device)
        
        valid_mask_batch = valid_mask[:batch_size]
        valid_mask_2d = valid_mask_batch.unsqueeze(1) * valid_mask.unsqueeze(0)  # [B, B+K]
        
        # 应用min_valid_ratio过滤：有效配对数比例低于阈值的anchor的loss不计入
        min_valid_ratio = getattr(self, 'min_valid_ratio', 0.0)
        valid_anchor_mask = compute_anchor_valid_mask_by_ratio(
            valid_mask_2d, valid_mask_batch, min_valid_ratio
        )
        
        # 当前实现约定:
        # logits = similarity / temperature - sigmoid_bias
        # 若换成 SigLIP 论文中的写法 logits = similarity * scale + b，
        # 则两者等价关系为 scale = 1 / temperature, b = -sigmoid_bias。
        logits = similarity - self.sigmoid_bias.float()
        
        # 方向1: 1→2
        logits_1to2 = logits[:batch_size, :]
        loss_1to2 = compute_single_direction_sigmoid(
            logits_1to2, positive_mask[:batch_size, :], valid_mask_2d, valid_anchor_mask
        )
        
        # 方向2: 2→1（转置）
        logits_2to1 = logits[:, :batch_size].t()
        loss_2to1 = compute_single_direction_sigmoid(
            logits_2to1, positive_mask[:, :batch_size].t(), valid_mask_2d, valid_anchor_mask
        )
        
        # 统计信息（使用原始valid_mask_batch计算，不受min_valid_ratio影响）
        pos_mask_batch = positive_mask[:batch_size, :batch_size]
        valid_2d_batch = valid_mask_batch.unsqueeze(1) * valid_mask_batch.unsqueeze(0)
        valid_pos_mask = (pos_mask_batch > 0) & (valid_2d_batch > 0)
        valid_neg_mask = (pos_mask_batch == 0) & (valid_2d_batch > 0)
        pos_logits = logits_1to2[:, :batch_size][valid_pos_mask]
        neg_logits = logits_1to2[:, :batch_size][valid_neg_mask]
        
        pos_mean_logit = pos_logits.mean() if len(pos_logits) > 0 else torch.tensor(0.0, device=device)
        neg_mean_logit = neg_logits.mean() if len(neg_logits) > 0 else torch.tensor(0.0, device=device)
        
        # 统计正负样本数：每个有效样本行的平均正样本数和负样本数
        num_pos_per_row = (pos_mask_batch * valid_2d_batch).sum(dim=1)  # [B]
        num_neg_per_row = ((1 - pos_mask_batch) * valid_2d_batch).sum(dim=1)  # [B]
        num_valid_rows = (valid_mask_batch > 0).sum().clamp(min=1)
        avg_pos_samples = num_pos_per_row.sum() / num_valid_rows
        avg_neg_samples = num_neg_per_row.sum() / num_valid_rows
        avg_total_samples = (num_pos_per_row + num_neg_per_row).sum() / num_valid_rows
        
        return (loss_1to2 + loss_2to1) / 2.0, {
            'pos_mean_logit': pos_mean_logit.detach(),
            'neg_mean_logit': neg_mean_logit.detach(),
            'avg_pos_samples': avg_pos_samples.detach(),
            'avg_neg_samples': avg_neg_samples.detach(),
            'avg_total_samples': avg_total_samples.detach(),
        }
    
    def _compute_infonce_loss(
        self,
        similarity: torch.Tensor,  # [B+K, B+K] logits（已除以temperature）
        batch_size: int,
        device: torch.device,
        positive_mask: Optional[torch.Tensor] = None,  # [B+K, B+K] 多正样本mask
        valid_mask: Optional[torch.Tensor] = None,  # [B+K] 有效样本mask
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算双向InfoNCE Loss（调用单向函数两次）
        """
        total_size = similarity.shape[0]
        
        if positive_mask is None:
            positive_mask = torch.eye(total_size, device=device)
        if valid_mask is None:
            valid_mask = torch.ones(total_size, device=device)
        
        valid_col_mask = valid_mask.unsqueeze(0)  # [1, B+K]
        valid_mask_batch = valid_mask[:batch_size]  # [B]
        
        # 计算2D mask用于min_valid_ratio过滤
        valid_mask_2d = valid_mask_batch.unsqueeze(1) * valid_mask.unsqueeze(0)  # [B, B+K]
        
        # 应用min_valid_ratio过滤：有效配对数比例低于阈值的anchor的loss不计入
        min_valid_ratio = getattr(self, 'min_valid_ratio', 0.0)
        valid_anchor_mask = compute_anchor_valid_mask_by_ratio(
            valid_mask_2d, valid_mask_batch, min_valid_ratio
        )
        
        # 方向1: 1→2
        logits_1to2 = similarity[:batch_size, :].clone()
        loss_1to2 = compute_single_direction_infonce(
            logits_1to2, positive_mask[:batch_size, :], valid_col_mask, valid_anchor_mask
        )
        
        # 方向2: 2→1（转置）
        logits_2to1 = similarity[:, :batch_size].t().clone()
        loss_2to1 = compute_single_direction_infonce(
            logits_2to1, positive_mask[:, :batch_size].t(), valid_col_mask, valid_anchor_mask
        )
        
        # 统计信息（使用原始valid_mask_batch计算，不受min_valid_ratio影响）
        pos_mask_batch = positive_mask[:batch_size, :batch_size]
        valid_2d_batch = valid_mask_batch.unsqueeze(1) * valid_mask_batch.unsqueeze(0)
        valid_pos_mask = (pos_mask_batch > 0) & (valid_2d_batch > 0)
        valid_neg_mask = (pos_mask_batch == 0) & (valid_2d_batch > 0)
        pos_logits = similarity[:batch_size, :batch_size][valid_pos_mask]
        neg_logits = similarity[:batch_size, :batch_size][valid_neg_mask]
        
        pos_mean_logit = pos_logits.mean() if len(pos_logits) > 0 else torch.tensor(0.0, device=device)
        neg_mean_logit = neg_logits.mean() if len(neg_logits) > 0 else torch.tensor(0.0, device=device)
        
        # 统计正负样本数：每个有效样本行的平均正样本数和负样本数
        num_pos_per_row = (pos_mask_batch * valid_2d_batch).sum(dim=1)  # [B]
        num_neg_per_row = ((1 - pos_mask_batch) * valid_2d_batch).sum(dim=1)  # [B]
        num_valid_rows = (valid_mask_batch > 0).sum().clamp(min=1)
        avg_pos_samples = num_pos_per_row.sum() / num_valid_rows
        avg_neg_samples = num_neg_per_row.sum() / num_valid_rows
        avg_total_samples = (num_pos_per_row + num_neg_per_row).sum() / num_valid_rows
        
        return (loss_1to2 + loss_2to1) / 2.0, {
            'pos_mean_logit': pos_mean_logit.detach(),
            'neg_mean_logit': neg_mean_logit.detach(),
            'avg_pos_samples': avg_pos_samples.detach(),
            'avg_neg_samples': avg_neg_samples.detach(),
            'avg_total_samples': avg_total_samples.detach(),
        }


# ============================================================
# AT2TT: FILIP风格细粒度对齐 (Action Token to Text Token)
# ============================================================

def compute_filip_similarity(
    action_tokens: torch.Tensor,   # [B, A, D] action token序列
    text_tokens: torch.Tensor,     # [N, T, D] text token序列 (N可以是B或B+K)
    text_mask: torch.Tensor = None,    # [N, T] text有效性mask
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FILIP风格的Token-wise Maximum Similarity计算
    
    核心思想：
    1. 对每个action token，找到最相似的text token，然后对所有action tokens求平均
    2. 对每个text token，找到最相似的action token，然后对所有text tokens求平均
    
    公式：
    - s_a2t(A, T) = (1/A) * Σ_i max_j sim(a_i, t_j)  # action-to-text
    - s_t2a(T, A) = (1/T) * Σ_j max_i sim(t_j, a_i)  # text-to-action
    
    Args:
        action_tokens: [B, A, D] L2归一化后的action token特征
        text_tokens: [N, T, D] L2归一化后的text token特征
        text_mask: [N, T] text有效性mask (1=有效, 0=padding)
        
    Returns:
        s_a2t: [B, N] action-to-text相似度矩阵
        s_t2a: [B, N] text-to-action相似度矩阵
    """
    B, A, D = action_tokens.shape
    N, T, _ = text_tokens.shape
    
    # 计算全量token-token相似度: [B, N, A, T]
    # action_tokens: [B, A, D] -> [B, 1, A, D]
    # text_tokens: [N, T, D] -> [1, N, T, D] -> [1, N, D, T]
    # 矩阵乘法得到 [B, N, A, T]
    sim_matrix = torch.matmul(
        action_tokens.unsqueeze(1),  # [B, 1, A, D]
        text_tokens.unsqueeze(0).permute(0, 1, 3, 2)  # [1, N, D, T]
    )  # [B, N, A, T]
    
    # ===== Action-to-Text方向 =====
    # 对每个action token，取最大的text token相似度
    if text_mask is not None:
        # 将padding位置的相似度设为很小的值
        text_mask_exp = text_mask.unsqueeze(0).unsqueeze(2)  # [1, N, 1, T]
        sim_for_a2t = sim_matrix.masked_fill(text_mask_exp == 0, -1e9)
    else:
        sim_for_a2t = sim_matrix
    
    max_sim_a2t, _ = sim_for_a2t.max(dim=-1)  # [B, N, A] 每个action token的最大相似度
    
    # 对action tokens求平均
    s_a2t = max_sim_a2t.mean(dim=-1)  # [B, N]
    
    # ===== Text-to-Action方向 =====
    # 对每个text token，取最大的action token相似度
    max_sim_t2a, _ = sim_matrix.max(dim=-2)  # [B, N, T] 每个text token的最大相似度
    
    # 对text tokens求平均
    if text_mask is not None:
        text_mask_exp = text_mask.unsqueeze(0)  # [1, N, T]
        max_sim_t2a = max_sim_t2a.masked_fill(text_mask_exp == 0, 0.0)
        num_valid_t = text_mask.sum(dim=-1, keepdim=True).unsqueeze(0).clamp(min=1)  # [1, N, 1]
        s_t2a = max_sim_t2a.sum(dim=-1) / num_valid_t.squeeze(-1)  # [B, N]
    else:
        s_t2a = max_sim_t2a.mean(dim=-1)  # [B, N]
    
    return s_a2t, s_t2a


def compute_wti_similarity(
    action_tokens: torch.Tensor,   # [B, A, D] L2-normalized action tokens
    text_tokens: torch.Tensor,     # [N, T, D] L2-normalized text tokens
    text_mask: torch.Tensor = None,  # [N, T]
    attn_temperature: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DRL-WTI 风格的软化版 FILIP（Weighted Token-wise Interaction）

    与 FILIP 的差别仅在最外层的"聚合方式"：
        FILIP : s_a2t = mean_a   (max_t sim[a,t])
        WTI   : s_a2t = Σ_a w_a * (max_t sim[a,t]),
                w_a = softmax_a( max_t sim[a,t] / τ_attn )

    直觉：保留"每个token找最相似对应"的fine-grained语义，
         但用softmax权重让"重要token"在聚合中占比更大，
         从而避免FILIP简单mean被无关/padding token拉低的问题。
    """
    sim_matrix = torch.matmul(
        action_tokens.unsqueeze(1),                       # [B, 1, A, D]
        text_tokens.unsqueeze(0).permute(0, 1, 3, 2),     # [1, N, D, T]
    )  # [B, N, A, T]

    # ===== a2t: 先对T取max（要mask掉padding text token），再对A加权聚合 =====
    if text_mask is not None:
        sim_for_a2t = sim_matrix.masked_fill(text_mask[None, :, None, :] == 0, -1e9)
    else:
        sim_for_a2t = sim_matrix
    max_a2t = sim_for_a2t.max(dim=-1)[0]                  # [B, N, A]
    alpha = F.softmax(max_a2t / attn_temperature, dim=-1) # [B, N, A] action token importance
    s_a2t = (alpha * max_a2t).sum(dim=-1)                 # [B, N]

    # ===== t2a: 先对A取max，再对T加权聚合（mask住padding token的权重） =====
    max_t2a = sim_matrix.max(dim=-2)[0]                   # [B, N, T]
    if text_mask is not None:
        max_t2a_for_softmax = max_t2a.masked_fill(text_mask[None, :, :] == 0, -1e9)
    else:
        max_t2a_for_softmax = max_t2a
    beta = F.softmax(max_t2a_for_softmax / attn_temperature, dim=-1)  # [B, N, T] text token importance
    # padding 位置 beta≈0，不贡献和；不需要再mask max_t2a
    s_t2a = (beta * max_t2a).sum(dim=-1)                  # [B, N]

    return s_a2t, s_t2a


def compute_aosm_similarity(
    action_tokens: torch.Tensor,   # [B, A, D] L2-normalized action tokens
    text_tokens: torch.Tensor,     # [N, T, D] L2-normalized text tokens
    text_mask: torch.Tensor = None,  # [N, T]
    attn_temperature: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X-CLIP AOSM 风格的纯 softmax 双层聚合（Frame-Word level，去掉learnable weight的最简版）

    与 FILIP/WTI 区别：完全不取max，每一层都用softmax加权聚合。
        a2t : 先对T softmax + 加权聚合 → [B,N,A]，再对A softmax + 加权聚合 → [B,N]
        t2a : 先对A softmax + 加权聚合 → [B,N,T]，再对T softmax + 加权聚合 → [B,N]

    论文: X-CLIP (ACM MM 2022) 中 _attenion_over_fine_grained_sim_matrix；
    去掉了原文的 word_mat_weight / frame_mat_weight 学习参数，保持最小化设计。
    """
    sim = torch.matmul(
        action_tokens.unsqueeze(1),                        # [B, 1, A, D]
        text_tokens.unsqueeze(0).permute(0, 1, 3, 2),      # [1, N, D, T]
    )  # [B, N, A, T]

    # ===== a2t: T方向先聚合，A方向后聚合 =====
    if text_mask is not None:
        sim_t = sim.masked_fill(text_mask[None, :, None, :] == 0, -1e9)
    else:
        sim_t = sim
    weights_t = F.softmax(sim_t / attn_temperature, dim=-1)        # [B, N, A, T]
    soft_per_action = (weights_t * sim).sum(dim=-1)                # [B, N, A] padding贡献0
    weights_a = F.softmax(soft_per_action / attn_temperature, dim=-1)  # [B, N, A]
    s_a2t = (weights_a * soft_per_action).sum(dim=-1)              # [B, N]

    # ===== t2a: A方向先聚合，T方向后聚合 =====
    weights_a2 = F.softmax(sim / attn_temperature, dim=-2)         # [B, N, A, T]
    soft_per_text = (weights_a2 * sim).sum(dim=-2)                 # [B, N, T]
    if text_mask is not None:
        soft_per_text_for_softmax = soft_per_text.masked_fill(text_mask[None, :, :] == 0, -1e9)
    else:
        soft_per_text_for_softmax = soft_per_text
    weights_t2 = F.softmax(soft_per_text_for_softmax / attn_temperature, dim=-1)
    s_t2a = (weights_t2 * soft_per_text).sum(dim=-1)               # [B, N]

    return s_a2t, s_t2a


def compute_at2tt_contrastive_loss(
    action_tokens: torch.Tensor,   # [B+K, A, D] 投影后的action tokens（包含feature bank）
    text_tokens: torch.Tensor,     # [B+K, T, D] 投影后的text tokens（包含feature bank）
    batch_size: int = None,        # 当前batch大小B（用于区分feature bank）
    text_mask: torch.Tensor = None,    # [B+K, T]
    temperature = 0.07,            # 温度参数（支持float或torch.Tensor，用于可学习温度）
    loss_type: str = 'infonce',
    sigmoid_bias: float = 0.0,
    positive_mask: torch.Tensor = None,  # [B+K, B+K] 多正样本mask
    valid_mask: torch.Tensor = None,     # [B+K] 有效样本mask
    min_valid_ratio: float = 0.0,  # 最小有效样本比例阈值
    aggregation: str = 'filip',    # 'filip'=mean(max), 'wti'=softmax-weighted(max)
    attn_temperature: float = 0.01,  # WTI 中 token importance softmax 的温度
) -> Tuple[torch.Tensor, Dict]:
    """
    AT2TT模式的FILIP风格对比学习损失（正确使用feature bank作为额外负样本）
    
    使用token-wise maximum similarity计算双向对比损失
    支持3种对齐模式：
    - positive_mask=None: 对角线为正样本（默认）
    - positive_mask提供: 根据mask确定正样本（task_id或task_index模式）
    
    Args:
        action_tokens: [B+K, A, D] action token特征（B=当前batch，K=feature bank）
        text_tokens: [B+K, T, D] text token特征
        batch_size: 当前batch大小B（只对前B个样本计算loss）
        text_mask: [B+K, T] text有效性mask
        temperature: 温度参数（float或torch.Tensor，支持可学习温度梯度回传）
        loss_type: 'infonce' 或 'sigmoid'
        sigmoid_bias: sigmoid loss的偏置
        positive_mask: [B+K, B+K] 多正样本mask（1=正样本对，0=负样本对）
        valid_mask: [B+K] 有效样本mask（1=有效，0=无效样本loss为0）
        
    Returns:
        loss: 双向平均损失
        stats: 统计信息
    """
    device = action_tokens.device
    B = action_tokens.shape[0] if batch_size is None else batch_size
    
    # 获取总样本数（包含feature bank）
    total_size = action_tokens.shape[0]  # B+K
    
    # 统一valid_mask：如果未提供则全部有效（大小为 B+K）
    if valid_mask is None:
        valid_mask = torch.ones(total_size, device=device)
    
    # 创建2D valid_mask（用于前B行）
    valid_mask_batch = valid_mask[:B]  # [B]
    valid_mask_2d = valid_mask_batch.unsqueeze(1) * valid_mask.unsqueeze(0)  # [B, B+K]
    valid_col_mask = valid_mask.unsqueeze(0)  # [1, B+K]
    
    # 应用min_valid_ratio过滤：有效配对数比例低于阈值的anchor的loss不计入
    valid_anchor_mask = compute_anchor_valid_mask_by_ratio(
        valid_mask_2d, valid_mask_batch, min_valid_ratio
    )
    
    # L2归一化
    action_norm = F.normalize(action_tokens, p=2, dim=-1)
    text_norm = F.normalize(text_tokens, p=2, dim=-1)

    # 根据 aggregation 选择 token-wise 聚合方式（作为 config 参数从外部传入）：
    #   'filip' = FILIP 原文 mean(max)，硬聚合
    #   'wti'   = DRL-WTI softmax-weighted(max)，软聚合 + 保留 max 语义
    #   'aosm'  = X-CLIP AOSM 双层 softmax，纯软聚合无 max
    if aggregation == 'wti':
        s_a2t, s_t2a = compute_wti_similarity(
            action_norm, text_norm, text_mask, attn_temperature
        )
    elif aggregation == 'aosm':
        s_a2t, s_t2a = compute_aosm_similarity(
            action_norm, text_norm, text_mask, attn_temperature
        )
    else:
        s_a2t, s_t2a = compute_filip_similarity(action_norm, text_norm, text_mask)
    
    # 根据positive_mask确定正样本（大小为 B+K x B+K）
    if positive_mask is None:
        positive_mask = torch.eye(total_size, device=device)
    
    # 统一使用除法应用温度
    logits_a2t = s_a2t[:B, :].clone() / temperature
    logits_t2a = s_t2a[:, :B].t().clone() / temperature
    
    if loss_type == 'sigmoid':
        # 与上面的全局分支保持一致：
        # logits = similarity / temperature - sigmoid_bias
        logits_a2t_with_bias = logits_a2t - sigmoid_bias
        logits_t2a_with_bias = logits_t2a - sigmoid_bias
        
        loss_a2t = compute_single_direction_sigmoid(
            logits_a2t_with_bias, positive_mask[:B, :], valid_mask_2d, valid_anchor_mask
        )
        loss_t2a = compute_single_direction_sigmoid(
            logits_t2a_with_bias, positive_mask[:, :B].t(), valid_mask_2d, valid_anchor_mask
        )
    else:
        # InfoNCE Loss
        loss_a2t = compute_single_direction_infonce(
            logits_a2t, positive_mask[:B, :], valid_col_mask, valid_anchor_mask
        )
        loss_t2a = compute_single_direction_infonce(
            logits_t2a, positive_mask[:, :B].t(), valid_col_mask, valid_anchor_mask
        )
    
    # 统计信息（只统计当前batch内的有效pair）
    pos_mask_batch = positive_mask[:B, :B]  # [B, B]
    valid_2d_batch = valid_mask_batch.unsqueeze(1) * valid_mask_batch.unsqueeze(0)  # [B, B]
    valid_pos_mask = (pos_mask_batch > 0) & (valid_2d_batch > 0)
    valid_neg_mask = (pos_mask_batch == 0) & (valid_2d_batch > 0)
    # 统一用除法后的logits统计
    raw_logits_a2t = s_a2t[:B, :B] / temperature
    if loss_type == 'sigmoid':
        raw_logits_a2t = raw_logits_a2t - sigmoid_bias
    pos_logits = raw_logits_a2t[valid_pos_mask]
    neg_logits = raw_logits_a2t[valid_neg_mask]
    
    # 统计正负样本数：每个有效样本行的平均正样本数和负样本数
    num_pos_per_row = (pos_mask_batch * valid_2d_batch).sum(dim=1)  # [B]
    num_neg_per_row = ((1 - pos_mask_batch) * valid_2d_batch).sum(dim=1)  # [B]
    num_valid_rows = (valid_mask_batch > 0).sum().clamp(min=1)
    avg_pos_samples = num_pos_per_row.sum() / num_valid_rows
    avg_neg_samples = num_neg_per_row.sum() / num_valid_rows
    avg_total_samples = (num_pos_per_row + num_neg_per_row).sum() / num_valid_rows
    
    return (loss_a2t + loss_t2a) / 2, {
        'pos_mean_logit': pos_logits.mean().detach() if len(pos_logits) > 0 else torch.tensor(0.0),
        'neg_mean_logit': neg_logits.mean().detach() if len(neg_logits) > 0 else torch.tensor(0.0),
        'loss_a2t': loss_a2t.detach(),
        'loss_t2a': loss_t2a.detach(),
        'avg_pos_samples': avg_pos_samples.detach(),
        'avg_neg_samples': avg_neg_samples.detach(),
        'avg_total_samples': avg_total_samples.detach(),
    }


# 导入 ClipLoss 供 EgoHOD 使用
from egovlpv2.model.egohod.loss import ClipLoss
