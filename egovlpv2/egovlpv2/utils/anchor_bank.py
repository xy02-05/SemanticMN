"""
anchor_bank.py — LIBERO 40 task × (paraphrase + counterfactual) anchor 特征银行

设计原则:
  1. **离线编码**: 由 build_anchor_features.py 用 Qwen3-VL-Embedding-8B 一次性编码到 .npz
  2. **运行期零开销**: AnchorBank 用 register_buffer 把 .npz 全量上 GPU,
     训练时按 task_id 索引 + 随机 sample K_neg_sample 个 counterfactual (不走全量)
  3. **多样性**: 每个训练 step 给同一 task 的 batch 抽不同 K 个 counterfactual,
     30 个候选里抽 8-12, 防止 q_a 记住固定的 30 个 hard-neg
  4. **向量化**: 全用 torch.gather / torch.randint, 无 Python for-loop

加载格式 (来自 libero_anchor_qwen3vl_features.npz):
  orig_emb     : (N=40, D=4096) float32, L2-normalized
  para_emb     : (N, K_p=10, D)  float32, L2-normalized
  cfact_emb    : (N, K_n=32, D)  float32, L2-normalized
  cfact_ood_mask: (N, K_n) bool, OOD-task 那 1-2 条最优质 counterfactual 的位置

用法 (在 fg_alignment_model.py):
  bank = AnchorBank(npz_path)
  bank.to(device)
  e_pos, e_para, e_cfact = bank.lookup(task_ids, K_neg_sample=10)
  # e_pos:    [B, D]
  # e_para:   [B, K_p, D]
  # e_cfact:  [B, K_neg_sample, D]  (从 K_n=32 中随机抽样)
"""
import numpy as np
import torch
import torch.nn as nn


class AnchorBank(nn.Module):
    """LIBERO instruction anchor bank (frozen, on GPU).

    内部存 3 个 buffer (orig/para/cfact), 加载后**不可训练**.
    支持: by-task lookup + 每 step 随机 sample K_neg.
    """

    def __init__(self, npz_path: str, prefer_ood: bool = True):
        """
        Args:
          npz_path: build_anchor_features.py 生成的 .npz 路径
          prefer_ood: True 时, sample 时优先包含 OOD-task 标记的 counterfactual (评测信号同构)
        """
        super().__init__()
        d = np.load(npz_path, allow_pickle=True)
        # 主要 tensor 用 register_buffer, 自动跟随 .to() / DDP / bf16
        self.register_buffer('orig_emb',  torch.from_numpy(d['orig_emb']).float())   # [N, D]
        self.register_buffer('para_emb',  torch.from_numpy(d['para_emb']).float())   # [N, K_p, D]
        self.register_buffer('cfact_emb', torch.from_numpy(d['cfact_emb']).float())  # [N, K_n, D]
        # OOD mask: 哪些 counterfactual 来自 OOD task (评测同构, 优先 sample)
        self.register_buffer('cfact_ood_mask', torch.from_numpy(d['cfact_ood_mask']).bool())  # [N, K_n]

        self.N, self.D = self.orig_emb.shape
        self.K_p = self.para_emb.shape[1]
        self.K_n = self.cfact_emb.shape[1]
        self.prefer_ood = prefer_ood

        # 不参与训练
        for buf in [self.orig_emb, self.para_emb, self.cfact_emb]:
            buf.requires_grad_(False)

    def lookup(self, task_ids: torch.Tensor, K_neg_sample: int = 10):
        """
        Args:
          task_ids: [B] long, batch 内每条样本的 task_index ∈ [0, N)
          K_neg_sample: 每条样本要抽几个 counterfactual (推荐 8-12)
        Returns:
          e_pos:   [B, D]              原句 embedding
          e_para:  [B, K_p, D]         全部 paraphrase (10 个全要, 多正样本)
          e_cfact: [B, K_neg_sample, D] 随机抽样的 counterfactual
        """
        B = task_ids.shape[0]
        device = task_ids.device

        # 索引: 直接用 task_ids 作 row 选择 (向量化)
        e_pos  = self.orig_emb[task_ids]                # [B, D]
        e_para = self.para_emb[task_ids]                # [B, K_p, D]
        cfact_all = self.cfact_emb[task_ids]            # [B, K_n, D]

        # ==== 随机 sample K_neg_sample 个 counterfactual ====
        # 策略: prefer_ood=True 时优先选 OOD-task 那几条 (1-2 条/task), 剩余从 LLM 生成的 ~30 条随机抽
        if self.prefer_ood:
            ood_mask = self.cfact_ood_mask[task_ids]    # [B, K_n] bool
            # 随机给每个 [b, k] 一个分数 ∈ [0, 1), OOD 行 +2 (保证排前)
            scores = torch.rand(B, self.K_n, device=device)
            scores = scores + ood_mask.float() * 2.0    # OOD 项分数 ≥ 1, 普通项 < 1
            # topk 选前 K_neg_sample 个 → indices [B, K_neg_sample]
            _, sample_idx = scores.topk(K_neg_sample, dim=-1)
        else:
            # 纯随机 (不优先 OOD): randperm-like via argsort(rand)
            scores = torch.rand(B, self.K_n, device=device)
            _, sample_idx = scores.topk(K_neg_sample, dim=-1)

        # gather: cfact_all [B, K_n, D] → e_cfact [B, K_neg_sample, D]
        # gather 需要 index 形状 [B, K_neg_sample, D] (广播 D 维)
        gather_idx = sample_idx.unsqueeze(-1).expand(-1, -1, self.D)
        e_cfact = cfact_all.gather(dim=1, index=gather_idx)

        return e_pos, e_para, e_cfact

    def get_full_cfact(self, task_ids: torch.Tensor):
        """返回全部 K_n=32 counterfactual, 不抽样. 用于 ablation 或 PKT-ext."""
        return self.cfact_emb[task_ids]                 # [B, K_n, D]

    def __repr__(self):
        return (f"AnchorBank(N={self.N}, K_p={self.K_p}, K_n={self.K_n}, "
                f"D={self.D}, prefer_ood={self.prefer_ood})")
