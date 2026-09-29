"""
Action ↔ Text 相似度指标计算（无需训练）

原理:
  直接计算 action features 与 text embeddings 之间的结构相似度，
  衡量 action representation 保持了多少语义信息。

  参考 "Embodied Representation Alignment with Mirror Neurons" (Zhu et al.)
  和分析代码 Codes/analysis/evaluation/metrics.py 中的指标实现。

指标:
  1. Linear CKA     — 核对齐度，旋转/缩放不变 [0,1]
  2. SVCCA / PWCCA   — SVD + CCA，共享子空间度量
  3. Effective Rank   — SVD 信息熵，越低 = 表征越坍缩
  4. KNN Accuracy     — K近邻分类准确率
  5. Intra/Inter Cos  — 类内/类间余弦相似度，gap 越大越好

输出:
  JSON 汇总 + 打印表格

用法:
    python eval_similarity.py --checkpoint pretrained --split test
    python eval_similarity.py --all --split test
"""
import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/root/data/xuyuan1/Codes/analysis")

from config import TEXT_EMBEDDINGS, FEATURE_DIR, RESULT_DIR, CHECKPOINTS, ALL_CHECKPOINTS

from evaluation.metrics import (
    linear_cka,
    svcca,
    pwcca,
    effective_rank,
    knn_classification,
    intra_inter_similarity,
)


# ===================== Platonic-rep metrics (inlined) =====================
# From "The Platonic Representation Hypothesis" (Huh et al., ICML 2024)

def _hsic_unbiased(K, L):
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)
    val = (
        torch.sum(K_tilde * L_tilde.T)
        + torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2))
        - 2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2)
    )
    return val / (m * (m - 3))


def compute_cknna(feats_A, feats_B, topk=10):
    n = feats_A.shape[0]
    if topk < 2:
        topk = 2
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def similarity(K, L, topk):
        K_hat = K.clone().fill_diagonal_(float("-inf"))
        L_hat = L.clone().fill_diagonal_(float("-inf"))
        _, topk_K_idx = torch.topk(K_hat, topk, dim=1)
        _, topk_L_idx = torch.topk(L_hat, topk, dim=1)
        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_idx, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_idx, 1)
        mask = mask_K * mask_L
        return _hsic_unbiased(mask * K, mask * L)

    sim_kl = similarity(K, L, topk)
    sim_kk = similarity(K, K, topk)
    sim_ll = similarity(L, L, topk)
    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def compute_mutual_knn(feats_A, feats_B, topk=10):
    n = feats_A.shape[0]
    knn_A = (feats_A @ feats_A.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    knn_B = (feats_B @ feats_B.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    rng = torch.arange(n, device=feats_A.device).unsqueeze(1)
    mask_A = torch.zeros(n, n, device=feats_A.device)
    mask_B = torch.zeros(n, n, device=feats_A.device)
    mask_A[rng, knn_A] = 1.0
    mask_B[rng, knn_B] = 1.0
    return ((mask_A * mask_B).sum(dim=1) / topk).mean().item()


def load_features(path):
    """加载 extract_features.py 产出的 .npz"""
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"],           # [N, L, D]
        "task_indices": data["task_indices"],     # [N]
        "layer_indices": data["layer_indices"],   # [L]
    }


def load_text_gallery(text_type="qwen3"):
    """加载 text embedding [40, D_text]"""
    cfg = TEXT_EMBEDDINGS[text_type]
    data = np.load(cfg["path"])
    return data[cfg["key"]]


def compute_all_metrics(action_feats, text_gallery, task_indices, layer_id,
                        train_task_set=None):
    """
    对一层 action features 计算所有相似度指标。

    Args:
        action_feats: [N, D_action=1024] 轨迹级 action features
        text_gallery: [40, D_text] text embedding gallery
        task_indices: [N] 每条轨迹的 task_index
        layer_id: 层号（用于打印）
        train_task_set: set of int — 训练集中包含的 task_index。
            如果提供（task 级别划分），KNN 评估时只用训练集 task 作为候选类别

    Returns:
        dict: 所有指标
    """
    N = len(action_feats)
    text_aligned = text_gallery[task_indices]  # [N, D_text]

    results = {"layer": int(layer_id), "N": N}

    # 1. CKA
    results["cka"] = linear_cka(action_feats, text_aligned)

    # 2. SVCCA
    svc_mean, _, svc_info = svcca(action_feats, text_aligned)
    results["svcca"] = svc_mean

    # 3. PWCCA
    pwc_val, _ = pwcca(action_feats, text_aligned)
    results["pwcca"] = pwc_val

    # 4. Effective Rank
    results["effective_rank"] = effective_rank(action_feats)

    # 5. KNN 分类
    task_labels = np.array([str(t) for t in task_indices])
    knn_acc, knn_std = knn_classification(action_feats, task_labels, k=5)
    results["knn_acc"] = knn_acc
    results["knn_std"] = knn_std

    # 6. 类内/类间余弦相似度
    intra, inter, gap = intra_inter_similarity(action_feats, task_labels)
    results["intra_cos"] = intra
    results["inter_cos"] = inter
    results["cos_gap"] = gap

    # 7. CKNNA + Mutual KNN (action↔text cross-modal alignment)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a_t = F.normalize(torch.from_numpy(action_feats).float().to(device), dim=-1)
        t_t = F.normalize(torch.from_numpy(text_aligned).float().to(device), dim=-1)
        topk = min(10, N - 1)
        results["cknna"] = compute_cknna(a_t, t_t, topk=topk)
        results["mutual_knn"] = compute_mutual_knn(a_t, t_t, topk=topk)
    except Exception as e:
        print(f"  CKNNA/mutual_knn failed: {e}")
        results["cknna"] = None
        results["mutual_knn"] = None

    # 8. Gallery Retrieval (task 级别划分 + 维度必须一致才能算余弦相似度)
    D_action = action_feats.shape[1]
    D_text = text_gallery.shape[1]
    if train_task_set is not None and D_action == D_text:
        norms_a = np.linalg.norm(action_feats, axis=1, keepdims=True)
        norms_a = np.clip(norms_a, 1e-8, None)
        action_norm = action_feats / norms_a
        norms_g = np.linalg.norm(text_gallery, axis=1, keepdims=True)
        norms_g = np.clip(norms_g, 1e-8, None)
        gallery_norm = text_gallery / norms_g
        # 余弦相似度 [N, 40]
        sim = action_norm @ gallery_norm.T
        preds = np.argsort(-sim, axis=1)
        for k in [1, 3, 5]:
            topk = preds[:, :k]
            correct = np.any(topk == task_indices[:, None], axis=1)
            results[f"retrieval_top{k}"] = float(correct.mean())

    return results


def evaluate_checkpoint(feature_path, text_type="qwen3", only_layer=None):
    """对一个 checkpoint 的 feature 文件计算指标。only_layer=10 只算该层。"""
    from config import SPLIT_PATH

    label = os.path.basename(feature_path).replace(".npz", "")
    data = np.load(feature_path, allow_pickle=True, mmap_mode='r')
    features = data["features"]         # [N, L, D]  (mmap, not loaded to RAM yet)
    task_indices = np.array(data["task_indices"])
    layer_indices = np.array(data["layer_indices"])

    text_gallery = load_text_gallery(text_type)  # [40, D_text]

    train_task_set = None
    split_mode = "unknown"
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            split_info = json.load(f)
        split_mode = split_info.get("mode", "episode")
        if split_mode == "task":
            train_task_set = set(split_info["train_tasks"])

    unique_test_tasks = sorted(set(task_indices.tolist()))

    print(f"\n{'='*70}")
    print(f"相似度评估: {label} (text={text_type}, split_mode={split_mode})")
    print(f"  Features: {features.shape}, Gallery: {text_gallery.shape}")
    if only_layer is not None:
        print(f"  Only layer: {only_layer}")
    print(f"{'='*70}")

    all_results = []
    for li_pos, layer_id in enumerate(layer_indices):
        if only_layer is not None and int(layer_id) != only_layer:
            continue
        layer_feat = np.array(features[:, li_pos, :])  # copy from mmap
        metrics = compute_all_metrics(
            layer_feat, text_gallery, task_indices, int(layer_id),
            train_task_set=train_task_set)
        all_results.append(metrics)

    # 打印汇总表
    has_retrieval = "retrieval_top1" in all_results[0]
    has_cknna = all_results[0].get("cknna") is not None
    header = (f"  {'Layer':>5s}  {'CKA':>6s}  {'CKNNA':>6s}  {'MutKNN':>6s}  {'SVCCA':>6s}"
              f"  {'EffRank':>7s}  {'KNN':>5s}  {'IntraCos':>8s}  {'InterCos':>8s}  {'Gap':>6s}")
    if has_retrieval:
        header += f"  {'R@1':>5s}  {'R@5':>5s}"
    print(f"\n{header}")
    for r in all_results:
        cknna_str = f"{r['cknna']:>6.4f}" if r.get('cknna') is not None else "   N/A"
        mknn_str = f"{r['mutual_knn']:>6.4f}" if r.get('mutual_knn') is not None else "   N/A"
        line = (f"  {r['layer']:>5d}  {r['cka']:>6.4f}  {cknna_str}  {mknn_str}  {r['svcca']:>6.4f}"
                f"  {r['effective_rank']:>7.1f}  {r['knn_acc']:>5.3f}"
                f"  {r['intra_cos']:>8.4f}  {r['inter_cos']:>8.4f}  {r['cos_gap']:>6.4f}")
        if has_retrieval:
            line += f"  {r['retrieval_top1']:>5.3f}  {r['retrieval_top5']:>5.3f}"
        print(line)

    # 保存 JSON
    os.makedirs(RESULT_DIR, exist_ok=True)
    summary = {
        "label": label,
        "text_type": text_type,
        "split_mode": split_mode,
        "feature_path": feature_path,
        "n_trajectories": int(len(task_indices)),
        "test_tasks": unique_test_tasks,
        "per_layer": {int(r["layer"]): {k: float(v) if isinstance(v, (float, np.floating)) else v
                      for k, v in r.items()} for r in all_results},
    }
    json_path = os.path.join(RESULT_DIR, f"similarity_{label}_{text_type}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n保存: {json_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="LIBERO Action-Text 相似度评估")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 名 (如 pretrained, step_5k, ..., 或 all)")
    parser.add_argument("--feature_path", type=str, default=None,
                        help="直接指定 feature npz 路径（覆盖 --checkpoint）")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--feature_mode", type=str, default="clean",
                        choices=["clean", "rollout"])
    parser.add_argument("--layer", type=int, default=None,
                        help="只评估指定层（如 10），不指定则评估所有层")
    parser.add_argument("--text_type", type=str, default="qwen3",
                        choices=["qwen3", "egohod"])
    args = parser.parse_args()

    mode_suffix = {"clean": "", "rollout": "_rollout"}[args.feature_mode]

    def get_feature_path(ckpt_name, split):
        suffix = f"_{split}" if split != "all" else ""
        return os.path.join(FEATURE_DIR, f"{ckpt_name}{suffix}{mode_suffix}.npz")

    if args.feature_path:
        evaluate_checkpoint(args.feature_path, args.text_type, only_layer=args.layer)
    elif args.checkpoint == "all":
        results = []
        for name in ALL_CHECKPOINTS:
            fp = get_feature_path(name, args.split)
            if os.path.exists(fp):
                r = evaluate_checkpoint(fp, args.text_type, only_layer=args.layer)
                results.append(r)
            else:
                print(f"跳过 {name}: {fp} 不存在")

        if results:
            target_layer = args.layer if args.layer is not None else 10
            print(f"\n{'='*80}")
            print(f"跨 Checkpoint 对比 (Layer {target_layer}, {args.feature_mode}, text={args.text_type})")
            print(f"  {'Label':<30s}  {'CKA':>6s}  {'CKNNA':>6s}  {'MutKNN':>6s}"
                  f"  {'KNN':>5s}  {'CosGap':>7s}")
            for r in results:
                l = r["per_layer"].get(target_layer, {})
                if l:
                    cknna = l.get('cknna')
                    mknn = l.get('mutual_knn')
                    print(f"  {r['label']:<30s}"
                          f"  {l.get('cka',0):>6.4f}"
                          f"  {cknna:>6.4f}" if cknna is not None else f"  {'N/A':>6s}"
                          f"  {mknn:>6.4f}" if mknn is not None else f"  {'N/A':>6s}"
                          f"  {l.get('knn_acc',0):>5.3f}"
                          f"  {l.get('cos_gap',0):>7.4f}")
            print(f"{'='*80}")

    elif args.checkpoint:
        fp = get_feature_path(args.checkpoint, args.split)
        evaluate_checkpoint(fp, args.text_type, only_layer=args.layer)
    else:
        parser.error("需要 --checkpoint <名称> 或 --feature_path")


if __name__ == "__main__":
    main()
