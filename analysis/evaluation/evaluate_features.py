"""
统一特征评估脚本

对任意模型的标准 npz 特征文件，计算:
  1. CKA / SVCCA / PWCCA (vs EgoHOD / Qwen text embeddings)
     - 三种聚合策略: all_expand / task_mean / single
  2. KNN task classification (k=5, 5-fold CV)
  3. Silhouette Score
  4. Intra/Inter-class cosine similarity
  5. Effective Rank

输入: 标准 npz 格式 (features [N,L,D], task_labels [N], task_indices [N], layer_indices [L])
输出: JSON 报告 + 控制台逐层输出

用法:
    python evaluate_features.py --feature_path PATH_TO_NPZ --output_dir DIR
    python evaluate_features.py --feature_path PATH_TO_NPZ --metrics cka svcca knn
"""
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
sys.path.insert(0, ANALYSIS_DIR)

from evaluation.metrics import (
    linear_cka, svcca, pwcca, compute_cka_svcca_pwcca,
    knn_classification, silhouette_score, intra_inter_similarity,
    calinski_harabasz, effective_rank,
    build_all_expand, build_task_mean, build_single,
    load_text_embeddings, build_task_text_dict,
)

# ============================================================
# 默认路径 (可通过参数覆盖)
# ============================================================
# 数据路径 (从 bridge_representation/config.py 同步)
DATA_ROOT = "/root/data/xuyuan1/dataset"
META_DIR = os.path.join(DATA_ROOT, "bridge_orig/bridge_orig_lerobot/meta")
DEFAULT_TASK_RLDS_PATH = os.path.join(META_DIR, "task_rlds.jsonl")
DEFAULT_EGOHOD_PATH = os.path.join(DATA_ROOT, "embedding/bridge_text_egohod_proj.npz")
DEFAULT_QWEN_PATH = os.path.join(DATA_ROOT, "embedding/bridge_text_embeddings.npz")
DEFAULT_OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "evaluation/outputs")

N_SINGLE_TRIALS = 3  # single 策略采样次数（大数据集下减少以加速）


# ============================================================
# 特征加载
# ============================================================

def load_feature_npz(path):
    """
    加载标准 npz 特征文件

    必须字段: features, task_labels, layer_indices
    可选字段: task_indices, task_ids, n_timesteps
    """
    data = np.load(path, allow_pickle=True)
    features = data['features']          # [N, L, D]
    task_labels = data['task_labels']    # [N] str
    layer_indices = data['layer_indices']  # [L]
    # 可选字段
    task_indices = data['task_indices'] if 'task_indices' in data.files else None
    task_ids = data['task_ids'] if 'task_ids' in data.files else None
    n_timesteps = data['n_timesteps'] if 'n_timesteps' in data.files else None

    print(f"  特征文件: {path}")
    print(f"    shape={features.shape}, dtype={features.dtype}")
    print(f"    unique_tasks={len(np.unique(task_labels))}, layers={list(layer_indices)}")
    return {
        'features': features,
        'task_labels': task_labels,
        'layer_indices': layer_indices,
        'task_indices': task_indices,
        'task_ids': task_ids,
        'n_timesteps': n_timesteps,
    }


# ============================================================
# CKA / SVCCA / PWCCA 评估
# ============================================================

def evaluate_cka_svcca(features, task_labels, layer_indices, text_dict, valid_tasks,
                       text_types=None, strategies=None):
    """
    逐层计算 CKA / SVCCA / PWCCA

    Args:
        features: [N, L, D]
        task_labels: [N] str
        layer_indices: [L]
        text_dict: {'egohod': {label: emb}, 'qwen': {label: emb}}
        valid_tasks: list of valid task labels
        text_types: ["ego", "qwen"] 或子集 (默认 ["ego"])
        strategies: ["all", "task_mean"] 或子集 (默认 ["all", "task_mean"])

    Returns:
        dict: per-layer 结果
    """
    if text_types is None:
        text_types = ["ego"]  # 默认只算 EgoHOD，Qwen 4096维 SVD 太慢
    if strategies is None:
        strategies = ["all", "task_mean"]  # 默认不计算 single (大数据集太慢)

    # 过滤到 valid_tasks
    mask = np.isin(task_labels, valid_tasks)
    feats = features[mask]
    labels = task_labels[mask]
    N_total = len(labels)
    T = len(valid_tasks)

    print(f"\n  CKA/SVCCA/PWCCA: {T} tasks, {N_total} trajectories")
    print(f"    text_types={text_types}, strategies={strategies}")

    results = {
        "n_tasks": T, "n_trajectories": N_total,
        "layers": [int(l) for l in layer_indices],
    }
    for tt in text_types:
        results[tt] = {agg: {"cka": [], "svcca": [], "pwcca": [],
                             "N": [], "kx": [], "ky": []}
                       for agg in strategies + (["single_mean", "single_std"] if "single" in strategies else [])}

    # 预构建 label index (只做一次, 避免每层重复)
    from evaluation.metrics import _build_label_index
    label_idx = _build_label_index(labels, valid_tasks)

    for i, li in enumerate(layer_indices):
        f_layer = feats[:, i, :]  # [N, D]

        for tt in text_types:
            t_dict = text_dict[tt.replace("ego", "egohod")]

            if "all" in strategies:
                # 策略1: all_expand — 直接用预建的 index 构建
                vla_rows, txt_rows = [], []
                for t in valid_tasks:
                    idxs = label_idx.get(t)
                    if idxs is None or len(idxs) == 0:
                        continue
                    vla_rows.append(f_layer[idxs])
                    txt_rows.append(np.tile(t_dict[t], (len(idxs), 1)))
                vla_all = np.concatenate(vla_rows)
                txt_all = np.concatenate(txt_rows)
                m_all = compute_cka_svcca_pwcca(vla_all, txt_all)
                for k in ["cka", "svcca", "pwcca", "N"]:
                    results[tt]["all"][k].append(m_all[k])
                results[tt]["all"]["kx"].append(m_all["svcca_kx"])
                results[tt]["all"]["ky"].append(m_all["svcca_ky"])

            if "task_mean" in strategies:
                # 策略2: task_mean
                vla_tm = np.stack([f_layer[label_idx[t]].mean(axis=0)
                                   for t in valid_tasks if t in label_idx])
                txt_tm = np.stack([t_dict[t] for t in valid_tasks if t in label_idx])
                m_tm = compute_cka_svcca_pwcca(vla_tm, txt_tm)
                for k in ["cka", "svcca", "pwcca", "N"]:
                    results[tt]["task_mean"][k].append(m_tm[k])
                results[tt]["task_mean"]["kx"].append(m_tm["svcca_kx"])
                results[tt]["task_mean"]["ky"].append(m_tm["svcca_ky"])

            if "single" in strategies:
                # 策略3: single (多次采样)
                rng = np.random.RandomState(42)
                s_cka, s_svc, s_pwc = [], [], []
                for _ in range(N_SINGLE_TRIALS):
                    vla_s = np.stack([f_layer[rng.choice(label_idx[t])]
                                      for t in valid_tasks if t in label_idx])
                    txt_s = np.stack([t_dict[t] for t in valid_tasks if t in label_idx])
                    ms = compute_cka_svcca_pwcca(vla_s, txt_s)
                    s_cka.append(ms["cka"]); s_svc.append(ms["svcca"]); s_pwc.append(ms["pwcca"])
                results[tt]["single_mean"]["cka"].append(float(np.mean(s_cka)))
                results[tt]["single_mean"]["svcca"].append(float(np.mean(s_svc)))
                results[tt]["single_mean"]["pwcca"].append(float(np.mean(s_pwc)))
                results[tt]["single_std"]["cka"].append(float(np.std(s_cka)))
                results[tt]["single_std"]["svcca"].append(float(np.std(s_svc)))
                results[tt]["single_std"]["pwcca"].append(float(np.std(s_pwc)))

        # 打印本层 (首个 text_type, all_expand 为主)
        tt0 = text_types[0]
        if "all" in strategies:
            ea = results[tt0]["all"]
            print(f"  L{li:>2d}  "
                  f"ALL(N={ea['N'][-1]}): CKA={ea['cka'][-1]:.4f} "
                  f"SVCCA={ea['svcca'][-1]:.4f}(kx={ea['kx'][-1]},ky={ea['ky'][-1]}) "
                  f"PWCCA={ea['pwcca'][-1]:.4f}", end="")
        if "task_mean" in strategies:
            et = results[tt0]["task_mean"]
            print(f"  |  MEAN(N={et['N'][-1]}): CKA={et['cka'][-1]:.4f} "
                  f"SVCCA={et['svcca'][-1]:.4f} PWCCA={et['pwcca'][-1]:.4f}", end="")
        print()
        sys.stdout.flush()

    return results


# ============================================================
# 聚类/分类指标评估
# ============================================================

def evaluate_cluster_metrics(features, task_labels, layer_indices):
    """
    逐层计算 KNN / Silhouette / Intra-Inter / CH / EffRank

    Returns:
        dict: per-layer 结果
    """
    results = {
        "layers": [int(l) for l in layer_indices],
        "knn_acc": [], "knn_std": [],
        "silhouette": [],
        "intra_sim": [], "inter_sim": [], "sim_gap": [],
        "calinski_harabasz": [],
        "effective_rank": [],
    }

    for i, li in enumerate(layer_indices):
        f_layer = features[:, i, :]  # [N, D]
        labels = task_labels

        # KNN
        knn_acc, knn_std = knn_classification(f_layer, labels)
        results["knn_acc"].append(knn_acc)
        results["knn_std"].append(knn_std)

        # Silhouette
        sil = silhouette_score(f_layer, labels)
        results["silhouette"].append(sil)

        # Intra/Inter
        intra, inter, gap = intra_inter_similarity(f_layer, labels)
        results["intra_sim"].append(intra)
        results["inter_sim"].append(inter)
        results["sim_gap"].append(gap)

        # Calinski-Harabasz
        ch = calinski_harabasz(f_layer, labels)
        results["calinski_harabasz"].append(ch)

        # Effective Rank
        er = effective_rank(f_layer)
        results["effective_rank"].append(er)

        print(f"  L{li:>2d}  KNN={knn_acc:.4f}±{knn_std:.3f}  "
              f"Sil={sil:.4f}  Gap={gap:.4f}  CH={ch:.0f}  ER={er:.1f}")

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="统一特征评估")
    parser.add_argument("--feature_path", type=str, required=True,
                        help="标准 npz 特征文件路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", type=str, default=None,
                        help="模型标签 (用于报告命名, 默认从文件名推断)")
    parser.add_argument("--metrics", nargs="+",
                        default=["cka", "svcca", "knn", "cluster"],
                        help="要计算的指标集: cka svcca knn cluster")
    parser.add_argument("--text_types", nargs="+", default=["ego"],
                        help="text embedding 类型: ego qwen (默认仅 ego)")
    parser.add_argument("--strategies", nargs="+", default=["all", "task_mean"],
                        help="聚合策略: all task_mean single (默认 all+task_mean)")
    parser.add_argument("--task_rlds_path", type=str, default=DEFAULT_TASK_RLDS_PATH)
    parser.add_argument("--egohod_path", type=str, default=DEFAULT_EGOHOD_PATH)
    parser.add_argument("--qwen_path", type=str, default=DEFAULT_QWEN_PATH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 推断标签
    if args.label is None:
        args.label = os.path.basename(args.feature_path).replace(".npz", "")

    print("=" * 90)
    print(f"统一特征评估: {args.label}")
    print(f"  特征文件: {args.feature_path}")
    print(f"  指标: {args.metrics}")
    print("=" * 90)

    # 加载特征
    data = load_feature_npz(args.feature_path)
    features = data['features']
    task_labels = data['task_labels']
    layer_indices = data['layer_indices']

    all_results = {"label": args.label, "timestamp": datetime.now().isoformat()}

    # ---- CKA / SVCCA / PWCCA ----
    need_text = any(m in args.metrics for m in ["cka", "svcca"])
    if need_text:
        print("\n加载 text embeddings...")
        text_emb_data = load_text_embeddings(
            args.task_rlds_path, args.egohod_path, args.qwen_path)
        task_dict = build_task_text_dict(text_emb_data, task_labels)
        valid_tasks = task_dict['valid_tasks']
        text_dict = {'egohod': task_dict['egohod'], 'qwen': task_dict['qwen']}
        print(f"  有效 tasks: {len(valid_tasks)} / {len(np.unique(task_labels))}")

        cka_results = evaluate_cka_svcca(
            features, task_labels, layer_indices, text_dict, valid_tasks,
            text_types=args.text_types, strategies=args.strategies)
        all_results["cka_svcca"] = cka_results

    # ---- KNN / Silhouette / Intra-Inter / CH / EffRank ----
    need_cluster = any(m in args.metrics for m in ["knn", "cluster"])
    if need_cluster:
        print("\n计算聚类/分类指标...")
        cluster_results = evaluate_cluster_metrics(features, task_labels, layer_indices)
        all_results["cluster"] = cluster_results

    # ---- 保存 JSON ----
    json_path = os.path.join(args.output_dir, f"eval_{args.label}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {json_path}")

    # ---- 最终摘要 ----
    print(f"\n{'=' * 90}")
    print(f"评估完成: {args.label}")
    if "cka_svcca" in all_results:
        ea = all_results["cka_svcca"]["ego"]["all"]
        best_idx = int(np.argmax(ea["cka"]))
        best_layer = all_results["cka_svcca"]["layers"][best_idx]
        print(f"  最佳 CKA (EgoHOD, all): L{best_layer} = {ea['cka'][best_idx]:.4f}")
        print(f"  对应 SVCCA = {ea['svcca'][best_idx]:.4f}, PWCCA = {ea['pwcca'][best_idx]:.4f}")
    if "cluster" in all_results:
        cr = all_results["cluster"]
        best_knn_idx = int(np.argmax(cr["knn_acc"]))
        best_knn_layer = cr["layers"][best_knn_idx]
        print(f"  最佳 KNN: L{best_knn_layer} = {cr['knn_acc'][best_knn_idx]:.4f}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
