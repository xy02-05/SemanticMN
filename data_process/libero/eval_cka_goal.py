"""
多粒度指标计算:
  - Task subset: all_40 / goal_10 / test_8
  - 粒度: trajectory (轨迹级 mean pool) / chunk (每帧)
  - Text: qwen3 / egohod
  - 指标: CKA, Silhouette, KNN, IntraCos, InterCos, CosGap, EffRank
"""
import os, sys, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, "/root/data/xuyuan1/Codes/analysis")

from config import TEXT_EMBEDDINGS, FEATURE_DIR, RESULT_DIR, LIBERO_SUITES, SPLIT_PATH
from evaluation.metrics import (
    linear_cka, effective_rank as eff_rank_fn,
    knn_classification, intra_inter_similarity,
)
from sklearn.metrics import silhouette_score

LAYER = 10

# 三组 task subset
GOAL_TASKS = set(LIBERO_SUITES["libero_goal"]["task_indices"])
ALL_TASKS = set(range(40))
if os.path.exists(SPLIT_PATH):
    with open(SPLIT_PATH) as _f:
        _split = json.load(_f)
    TEST_TASKS = set(_split.get("test_tasks", []))
else:
    TEST_TASKS = set()

SUBSETS = {"all_40": ALL_TASKS, "goal_10": GOAL_TASKS, "test_8": TEST_TASKS}


def load_features(ckpt_name, mode, granularity):
    """加载 train+test，返回 (feat, task_indices)
    granularity: 'trajectory' 用 features key, 'chunk' 用 chunk_features key
    """
    suffix = "" if mode == "clean" else "_rollout"
    feat_key = "features" if granularity == "trajectory" else "chunk_features"
    task_key = "task_indices" if granularity == "trajectory" else "chunk_task_indices"

    parts = []
    for split in ["train", "test"]:
        p = os.path.join(FEATURE_DIR, f"{ckpt_name}_{split}{suffix}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        if feat_key not in d:
            return None, None
        layers = d["layer_indices"].tolist()
        li = layers.index(LAYER)
        parts.append((d[feat_key][:, li, :], d[task_key]))

    if not parts:
        return None, None
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


def compute_metrics(feat, tasks, text_gallery):
    """对一组 features 计算全部指标"""
    text_aligned = text_gallery[tasks]
    n_tasks = len(set(tasks))

    cka = linear_cka(feat, text_aligned)
    sil = silhouette_score(feat, tasks, metric="cosine") if n_tasks >= 2 else float('nan')
    task_labels = np.array([str(t) for t in tasks])
    knn_acc, _ = knn_classification(feat, task_labels, k=5)
    intra, inter, gap = intra_inter_similarity(feat, task_labels)
    erank = eff_rank_fn(feat)

    return {
        "cka": float(cka), "silhouette": float(sil), "knn": float(knn_acc),
        "intra_cos": float(intra), "inter_cos": float(inter), "cos_gap": float(gap),
        "effective_rank": float(erank), "n_samples": len(feat), "n_tasks": n_tasks,
    }


def main():
    ckpts = [
        ("pretrained", "rollout"),
        ("step_5k", "rollout"),
        ("step_10k", "rollout"),
        ("step_15k", "rollout"),
        ("step_20k", "rollout"),
        ("step_25k", "rollout"),
        ("step_30k", "rollout"),
        ("new_dsn_new_5k", "rollout"),
        ("new_dsn_new_10k", "rollout"),
        ("new_dsn_new_15k", "rollout"),
        ("new_dsn_new_20k", "rollout"),
        ("new_dsn_new_25k", "rollout"),
        ("new_dsn_new_30k", "rollout"),
    ]

    all_results = []

    for text_type in ["qwen3", "egohod"]:
        text_gallery = np.load(TEXT_EMBEDDINGS[text_type]["path"])[TEXT_EMBEDDINGS[text_type]["key"]]

        for granularity in ["trajectory", "chunk"]:
            for subset_name, subset_tasks in SUBSETS.items():
                if not subset_tasks:
                    continue

                print(f"\n{'='*80}")
                print(f"text={text_type}, granularity={granularity}, subset={subset_name} ({len(subset_tasks)} tasks)")
                print(f"{'='*80}")
                print(f"  {'Checkpoint':<20s} {'N':>7s} {'CKA':>7s} {'Sil':>7s} {'KNN':>6s} "
                      f"{'Intra':>7s} {'Inter':>7s} {'Gap':>7s} {'ERank':>6s}")

                for ckpt_name, mode in ckpts:
                    feat, tasks = load_features(ckpt_name, mode, granularity)
                    if feat is None:
                        continue

                    # 筛选 subset
                    mask = np.isin(tasks, list(subset_tasks))
                    if mask.sum() == 0:
                        continue
                    feat_sub, tasks_sub = feat[mask], tasks[mask]

                    # chunk 级样本量很大，为加速对 all_40 做采样
                    if granularity == "chunk" and len(feat_sub) > 10000:
                        rng = np.random.RandomState(42)
                        idx = rng.choice(len(feat_sub), 10000, replace=False)
                        feat_sub, tasks_sub = feat_sub[idx], tasks_sub[idx]

                    m = compute_metrics(feat_sub, tasks_sub, text_gallery)
                    m.update({"checkpoint": ckpt_name, "mode": mode,
                              "text_type": text_type, "granularity": granularity,
                              "subset": subset_name})
                    all_results.append(m)

                    print(f"  {ckpt_name:<20s} {m['n_samples']:>7d} {m['cka']:>7.4f} {m['silhouette']:>7.4f} "
                          f"{m['knn']:>6.3f} {m['intra_cos']:>7.4f} {m['inter_cos']:>7.4f} "
                          f"{m['cos_gap']:>7.4f} {m['effective_rank']:>6.1f}")

    # 保存 JSON
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, "metrics_multi_subset.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")

    # 生成 markdown
    md_path = os.path.join(RESULT_DIR, "metrics_multi_subset.md")
    with open(md_path, "w") as f:
        f.write("# 多粒度指标对比 (Layer 10, Rollout)\n\n")
        f.write("维度: task subset (all_40 / goal_10 / test_8) × 粒度 (trajectory / chunk) × text (qwen3 / egohod)\n\n")

        for text_type in ["qwen3", "egohod"]:
            f.write(f"## Text: {text_type}\n\n")
            for granularity in ["trajectory", "chunk"]:
                f.write(f"### {granularity}\n\n")
                for subset_name in ["all_40", "goal_10", "test_8"]:
                    sub = [r for r in all_results
                           if r["text_type"] == text_type and r["granularity"] == granularity
                           and r["subset"] == subset_name]
                    if not sub:
                        continue
                    f.write(f"**{subset_name}** ({sub[0]['n_tasks']} tasks)\n\n")
                    f.write("| Checkpoint | N | CKA | Sil | KNN | IntraCos | InterCos | Gap | EffRank |\n")
                    f.write("|---|---|---|---|---|---|---|---|---|\n")
                    for r in sub:
                        f.write(f"| {r['checkpoint']} | {r['n_samples']} | {r['cka']:.4f} | "
                                f"{r['silhouette']:.4f} | {r['knn']:.3f} | {r['intra_cos']:.4f} | "
                                f"{r['inter_cos']:.4f} | {r['cos_gap']:.4f} | {r['effective_rank']:.1f} |\n")
                    f.write("\n")

    print(f"保存 markdown: {md_path}")


if __name__ == "__main__":
    main()
