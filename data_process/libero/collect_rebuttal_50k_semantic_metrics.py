"""
汇总 Rebuttal 5k-50k 的真实语义结构曲线。

本脚本不使用论文绘图文件中的手填数值，而是从 rollout NPZ 和 probe JSON
重新计算/读取：
1. Layer-10 trajectory CKA（Raw/Ours，train+test 共 40 个任务）；
2. 与现有 5k-30k 相同协议的 qwen3 rollout probe test top-1。
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import ANALYSIS_ROOT, FEATURE_DIR, PROBE_DIR, RESULT_DIR, TEXT_EMBEDDINGS


sys.path.insert(0, ANALYSIS_ROOT)
from evaluation.metrics import linear_cka


STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
LAYER = 10


def checkpoint_name(kind, step):
    if kind == "raw":
        return f"step_{step}k"
    return f"new_dsn_new_{step}k"


def load_layer10_all_tasks(name):
    """合并 task-disjoint train/test，构造同一 40-task CKA 口径。"""
    features = []
    tasks = []
    for split in ("train", "test"):
        path = os.path.join(FEATURE_DIR, f"{name}_{split}_rollout.npz")
        data = np.load(path, allow_pickle=False)
        layer_indices = data["layer_indices"].tolist()
        layer_position = layer_indices.index(LAYER)
        features.append(data["features"][:, layer_position, :])
        tasks.append(data["task_indices"])
    return np.concatenate(features), np.concatenate(tasks)


def compute_cka(name, text_gallery):
    features, tasks = load_layer10_all_tasks(name)
    text_features = text_gallery[tasks]
    return {
        "cka": float(linear_cka(features, text_features)),
        "n_trajectories": int(len(features)),
        "n_tasks": int(len(np.unique(tasks))),
    }


def load_probe(name):
    path = os.path.join(PROBE_DIR, f"{name}_qwen3_rollout", "probe_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as file:
        result = json.load(file)
    layer_result = result["per_layer_results"][str(LAYER)]
    return {
        "test_top1": float(layer_result["test_metrics"]["top1"]),
        "test_cosine": float(layer_result["test_metrics"]["mean_cos_sim"]),
        "best_epoch": int(layer_result["best_epoch"]),
        "n_train": int(result["n_train"]),
        "n_test": int(result["n_test"]),
    }


def plot_metric(rows, key, ylabel, output_name):
    fig, axis = plt.subplots(figsize=(8, 5))
    for kind, color, label in (
        ("raw", "#888888", "Vanilla FT"),
        ("ours", "#3A6FB0", "Ours"),
    ):
        selected = [row for row in rows if row["kind"] == kind and row.get(key) is not None]
        axis.plot(
            [row["step_k"] for row in selected],
            [row[key] for row in selected],
            "o-",
            color=color,
            linewidth=2.5,
            label=label,
        )
    axis.set_xlabel("Training Steps (k)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(STEPS)
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f"{output_name}.png"), dpi=200)
    fig.savefig(os.path.join(RESULT_DIR, f"{output_name}.pdf"))
    plt.close(fig)


def main():
    text_config = TEXT_EMBEDDINGS["qwen3"]
    text_gallery = np.load(text_config["path"])[text_config["key"]]
    rows = []

    for kind in ("raw", "ours"):
        for step in STEPS:
            name = checkpoint_name(kind, step)
            train_path = os.path.join(FEATURE_DIR, f"{name}_train_rollout.npz")
            test_path = os.path.join(FEATURE_DIR, f"{name}_test_rollout.npz")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                continue

            cka_result = compute_cka(name, text_gallery)
            probe_result = load_probe(name)
            row = {
                "kind": kind,
                "checkpoint": name,
                "step_k": step,
                **cka_result,
                "probe_test_top1": None,
                "probe_test_cosine": None,
                "probe_best_epoch": None,
            }
            if probe_result is not None:
                row.update({
                    "probe_test_top1": probe_result["test_top1"],
                    "probe_test_cosine": probe_result["test_cosine"],
                    "probe_best_epoch": probe_result["best_epoch"],
                    "probe_n_train": probe_result["n_train"],
                    "probe_n_test": probe_result["n_test"],
                })
            rows.append(row)

    os.makedirs(RESULT_DIR, exist_ok=True)
    output = {
        "metric_definition": {
            "semantic_main": "linear CKA on layer-10 rollout trajectory features, train+test 40 tasks",
            "probe_auxiliary": "qwen3 rollout-to-rollout layer-10 probe, existing train_probe.py protocol",
        },
        "provenance_warning": (
            "The legacy Figure-2 retrieval values are hardcoded in plot_diagnostic_*.py "
            "and have no traceable raw result file; they are not extended here."
        ),
        "rows": rows,
    }
    json_path = os.path.join(RESULT_DIR, "rebuttal_50k_semantic_curves.json")
    with open(json_path, "w") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    markdown = [
        "# Rebuttal 5k-50k Semantic Curves",
        "",
        "> All values below are regenerated from rollout NPZ/probe JSON.",
        "> Legacy hardcoded Figure-2 retrieval numbers are not mixed into this table.",
        "",
        "| Kind | Step | CKA | Probe Top-1 | Probe Cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        top1 = "—" if row["probe_test_top1"] is None else f"{row['probe_test_top1']:.4f}"
        cosine = "—" if row["probe_test_cosine"] is None else f"{row['probe_test_cosine']:.4f}"
        markdown.append(
            f"| {row['kind']} | {row['step_k']}k | {row['cka']:.4f} | {top1} | {cosine} |"
        )
    md_path = os.path.join(RESULT_DIR, "rebuttal_50k_semantic_curves.md")
    with open(md_path, "w") as file:
        file.write("\n".join(markdown) + "\n")

    plot_metric(rows, "cka", "Layer-10 Rollout CKA", "rebuttal_50k_cka")
    plot_metric(rows, "probe_test_top1", "Rollout Probe Top-1", "rebuttal_50k_probe_top1")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
