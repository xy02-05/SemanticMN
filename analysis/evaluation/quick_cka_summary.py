"""
快速 CKA 汇总: 6模型全层 + layer10 对比
只算 Linear CKA (跳过耗时的 SVCCA/PWCCA)

用法:
    cd /root/data/xuyuan1/Codes/analysis
    python evaluation/quick_cka_summary.py
"""
import sys
import os
import time
import json
import numpy as np

ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
sys.path.insert(0, ANALYSIS_DIR)

from evaluation.metrics import (
    load_text_embeddings, build_task_text_dict,
    linear_cka,
    build_all_expand, build_task_mean,
)
from bridge_representation.config import TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH

# ====================================================================
# 6 个模型的特征文件
# ====================================================================
FEATURE_FILES = {
    "SVLA_pretrained": os.path.join(
        ANALYSIS_DIR, "bridge_representation/outputs/features/pretrained_chunk4_features.npz"),
    "SVLA_raw_ft": os.path.join(
        ANALYSIS_DIR, "bridge_representation/outputs/features/raw_ft_chunk4_features.npz"),
    "SVLA_cotrain_fg": os.path.join(
        ANALYSIS_DIR, "bridge_representation/outputs/features/cotrain_fg_chunk4_features.npz"),
    "OpenPI_pretrained": os.path.join(
        ANALYSIS_DIR, "openpi_representation/outputs/features/pretrained_chunk5_features.npz"),
    "OpenPI_raw_ft": os.path.join(
        ANALYSIS_DIR, "openpi_representation/outputs/features/raw_ft_chunk5_features.npz"),
    "OpenPI_aligned": os.path.join(
        ANALYSIS_DIR, "openpi_representation/outputs/features/aligned_chunk5_features.npz"),
}

DISPLAY_NAMES = {
    "SVLA_pretrained":   "SpatialVLA Pretrained",
    "SVLA_raw_ft":       "SpatialVLA Raw FT",
    "SVLA_cotrain_fg":   "SpatialVLA Ours (Cotrain+FG)",
    "OpenPI_pretrained": "OpenPI Pretrained",
    "OpenPI_raw_ft":     "OpenPI Raw FT",
    "OpenPI_aligned":    "OpenPI Ours (Aligned)",
}

OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "evaluation/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 80)
    print("快速 CKA 汇总 — 6模型 67-task 子集")
    print("=" * 80)

    # 1. 加载 text embeddings
    print("\n[1] 加载 text embeddings...")
    text_emb = load_text_embeddings(TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH)
    print(f"  EgoHOD: {text_emb['egohod'].shape}")

    # 2. 加载所有模型特征
    print("\n[2] 加载 6 个模型特征...")
    model_data = {}
    for key, path in FEATURE_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠ {key}: 文件不存在 → {path}")
            continue
        data = np.load(path, allow_pickle=True)
        feats = data['features']
        labels = data['task_labels']
        layer_indices = data['layer_indices']
        model_data[key] = {
            'features': feats,
            'task_labels': labels,
            'layer_indices': list(layer_indices),
        }
        print(f"  ✓ {key}: shape={feats.shape}, unique_tasks={len(set(labels))}, layers={list(layer_indices)}")

    # 3. 构建 text dict
    print("\n[3] 构建 task-text 映射...")
    model_text = {}
    for key, md in model_data.items():
        tdict = build_task_text_dict(text_emb, md['task_labels'])
        model_text[key] = tdict
        print(f"  {key}: valid_tasks={len(tdict['valid_tasks'])} / {len(set(md['task_labels']))}")

    # 4. 逐模型逐层计算 CKA (仅 Linear CKA, 非常快)
    print("\n" + "=" * 80)
    print("[4] 逐层 Linear CKA 计算 (EgoHOD)")
    print("=" * 80)

    all_results = {}  # {model_key: {layer_idx: {cka_all, cka_mean}}}

    for key in FEATURE_FILES:
        if key not in model_data:
            continue
        md = model_data[key]
        td = model_text[key]
        feats = md['features']
        labels = md['task_labels']
        layer_indices = md['layer_indices']
        valid_tasks = td['valid_tasks']
        ego_dict = td['egohod']

        # 过滤到 valid tasks
        mask = np.isin(labels, valid_tasks)
        feats_valid = feats[mask]
        labels_valid = labels[mask]

        all_results[key] = {}
        t1 = time.time()
        print(f"\n  --- {DISPLAY_NAMES[key]} ---")
        print(f"  {'Layer':>6s} | {'CKA_all':>10s} | {'CKA_mean':>10s}")
        print(f"  {'-'*38}")

        for li_idx, li in enumerate(layer_indices):
            f_layer = feats_valid[:, li_idx, :]

            # all_expand
            vla_ae, txt_ae = build_all_expand(f_layer, labels_valid, valid_tasks, ego_dict)
            cka_all = linear_cka(vla_ae, txt_ae)

            # task_mean
            vla_tm, txt_tm = build_task_mean(f_layer, labels_valid, valid_tasks, ego_dict)
            cka_mean = linear_cka(vla_tm, txt_tm)

            all_results[key][int(li)] = {
                'cka_all': cka_all,
                'cka_mean': cka_mean,
            }

            print(f"  L{li:>2d}   | {cka_all:>10.6f} | {cka_mean:>10.6f}")

        dt = time.time() - t1
        print(f"  ⏱ {dt:.1f}s")

    # ================================================================
    # 5. 汇总: Layer 10 对比 (6模型)
    # ================================================================
    print("\n" + "=" * 80)
    print("[5] 6模型 Layer 10 CKA 对比")
    print("=" * 80)

    # SpatialVLA 的 layer 10 在 layer_indices 中的 actual layer index = 10
    # OpenPI 的 layer 10 在 layer_indices 中的 actual layer index = 10
    print(f"\n  {'Model':<35s} | {'CKA_all':>10s} | {'CKA_mean':>10s} | {'L10 存在':>8s}")
    print(f"  {'-'*72}")
    for key in FEATURE_FILES:
        if key not in all_results:
            continue
        layers_res = all_results[key]
        if 10 in layers_res:
            r = layers_res[10]
            print(f"  {DISPLAY_NAMES[key]:<35s} | {r['cka_all']:>10.6f} | {r['cka_mean']:>10.6f} | {'✓':>8s}")
        else:
            # 找最近的层
            available = sorted(layers_res.keys())
            closest = min(available, key=lambda x: abs(x - 10))
            r = layers_res[closest]
            print(f"  {DISPLAY_NAMES[key]:<35s} | {r['cka_all']:>10.6f} | {r['cka_mean']:>10.6f} | L{closest} (近似)")

    # ================================================================
    # 6. 全层值: SpatialVLA pretrained + OpenPI pretrained
    # ================================================================
    print("\n" + "=" * 80)
    print("[6] SpatialVLA Pretrained — 全层 CKA")
    print("=" * 80)
    if "SVLA_pretrained" in all_results:
        layers_res = all_results["SVLA_pretrained"]
        print(f"  {'Layer':>6s} | {'CKA_all':>10s} | {'CKA_mean':>10s}")
        print(f"  {'-'*38}")
        for li in sorted(layers_res.keys()):
            r = layers_res[li]
            print(f"  L{li:>2d}   | {r['cka_all']:>10.6f} | {r['cka_mean']:>10.6f}")

    print("\n" + "=" * 80)
    print("[7] OpenPI Pretrained — 全层 CKA")
    print("=" * 80)
    if "OpenPI_pretrained" in all_results:
        layers_res = all_results["OpenPI_pretrained"]
        print(f"  {'Layer':>6s} | {'CKA_all':>10s} | {'CKA_mean':>10s}")
        print(f"  {'-'*38}")
        for li in sorted(layers_res.keys()):
            r = layers_res[li]
            print(f"  L{li:>2d}   | {r['cka_all']:>10.6f} | {r['cka_mean']:>10.6f}")

    # ================================================================
    # 7. 最佳层对比
    # ================================================================
    print("\n" + "=" * 80)
    print("[8] 6模型最佳层 CKA 对比")
    print("=" * 80)
    print(f"\n  === all_expand ===")
    print(f"  {'Model':<35s} | {'Best Layer':>10s} | {'CKA':>10s}")
    print(f"  {'-'*62}")
    for key in FEATURE_FILES:
        if key not in all_results:
            continue
        layers_res = all_results[key]
        best_layer = max(layers_res, key=lambda l: layers_res[l]['cka_all'])
        r = layers_res[best_layer]
        print(f"  {DISPLAY_NAMES[key]:<35s} | L{best_layer:>8d} | {r['cka_all']:>10.6f}")

    print(f"\n  === task_mean ===")
    print(f"  {'Model':<35s} | {'Best Layer':>10s} | {'CKA':>10s}")
    print(f"  {'-'*62}")
    for key in FEATURE_FILES:
        if key not in all_results:
            continue
        layers_res = all_results[key]
        best_layer = max(layers_res, key=lambda l: layers_res[l]['cka_mean'])
        r = layers_res[best_layer]
        print(f"  {DISPLAY_NAMES[key]:<35s} | L{best_layer:>8d} | {r['cka_mean']:>10.6f}")

    # ================================================================
    # 保存 JSON
    # ================================================================
    json_path = os.path.join(OUTPUT_DIR, "quick_cka_summary.json")
    json_data = {}
    for key, layers_res in all_results.items():
        json_data[key] = {
            "display_name": DISPLAY_NAMES[key],
            "layers": {str(l): v for l, v in layers_res.items()},
        }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {json_path}")

    total = time.time() - t0
    print(f"总耗时: {total:.1f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
