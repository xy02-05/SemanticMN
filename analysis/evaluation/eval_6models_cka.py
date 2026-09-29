"""
6模型 CKA 对比评测

对 SpatialVLA (pretrained, raw_ft, cotrain_fg) + OpenPI (pretrained, raw_ft, aligned)
在 67-task 子集上计算 CKA / SVCCA / PWCCA (vs EgoHOD text embeddings)

使用 all_expand + task_mean 两种聚合策略

用法:
    conda activate gaze
    cd /root/data/xuyuan1/Codes/analysis
    python evaluation/eval_6models_cka.py
"""
import sys
import os
import time
import numpy as np

ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
sys.path.insert(0, ANALYSIS_DIR)

from evaluation.metrics import (
    load_text_embeddings, build_task_text_dict,
    linear_cka, compute_cka_svcca_pwcca,
    build_all_expand, build_task_mean,
)
from bridge_representation.config import TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH

# ====================================================================
# 6 个模型的特征文件定义
# ====================================================================
FEATURE_FILES = {
    # SpatialVLA: chunk4 (mean over 4 action tokens)
    "SVLA_pretrained": os.path.join(
        ANALYSIS_DIR, "bridge_representation/outputs/features/pretrained_chunk4_features.npz"),
    "SVLA_raw_ft": os.path.join(
        ANALYSIS_DIR, "bridge_representation/outputs/features/raw_ft_chunk4_features.npz"),
    "SVLA_cotrain_fg": os.path.join(
        ANALYSIS_DIR, "bridge_representation/outputs/features/cotrain_fg_chunk4_features.npz"),
    # OpenPI: chunk5 (mean over 5 action tokens)
    "OpenPI_pretrained": os.path.join(
        ANALYSIS_DIR, "openpi_representation/outputs/features/pretrained_chunk5_features.npz"),
    "OpenPI_raw_ft": os.path.join(
        ANALYSIS_DIR, "openpi_representation/outputs/features/raw_ft_chunk5_features.npz"),
    "OpenPI_aligned": os.path.join(
        ANALYSIS_DIR, "openpi_representation/outputs/features/aligned_chunk5_features.npz"),
}

# 友好显示名
DISPLAY_NAMES = {
    "SVLA_pretrained":  "SpatialVLA Pretrained",
    "SVLA_raw_ft":      "SpatialVLA Raw FT",
    "SVLA_cotrain_fg":  "SpatialVLA Ours (Cotrain+FG)",
    "OpenPI_pretrained": "OpenPI Pretrained",
    "OpenPI_raw_ft":     "OpenPI Raw FT",
    "OpenPI_aligned":    "OpenPI Ours (Aligned)",
}

# ====================================================================
# 日志辅助
# ====================================================================
LOG_DIR = os.path.join(ANALYSIS_DIR, "evaluation/outputs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "eval_6models_cka.log")
log_file = open(LOG_PATH, "w")

def P(s=""):
    print(s, flush=True)
    log_file.write(s + "\n")
    log_file.flush()


# ====================================================================
# 主函数
# ====================================================================
def main():
    t_start = time.time()
    P(f"=" * 100)
    P(f"6模型 CKA/SVCCA/PWCCA 对比评测")
    P(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    P(f"日志: {LOG_PATH}")
    P(f"=" * 100)

    # ---- 1. 加载 text embeddings ----
    P("\n[1] 加载 text embeddings...")
    text_emb = load_text_embeddings(TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH)
    P(f"  EgoHOD: {text_emb['egohod'].shape}")

    # ---- 2. 加载所有模型特征 ----
    P("\n[2] 加载 6 个模型特征...")
    model_data = {}
    for key, path in FEATURE_FILES.items():
        if not os.path.exists(path):
            P(f"  ⚠ {key}: 文件不存在 → {path}")
            continue
        data = np.load(path, allow_pickle=True)
        feats = data['features']          # [N, L, D]
        labels = data['task_labels']      # [N]
        layer_indices = data['layer_indices']  # [L]
        model_data[key] = {
            'features': feats,
            'task_labels': labels,
            'layer_indices': layer_indices,
        }
        P(f"  ✓ {key}: shape={feats.shape}, "
          f"unique_tasks={len(set(labels))}, layers={list(layer_indices)}")

    if not model_data:
        P("没有可用模型，退出。")
        return

    # ---- 3. 对每个模型构建 text dict ----
    P("\n[3] 构建 task-text 映射...")
    model_text = {}
    for key, md in model_data.items():
        tdict = build_task_text_dict(text_emb, md['task_labels'])
        model_text[key] = tdict
        P(f"  {key}: valid_tasks={len(tdict['valid_tasks'])} / {len(set(md['task_labels']))}")

    # ---- 4. 逐层 CKA/SVCCA/PWCCA (all_expand + task_mean) ----
    P("\n" + "=" * 100)
    P("[4] 逐层 CKA / SVCCA / PWCCA 计算 (EgoHOD, all_expand)")
    P("=" * 100)

    # 为每个模型的每一层计算
    all_results = {}  # {model_key: {layer_idx: {metric: value}}}

    for key in model_data:
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
        P(f"\n  --- {DISPLAY_NAMES[key]} ---")
        P(f"  {'Layer':>6s} | {'CKA_all':>10s} {'SVCCA_all':>10s} {'PWCCA_all':>10s} | "
          f"{'CKA_mean':>10s} {'SVCCA_mean':>10s} {'PWCCA_mean':>10s}")
        P(f"  {'-'*80}")

        for li_idx, li in enumerate(layer_indices):
            f_layer = feats_valid[:, li_idx, :]

            # all_expand
            vla_ae, txt_ae = build_all_expand(f_layer, labels_valid, valid_tasks, ego_dict)
            m_ae = compute_cka_svcca_pwcca(vla_ae, txt_ae)

            # task_mean
            vla_tm, txt_tm = build_task_mean(f_layer, labels_valid, valid_tasks, ego_dict)
            m_tm = compute_cka_svcca_pwcca(vla_tm, txt_tm)

            all_results[key][int(li)] = {
                'all_cka': m_ae['cka'], 'all_svcca': m_ae['svcca'], 'all_pwcca': m_ae['pwcca'],
                'mean_cka': m_tm['cka'], 'mean_svcca': m_tm['svcca'], 'mean_pwcca': m_tm['pwcca'],
            }

            P(f"  L{li:>2d}   | {m_ae['cka']:>10.6f} {m_ae['svcca']:>10.4f} {m_ae['pwcca']:>10.4f} | "
              f"{m_tm['cka']:>10.6f} {m_tm['svcca']:>10.4f} {m_tm['pwcca']:>10.4f}")

    # ---- 5. 汇总对比表: 最佳层 CKA ----
    P("\n" + "=" * 100)
    P("[5] 6模型横向对比 — 最佳层 CKA (EgoHOD)")
    P("=" * 100)
    P(f"\n  === all_expand 策略 ===")
    P(f"  {'Model':<30s} | {'Best Layer':>10s} | {'CKA':>8s} | {'SVCCA':>8s} | {'PWCCA':>8s}")
    P(f"  {'-'*75}")
    for key in FEATURE_FILES:
        if key not in all_results:
            continue
        layers_res = all_results[key]
        best_layer = max(layers_res, key=lambda l: layers_res[l]['all_cka'])
        br = layers_res[best_layer]
        P(f"  {DISPLAY_NAMES[key]:<30s} | L{best_layer:>8d} | {br['all_cka']:>8.4f} | "
          f"{br['all_svcca']:>8.4f} | {br['all_pwcca']:>8.4f}")

    P(f"\n  === task_mean 策略 ===")
    P(f"  {'Model':<30s} | {'Best Layer':>10s} | {'CKA':>8s} | {'SVCCA':>8s} | {'PWCCA':>8s}")
    P(f"  {'-'*75}")
    for key in FEATURE_FILES:
        if key not in all_results:
            continue
        layers_res = all_results[key]
        best_layer = max(layers_res, key=lambda l: layers_res[l]['mean_cka'])
        br = layers_res[best_layer]
        P(f"  {DISPLAY_NAMES[key]:<30s} | L{best_layer:>8d} | {br['mean_cka']:>8.4f} | "
          f"{br['mean_svcca']:>8.4f} | {br['mean_pwcca']:>8.4f}")

    # ---- 6. 汇总对比表: 固定层对比 ----
    P("\n" + "=" * 100)
    P("[6] 6模型固定层对比 — CKA (all_expand, EgoHOD)")
    P("=" * 100)

    # SpatialVLA 和 OpenPI 层索引不同，分开展示
    # SpatialVLA layers
    svla_keys = [k for k in FEATURE_FILES if k.startswith("SVLA_") and k in all_results]
    openpi_keys = [k for k in FEATURE_FILES if k.startswith("OpenPI_") and k in all_results]

    if svla_keys:
        svla_layers = sorted(all_results[svla_keys[0]].keys())
        P(f"\n  --- SpatialVLA (layers: {svla_layers}) ---")
        header = f"  {'Layer':>6s}"
        for key in svla_keys:
            header += f" | {DISPLAY_NAMES[key]:>25s}"
        P(header)
        P(f"  {'-'*(8 + 28*len(svla_keys))}")
        for li in svla_layers:
            row = f"  L{li:>4d}"
            for key in svla_keys:
                row += f" | {all_results[key][li]['all_cka']:>25.6f}"
            P(row)

    if openpi_keys:
        openpi_layers = sorted(all_results[openpi_keys[0]].keys())
        P(f"\n  --- OpenPI (layers: {openpi_layers}) ---")
        header = f"  {'Layer':>6s}"
        for key in openpi_keys:
            header += f" | {DISPLAY_NAMES[key]:>25s}"
        P(header)
        P(f"  {'-'*(8 + 28*len(openpi_keys))}")
        for li in openpi_layers:
            row = f"  L{li:>4d}"
            for key in openpi_keys:
                row += f" | {all_results[key][li]['all_cka']:>25.6f}"
            P(row)

    # ---- 7. 保存结果 JSON ----
    import json
    json_path = os.path.join(LOG_DIR, "eval_6models_cka.json")
    # 转换为可序列化格式
    json_results = {}
    for key, layers_res in all_results.items():
        json_results[key] = {
            "display_name": DISPLAY_NAMES[key],
            "layers": {str(l): v for l, v in layers_res.items()},
        }
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    P(f"\n结果已保存: {json_path}")

    total_time = time.time() - t_start
    P(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    P(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_file.close()


if __name__ == "__main__":
    main()
