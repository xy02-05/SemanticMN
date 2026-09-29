"""
4模型 Layer 10 CKA/SVCCA/PWCCA — SpatialVLA + OpenPI×3

文本: EgoHOD (512d) + Qwen (4096d, 不降维)
聚合: all_expand（每条轨迹独立，不做 task mean）
SVD阈值: 0.9999

用法:
    conda activate openpi
    python evaluation/eval_4models_streaming_cka.py
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
    compute_cka_svcca_pwcca, build_all_expand,
)
from bridge_representation.config import TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH

TARGET_LAYER = 10

MODELS = {
    "svla_pretrained": {
        "path": os.path.join(ANALYSIS_DIR, "bridge_representation/outputs/streaming/features/pretrained_streaming_features.npz"),
        "display": "SVLA Pretrained",
    },
    "pi0_pretrained": {
        "path": os.path.join(ANALYSIS_DIR, "openpi_representation/outputs/streaming/features/pretrained_streaming_features.npz"),
        "display": "PI0 Pretrained",
    },
    "pi0_raw_ft": {
        "path": os.path.join(ANALYSIS_DIR, "openpi_representation/outputs/streaming/features/raw_ft_streaming_features.npz"),
        "display": "PI0 Raw FT",
    },
    "pi0_aligned": {
        "path": os.path.join(ANALYSIS_DIR, "openpi_representation/outputs/streaming/features/aligned_streaming_features.npz"),
        "display": "PI0 Aligned",
    },
}

OUT_DIR = os.path.join(ANALYSIS_DIR, "evaluation/outputs")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "eval_4models_streaming_cka.log")
JSON_PATH = os.path.join(OUT_DIR, "eval_4models_streaming_cka.json")

log_file = open(LOG_PATH, "w")
def P(s=""):
    print(s, flush=True)
    log_file.write(s + "\n"); log_file.flush()


def main():
    t0 = time.time()
    P("=" * 90)
    P(f"4模型 Layer {TARGET_LAYER} — CKA / SVCCA / PWCCA (all_expand, SVD threshold=0.9999)")
    P(f"文本: EgoHOD 512d + Qwen 4096d (不降维) | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    P("=" * 90)

    text_emb = load_text_embeddings(TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH)
    P(f"EgoHOD: {text_emb['egohod'].shape}, Qwen: {text_emb['qwen'].shape}")

    all_results = {}
    for key, cfg in MODELS.items():
        path = cfg["path"]
        name = cfg["display"]
        P(f"\n{'='*80}\n[{name}]\n{'='*80}")
        if not os.path.exists(path):
            P(f"  不存在: {path}"); continue

        t_m = time.time()
        data = np.load(path, allow_pickle=True)
        feats, labels = data['features'], data['task_labels']
        layer_indices = list(data['layer_indices'])
        N = feats.shape[0]
        D = feats.shape[2]
        P(f"  shape={feats.shape}, D={D}, tasks={len(set(labels))}")

        if TARGET_LAYER not in layer_indices:
            P(f"  Layer {TARGET_LAYER} 不在 {layer_indices}"); continue
        li = layer_indices.index(TARGET_LAYER)
        f_layer = feats[:, li, :]

        tdict = build_task_text_dict(text_emb, labels)
        valid_tasks = tdict['valid_tasks']
        valid_set = set(valid_tasks)
        mask = np.array([l in valid_set for l in labels])
        f_v, l_v = f_layer[mask], labels[mask]
        P(f"  有效: {len(l_v)}/{N} traj, {len(valid_tasks)} tasks")

        r = {"n_traj": int(N), "n_valid": int(len(l_v)),
             "layer": TARGET_LAYER, "D": D}

        for emb_name, emb_key, emb_dim in [("EgoHOD", "egohod", 512), ("Qwen", "qwen", 4096)]:
            P(f"\n  --- {emb_name} {emb_dim}d ---")
            t1 = time.time()
            vla, txt = build_all_expand(f_v, l_v, valid_tasks, tdict[emb_key])
            m = compute_cka_svcca_pwcca(vla, txt)
            elapsed = time.time() - t1
            P(f"  CKA={m['cka']:.6f}  SVCCA={m['svcca']:.4f}  PWCCA={m['pwcca']:.4f}  ({elapsed:.1f}s)")
            P(f"  SVD降维: kx(VLA)={m['svcca_kx']}, ky(Text)={m['svcca_ky']}, n_corr={m['svcca_n_corr']}")
            r[emb_key] = {
                "cka": m['cka'], "svcca": m['svcca'], "pwcca": m['pwcca'],
                "N": m['N'], "svcca_kx": m['svcca_kx'], "svcca_ky": m['svcca_ky'],
                "svcca_n_corr": m['svcca_n_corr'],
            }

        all_results[key] = r
        P(f"\n  模型耗时: {time.time()-t_m:.1f}s")
        del feats, labels, data

    # 汇总
    P(f"\n{'='*90}")
    P(f"汇总 Layer {TARGET_LAYER} (SVD threshold=0.9999)")
    P(f"{'='*90}")
    for tag, label in [("egohod", "EgoHOD 512d"), ("qwen", "Qwen 4096d")]:
        P(f"\n  === {label} ===")
        P(f"  {'Model':<18} | {'D':>5} | {'CKA':>10} | {'SVCCA':>10} | {'PWCCA':>10} | {'kx':>5} {'ky':>5} {'n_corr':>6}")
        P(f"  {'-'*82}")
        for key in MODELS:
            if key not in all_results or tag not in all_results[key]:
                continue
            m = all_results[key][tag]
            d = all_results[key].get("D", "?")
            P(f"  {MODELS[key]['display']:<18} | {d:>5} | {m['cka']:>10.6f} | {m['svcca']:>10.4f} | {m['pwcca']:>10.4f} | {m['svcca_kx']:>5} {m['svcca_ky']:>5} {m['svcca_n_corr']:>6}")

    def _c(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, dict): return {str(k): _c(v) for k, v in o.items()}
        return o
    with open(JSON_PATH, "w") as f:
        json.dump(_c(all_results), f, indent=2)
    P(f"\nJSON: {JSON_PATH}")
    P(f"总耗时: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
    log_file.close()


if __name__ == "__main__":
    main()
