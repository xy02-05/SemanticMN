"""
SpatialVLA pretrained Layer 10 CKA/SVCCA/PWCCA — 与 OpenPI 同口径对比

文本: EgoHOD (512d) + Qwen (4096d, 不降维)
SVD阈值: 0.9995

用法:
    python evaluation/eval_spatialvla_streaming_cka.py
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

FEATURE_PATH = os.path.join(
    ANALYSIS_DIR, "bridge_representation/outputs/streaming/features/pretrained_streaming_features.npz"
)

OUT_DIR = os.path.join(ANALYSIS_DIR, "evaluation/outputs")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "eval_spatialvla_streaming_cka.log")
JSON_PATH = os.path.join(OUT_DIR, "eval_spatialvla_streaming_cka.json")

log_file = open(LOG_PATH, "w")
def P(s=""):
    print(s, flush=True)
    log_file.write(s + "\n"); log_file.flush()


def main():
    t0 = time.time()
    P("=" * 90)
    P(f"SpatialVLA Pretrained Layer {TARGET_LAYER} — CKA / SVCCA / PWCCA (all_expand, SVD threshold=0.9995)")
    P(f"文本: EgoHOD 512d + Qwen 4096d (不降维) | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    P("=" * 90)

    text_emb = load_text_embeddings(TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH)
    P(f"EgoHOD: {text_emb['egohod'].shape}, Qwen: {text_emb['qwen'].shape}")

    P(f"\n加载: {FEATURE_PATH}")
    data = np.load(FEATURE_PATH, allow_pickle=True)
    feats, labels = data['features'], data['task_labels']
    layer_indices = list(data['layer_indices'])
    N = feats.shape[0]
    P(f"shape={feats.shape}, tasks={len(set(labels))}, layers={layer_indices}")

    li = layer_indices.index(TARGET_LAYER)
    f_layer = feats[:, li, :]
    P(f"Layer {TARGET_LAYER} → index {li}, D={f_layer.shape[1]}")

    tdict = build_task_text_dict(text_emb, labels)
    valid_tasks = tdict['valid_tasks']
    valid_set = set(valid_tasks)
    mask = np.array([l in valid_set for l in labels])
    f_v, l_v = f_layer[mask], labels[mask]
    P(f"有效: {len(l_v)}/{N} traj, {len(valid_tasks)} tasks")

    r = {"model": "SpatialVLA Pretrained", "n_traj": int(N),
         "n_valid": int(len(l_v)), "layer": TARGET_LAYER, "D": int(f_layer.shape[1])}

    for emb_name, emb_key, emb_dim in [("EgoHOD", "egohod", 512), ("Qwen", "qwen", 4096)]:
        P(f"\n--- {emb_name} {emb_dim}d ---")
        t1 = time.time()
        vla, txt = build_all_expand(f_v, l_v, valid_tasks, tdict[emb_key])
        m = compute_cka_svcca_pwcca(vla, txt)
        P(f"CKA={m['cka']:.6f}  SVCCA={m['svcca']:.4f}  PWCCA={m['pwcca']:.4f}  ({time.time()-t1:.1f}s)")
        P(f"SVD降维: kx(VLA)={m['svcca_kx']}, ky(Text)={m['svcca_ky']}, n_corr={m['svcca_n_corr']}")
        r[emb_key] = {
            "cka": m['cka'], "svcca": m['svcca'], "pwcca": m['pwcca'],
            "N": m['N'], "svcca_kx": m['svcca_kx'], "svcca_ky": m['svcca_ky'],
            "svcca_n_corr": m['svcca_n_corr'],
        }

    # 汇总对比（加载 OpenPI 结果）
    openpi_json = os.path.join(OUT_DIR, "eval_openpi_streaming_cka.json")
    if os.path.exists(openpi_json):
        with open(openpi_json) as f:
            openpi = json.load(f)
        P(f"\n{'='*90}")
        P("4模型横向对比 — Layer 10 (all_expand, 每条轨迹独立)")
        P(f"{'='*90}")

        all_models = [
            ("SVLA Pretrained", r),
            ("PI0 Pretrained", openpi.get("pretrained", {})),
            ("PI0 Raw FT", openpi.get("raw_ft", {})),
            ("PI0 Aligned", openpi.get("aligned", {})),
        ]

        for tag, label in [("egohod", "EgoHOD 512d"), ("qwen", "Qwen 4096d")]:
            P(f"\n  === {label} ===")
            P(f"  {'Model':<18} | {'CKA':>10} | {'SVCCA':>10} | {'PWCCA':>10}")
            P(f"  {'-'*58}")
            for name, mr in all_models:
                em = mr.get(tag, {})
                if not em: continue
                P(f"  {name:<18} | {em['cka']:>10.6f} | {em['svcca']:>10.4f} | {em['pwcca']:>10.4f}")

    def _c(o):
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,dict): return {str(k):_c(v) for k,v in o.items()}
        return o
    with open(JSON_PATH,"w") as f: json.dump(_c(r),f,indent=2)
    P(f"\nJSON: {JSON_PATH}")
    P(f"总耗时: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
    log_file.close()

if __name__ == "__main__":
    main()
