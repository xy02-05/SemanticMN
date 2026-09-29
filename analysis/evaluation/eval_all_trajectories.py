"""
全量轨迹评测: CKA / Whitened CKA / SVCCA / PWCCA
只测全部 38660 条轨迹, 10 层, EgoHOD text
所有输出同时打印到 stdout 和 log 文件
"""
import numpy as np
import sys, os, time

sys.path.insert(0, "/root/data/xuyuan1/Codes/analysis")

from bridge_representation.config import TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH
from evaluation.metrics import (
    load_text_embeddings, build_task_text_dict,
    linear_cka, whitened_cka, whitened_cka_thresholded,
    svcca, pwcca, _svcca_pwcca_shared,
    build_all_expand, build_task_mean, effective_rank,
)
from openpi_representation.config import LAYER_INDICES

# ====================================================================
# 日志输出: 同时写 stdout 和 log 文件
# ====================================================================
LOG_DIR = "/root/data/xuyuan1/Codes/analysis/doc"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "eval_all_trajectories.log")
log_file = open(LOG_PATH, "w")

def P(s=""):
    print(s, flush=True)
    log_file.write(s + "\n")
    log_file.flush()


P(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
P(f"日志路径: {LOG_PATH}")
P(f"LAYER_INDICES = {LAYER_INDICES}")

# ====================================================================
# 加载数据
# ====================================================================
P("\n[1] 加载数据...")
t0 = time.time()

text_emb = load_text_embeddings(TASK_RLDS_PATH, EGOHOD_EMB_PATH, QWEN_EMB_PATH)

# raw_ft streaming 全量
feat_path = "openpi_representation/outputs/streaming/features/raw_ft_streaming_features.npz"
P(f"  特征文件: {feat_path}")
data = np.load(feat_path, allow_pickle=True)
feats, labels = data["features"], data["task_labels"]
N, N_layers, D = feats.shape
P(f"  feats shape: {feats.shape}  (N={N}, layers={N_layers}, D={D})")

tdict = build_task_text_dict(text_emb, labels)
valid_tasks = tdict["valid_tasks"]
P(f"  unique labels: {len(set(labels))}, valid tasks (有text embedding): {len(valid_tasks)}")
P(f"  数据加载耗时: {time.time()-t0:.1f}s")

# ====================================================================
# Part 1: 基本统计
# ====================================================================
P("\n" + "=" * 100)
P("Part 1: 每层特征基本统计 (全量 38660 轨迹)")
P("=" * 100)
P(f"{'Layer':>6s} | {'norm_mean':>10s} {'norm_std':>10s} {'per_dim_std':>12s} {'EffRank':>8s} {'centered_norm':>14s}")
P("-" * 75)
for li_idx in range(N_layers):
    li = LAYER_INDICES[li_idx]
    f = feats[:, li_idx, :]
    norms = np.linalg.norm(f, axis=1)
    f_c = f - f.mean(0)
    cn = np.linalg.norm(f_c, axis=1)
    er = effective_rank(f)
    P(f"  L{li:>2d}  | {norms.mean():>10.2f} {norms.std():>10.4f} {f.std(axis=0).mean():>12.6f} {er:>8.1f} {cn.mean():>8.2f}±{cn.std():>5.2f}")


# ====================================================================
# Part 2: CKA (标准) — all_expand + task_mean
# ====================================================================
P("\n" + "=" * 100)
P("Part 2: Linear CKA (标准, EgoHOD)")
P("=" * 100)
P(f"{'Layer':>6s} | {'all_expand':>14s} {'time':>6s} | {'task_mean':>14s} {'time':>6s}")
P("-" * 60)
for li_idx in range(N_layers):
    li = LAYER_INDICES[li_idx]
    f_layer = feats[:, li_idx, :]

    t1 = time.time()
    vla_ae, txt_ae = build_all_expand(f_layer, labels, valid_tasks, tdict["egohod"])
    cka_ae = linear_cka(vla_ae, txt_ae)
    dt_ae = time.time() - t1

    t1 = time.time()
    vla_tm, txt_tm = build_task_mean(f_layer, labels, valid_tasks, tdict["egohod"])
    cka_tm = linear_cka(vla_tm, txt_tm)
    dt_tm = time.time() - t1

    P(f"  L{li:>2d}  | {cka_ae:>14.6f} {dt_ae:>5.1f}s | {cka_tm:>14.6f} {dt_tm:>5.1f}s")


# ====================================================================
# Part 3: Whitened CKA (全维度) — all_expand + task_mean
# ====================================================================
P("\n" + "=" * 100)
P("Part 3: Whitened CKA (全维度白化, EgoHOD)")
P("=" * 100)
P(f"{'Layer':>6s} | {'all_expand':>14s} {'kx':>5s} {'ky':>5s} {'time':>6s} | {'task_mean':>14s} {'kx':>5s} {'ky':>5s} {'time':>6s}")
P("-" * 90)
for li_idx in range(N_layers):
    li = LAYER_INDICES[li_idx]
    f_layer = feats[:, li_idx, :]

    t1 = time.time()
    vla_ae, txt_ae = build_all_expand(f_layer, labels, valid_tasks, tdict["egohod"])
    wcka_ae, info_ae = whitened_cka(vla_ae, txt_ae)
    dt_ae = time.time() - t1

    t1 = time.time()
    vla_tm, txt_tm = build_task_mean(f_layer, labels, valid_tasks, tdict["egohod"])
    wcka_tm, info_tm = whitened_cka(vla_tm, txt_tm)
    dt_tm = time.time() - t1

    P(f"  L{li:>2d}  | {wcka_ae:>14.6f} {info_ae['kx']:>5d} {info_ae['ky']:>5d} {dt_ae:>5.1f}s | {wcka_tm:>14.6f} {info_tm['kx']:>5d} {info_tm['ky']:>5d} {dt_tm:>5.1f}s")


# ====================================================================
# Part 4: Whitened CKA (阈值版) — 多个 threshold
# ====================================================================
P("\n" + "=" * 100)
P("Part 4: Whitened CKA (阈值版, all_expand, EgoHOD)")
P("=" * 100)

for threshold in [0.99, 0.999, 0.9999]:
    P(f"\n  --- threshold = {threshold} ---")
    P(f"{'Layer':>6s} | {'wcka':>10s} {'kx':>5s} {'ky':>5s} {'time':>6s}")
    P("-" * 40)
    for li_idx in range(N_layers):
        li = LAYER_INDICES[li_idx]
        f_layer = feats[:, li_idx, :]
        t1 = time.time()
        vla, txt = build_all_expand(f_layer, labels, valid_tasks, tdict["egohod"])
        wcka_val, info = whitened_cka_thresholded(vla, txt, threshold=threshold)
        dt = time.time() - t1
        P(f"  L{li:>2d}  | {wcka_val:>10.6f} {info['kx']:>5d} {info['ky']:>5d} {dt:>5.1f}s")


# ====================================================================
# Part 5: Whitened CKA (固定维度) — 50, 100, 200, 500
# ====================================================================
P("\n" + "=" * 100)
P("Part 5: Whitened CKA (固定维度, all_expand, EgoHOD)")
P("=" * 100)

for n_comp in [50, 100, 200, 500]:
    P(f"\n  --- n_components = {n_comp} ---")
    P(f"{'Layer':>6s} | {'wcka':>10s} {'time':>6s}")
    P("-" * 30)
    for li_idx in range(N_layers):
        li = LAYER_INDICES[li_idx]
        f_layer = feats[:, li_idx, :]
        t1 = time.time()
        vla, txt = build_all_expand(f_layer, labels, valid_tasks, tdict["egohod"])
        wcka_val, _ = whitened_cka(vla, txt, n_components=n_comp)
        dt = time.time() - t1
        P(f"  L{li:>2d}  | {wcka_val:>10.6f} {dt:>5.1f}s")


# ====================================================================
# Part 6: SVCCA + PWCCA (all_expand, threshold=0.99)
# ====================================================================
P("\n" + "=" * 100)
P("Part 6: SVCCA + PWCCA (all_expand, EgoHOD, threshold=0.99)")
P("=" * 100)
P(f"{'Layer':>6s} | {'svcca':>8s} {'pwcca':>8s} {'kx':>5s} {'ky':>5s} {'time':>6s}")
P("-" * 50)
for li_idx in range(N_layers):
    li = LAYER_INDICES[li_idx]
    f_layer = feats[:, li_idx, :]
    t1 = time.time()
    vla, txt = build_all_expand(f_layer, labels, valid_tasks, tdict["egohod"])
    svc_val, pwc_val, info = _svcca_pwcca_shared(vla, txt, threshold=0.99)
    dt = time.time() - t1
    P(f"  L{li:>2d}  | {svc_val:>8.4f} {pwc_val:>8.4f} {info['kx']:>5d} {info['ky']:>5d} {dt:>5.1f}s")


# ====================================================================
# Part 7: SVCCA + PWCCA (all_expand, threshold=0.999)
# ====================================================================
P("\n" + "=" * 100)
P("Part 7: SVCCA + PWCCA (all_expand, EgoHOD, threshold=0.999)")
P("=" * 100)
P(f"{'Layer':>6s} | {'svcca':>8s} {'pwcca':>8s} {'kx':>5s} {'ky':>5s} {'time':>6s}")
P("-" * 50)
for li_idx in range(N_layers):
    li = LAYER_INDICES[li_idx]
    f_layer = feats[:, li_idx, :]
    t1 = time.time()
    vla, txt = build_all_expand(f_layer, labels, valid_tasks, tdict["egohod"])
    svc_val, pwc_val, info = _svcca_pwcca_shared(vla, txt, threshold=0.999)
    dt = time.time() - t1
    P(f"  L{li:>2d}  | {svc_val:>8.4f} {pwc_val:>8.4f} {info['kx']:>5d} {info['ky']:>5d} {dt:>5.1f}s")


# ====================================================================
# Part 8: SVCCA + PWCCA (task_mean, threshold=0.99)
# ====================================================================
P("\n" + "=" * 100)
P("Part 8: SVCCA + PWCCA (task_mean, EgoHOD, threshold=0.99)")
P("=" * 100)
P(f"{'Layer':>6s} | {'svcca':>8s} {'pwcca':>8s} {'kx':>5s} {'ky':>5s} {'time':>6s}")
P("-" * 50)
for li_idx in range(N_layers):
    li = LAYER_INDICES[li_idx]
    f_layer = feats[:, li_idx, :]
    t1 = time.time()
    vla, txt = build_task_mean(f_layer, labels, valid_tasks, tdict["egohod"])
    svc_val, pwc_val, info = _svcca_pwcca_shared(vla, txt, threshold=0.99)
    dt = time.time() - t1
    P(f"  L{li:>2d}  | {svc_val:>8.4f} {pwc_val:>8.4f} {info['kx']:>5d} {info['ky']:>5d} {dt:>5.1f}s")


# ====================================================================
# Part 9: 横向对比汇总表 (all_expand, 每层一行)
# ====================================================================
P("\n" + "=" * 100)
P("Part 9: 汇总对比表 (all_expand, EgoHOD)")
P("=" * 100)
P(f"{'Layer':>6s} | {'CKA':>8s} | {'WCKA_full':>10s} | {'WCKA_999':>10s} | {'WCKA_200':>10s} | {'SVCCA_99':>10s} | {'PWCCA_99':>10s}")
P("-" * 80)

for li_idx in range(N_layers):
    li = LAYER_INDICES[li_idx]
    f_layer = feats[:, li_idx, :]
    vla, txt = build_all_expand(f_layer, labels, valid_tasks, tdict["egohod"])

    cka_val = linear_cka(vla, txt)
    wcka_full, _ = whitened_cka(vla, txt)
    wcka_999, _ = whitened_cka_thresholded(vla, txt, threshold=0.999)
    wcka_200, _ = whitened_cka(vla, txt, n_components=200)
    svc_val, pwc_val, _ = _svcca_pwcca_shared(vla, txt, threshold=0.99)

    P(f"  L{li:>2d}  | {cka_val:>8.4f} | {wcka_full:>10.4f} | {wcka_999:>10.4f} | {wcka_200:>10.4f} | {svc_val:>10.4f} | {pwc_val:>10.4f}")


# ====================================================================
# 完成
# ====================================================================
total_time = time.time() - t0
P(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
P(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
P(f"日志已保存到: {LOG_PATH}")
log_file.close()
