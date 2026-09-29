"""
SimplerEnv 表征分析（action-action + action-text）

一、Action-Action（仅看轨迹间 cosine，与 text 无关）
  1. 等量成功/失败子集的轨迹两两相似度
  2. 全量成功/失败子集的轨迹两两相似度
  3. 任务内全体轨迹两两相似度 vs 成功率

二、Action-Text（DualAligner: proj_action / proj_text 后 cosine，即「与指令的对齐度」）
  1. 等量成功/失败：分别对成功轨迹、失败轨迹求「action–text」相似度均值（4a）
  2. 不等量：全量成功/失败各自的 action–text 均值（4b）
  3. 每任务全体轨迹的 action–text 均值 vs 成功率 Pearson（4c）

Text 向量来源（512 维）:
  - 优先 dataset/embedding/simpler_text_egohod.npz（gaze 下 encode_simpler_text.py 真编码）
  - 否则从 bridge_text_egohod_proj.npz 按语义代理 task_index 取行（与训练分布一致）
Qwen 对齐模型（4096 维）自动用 bridge_text_embeddings.npz 同索引代理。

用法:
  python evaluation/analyze_simpler_aligned.py \\
      --aligner_path alignment/outputs/.../aligner_..._layer10.pt \\
      --layer_idx 3 --no_raw
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn

ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
sys.path.insert(0, ANALYSIS_DIR)

# ===================== DualAligner 定义（复用 train_alignment.py） =====================
class DualAligner(nn.Module):
    def __init__(self, d_action, d_text, d_shared=512):
        super().__init__()
        self.proj_action = nn.Linear(d_action, d_shared)
        self.proj_text = nn.Linear(d_text, d_shared)

    def forward_action(self, x):
        return self.proj_action(x)

    def forward_text(self, x):
        return self.proj_text(x)


# ===================== 工具函数 =====================
def cosine_sim_matrix(A, B):
    """A: [N,D], B: [M,D] → [N,M] cosine similarity"""
    A_n = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-8, None)
    B_n = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-8, None)
    return A_n @ B_n.T

def mean_upper_tri(M):
    n = M.shape[0]
    if n < 2:
        return float('nan')
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(M[mask].mean())

def project_features(feats, aligner, device='cpu'):
    """用 aligner 投影 action features: [N, D] → [N, D_shared]"""
    with torch.no_grad():
        x = torch.from_numpy(feats).float().to(device)
        proj = aligner.forward_action(x)
        proj = torch.nn.functional.normalize(proj, dim=1)
        return proj.cpu().numpy()


# ===================== 加载 & 合并特征 =====================

# 每个数据源: (npz路径, 仅使用的task列表或None表示全部)
DEFAULT_SOURCES = [
    # spoon 来自 backup (24 ep)
    ("/root/data/xuyuan1/Codes/INT-ACT/ENV/LOGS/simpler_features/"
     "spatialvla-4b-224-pt_chunk4_features_task1_backup.npz",
     ["widowx_spoon_on_towel"]),
    # eggplant 专属运行 (24 ep)
    ("/root/data/xuyuan1/Codes/INT-ACT/ENV/LOGS_eggplant/simpler_features/"
     "spatialvla-4b-224-pt_chunk4_features.npz",
     ["widowx_put_eggplant_in_basket"]),
    # carrot + stack_cube 专属运行 (各 24 ep)
    ("/root/data/xuyuan1/Codes/INT-ACT/ENV/LOGS_carrot_stack/simpler_features/"
     "spatialvla-4b-224-pt_chunk4_features.npz",
     ["widowx_carrot_on_plate", "widowx_stack_cube"]),
]

def load_and_merge_features(sources=None):
    """从多个 npz 加载并合并，可选只取指定 task"""
    if sources is None:
        sources = DEFAULT_SOURCES
    all_feats, all_labels, all_success = [], [], []
    layer_indices = None

    for path, task_filter in sources:
        data = np.load(path, allow_pickle=True)
        feats = data['features']
        labels = data['task_labels']
        succ = data['success']
        if layer_indices is None:
            layer_indices = data['layer_indices']

        if task_filter is not None:
            mask = np.isin(labels, task_filter)
            feats, labels, succ = feats[mask], labels[mask], succ[mask]
        all_feats.append(feats)
        all_labels.append(labels)
        all_success.append(succ)
        print(f"  加载 {os.path.basename(path)}: {feats.shape[0]} episodes, tasks={list(np.unique(labels))}")

    features = np.concatenate(all_feats, axis=0)
    task_labels = np.concatenate(all_labels)
    success = np.concatenate(all_success)
    print(f"  合并: {features.shape[0]} episodes, {len(np.unique(task_labels))} tasks")
    return features, task_labels, success, layer_indices


def group_by_task(features, task_labels, success, layer_idx):
    """按 task 分组，提取指定层的特征"""
    task_data = {}
    for task in np.unique(task_labels):
        mask = task_labels == task
        s_mask = mask & success
        f_mask = mask & (~success)
        n_s, n_f = int(s_mask.sum()), int(f_mask.sum())
        sr = n_s / mask.sum() if mask.sum() > 0 else 0

        task_data[task] = {
            'succ_feats': features[s_mask, layer_idx, :],  # [n_s, D]
            'fail_feats': features[f_mask, layer_idx, :],  # [n_f, D]
            'all_feats': features[mask, layer_idx, :],      # [n_all, D]
            'n_success': n_s,
            'n_fail': n_f,
            'success_rate': float(sr),
        }
    return task_data


# ===================== 分析函数 =====================
def analyze_balanced(s_feats, f_feats):
    """分析 1: 等量成功/失败轨迹的相似度"""
    n = min(len(s_feats), len(f_feats))
    if n < 2:
        return None
    s, f = s_feats[:n], f_feats[:n]
    intra_s = mean_upper_tri(cosine_sim_matrix(s, s))
    intra_f = mean_upper_tri(cosine_sim_matrix(f, f))
    cross = float(cosine_sim_matrix(s, f).mean())
    return {
        'n_per_group': n,
        'intra_success': intra_s,
        'intra_fail': intra_f,
        'cross': cross,
        'gap_success': intra_s - cross,
        'gap_fail': intra_f - cross,
    }


def analyze_unbalanced(s_feats, f_feats):
    """分析 2: 不限数量的成功/失败轨迹相似度"""
    if len(s_feats) < 1 or len(f_feats) < 1:
        return None
    intra_s = mean_upper_tri(cosine_sim_matrix(s_feats, s_feats)) if len(s_feats) >= 2 else float('nan')
    intra_f = mean_upper_tri(cosine_sim_matrix(f_feats, f_feats)) if len(f_feats) >= 2 else float('nan')
    cross = float(cosine_sim_matrix(s_feats, f_feats).mean())
    return {
        'n_success': len(s_feats),
        'n_fail': len(f_feats),
        'intra_success': intra_s,
        'intra_fail': intra_f,
        'cross': cross,
        'gap_success': intra_s - cross if not np.isnan(intra_s) else float('nan'),
        'gap_fail': intra_f - cross if not np.isnan(intra_f) else float('nan'),
    }


def analyze_overall(all_feats):
    """分析 3: task 内所有轨迹的整体相似度"""
    if len(all_feats) < 2:
        return None
    sim = cosine_sim_matrix(all_feats, all_feats)
    return {
        'n_traj': len(all_feats),
        'mean_sim': mean_upper_tri(sim),
    }


# ===================== SimplerEnv 任务 text embedding =====================
DEFAULT_SIMPLER_TEXT_NPZ = "/root/data/xuyuan1/dataset/embedding/simpler_text_egohod.npz"
BRIDGE_EGOHOD_NPZ = "/root/data/xuyuan1/dataset/embedding/bridge_text_egohod_proj.npz"
BRIDGE_QWEN_NPZ = "/root/data/xuyuan1/dataset/embedding/bridge_text_embeddings.npz"

# Bridge 上与 Simpler 指令语义最接近的 task_index（与 bridge_text_*.npz 行号一致）
# 说明: stack / eggplant 在 Bridge 无完全同句，用最近任务代理；若已有 simpler_text_egohod.npz 则优先用真 EgoHOD 编码
SIMPLER_BRIDGE_FALLBACK_IDX = {
    "widowx_spoon_on_towel": 17156,              # put the spoon on the towel
    "widowx_carrot_on_plate": 172,               # put carrot on plate
    "widowx_stack_cube": 1376,                   # put the green block on top of the yellow block
    "widowx_put_eggplant_in_basket": 329,        # put eggplant into pan（篮子任务无完全匹配，用容器类代理）
}


def load_simpler_text_embeddings(npz_path=DEFAULT_SIMPLER_TEXT_NPZ):
    """加载 encode_simpler_text.py 生成的 EgoHOD 真编码 → {task_name: [512]}"""
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    for i, name in enumerate(data["task_names"]):
        out[str(name)] = data["embeddings"][i].astype(np.float32)
    print(f"  加载 text embeddings (Simpler EgoHOD 直编): {npz_path} ({len(out)} tasks)")
    return out


def load_simpler_text_from_bridge(npz_path, name="bridge"):
    """从 Bridge 预计算 npz 按 task_index 取行，与 alignment 训练时 text 空间一致"""
    mat = np.load(npz_path)["embeddings"].astype(np.float32)
    out = {}
    for task, idx in SIMPLER_BRIDGE_FALLBACK_IDX.items():
        out[task] = mat[idx].copy()
    print(f"  加载 text embeddings (Bridge 语义代理, {name}): {npz_path}, dim={mat.shape[1]}")
    return out


def get_task_text_embeddings(d_text: int):
    """
    按 aligner 的 d_text 返回 {task: vec}，保证与 DualAligner.proj_text 输入维一致。
    - 512: 优先 simpler_text_egohod.npz，否则 Bridge EgoHOD 代理行
    - 4096: Bridge Qwen 代理行（与 train 时 egohod_path=bridge_text_embeddings.npz 一致）
    """
    if d_text == 512:
        if os.path.isfile(DEFAULT_SIMPLER_TEXT_NPZ):
            return load_simpler_text_embeddings(DEFAULT_SIMPLER_TEXT_NPZ)
        return load_simpler_text_from_bridge(BRIDGE_EGOHOD_NPZ, "EgoHOD-512")
    if d_text == 4096:
        return load_simpler_text_from_bridge(BRIDGE_QWEN_NPZ, "Qwen-4096")
    raise ValueError(f"不支持的 d_text={d_text}，当前仅支持 512(EgoHOD) 与 4096(Qwen)")


def compute_action_text_similarity(task_data_raw, aligner, task_text_emb):
    """
    每条轨迹的 action 经 proj_action，task text 经 proj_text，在共享空间算 cosine
    task_text_emb: {task_name: np.array [D_text]}
    """
    results = {}
    for task_name, d in sorted(task_data_raw.items()):
        if task_name not in task_text_emb:
            print(f"  {task_name}: 无 text embedding，跳过")
            continue

        text_vec = task_text_emb[task_name].reshape(1, -1)
        with torch.no_grad():
            text_proj = aligner.forward_text(torch.from_numpy(text_vec).float())
            text_proj = torch.nn.functional.normalize(text_proj, dim=1).numpy()

        n_s, n_f = d['n_success'], d['n_fail']
        if n_s > 0:
            with torch.no_grad():
                s_proj = aligner.forward_action(torch.from_numpy(d['succ_feats']).float())
                s_proj = torch.nn.functional.normalize(s_proj, dim=1).numpy()
            s_sims = (s_proj @ text_proj.T).reshape(-1)
        else:
            s_sims = np.array([])

        if n_f > 0:
            with torch.no_grad():
                f_proj = aligner.forward_action(torch.from_numpy(d['fail_feats']).float())
                f_proj = torch.nn.functional.normalize(f_proj, dim=1).numpy()
            f_sims = (f_proj @ text_proj.T).reshape(-1)
        else:
            f_sims = np.array([])

        all_sims = np.concatenate([s_sims, f_sims]) if (len(s_sims) + len(f_sims)) > 0 else np.array([])

        # balanced: 取等量
        n_bal = min(n_s, n_f)
        if n_bal > 0:
            bal_s = float(s_sims[:n_bal].mean())
            bal_f = float(f_sims[:n_bal].mean())
        else:
            bal_s = bal_f = float('nan')

        results[task_name] = {
            'success_rate': d['success_rate'],
            'n_success': n_s,
            'n_fail': n_f,
            'mean_sim_success': float(s_sims.mean()) if len(s_sims) > 0 else float('nan'),
            'mean_sim_fail': float(f_sims.mean()) if len(f_sims) > 0 else float('nan'),
            'mean_sim_all': float(all_sims.mean()) if len(all_sims) > 0 else float('nan'),
            'gap_succ_minus_fail': float(s_sims.mean() - f_sims.mean())
                if len(s_sims) > 0 and len(f_sims) > 0 else float('nan'),
            'balanced_n': n_bal,
            'balanced_sim_success': bal_s,
            'balanced_sim_fail': bal_f,
            'balanced_gap': bal_s - bal_f if n_bal > 0 else float('nan'),
        }

    return results


def compute_action_text_matching(task_data_raw, aligner, task_text_emb):
    """
    Action-Text 匹配正确率: 每条 action 对 4 个 task text 算 cosine，
    取最相似的 text 作为预测，与真实 task 比较，得到 top-1 正确率。
    分 success / fail / all 三组统计。
    """
    task_names = sorted(task_text_emb.keys())
    # 构建 text gallery: [4, D_shared]
    with torch.no_grad():
        text_vecs = np.stack([task_text_emb[t] for t in task_names])
        text_proj = aligner.forward_text(torch.from_numpy(text_vecs).float())
        text_proj = torch.nn.functional.normalize(text_proj, dim=1).numpy()

    # 对每条轨迹的 action 投影并匹配
    s_correct, s_total = 0, 0
    f_correct, f_total = 0, 0
    per_task = {}

    for task_name, d in sorted(task_data_raw.items()):
        if task_name not in task_text_emb:
            continue
        gt_idx = task_names.index(task_name)
        tc, sc, fc = 0, 0, 0
        tn, sn, fn = 0, 0, 0

        for label, feats in [('success', d['succ_feats']), ('fail', d['fail_feats'])]:
            if len(feats) == 0:
                continue
            with torch.no_grad():
                a_proj = aligner.forward_action(torch.from_numpy(feats).float())
                a_proj = torch.nn.functional.normalize(a_proj, dim=1).numpy()
            # [N, 4] cosine similarity
            sims = a_proj @ text_proj.T
            preds = sims.argmax(axis=1)
            correct = int((preds == gt_idx).sum())
            n = len(feats)
            tc += correct; tn += n
            if label == 'success':
                sc += correct; sn += n
                s_correct += correct; s_total += n
            else:
                fc += correct; fn += n
                f_correct += correct; f_total += n

        per_task[task_name] = {
            'success_rate': d['success_rate'],
            'acc_success': sc / sn if sn > 0 else float('nan'),
            'acc_fail': fc / fn if fn > 0 else float('nan'),
            'acc_all': tc / tn if tn > 0 else float('nan'),
            'n_success': sn, 'n_fail': fn,
        }

    total = s_total + f_total
    return {
        'per_task': per_task,
        'overall_acc_success': s_correct / s_total if s_total > 0 else float('nan'),
        'overall_acc_fail': f_correct / f_total if f_total > 0 else float('nan'),
        'overall_acc_all': (s_correct + f_correct) / total if total > 0 else float('nan'),
        'n_success': s_total, 'n_fail': f_total,
    }


def linear_cka(X, Y):
    """Linear CKA: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)"""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XTX = X.T @ X
    YTY = Y.T @ Y
    YTX = Y.T @ X
    num = np.linalg.norm(YTX, 'fro') ** 2
    den = np.linalg.norm(XTX, 'fro') * np.linalg.norm(YTY, 'fro')
    return float(num / den) if den > 1e-10 else 0.0


def compute_action_text_cka(task_data_raw, task_text_emb):
    """
    跨任务 pool 计算 CKA(action, text)，分 success/fail/all 三组。
    CKA 对线性变换不变，直接用原始特征。
    单个任务内 text 全相同 → centering 后贡献为零，所以必须跨任务。
    """
    s_actions, s_texts = [], []
    f_actions, f_texts = [], []

    for task_name, d in sorted(task_data_raw.items()):
        if task_name not in task_text_emb:
            continue
        text_vec = task_text_emb[task_name]
        if d['n_success'] > 0:
            s_actions.append(d['succ_feats'])
            s_texts.append(np.tile(text_vec, (d['n_success'], 1)))
        if d['n_fail'] > 0:
            f_actions.append(d['fail_feats'])
            f_texts.append(np.tile(text_vec, (d['n_fail'], 1)))

    results = {}
    if s_actions:
        sa, st = np.concatenate(s_actions), np.concatenate(s_texts)
        results['cka_success'] = linear_cka(sa, st)
        results['n_success'] = len(sa)
    if f_actions:
        fa, ft = np.concatenate(f_actions), np.concatenate(f_texts)
        results['cka_fail'] = linear_cka(fa, ft)
        results['n_fail'] = len(fa)
    if s_actions and f_actions:
        aa = np.concatenate([sa, fa])
        at = np.concatenate([st, ft])
        results['cka_all'] = linear_cka(aa, at)
        results['n_all'] = len(aa)
        results['cka_gap'] = results['cka_success'] - results['cka_fail']

    # balanced: 每个任务取 min(n_s, n_f) 条
    bs_actions, bs_texts, bf_actions, bf_texts = [], [], [], []
    for task_name, d in sorted(task_data_raw.items()):
        if task_name not in task_text_emb:
            continue
        n_bal = min(d['n_success'], d['n_fail'])
        if n_bal < 1:
            continue
        text_vec = task_text_emb[task_name]
        bs_actions.append(d['succ_feats'][:n_bal])
        bs_texts.append(np.tile(text_vec, (n_bal, 1)))
        bf_actions.append(d['fail_feats'][:n_bal])
        bf_texts.append(np.tile(text_vec, (n_bal, 1)))
    if bs_actions:
        bsa, bst = np.concatenate(bs_actions), np.concatenate(bs_texts)
        bfa, bft = np.concatenate(bf_actions), np.concatenate(bf_texts)
        results['balanced_cka_success'] = linear_cka(bsa, bst)
        results['balanced_cka_fail'] = linear_cka(bfa, bft)
        results['balanced_n_per_group'] = len(bsa)
        results['balanced_cka_gap'] = results['balanced_cka_success'] - results['balanced_cka_fail']

    return results


# ===================== 单个 aligner 分析 =====================
def run_analysis(task_data_raw, aligner_path, layer_indices, li, output_tag=None):
    """对一个 aligner checkpoint 执行分析；text 向量按 checkpoint 的 d_text 自动选取"""
    aligner = None
    # 有 aligner 时 text 维与训练一致；raw 基线无 action-text 对齐，CKA 仍用 512 维 Bridge 代理便于对比
    task_text_emb = None
    if aligner_path is not None:
        ckpt = torch.load(aligner_path, map_location='cpu')
        d_action = ckpt['d_action']
        d_text = ckpt['d_text']
        task_text_emb = get_task_text_embeddings(int(d_text))
        state = ckpt['model_state_dict']
        d_shared = state['proj_action.weight'].shape[0]
        aligner = DualAligner(d_action, d_text, d_shared)
        aligner.load_state_dict(state)
        aligner.eval()
        epoch = ckpt.get('epoch', '?')
        val_top1 = ckpt.get('val_metrics', {}).get('top1', 0)
        space_label = f"aligned_ep{epoch}"
        print(f"\n  Aligner: {d_action}→{d_shared}, epoch={epoch}, val_top1={val_top1:.2%}")

        task_data = {}
        for t, d in task_data_raw.items():
            task_data[t] = {
                'succ_feats': project_features(d['succ_feats'], aligner) if d['n_success'] > 0 else d['succ_feats'],
                'fail_feats': project_features(d['fail_feats'], aligner) if d['n_fail'] > 0 else d['fail_feats'],
                'all_feats': project_features(d['all_feats'], aligner),
                'n_success': d['n_success'],
                'n_fail': d['n_fail'],
                'success_rate': d['success_rate'],
            }
    else:
        task_data = task_data_raw
        space_label = "raw"
        epoch = None
        task_text_emb = get_task_text_embeddings(512)
        print(f"\n  模式: 原始特征 (无 alignment 投影)；CKA 使用 EgoHOD 维 text 代理")

    # 分析 1: 等量成功/失败
    print(f"\n{'='*80}")
    print(f"分析 1: 等量成功/失败轨迹相似度 [{space_label}]")
    print(f"{'='*80}")
    print(f"  {'Task':<35} {'n':>3} | {'intra_S':>8} {'intra_F':>8} {'cross':>8} | {'gap_S':>7} {'gap_F':>7}")
    print(f"  {'-'*90}")

    results_balanced = {}
    for t in sorted(task_data.keys()):
        d = task_data[t]
        r = analyze_balanced(d['succ_feats'], d['fail_feats'])
        if r is None:
            continue
        results_balanced[t] = {**r, 'success_rate': d['success_rate']}
        print(f"  {t:<35} {r['n_per_group']:>3} | "
              f"{r['intra_success']:>8.4f} {r['intra_fail']:>8.4f} {r['cross']:>8.4f} | "
              f"{r['gap_success']:>7.4f} {r['gap_fail']:>7.4f}")

    # 分析 2: 不等量成功/失败
    print(f"\n{'='*80}")
    print(f"分析 2: 不等量成功/失败轨迹相似度 [{space_label}]")
    print(f"{'='*80}")
    print(f"  {'Task':<35} {'S':>3} {'F':>3} | {'intra_S':>8} {'intra_F':>8} {'cross':>8} | {'gap_S':>7} {'gap_F':>7}")
    print(f"  {'-'*95}")

    results_unbalanced = {}
    for t in sorted(task_data.keys()):
        d = task_data[t]
        r = analyze_unbalanced(d['succ_feats'], d['fail_feats'])
        if r is None:
            continue
        results_unbalanced[t] = {**r, 'success_rate': d['success_rate']}
        i_s = f"{r['intra_success']:>8.4f}" if not np.isnan(r['intra_success']) else "     N/A"
        i_f = f"{r['intra_fail']:>8.4f}" if not np.isnan(r['intra_fail']) else "     N/A"
        g_s = f"{r['gap_success']:>7.4f}" if not np.isnan(r['gap_success']) else "    N/A"
        g_f = f"{r['gap_fail']:>7.4f}" if not np.isnan(r['gap_fail']) else "    N/A"
        print(f"  {t:<35} {r['n_success']:>3} {r['n_fail']:>3} | "
              f"{i_s} {i_f} {r['cross']:>8.4f} | {g_s} {g_f}")

    # 分析 3: 整体相似度 vs 成功率
    print(f"\n{'='*80}")
    print(f"分析 3: 整体相似度 vs 成功率 [{space_label}]")
    print(f"{'='*80}")
    print(f"  {'Task':<35} {'SR':>5} | {'mean_sim':>8} {'n_traj':>6}")
    print(f"  {'-'*65}")

    results_overall = {}
    srs, sims = [], []
    for t in sorted(task_data.keys()):
        d = task_data[t]
        r = analyze_overall(d['all_feats'])
        if r is None:
            continue
        results_overall[t] = {**r, 'success_rate': d['success_rate']}
        srs.append(d['success_rate'])
        sims.append(r['mean_sim'])
        print(f"  {t:<35} {d['success_rate']:>4.0%} | {r['mean_sim']:>8.4f} {r['n_traj']:>6}")

    if len(srs) >= 3:
        corr = np.corrcoef(srs, sims)[0, 1]
        print(f"\n  Pearson 相关系数 (成功率 vs 整体相似度): r = {corr:.4f}")

    # 分析 4: action-text 相似度
    results_action_text = {}
    if aligner is not None and task_text_emb is not None:
        print(f"\n{'='*80}")
        print(f"分析 4: Action-Text 相似度 [{space_label}]")
        print(f"  (成功轨迹的 action 是否和 task text 更对齐?)")
        print(f"{'='*80}")

        results_action_text = compute_action_text_similarity(
            task_data_raw, aligner, task_text_emb)

        # 4a: balanced (同等数量)
        print(f"\n  4a. Balanced (等量成功/失败):")
        print(f"  {'Task':<35} {'n':>3} | {'sim_S':>7} {'sim_F':>7} {'gap':>7}")
        print(f"  {'-'*65}")
        for t in sorted(results_action_text.keys()):
            r = results_action_text[t]
            if r['balanced_n'] > 0:
                print(f"  {t:<35} {r['balanced_n']:>3} | "
                      f"{r['balanced_sim_success']:>7.4f} {r['balanced_sim_fail']:>7.4f} {r['balanced_gap']:>7.4f}")

        # 4b: unbalanced (全量)
        print(f"\n  4b. Unbalanced (全量):")
        print(f"  {'Task':<35} {'S':>3} {'F':>3} | {'sim_S':>7} {'sim_F':>7} {'gap':>7} | {'sim_all':>7}")
        print(f"  {'-'*80}")
        for t in sorted(results_action_text.keys()):
            r = results_action_text[t]
            s_str = f"{r['mean_sim_success']:>7.4f}" if not np.isnan(r['mean_sim_success']) else "    N/A"
            f_str = f"{r['mean_sim_fail']:>7.4f}" if not np.isnan(r['mean_sim_fail']) else "    N/A"
            g_str = f"{r['gap_succ_minus_fail']:>7.4f}" if not np.isnan(r['gap_succ_minus_fail']) else "    N/A"
            print(f"  {t:<35} {r['n_success']:>3} {r['n_fail']:>3} | "
                  f"{s_str} {f_str} {g_str} | {r['mean_sim_all']:>7.4f}")

        # 4c: 成功率 vs 整体相似度
        print(f"\n  4c. 成功率 vs Action-Text 相似度:")
        print(f"  {'Task':<35} {'SR':>5} | {'sim_all':>7}")
        print(f"  {'-'*55}")
        at_srs, at_sims = [], []
        for t in sorted(results_action_text.keys()):
            r = results_action_text[t]
            print(f"  {t:<35} {r['success_rate']:>4.0%} | {r['mean_sim_all']:>7.4f}")
            at_srs.append(r['success_rate'])
            at_sims.append(r['mean_sim_all'])
        if len(at_srs) >= 3:
            corr = np.corrcoef(at_srs, at_sims)[0, 1]
            print(f"\n  Pearson (成功率 vs action-text sim): r = {corr:.4f}")

        # 汇总
        gaps = [r['gap_succ_minus_fail'] for r in results_action_text.values()
                if not np.isnan(r['gap_succ_minus_fail'])]
        if gaps:
            print(f"\n  平均 gap (sim_success - sim_fail): {np.mean(gaps):.4f}")
            print(f"  → {'成功轨迹更对齐 text' if np.mean(gaps) > 0 else '失败轨迹更对齐 text'}")

    # 分析 5: Action-Text Matching Accuracy (top-1 retrieval)
    results_matching = {}
    if aligner is not None and task_text_emb is not None:
        print(f"\n{'='*80}")
        print(f"分析 5: Action-Text Matching Accuracy [{space_label}]")
        print(f"  (每条 action 对 {len(task_text_emb)} 个 task text 检索 top-1)")
        print(f"{'='*80}")

        results_matching = compute_action_text_matching(
            task_data_raw, aligner, task_text_emb)

        print(f"\n  {'Task':<35} {'SR':>5} | {'acc_S':>7} {'acc_F':>7} {'acc_all':>7}")
        print(f"  {'-'*70}")
        for t in sorted(results_matching['per_task'].keys()):
            r = results_matching['per_task'][t]
            as_str = f"{r['acc_success']:>6.1%}" if not np.isnan(r['acc_success']) else "   N/A"
            af_str = f"{r['acc_fail']:>6.1%}" if not np.isnan(r['acc_fail']) else "   N/A"
            print(f"  {t:<35} {r['success_rate']:>4.0%} | {as_str} {af_str} {r['acc_all']:>6.1%}")

        print(f"\n  Overall: acc_S={results_matching['overall_acc_success']:.1%}  "
              f"acc_F={results_matching['overall_acc_fail']:.1%}  "
              f"acc_all={results_matching['overall_acc_all']:.1%}")

    # 分析 6: Action-Text CKA (CKA 对线性变换不变，直接用原始特征)
    results_cka = {}
    if task_text_emb is not None:
        print(f"\n{'='*80}")
        print(f"分析 5: Action-Text CKA [{space_label}]")
        print(f"  (CKA 对线性变换不变，使用原始 action 特征)")
        print(f"{'='*80}")

        results_cka = compute_action_text_cka(task_data_raw, task_text_emb)

        if 'balanced_cka_success' in results_cka:
            print(f"\n  Balanced (每task等量, 共 {results_cka['balanced_n_per_group']} 条/组):")
            print(f"    CKA_success = {results_cka['balanced_cka_success']:.6f}")
            print(f"    CKA_fail    = {results_cka['balanced_cka_fail']:.6f}")
            print(f"    gap (S-F)   = {results_cka['balanced_cka_gap']:+.6f}")

        if 'cka_success' in results_cka:
            print(f"\n  Unbalanced (全量):")
            print(f"    CKA_success = {results_cka['cka_success']:.6f}  (n={results_cka['n_success']})")
            print(f"    CKA_fail    = {results_cka['cka_fail']:.6f}  (n={results_cka['n_fail']})")
            print(f"    CKA_all     = {results_cka['cka_all']:.6f}  (n={results_cka['n_all']})")
            print(f"    gap (S-F)   = {results_cka['cka_gap']:+.6f}")

        verdict = ""
        gap_key = 'balanced_cka_gap' if 'balanced_cka_gap' in results_cka else 'cka_gap'
        if gap_key in results_cka:
            g = results_cka[gap_key]
            verdict = "成功轨迹 action 与 text 的 CKA 更高" if g > 0 else "失败轨迹 action 与 text 的 CKA 更高"
            print(f"\n  → {verdict}")

    return {
        'space': space_label,
        'epoch': epoch,
        'balanced': results_balanced,
        'unbalanced': results_unbalanced,
        'overall_vs_sr': results_overall,
        'action_text': results_action_text,
        'action_text_matching': results_matching,
        'action_text_cka': results_cka,
    }


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description="SimplerEnv aligned similarity analysis")
    parser.add_argument("--aligner_path", type=str, default=None,
                        help="单个 DualAligner .pt 路径")
    parser.add_argument("--aligner_dir", type=str, default=None,
                        help="扫描目录下所有 .pt checkpoint 批量分析")
    parser.add_argument("--chunk", type=str, default="chunk4")
    parser.add_argument("--layer_idx", type=int, default=3,
                        help="layer_indices 数组中的位置索引 (默认 3 = Layer 10)")
    parser.add_argument("--output_tag", type=str, default=None,
                        help="输出文件标签")
    parser.add_argument("--no_raw", action="store_true",
                        help="不跑 raw 基线（避免覆盖 aligned_analysis_raw_L*.json）")
    args = parser.parse_args()

    # 加载合并特征 (只加载一次)
    print("=" * 80)
    print("SimplerEnv 成功/失败轨迹表征分析")
    print("=" * 80)
    features, task_labels, success, layer_indices = load_and_merge_features()
    li = args.layer_idx
    print(f"  使用层: Layer {layer_indices[li]} (index={li})")

    task_data_raw = group_by_task(features, task_labels, success, li)
    for t, d in sorted(task_data_raw.items()):
        print(f"  {t}: {d['n_success']}成功 {d['n_fail']}失败 (SR={d['success_rate']:.0%})")

    # 收集要分析的 checkpoint 列表
    import glob
    ckpt_list = []
    if args.aligner_dir:
        pts = sorted(glob.glob(os.path.join(args.aligner_dir, "*.pt")))
        ckpt_list = [(p, os.path.basename(p).replace('.pt', '')) for p in pts]
        print(f"\n  扫描到 {len(ckpt_list)} 个 checkpoint: {[c[1] for c in ckpt_list]}")
    elif args.aligner_path:
        tag = args.output_tag or os.path.basename(args.aligner_path).replace('.pt', '')
        ckpt_list = [(args.aligner_path, tag)]
    if not args.no_raw:
        ckpt_list.append((None, 'raw'))

    # 逐个 checkpoint 分析
    out_dir = "/root/data/xuyuan1/Codes/INT-ACT/ENV/LOGS/simpler_features/analysis"
    os.makedirs(out_dir, exist_ok=True)
    all_summaries = []

    def _c(obj):
        if isinstance(obj, (np.floating, float)):
            return round(float(obj), 6)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _c(v) for k, v in obj.items()}
        if isinstance(obj, (list, np.ndarray)):
            return [_c(x) for x in obj]
        return obj

    for ckpt_path, tag in ckpt_list:
        print(f"\n{'#'*80}")
        print(f"# Checkpoint: {tag}")
        print(f"{'#'*80}")

        result = run_analysis(task_data_raw, ckpt_path, layer_indices, li, tag)

        out = {
            'config': {
                'aligner': ckpt_path or 'none',
                'layer': int(layer_indices[li]),
                'space': result['space'],
                'epoch': result['epoch'],
            },
            'balanced': _c(result['balanced']),
            'unbalanced': _c(result['unbalanced']),
            'overall_vs_sr': _c(result['overall_vs_sr']),
            'action_text': _c(result['action_text']),
            'action_text_matching': _c(result['action_text_matching']),
            'action_text_cka': _c(result['action_text_cka']),
        }
        json_path = os.path.join(out_dir, f"aligned_analysis_{tag}_L{layer_indices[li]}.json")
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  → 保存: {json_path}")
        all_summaries.append((tag, result))

    # 汇总对比表: 每个 checkpoint 的 gap_S 均值
    if len(all_summaries) > 1:
        print(f"\n{'='*80}")
        print("汇总: 各 checkpoint 的平均 gap_S (balanced)")
        print(f"{'='*80}")
        print(f"  {'Checkpoint':<50} | {'avg_gap_S':>9} {'avg_gap_F':>9} {'avg_cross':>9}")
        print(f"  {'-'*85}")
        for tag, res in all_summaries:
            gaps_s = [v['gap_success'] for v in res['balanced'].values()]
            gaps_f = [v['gap_fail'] for v in res['balanced'].values()]
            crosses = [v['cross'] for v in res['balanced'].values()]
            if gaps_s:
                print(f"  {tag:<50} | {np.mean(gaps_s):>9.4f} {np.mean(gaps_f):>9.4f} {np.mean(crosses):>9.4f}")

    print(f"\n全部分析完成!")


if __name__ == "__main__":
    main()
