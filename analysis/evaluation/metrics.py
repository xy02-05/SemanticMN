"""
统一指标计算模块

所有函数接受纯 numpy 数组，无副作用，可独立使用。
包含以下指标:
  1. Linear CKA  — 表征-文本结构相似度（核对齐）
  2. SVCCA       — SVD + CCA，度量共享子空间
  3. PWCCA       — 投影加权 CCA
  4. KNN         — K近邻任务分类准确率
  5. Silhouette  — 聚类分离度
  6. Intra/Inter — 类内/类间余弦相似度
  7. Calinski-Harabasz — 方差比
  8. Effective Rank    — 特征有效秩（SVD 信息熵）

来源: 合并自 analyze_representation.py 和 compare_cka_svcca_aggregation.py
"""
import numpy as np


# ====================================================================
# 1. Linear CKA (Centered Kernel Alignment)
# ====================================================================

def linear_cka(X, Y):
    """
    Linear CKA: 衡量两个表征矩阵的线性核对齐度

    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    范围 [0, 1]，不受线性变换影响

    Args:
        X: [N, D1]  (如 VLA action features)
        Y: [N, D2]  (如 text embeddings)
    Returns:
        float: CKA 值
    """
    n = X.shape[0]
    assert Y.shape[0] == n, f"Shape mismatch: X={X.shape}, Y={Y.shape}"
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XTX = X.T @ X
    YTY = Y.T @ Y
    YTX = Y.T @ X
    numerator = np.linalg.norm(YTX, 'fro') ** 2
    denominator = np.linalg.norm(XTX, 'fro') * np.linalg.norm(YTY, 'fro')
    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


# ====================================================================
# 1b. Whitened CKA (去除方差主导效应的 CKA)
# ====================================================================

def whitened_cka(X, Y, n_components=None):
    """
    Whitened CKA: 先对特征做 SVD 白化（去除各方向方差差异），再计算 CKA

    原理:
      标准 CKA 中，方差最大的几个主成分会主导结果。
      白化就是做 X = U S V^T → X_w = U（丢掉 S），
      使得各方向方差相等，所有维度平等参与 CKA 计算。

    白化后 X_w^T X_w = I, 所以:
      CKA_whitened = ||Y_w^T X_w||_F^2 / sqrt(kx * ky)

    这与 RV coefficient (sum of squared canonical correlations / sqrt(kx*ky)) 等价。

    Args:
        X: [N, D1], Y: [N, D2]
        n_components: 保留的成分数 (None=全部, 或指定固定维度)
    Returns:
        float: Whitened CKA 值, info: dict
    """
    n = X.shape[0]
    assert Y.shape[0] == n
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # SVD 白化: X = U S V^T → X_w = U[:, :k]  (各方向等权)
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)

    # 决定保留多少成分
    if n_components is not None:
        kx = min(n_components, len(Sx))
        ky = min(n_components, len(Sy))
    else:
        kx = len(Sx)
        ky = len(Sy)

    X_w = Ux[:, :kx]  # [N, kx] — 白化后，各方向方差=1
    Y_w = Uy[:, :ky]  # [N, ky]

    # CKA 公式 (在白化空间)
    # X_w^T X_w = I_kx, Y_w^T Y_w = I_ky (正交列)
    # 所以 ||X_w^T X_w||_F = sqrt(kx), ||Y_w^T Y_w||_F = sqrt(ky)
    YTX = Y_w.T @ X_w  # [ky, kx]
    numerator = np.linalg.norm(YTX, 'fro') ** 2
    denominator = np.sqrt(float(kx) * float(ky))
    if denominator < 1e-10:
        return 0.0, {"kx": kx, "ky": ky}
    return float(numerator / denominator), {"kx": kx, "ky": ky}


def whitened_cka_thresholded(X, Y, threshold=0.999):
    """
    基于方差阈值的 Whitened CKA
    先做 SVD 降维到保留 threshold 方差，再白化计算 CKA

    比全维度 whitened_cka 更鲁棒 (去掉噪声维度)

    Args:
        X: [N, D1], Y: [N, D2]
        threshold: SVD 保留方差比例 (默认 0.999)
    Returns:
        float: Whitened CKA 值, info: dict
    """
    n = X.shape[0]
    assert Y.shape[0] == n
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)

    # 按方差阈值截断
    var_x = np.cumsum(Sx ** 2) / np.sum(Sx ** 2)
    kx = int(np.searchsorted(var_x, threshold)) + 1
    kx = min(max(kx, 1), len(Sx))

    var_y = np.cumsum(Sy ** 2) / np.sum(Sy ** 2)
    ky = int(np.searchsorted(var_y, threshold)) + 1
    ky = min(max(ky, 1), len(Sy))

    X_w = Ux[:, :kx]
    Y_w = Uy[:, :ky]

    YTX = Y_w.T @ X_w
    numerator = np.linalg.norm(YTX, 'fro') ** 2
    denominator = np.sqrt(float(kx) * float(ky))
    if denominator < 1e-10:
        return 0.0, {"kx": kx, "ky": ky}
    return float(numerator / denominator), {"kx": kx, "ky": ky}


# ====================================================================
# 2-3. SVCCA / PWCCA
# ====================================================================

def _svd_reduce(M, threshold=0.9999):
    """
    SVD 降维: 保留解释 threshold 方差的前 k 个主成分

    Args:
        M: [N, D] 已中心化的数据
        threshold: 累积方差保留比例 (默认 0.99)
    Returns:
        M_red: [N, k], k: int, S_kept: [k] 奇异值
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    var_ratio = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(var_ratio, threshold)) + 1
    k = max(k, 1)
    k = min(k, len(S))
    return U[:, :k] * S[:k], k, S[:k]


def _qr_cca(X_red, Y_red):
    """
    QR 分解法求 CCA (比白化法数值更稳定)

    原理: X=Qx·Rx, Y=Qy·Ry → canonical correlations = SVD(Qx^T Qy)
    Returns: correlations, Uc (left singular vecs), Rx
    """
    Qx, Rx = np.linalg.qr(X_red, mode='reduced')
    Qy, Ry = np.linalg.qr(Y_red, mode='reduced')
    Uc, s, Vct = np.linalg.svd(Qx.T @ Qy, full_matrices=False)
    return np.clip(s, 0.0, 1.0), Uc, Rx


def svcca(X, Y, threshold=0.9999):
    """
    Singular Vector CCA (Raghu et al., 2017)

    步骤:
      1. 中心化
      2. SVD 降维特征维度: D_x → k_x, D_y → k_y (保留 threshold 方差)
      3. 若 max(k_x, k_y) > N//2, cap 防止 CCA 过拟合
      4. QR-CCA 求 canonical correlations
      5. 返回 mean(correlations)

    Args:
        X: [N, D1], Y: [N, D2]
    Returns:
        mean_corr: float, correlations: ndarray, info: dict
    """
    N = X.shape[0]
    assert Y.shape[0] == N
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    X_red, kx, _ = _svd_reduce(X, threshold)
    Y_red, ky, _ = _svd_reduce(Y, threshold)
    # 小样本保护: CCA 至少需要 N/2 自由度
    max_k = max(1, N // 2)
    if kx > max_k:
        X_red = X_red[:, :max_k]; kx = max_k
    if ky > max_k:
        Y_red = Y_red[:, :max_k]; ky = max_k
    correlations, _, _ = _qr_cca(X_red, Y_red)
    info = {"N": N, "kx": kx, "ky": ky, "n_corr": len(correlations)}
    return float(np.mean(correlations)), correlations, info


def pwcca(X, Y, threshold=0.9999):
    """
    Projection-Weighted CCA (Morcos et al., 2018)

    用 X 的 SVD 奇异值对 CCA 方向加权，反映原始空间方差贡献:
      α_i = Σ_j S_x[j]² · Uc[j,i]²

    Returns:
        pwcca_score: float, info: dict
    """
    N = X.shape[0]
    assert Y.shape[0] == N
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    X_red, kx, Sx = _svd_reduce(X, threshold)
    Y_red, ky, Sy = _svd_reduce(Y, threshold)
    max_k = max(1, N // 2)
    if kx > max_k:
        X_red = X_red[:, :max_k]; Sx = Sx[:max_k]; kx = max_k
    if ky > max_k:
        Y_red = Y_red[:, :max_k]; Sy = Sy[:max_k]; ky = max_k
    correlations, Uc, Rx = _qr_cca(X_red, Y_red)
    n_corr = len(correlations)
    info = {"N": N, "kx": kx, "ky": ky, "n_corr": n_corr}
    if n_corr == 0:
        return 0.0, info
    weights = np.sum((Sx[:, None] ** 2) * (Uc[:, :n_corr] ** 2), axis=0)
    weights = weights / (weights.sum() + 1e-10)
    return float(np.sum(weights * correlations)), info


def _svcca_pwcca_shared(X, Y, threshold=0.9999):
    """
    共享 SVD 的 SVCCA + PWCCA 联合计算

    PWCCA 权重修正 (Morcos et al., 2018):
      原始实现中 projections = U_k @ Uc 列正交导致权重恒等，退化为 SVCCA。
      正确做法: 用 SVD 奇异值 S_k 加权 CCA 方向，反映原始空间中的方差贡献:
        α_i = ||diag(S_k) @ Uc[:,i]||² = Σ_j S_k[j]² · Uc[j,i]²
      高方差 SVD 方向上的 canonical correlation 获得更大权重。

    Returns:
        svc_mean, pwc_val, info
    """
    N = X.shape[0]
    assert Y.shape[0] == N
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    X_red, kx, Sx = _svd_reduce(X, threshold)
    Y_red, ky, Sy = _svd_reduce(Y, threshold)

    max_k = max(1, N // 2)
    if kx > max_k:
        X_red = X_red[:, :max_k]; Sx = Sx[:max_k]; kx = max_k
    if ky > max_k:
        Y_red = Y_red[:, :max_k]; Sy = Sy[:max_k]; ky = max_k

    correlations, Uc, Rx = _qr_cca(X_red, Y_red)
    n_corr = len(correlations)
    info = {"N": N, "kx": kx, "ky": ky, "n_corr": n_corr}

    svc_mean = float(np.mean(correlations)) if n_corr > 0 else 0.0

    if n_corr == 0:
        pwc_val = 0.0
    else:
        # α_i = Σ_j S_x[j]² · Uc[j,i]²
        # Uc: [kx, n_corr], Sx: [kx]
        weights = np.sum((Sx[:, None] ** 2) * (Uc[:, :n_corr] ** 2), axis=0)
        weights = weights / (weights.sum() + 1e-10)
        pwc_val = float(np.sum(weights * correlations))

    return svc_mean, pwc_val, info


def compute_cka_svcca_pwcca(vla_feats, txt_feats, tag=""):
    """
    一次性计算 CKA + SVCCA + PWCCA (SVD 共享版, 约 2x 快于独立调用)

    Args:
        vla_feats: [N, D1]  action features
        txt_feats: [N, D2]  text embeddings (已按 task 对齐)
    Returns:
        dict: {cka, svcca, pwcca, N, svcca_kx, svcca_ky, svcca_n_corr}
    """
    cka_val = linear_cka(vla_feats, txt_feats)
    svc_mean, pwc_val, info = _svcca_pwcca_shared(vla_feats, txt_feats)
    return {
        "cka": float(cka_val),
        "svcca": svc_mean,
        "pwcca": pwc_val,
        "N": vla_feats.shape[0],
        "svcca_kx": info["kx"],
        "svcca_ky": info["ky"],
        "svcca_n_corr": info["n_corr"],
    }


# ====================================================================
# 4. KNN Task Classification
# ====================================================================

def knn_classification(features, labels, k=5, n_splits=5, max_samples=10000):
    """
    KNN 分类准确率 (Stratified K-Fold CV)

    使用 cosine 距离做 KNN, 衡量特征能否区分不同 task
    当 N > max_samples 时, 随机采样避免过慢

    Args:
        features: [N, D]
        labels: [N] — task 标签 (str 或 int)
        k: 邻居数
        n_splits: CV 折数
        max_samples: 最大样本数 (超过则采样)
    Returns:
        mean_acc, std_acc
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from sklearn.preprocessing import LabelEncoder

    N = len(features)
    if N > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    accuracies = []
    for train_idx, test_idx in cv.split(features, y):
        knn.fit(features[train_idx], y[train_idx])
        acc = knn.score(features[test_idx], y[test_idx])
        accuracies.append(acc)
    return float(np.mean(accuracies)), float(np.std(accuracies))


# ====================================================================
# 5. Silhouette Score
# ====================================================================

def silhouette_score(features, labels, max_samples=5000):
    """
    Silhouette Score: [-1, 1], 使用 cosine 距离
    越高 = 类内紧凑 + 类间分离
    当 N > max_samples 时采样
    """
    from sklearn.metrics import silhouette_score as _sil
    from sklearn.preprocessing import LabelEncoder

    N = len(features)
    if N > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(np.unique(y)) < 2:
        return 0.0
    return float(_sil(features, y, metric='cosine'))


# ====================================================================
# 6. Intra/Inter-class Cosine Similarity
# ====================================================================

def intra_inter_similarity(features, labels, max_samples=5000):
    """
    类内/类间 cosine 相似度

    当 N > max_samples 时, 随机采样 max_samples 条避免 O(N^2) 内存/计算

    Returns:
        intra_mean, inter_mean, gap (intra - inter, 越大越好)
    """
    N = len(features)
    if N > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    features_norm = features / norms
    cos_sim = features_norm @ features_norm.T
    label_eq = labels[:, None] == labels[None, :]
    upper_tri = np.triu(np.ones_like(label_eq, dtype=bool), k=1)
    intra_mask = label_eq & upper_tri
    inter_mask = (~label_eq) & upper_tri
    intra_mean = float(cos_sim[intra_mask].mean()) if intra_mask.any() else 0.0
    inter_mean = float(cos_sim[inter_mask].mean()) if inter_mask.any() else 0.0
    return intra_mean, inter_mean, float(intra_mean - inter_mean)


# ====================================================================
# 7. Calinski-Harabasz Index
# ====================================================================

def calinski_harabasz(features, labels):
    """方差比准则, 越高 = 聚类越好"""
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(np.unique(y)) < 2:
        return 0.0
    return float(calinski_harabasz_score(features, y))


# ====================================================================
# 8. Effective Rank
# ====================================================================

def effective_rank(features):
    """
    特征矩阵的 Effective Rank (基于 SVD 奇异值的信息熵)
    ER = exp(-sum(p_i * log(p_i))), p_i = sigma_i / sum(sigma_j)
    越高 = 表征越多样; 越低 = 表征越坍缩
    """
    features_centered = features - features.mean(axis=0)
    U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)
    S_pos = S[S > 1e-10]
    p = S_pos / S_pos.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


# ====================================================================
# 聚合策略辅助函数 (用于 CKA/SVCCA/PWCCA)
# ====================================================================

def _build_label_index(labels, valid_tasks):
    """
    构建 label → sample indices 的索引 (避免重复 O(N) 字符串比较)
    对大数据集 (N>10000, T>5000) 性能提升显著

    Returns:
        label_to_indices: {task_label: np.array of indices}
    """
    from collections import defaultdict
    idx_map = defaultdict(list)
    valid_set = set(valid_tasks)
    for i, lab in enumerate(labels):
        if lab in valid_set:
            idx_map[lab].append(i)
    return {k: np.array(v) for k, v in idx_map.items()}


def build_all_expand(feats, labels, valid_tasks, text_dict):
    """
    all_expand 策略: 全部轨迹保留, text 按 task 复制到对应轨迹数
    返回 vla [N_total, D], txt [N_total, D_text]
    """
    idx_map = _build_label_index(labels, valid_tasks)
    vla_rows, txt_rows = [], []
    for t in valid_tasks:
        idxs = idx_map.get(t)
        if idxs is None or len(idxs) == 0:
            continue
        vla_rows.append(feats[idxs])
        txt_rows.append(np.tile(text_dict[t], (len(idxs), 1)))
    return np.concatenate(vla_rows), np.concatenate(txt_rows)


def build_task_mean(feats, labels, valid_tasks, text_dict):
    """
    task_mean 策略: 每 task 的所有轨迹取均值
    返回 vla [T, D], txt [T, D_text]
    """
    idx_map = _build_label_index(labels, valid_tasks)
    vla = np.stack([feats[idx_map[t]].mean(axis=0) for t in valid_tasks if t in idx_map])
    txt = np.stack([text_dict[t] for t in valid_tasks if t in idx_map])
    return vla, txt


def build_single(feats, labels, valid_tasks, text_dict, rng):
    """
    single 策略: 每 task 随机取 1 条轨迹
    返回 vla [T, D], txt [T, D_text]
    """
    idx_map = _build_label_index(labels, valid_tasks)
    vla = np.stack([feats[rng.choice(idx_map[t])] for t in valid_tasks if t in idx_map])
    txt = np.stack([text_dict[t] for t in valid_tasks if t in idx_map])
    return vla, txt


# ====================================================================
# Text Embedding 加载 (通用)
# ====================================================================

def load_text_embeddings(task_rlds_path, egohod_emb_path, qwen_emb_path):
    """
    加载 EgoHOD / Qwen text embeddings，按 task_index 索引

    Args:
        task_rlds_path: task_rlds.jsonl 路径
        egohod_emb_path: EgoHOD embedding npz 路径
        qwen_emb_path: Qwen embedding npz 路径

    Returns:
        dict: {
            'egohod': ndarray [21938, 512],
            'qwen': ndarray [21938, 4096],
            'text_to_task_index': {text_lower: task_index},
        }
    """
    import json
    text_to_task_index = {}
    with open(task_rlds_path) as f:
        for line in f:
            d = json.loads(line)
            text_to_task_index[d['task'].lower()] = d['task_index']

    egohod_emb = np.load(egohod_emb_path)['embeddings']  # [21938, 512]
    qwen_emb = np.load(qwen_emb_path)['embeddings']      # [21938, 4096]

    return {
        'egohod': egohod_emb,
        'qwen': qwen_emb,
        'text_to_task_index': text_to_task_index,
    }


def build_task_text_dict(text_emb_data, task_labels):
    """
    从 task_labels (str) 构建 canonical_label → embedding 的查表字典

    适配不同来源的 task_labels:
      - 直接用 text_to_task_index 查 task_index → embedding
      - 返回有效的 task 集合

    Args:
        text_emb_data: load_text_embeddings() 的返回值
        task_labels: [N] str — 特征文件中的 task_labels

    Returns:
        dict: {
            'egohod': {task_label: emb [D]},
            'qwen': {task_label: emb [D]},
            'valid_tasks': sorted list of valid task labels
        }
    """
    t2i = text_emb_data['text_to_task_index']
    egohod = text_emb_data['egohod']
    qwen = text_emb_data['qwen']

    unique_labels = sorted(set(task_labels))
    result = {'egohod': {}, 'qwen': {}, 'valid_tasks': []}

    for label in unique_labels:
        idx = t2i.get(label.lower())
        if idx is not None and idx < len(egohod):
            result['egohod'][label] = egohod[idx]
            result['qwen'][label] = qwen[idx]
            result['valid_tasks'].append(label)

    return result
