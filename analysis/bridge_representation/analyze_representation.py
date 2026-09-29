"""
表征分析脚本: 从提取的特征计算多维度表征质量指标

指标体系:
  1. KNN Task Classification (k=5, 5-fold CV)
     - 表征能否区分不同task → 越高=结构越好
  2. Silhouette Score
     - 聚类分离度 [-1, 1] → 越高=类内紧凑+类间分离
  3. Intra/Inter class cosine similarity
     - 类内cos相似度 vs 类间cos相似度
     - Gap = intra - inter → 越大=类内紧凑且类间分离
  4. Calinski-Harabasz Index (Variance Ratio)
     - 越高=聚类越好
  5. Effective Rank (via SVD)
     - 特征矩阵的有效秩 → 越低=表征坍缩越严重
  6. t-SNE / UMAP 可视化
     - 定性对比三模型的聚类结构
  7. Per-layer 分析
     - 每个层单独计算上述指标，画曲线

用法:
    # 分析所有已提取的模型 (自动检测outputs/features/下的npz)
    python analyze_representation.py

    # 只分析指定模型
    python analyze_representation.py --models pretrained raw_ft cotrain_fg

    # 指定层索引
    python analyze_representation.py --layer_idx 10
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ========== Config ==========
sys.path.insert(0, "/root/data/xuyuan1/Codes/analysis")
from bridge_representation.config import (
    MODEL_CONFIGS, SELECTED_TASKS, HARD_GROUPS,
    LAYER_INDICES, FEATURE_DIR, FIGURE_DIR, OUTPUT_DIR,
    EGOHOD_EMB_PATH, QWEN_EMB_PATH, TASK_RLDS_PATH,
)


# =============================================================================
# Data Loading
# =============================================================================

def load_features(model_name: str, chunk_tag: str = "chunk4", feature_dir: str = None):
    """
    加载模型特征文件

    Args:
        model_name: 模型名 (pretrained, raw_ft, cotrain_fg)
        chunk_tag: "chunk1" (仅当前步 3 tokens) 或 "chunk4" (全部 4 步 12 tokens)
        feature_dir: 特征目录 (默认: FEATURE_DIR)

    Returns:
        features: [N, L, D] — N条轨迹, L层, D维
        task_labels: [N] — 每条轨迹的canonical task label (字符串)
        layer_indices: [L] — 层索引
    """
    if feature_dir is None:
        feature_dir = FEATURE_DIR

    # 优先查找 chunk 版本
    path = os.path.join(feature_dir, f"{model_name}_{chunk_tag}_features.npz")
    if not os.path.exists(path):
        path = os.path.join(feature_dir, f"{model_name}_{chunk_tag}_checkpoint.npz")
    if not os.path.exists(path):
        path = os.path.join(feature_dir, f"{model_name}_features.npz")
    if not os.path.exists(path):
        path = os.path.join(feature_dir, f"{model_name}_checkpoint.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No features found for {model_name} ({chunk_tag}) at {feature_dir}")

    data = np.load(path, allow_pickle=True)
    features = data['features']          # [N, L, D]
    task_labels = data['task_labels']     # [N] canonical task text strings
    layer_indices = data['layer_indices'] # [L]

    unique_tasks = np.unique(task_labels)
    print(f"  Loaded {model_name} ({chunk_tag}): features={features.shape}, "
          f"unique_tasks={len(unique_tasks)}, layers={list(layer_indices)}")
    return features, task_labels, layer_indices


def get_layer_features(features, layer_indices, target_layer):
    """提取指定层的特征 [N, D]"""
    layer_list = list(layer_indices)
    if target_layer not in layer_list:
        raise ValueError(f"Layer {target_layer} not in {layer_list}")
    idx = layer_list.index(target_layer)
    return features[:, idx, :]


# =============================================================================
# Metric 1: KNN Task Classification
# =============================================================================

def knn_classification(features, labels, k=5, n_splits=5):
    """
    KNN分类准确率 (5-fold cross-validation)

    Args:
        features: [N, D]
        labels: [N] — task indices as class labels
        k: number of neighbors
        n_splits: CV folds

    Returns:
        mean_accuracy, std_accuracy
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # 检查每个类别至少有n_splits个样本
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        print(f"    ⚠️ Some classes have <2 samples, using LOO")
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')

    accuracies = []
    for train_idx, test_idx in cv.split(features, y):
        knn.fit(features[train_idx], y[train_idx])
        acc = knn.score(features[test_idx], y[test_idx])
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


# =============================================================================
# Metric 2: Silhouette Score
# =============================================================================

def silhouette_analysis(features, labels):
    """
    Silhouette Score: [-1, 1], 越高越好
    使用cosine距离

    Returns:
        score: float
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return 0.0

    score = silhouette_score(features, y, metric='cosine')
    return score


# =============================================================================
# Metric 3: Intra/Inter class cosine similarity
# =============================================================================

def intra_inter_similarity(features, labels):
    """
    计算类内和类间的cosine相似度

    Returns:
        intra_mean: 类内平均cosine similarity
        inter_mean: 类间平均cosine similarity
        separation_ratio: intra / inter (越大=类内更紧凑相对于类间)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Normalize for cosine
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    features_norm = features / norms

    cos_sim = features_norm @ features_norm.T  # [N, N]

    unique_labels = np.unique(labels)
    intra_sims = []
    inter_sims = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            sim = cos_sim[i, j]
            if labels[i] == labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    intra_mean = np.mean(intra_sims) if intra_sims else 0.0
    inter_mean = np.mean(inter_sims) if inter_sims else 0.0

    # 避免除零
    separation = intra_mean / max(inter_mean, 1e-8)

    return intra_mean, inter_mean, separation


def intra_inter_similarity_fast(features, labels):
    """
    快速版本: 使用矩阵运算避免O(N^2)循环

    Returns:
        intra_mean, inter_mean, gap (intra - inter)
        gap越大 = 类内紧凑且类间分离
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    features_norm = features / norms

    cos_sim = features_norm @ features_norm.T  # [N, N]

    # 构建mask
    label_eq = labels[:, None] == labels[None, :]  # [N, N]
    upper_tri = np.triu(np.ones_like(label_eq, dtype=bool), k=1)

    intra_mask = label_eq & upper_tri
    inter_mask = (~label_eq) & upper_tri

    intra_mean = cos_sim[intra_mask].mean() if intra_mask.any() else 0.0
    inter_mean = cos_sim[inter_mask].mean() if inter_mask.any() else 0.0
    gap = intra_mean - inter_mean  # 越大越好

    return float(intra_mean), float(inter_mean), float(gap)


# =============================================================================
# Metric 4: Calinski-Harabasz Index
# =============================================================================

def calinski_harabasz(features, labels):
    """
    Calinski-Harabasz Index (Variance Ratio Criterion)
    越高 = 聚类越好
    """
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    if len(np.unique(y)) < 2:
        return 0.0

    return calinski_harabasz_score(features, y)


# =============================================================================
# Metric 5: Effective Rank
# =============================================================================

def effective_rank(features):
    """
    特征矩阵的Effective Rank (基于SVD奇异值的信息熵)

    ER = exp(-sum(p_i * log(p_i)))
    其中 p_i = sigma_i / sum(sigma_j)

    越低 = 表征越坍缩（所有特征挤在少数方向）
    越高 = 表征越丰富多样
    """
    # Center features
    features_centered = features - features.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)

    # Normalize singular values to probabilities
    S_pos = S[S > 1e-10]
    p = S_pos / S_pos.sum()

    # Shannon entropy
    entropy = -np.sum(p * np.log(p))
    eff_rank = np.exp(entropy)

    return eff_rank


# =============================================================================
# Metric 6: CKA (Centered Kernel Alignment) with Text Embeddings
# =============================================================================

def _centering_matrix(n):
    """H = I - 1/n * 11^T"""
    return np.eye(n) - np.ones((n, n)) / n

def linear_cka(X, Y):
    """
    Linear CKA between two representation matrices.

    X: [N, D1]  (e.g. VLA action features)
    Y: [N, D2]  (e.g. text embeddings)

    Returns: CKA similarity in [0, 1]
    """
    n = X.shape[0]
    assert Y.shape[0] == n, f"Shape mismatch: X={X.shape}, Y={Y.shape}"

    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Linear kernel: K = XX^T, L = YY^T
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    XTX = X.T @ X  # [D1, D1]
    YTY = Y.T @ Y  # [D2, D2]
    YTX = Y.T @ X  # [D2, D1]

    numerator = np.linalg.norm(YTX, 'fro') ** 2
    denominator = np.linalg.norm(XTX, 'fro') * np.linalg.norm(YTY, 'fro')

    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


def load_text_embeddings():
    """
    加载 EgoHOD 和 Qwen 的 task-level text embeddings。

    embeddings 按 task_index 索引 (egohod_emb[task_index] → embedding)。
    通过 task_rlds.jsonl 建立 text → task_index 映射，
    再通过 SELECTED_TASKS 建立 canonical_label → task_index。

    Returns:
        dict: {
            'egohod': {canonical_label: np.array([D])},
            'qwen': {canonical_label: np.array([D])},
        }
    """
    import json

    # 加载 text → task_index 映射
    text_to_task_index = {}
    try:
        with open(TASK_RLDS_PATH) as f:
            for line in f:
                d = json.loads(line)
                text_to_task_index[d['task'].lower()] = d['task_index']
    except FileNotFoundError:
        print("  ⚠️ task_rlds.jsonl not found, skipping CKA")
        return None

    # 加载 embeddings
    try:
        egohod_emb = np.load(EGOHOD_EMB_PATH)['embeddings']  # [21938, 512]
        qwen_emb = np.load(QWEN_EMB_PATH)['embeddings']      # [21938, 4096]
    except FileNotFoundError as e:
        print(f"  ⚠️ Embedding file not found: {e}, skipping CKA")
        return None

    # 构建 canonical_label → embedding
    result = {'egohod': {}, 'qwen': {}}
    for t in SELECTED_TASKS:
        text_lower = t['text'].lower()
        task_idx = text_to_task_index.get(text_lower, t.get('task_index'))
        if task_idx is not None and task_idx < len(egohod_emb):
            result['egohod'][t['text']] = egohod_emb[task_idx]
            result['qwen'][t['text']] = qwen_emb[task_idx]

    print(f"  Loaded text embeddings: EgoHOD={len(result['egohod'])} tasks, "
          f"Qwen={len(result['qwen'])} tasks")
    return result


def compute_cka_with_text(features, task_labels, layer_indices, text_embeddings):
    """
    计算 VLA 每层特征与 text embeddings 的 CKA。

    采用 task-level 表征:
      - VLA: 每个 task 的所有轨迹 mean pool → [N_tasks, D_vla]
      - Text: 每个 task 一个 embedding → [N_tasks, D_text]

    Args:
        features: [N, L, D]
        task_labels: [N] canonical labels
        layer_indices: [L]
        text_embeddings: dict from load_text_embeddings()

    Returns:
        dict: {
            'egohod_cka': [per-layer CKA values],
            'qwen_cka': [per-layer CKA values],
            'layer_indices': list,
        }
    """
    unique_labels = sorted(set(task_labels))

    # 构建 task-level text embedding 矩阵
    egohod_list, qwen_list = [], []
    valid_labels = []
    for label in unique_labels:
        if label in text_embeddings['egohod'] and label in text_embeddings['qwen']:
            egohod_list.append(text_embeddings['egohod'][label])
            qwen_list.append(text_embeddings['qwen'][label])
            valid_labels.append(label)

    if len(valid_labels) < 3:
        print("  ⚠️ Too few matching tasks for CKA")
        return None

    text_egohod = np.stack(egohod_list)  # [N_tasks, 512]
    text_qwen = np.stack(qwen_list)      # [N_tasks, 4096]

    result = {'egohod_cka': [], 'qwen_cka': [], 'layer_indices': list(layer_indices)}

    for i, layer_idx in enumerate(layer_indices):
        # 构建 VLA task-level features: mean pool per task
        vla_task_feats = []
        for label in valid_labels:
            mask = task_labels == label
            vla_task_feats.append(features[mask, i, :].mean(axis=0))
        vla_mat = np.stack(vla_task_feats)  # [N_tasks, D_vla]

        # CKA
        cka_ego = linear_cka(vla_mat, text_egohod)
        cka_qwen = linear_cka(vla_mat, text_qwen)
        result['egohod_cka'].append(cka_ego)
        result['qwen_cka'].append(cka_qwen)

        print(f"    Layer {layer_idx:>2}: CKA(EgoHOD)={cka_ego:.4f}, CKA(Qwen)={cka_qwen:.4f}")

    return result


def plot_cka_comparison(all_cka_results, model_names, save_dir=None):
    """
    绘制 per-layer CKA 对比图 (VLA vs EgoHOD / Qwen)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for ax, text_key, text_name in zip(axes, ['egohod_cka', 'qwen_cka'], ['EgoHOD', 'Qwen']):
        for i, (result, name) in enumerate(zip(all_cka_results, model_names)):
            if result is None:
                continue
            layers = result['layer_indices']
            values = result[text_key]
            ax.plot(layers, values, marker=markers[i % len(markers)],
                   color=colors[i % len(colors)], label=name, linewidth=2, markersize=6)

        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel('Linear CKA', fontsize=11)
        ax.set_title(f'CKA with {text_name} Text Embeddings', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "cka_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved CKA plot: {path}")

    plt.close(fig)


# =============================================================================
# Metric 7: t-SNE / UMAP Visualization
# =============================================================================

def tsne_visualization(all_model_features, all_model_labels, all_model_names,
                       layer_idx, save_dir=None, perplexity=30):
    """
    为多个模型生成t-SNE可视化对比图

    Args:
        all_model_features: list of [N, D] arrays
        all_model_labels: list of [N] arrays (task indices)
        all_model_names: list of str
        layer_idx: which layer
        save_dir: output directory
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n_models = len(all_model_features)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    # 统一颜色映射
    all_labels_combined = np.concatenate(all_model_labels)
    unique_labels = np.unique(all_labels_combined)
    n_classes = len(unique_labels)

    # 使用tab20 + 额外颜色
    if n_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    label_to_color = {l: cmap(i / max(n_classes - 1, 1)) for i, l in enumerate(unique_labels)}
    # labels已经是文本字符串，截断显示
    label_to_short = {l: (l[:20] if isinstance(l, str) else str(l)[:20]) for l in unique_labels}

    for ax, feats, labels, name in zip(axes, all_model_features, all_model_labels, all_model_names):
        # t-SNE
        n_samples = len(feats)
        actual_perp = min(perplexity, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=actual_perp, random_state=42,
                     max_iter=1000, metric='cosine')
        embedded = tsne.fit_transform(feats)

        for label in unique_labels:
            mask = labels == label
            if not mask.any():
                continue
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                      c=[label_to_color[label]], s=15, alpha=0.7,
                      label=label_to_short[label])

        ax.set_title(f"{name} (Layer {layer_idx})", fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # 统一legend（只在最后一个subplot）
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=7, ncol=1, markerscale=1.5)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"tsne_layer{layer_idx}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved t-SNE: {path}")

    plt.close(fig)


# =============================================================================
# Per-Layer Analysis
# =============================================================================

def compute_all_metrics_per_layer(features, labels, layer_indices):
    """
    对每一层分别计算所有指标

    Returns:
        results: dict of {metric_name: [value_per_layer]}
    """
    results = {
        'knn_acc': [],
        'knn_std': [],
        'silhouette': [],
        'intra_sim': [],
        'inter_sim': [],
        'gap': [],
        'calinski_harabasz': [],
        'effective_rank': [],
        'layer_indices': list(layer_indices),
    }

    for i, layer_idx in enumerate(layer_indices):
        feats = features[:, i, :]  # [N, D]

        # KNN
        knn_acc, knn_std = knn_classification(feats, labels, k=5, n_splits=5)
        results['knn_acc'].append(knn_acc)
        results['knn_std'].append(knn_std)

        # Silhouette
        sil = silhouette_analysis(feats, labels)
        results['silhouette'].append(sil)

        # Intra/Inter similarity
        intra, inter, gap = intra_inter_similarity_fast(feats, labels)
        results['intra_sim'].append(intra)
        results['inter_sim'].append(inter)
        results['gap'].append(gap)

        # Calinski-Harabasz
        ch = calinski_harabasz(feats, labels)
        results['calinski_harabasz'].append(ch)

        # Effective Rank
        er = effective_rank(feats)
        results['effective_rank'].append(er)

        print(f"    Layer {layer_idx:>2}: KNN={knn_acc:.3f}±{knn_std:.3f}, "
              f"Sil={sil:.3f}, Intra={intra:.3f}, Inter={inter:.3f}, "
              f"Gap={gap:.3f}, CH={ch:.1f}, ER={er:.1f}")

    return results


def plot_per_layer_comparison(all_results, model_names, save_dir=None):
    """
    绘制per-layer指标对比曲线

    Args:
        all_results: list of per-layer results dicts
        model_names: list of model names
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ('knn_acc', 'KNN Accuracy', True),
        ('silhouette', 'Silhouette Score', True),
        ('gap', 'Intra-Inter Gap', True),
        ('calinski_harabasz', 'Calinski-Harabasz', True),
        ('effective_rank', 'Effective Rank', True),
    ]

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for ax, (metric_key, metric_name, higher_better) in zip(axes, metrics_to_plot):
        for i, (result, name) in enumerate(zip(all_results, model_names)):
            layers = result['layer_indices']
            values = result[metric_key]

            ax.plot(layers, values, marker=markers[i % len(markers)],
                   color=colors[i % len(colors)], label=name, linewidth=2,
                   markersize=6)

            if metric_key == 'knn_acc' and 'knn_std' in result:
                std = result['knn_std']
                values_arr = np.array(values)
                std_arr = np.array(std)
                ax.fill_between(layers, values_arr - std_arr, values_arr + std_arr,
                              alpha=0.15, color=colors[i % len(colors)])

        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "per_layer_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved per-layer plot: {path}")

    plt.close(fig)


def plot_intra_inter_comparison(all_results, model_names, save_dir=None):
    """
    绘制 intra vs inter similarity 对比图 (per layer)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, result, name in zip(axes, all_results, model_names):
        layers = result['layer_indices']
        ax.plot(layers, result['intra_sim'], 'o-', label='Intra-class', color='#2ca02c', linewidth=2)
        ax.plot(layers, result['inter_sim'], 's-', label='Inter-class', color='#d62728', linewidth=2)
        ax.fill_between(layers, result['intra_sim'], result['inter_sim'],
                        alpha=0.15, color='gray')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "intra_inter_similarity.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved intra/inter plot: {path}")

    plt.close(fig)


# =============================================================================
# Summary Report
# =============================================================================

def generate_summary(all_results, model_names, key_layer=10, output_dir=None):
    """
    生成关键层的汇总表格

    Args:
        key_layer: 重点关注的层 (通常是alignment层)
    """
    print(f"\n{'='*80}")
    print(f"SUMMARY — Key Layer {key_layer}")
    print(f"{'='*80}")

    # 找到key_layer对应的index
    layer_list = all_results[0]['layer_indices']
    if key_layer in layer_list:
        li = layer_list.index(key_layer)
    else:
        print(f"  ⚠️ Layer {key_layer} not found, using last layer")
        li = -1

    header = f"{'Model':<15} {'KNN↑':>10} {'Silh↑':>10} {'Intra':>10} {'Inter':>10} {'Gap↑':>10} {'CH↑':>10} {'ER↑':>10}"
    print(header)
    print("-" * len(header))

    for result, name in zip(all_results, model_names):
        row = (f"{name:<15} "
               f"{result['knn_acc'][li]:.3f}±{result['knn_std'][li]:.3f} "
               f"{result['silhouette'][li]:>10.3f} "
               f"{result['intra_sim'][li]:>10.3f} "
               f"{result['inter_sim'][li]:>10.3f} "
               f"{result['gap'][li]:>10.3f} "
               f"{result['calinski_harabasz'][li]:>10.1f} "
               f"{result['effective_rank'][li]:>10.1f}")
        print(row)

    print(f"{'='*80}")

    # 保存JSON
    summary = {}
    for result, name in zip(all_results, model_names):
        summary[name] = {
            'knn_acc': float(result['knn_acc'][li]),
            'knn_std': float(result['knn_std'][li]),
            'silhouette': float(result['silhouette'][li]),
            'intra_sim': float(result['intra_sim'][li]),
            'inter_sim': float(result['inter_sim'][li]),
            'gap': float(result['gap'][li]),
            'calinski_harabasz': float(result['calinski_harabasz'][li]),
            'effective_rank': float(result['effective_rank'][li]),
        }

    if output_dir is None:
        output_dir = OUTPUT_DIR
    json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {json_path}")

    return summary


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(all_results, model_names, display_names, all_layer_indices,
                    chunk_tag, key_layer=10, save_dir=None):
    """
    生成完整的文本分析报告

    Args:
        all_results: list of per-layer results dicts
        model_names: list of internal model names
        display_names: list of display labels
        all_layer_indices: layer indices array
        chunk_tag: "chunk1" or "chunk4"
        key_layer: 重点层
        save_dir: 输出目录
    """
    from datetime import datetime

    lines = []
    lines.append("=" * 80)
    lines.append(f"Bridge Representation Quality Analysis Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Chunk mode: {chunk_tag}")
    lines.append(f"Models: {', '.join(display_names)}")
    lines.append(f"Layers: {list(all_layer_indices)}")
    lines.append("=" * 80)

    # ---- Per-model per-layer details ----
    lines.append("")
    lines.append("=" * 80)
    lines.append("PER-LAYER METRICS")
    lines.append("=" * 80)

    for result, dname in zip(all_results, display_names):
        lines.append(f"\n  [{dname}]")
        layers = result['layer_indices']
        header = f"  {'Layer':>6} {'KNN':>12} {'Silh':>8} {'Intra':>8} {'Inter':>8} {'Gap':>8} {'CH':>10} {'ER':>8}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for i, layer in enumerate(layers):
            marker = " ***" if layer == key_layer else ""
            lines.append(
                f"  {layer:>6} "
                f"{result['knn_acc'][i]:.3f}±{result['knn_std'][i]:.3f} "
                f"{result['silhouette'][i]:>8.3f} "
                f"{result['intra_sim'][i]:>8.3f} "
                f"{result['inter_sim'][i]:>8.3f} "
                f"{result['gap'][i]:>8.3f} "
                f"{result['calinski_harabasz'][i]:>10.1f} "
                f"{result['effective_rank'][i]:>8.1f}"
                f"{marker}"
            )

    # ---- Key layer summary table ----
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"SUMMARY — Key Layer {key_layer}")
    lines.append("=" * 80)

    layer_list = all_results[0]['layer_indices']
    li = layer_list.index(key_layer) if key_layer in layer_list else -1

    header = f"{'Model':<20} {'KNN↑':>12} {'Silh↑':>8} {'Intra':>8} {'Inter':>8} {'Gap↑':>8} {'CH↑':>10} {'ER↑':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for result, dname in zip(all_results, display_names):
        lines.append(
            f"{dname:<20} "
            f"{result['knn_acc'][li]:.3f}±{result['knn_std'][li]:.3f} "
            f"{result['silhouette'][li]:>8.3f} "
            f"{result['intra_sim'][li]:>8.3f} "
            f"{result['inter_sim'][li]:>8.3f} "
            f"{result['gap'][li]:>8.3f} "
            f"{result['calinski_harabasz'][li]:>10.1f} "
            f"{result['effective_rank'][li]:>8.1f}"
        )

    # ---- Cross-model comparison (if >1 model) ----
    if len(all_results) > 1:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"CROSS-MODEL COMPARISON (Layer {key_layer})")
        lines.append("=" * 80)

        ref_name = display_names[0]
        ref = all_results[0]
        for result, dname in zip(all_results[1:], display_names[1:]):
            lines.append(f"\n  {dname} vs {ref_name}:")
            for metric in ['knn_acc', 'silhouette', 'gap', 'calinski_harabasz', 'effective_rank']:
                v_ref = result[metric][li] if isinstance(result[metric], list) else result[metric]
                v_cur = ref[metric][li] if isinstance(ref[metric], list) else ref[metric]
                if isinstance(v_ref, (int, float)) and isinstance(v_cur, (int, float)):
                    delta = float(result[metric][li]) - float(ref[metric][li])
                    pct = delta / max(abs(float(ref[metric][li])), 1e-8) * 100
                    arrow = "↑" if delta > 0 else "↓"
                    lines.append(f"    {metric:<20}: {delta:+.4f} ({pct:+.1f}%) {arrow}")

    # ---- Metric definitions ----
    lines.append("")
    lines.append("=" * 80)
    lines.append("METRIC DEFINITIONS")
    lines.append("=" * 80)
    lines.append("  KNN  : K-Nearest Neighbor task classification accuracy (k=5, 5-fold CV)")
    lines.append("  Silh : Silhouette Score [-1,1] — cluster separation quality")
    lines.append("  Intra: Intra-class cosine similarity — within-task coherence")
    lines.append("  Inter: Inter-class cosine similarity — between-task similarity")
    lines.append("  Gap  : Intra - Inter — larger = better task discrimination")
    lines.append("  CH   : Calinski-Harabasz Index — variance ratio criterion")
    lines.append("  ER   : Effective Rank — representation diversity (higher = less collapse)")
    lines.append("  CKA  : Centered Kernel Alignment — structural similarity with text embeddings")
    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, f"analysis_report_{chunk_tag}.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"  Saved report: {report_path}")

    return report_text


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="模型名 (默认: 自动检测已提取的)")
    parser.add_argument("--key_layer", type=int, default=10,
                        help="重点展示的层 (默认: 10)")
    parser.add_argument("--tsne_layers", nargs="+", type=int, default=[10, 20, 26],
                        help="生成t-SNE的层 (默认: 10 20 26)")
    parser.add_argument("--skip_tsne", action="store_true",
                        help="跳过t-SNE (较慢)")
    parser.add_argument("--chunk", type=str, default="chunk4",
                        choices=["chunk1", "chunk4"],
                        help="chunk1 (当前步) 或 chunk4 (全部4步)")
    # 可选: 自定义目录 (用于 val 集分析)
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="特征目录 (默认: train features)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="JSON 输出目录 (默认: train outputs)")
    parser.add_argument("--figure_dir", type=str, default=None,
                        help="图表输出目录 (默认: train figures)")
    # 可选: 自定义 hard groups JSON
    parser.add_argument("--hard_groups_json", type=str, default=None,
                        help="自定义 hard groups JSON 文件")
    args = parser.parse_args()

    # 目录: 优先用命令行参数, 否则用默认值
    feat_dir = args.feature_dir or FEATURE_DIR
    out_dir = args.output_dir or OUTPUT_DIR
    fig_dir = args.figure_dir or FIGURE_DIR

    # 确定要分析的模型
    if args.models:
        model_names = args.models
    else:
        # 自动检测
        model_names = []
        for name in MODEL_CONFIGS.keys():
            for pattern in [f"{name}_{args.chunk}_features.npz",
                           f"{name}_{args.chunk}_checkpoint.npz",
                           f"{name}_features.npz",
                           f"{name}_checkpoint.npz"]:
                if os.path.exists(os.path.join(feat_dir, pattern)):
                    model_names.append(name)
                    break

    if not model_names:
        print(f"No feature files found in {feat_dir}. Run extract_features.py first.")
        return

    print(f"Analyzing models: {model_names}")
    print(f"Chunk mode: {args.chunk}")
    print(f"Key layer: {args.key_layer}")
    print(f"t-SNE layers: {args.tsne_layers}")
    print(f"Feature dir: {feat_dir}")
    print(f"Output dir: {out_dir}")

    # 加载所有模型特征 (使用指定的 feature_dir)
    all_features = {}
    all_labels = {}
    all_layer_indices = None

    for name in model_names:
        features, task_labels, layer_indices = load_features(
            name, chunk_tag=args.chunk, feature_dir=feat_dir)
        all_features[name] = features
        all_labels[name] = task_labels  # canonical text strings
        if all_layer_indices is None:
            all_layer_indices = layer_indices

    # Per-layer 分析
    print(f"\n{'='*60}")
    print("Per-layer metric computation")
    print(f"{'='*60}")

    all_results = []
    display_names = []
    for name in model_names:
        label = MODEL_CONFIGS[name]['label']
        print(f"\n  [{label}]")
        result = compute_all_metrics_per_layer(
            all_features[name], all_labels[name], all_layer_indices
        )
        all_results.append(result)
        display_names.append(label)

    # 汇总表
    generate_summary(all_results, display_names, key_layer=args.key_layer,
                     output_dir=out_dir)

    # Per-layer 对比图
    print(f"\n  Generating per-layer comparison plots...")
    plot_per_layer_comparison(all_results, display_names, save_dir=fig_dir)
    plot_intra_inter_comparison(all_results, display_names, save_dir=fig_dir)

    # ===== CKA with Text Embeddings =====
    print(f"\n{'='*60}")
    print("CKA with Text Embeddings (EgoHOD / Qwen)")
    print(f"{'='*60}")

    text_emb = load_text_embeddings()
    all_cka_results = []
    if text_emb is not None:
        for name in model_names:
            label = MODEL_CONFIGS[name]['label']
            print(f"\n  [{label}]")
            cka_result = compute_cka_with_text(
                all_features[name], all_labels[name], all_layer_indices, text_emb
            )
            all_cka_results.append(cka_result)

        # CKA 对比图
        print(f"\n  Generating CKA comparison plot...")
        plot_cka_comparison(all_cka_results, display_names, save_dir=fig_dir)

        # 保存 CKA 结果
        cka_save = {}
        for name, cka_r in zip(model_names, all_cka_results):
            if cka_r is not None:
                cka_save[name] = {k: [float(v) for v in vals] if isinstance(vals, list) else vals
                                  for k, vals in cka_r.items()}
        cka_json_path = os.path.join(out_dir, f"cka_metrics_{args.chunk}.json")
        with open(cka_json_path, 'w') as f:
            json.dump(cka_save, f, indent=2)
        print(f"  Saved CKA results: {cka_json_path}")

    # t-SNE
    if not args.skip_tsne:
        for layer_idx in args.tsne_layers:
            if layer_idx not in list(all_layer_indices):
                print(f"  ⚠️ Layer {layer_idx} not available, skipping t-SNE")
                continue

            print(f"\n  Generating t-SNE for layer {layer_idx}...")
            tsne_feats = []
            tsne_labels = []
            for name in model_names:
                feats = get_layer_features(all_features[name], all_layer_indices, layer_idx)
                tsne_feats.append(feats)
                tsne_labels.append(all_labels[name])

            # 使用命令行传入的 figure_dir，避免写死到默认 FIGURE_DIR
            tsne_visualization(tsne_feats, tsne_labels, display_names,
                             layer_idx, save_dir=fig_dir)

    # ===== Hard Mode 分析 =====
    # 支持自定义 hard groups (如 val 集的 hard groups)
    hard_groups_to_use = HARD_GROUPS
    if args.hard_groups_json:
        import json as _json
        with open(args.hard_groups_json) as _f:
            hard_groups_to_use = _json.load(_f)
        print(f"  Using custom hard groups from: {args.hard_groups_json}")

    if hard_groups_to_use:
        print(f"\n{'='*60}")
        print("HARD MODE ANALYSIS (语义相近任务组)")
        print(f"{'='*60}")

        all_labels_set = set()
        for name in model_names:
            all_labels_set.update(all_labels[name])

        li = list(all_layer_indices).index(args.key_layer) if args.key_layer in list(all_layer_indices) else -1

        hard_results = {}  # group -> {model_name -> metrics}
        for group_name, group_tasks in hard_groups_to_use.items():
            available = [t for t in group_tasks if t in all_labels_set]
            if len(available) < 2:
                continue

            print(f"\n  [{group_name}] ({len(available)} tasks)")

            hard_results[group_name] = {'tasks': available, 'models': {}}

            for name in model_names:
                label = MODEL_CONFIGS[name]['label']
                feats = all_features[name]
                labels = all_labels[name]
                mask = np.isin(labels, available)
                group_feats = feats[mask, li, :]
                group_labels = labels[mask]

                if len(np.unique(group_labels)) < 2:
                    continue

                knn_acc, knn_std = knn_classification(group_feats, group_labels, k=5, n_splits=5)
                sil = silhouette_analysis(group_feats, group_labels)
                intra, inter, gap = intra_inter_similarity_fast(group_feats, group_labels)
                ch = calinski_harabasz(group_feats, group_labels)
                er = effective_rank(group_feats)

                hard_results[group_name]['models'][label] = {
                    'knn_acc': float(knn_acc), 'knn_std': float(knn_std),
                    'silhouette': float(sil), 'intra_sim': float(intra),
                    'inter_sim': float(inter), 'gap': float(gap),
                    'calinski_harabasz': float(ch), 'effective_rank': float(er),
                    'n_samples': int(mask.sum()),
                }

                print(f"    {label:<15} KNN={knn_acc:.3f} Sil={sil:.3f} "
                      f"Gap={gap:.3f} CH={ch:.1f} N={mask.sum()}")

        # Hard mode t-SNE: 对每个组生成一个
        if not args.skip_tsne and hard_results:
            hard_fig_dir = os.path.join(fig_dir, "hard_mode")
            os.makedirs(hard_fig_dir, exist_ok=True)

            for group_name, gdata in hard_results.items():
                available = gdata['tasks']
                if len(available) < 2:
                    continue

                print(f"\n  Hard t-SNE: {group_name} (layer {args.key_layer})...")
                tsne_feats, tsne_labels = [], []
                skip_tsne_group = False
                for name in model_names:
                    labels = all_labels[name]
                    mask = np.isin(labels, available)
                    if mask.sum() == 0:
                        print(f"    ⚠️ {name} has 0 samples for {group_name}, skipping t-SNE")
                        skip_tsne_group = True
                        break
                    feats = get_layer_features(all_features[name], all_layer_indices, args.key_layer)
                    tsne_feats.append(feats[mask])
                    tsne_labels.append(labels[mask])

                if skip_tsne_group or not tsne_feats:
                    continue

                min_n = min(len(f) for f in tsne_feats)
                if min_n < 3:
                    print(f"    ⚠️ Too few samples ({min_n}) for t-SNE, skipping")
                    continue

                tsne_visualization(tsne_feats, tsne_labels, display_names,
                                 args.key_layer, save_dir=hard_fig_dir,
                                 perplexity=min(30, min_n - 1))

                # 重命名文件加上组名
                old_path = os.path.join(hard_fig_dir, f"tsne_layer{args.key_layer}.png")
                new_path = os.path.join(hard_fig_dir, f"tsne_{group_name}_layer{args.key_layer}.png")
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)

        # 保存 hard mode 结果
        hard_json_path = os.path.join(out_dir, f"hard_mode_metrics_{args.chunk}.json")
        with open(hard_json_path, 'w') as f:
            json.dump(hard_results, f, indent=2)
        print(f"\n  Saved hard mode results: {hard_json_path}")

    # ===== 保存完整per-layer结果 (转换numpy类型为Python原生类型) =====
    def _to_native(v):
        if isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        return v

    full_results = {}
    for name, result in zip(model_names, all_results):
        full_results[name] = {
            k: [_to_native(v) for v in vals] if isinstance(vals, list) else _to_native(vals)
            for k, vals in result.items()
        }
    json_path = os.path.join(out_dir, f"per_layer_metrics_{args.chunk}.json")
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Saved full results: {json_path}")

    # 生成文本报告
    report = generate_report(
        all_results, model_names, display_names, all_layer_indices,
        chunk_tag=args.chunk, key_layer=args.key_layer, save_dir=out_dir,
    )
    print(report)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"  Figures: {fig_dir}")
    print(f"  Metrics: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
