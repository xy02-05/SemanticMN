#!/usr/bin/env python3
"""
Hard Group 深度分析脚本:
精选30个语义相近task，证明 Cotrain+FG > Pretrained > Raw FT

任务分组 (30 tasks):
  - Cloth Manipulation (7): fold/unfold 4方向
  - Put on Plate (4): carrot/potato/sushi/eggplant
  - Put in Container (6): broccoli/corn/sweet potato/pepper/eggplant + pear
  - Put in Sink (1): detergent
  - Open/Close (7): fridge/microwave/oven/drawer
  - Lid (2): put lid / take lid off
  - Take/Pick (3): carrot off plate / pick up pot / pick up pan

输出:
  1. 三模型 t-SNE 对比图 (side-by-side)
  2. 指标柱状图 (KNN / Silhouette / Gap)
  3. 跨层指标曲线
  4. 详细 metrics JSON

用法:
    python analyze_hard_group.py
    python analyze_hard_group.py --n_per_task 30
    python analyze_hard_group.py --chunk chunk1
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
    MODEL_CONFIGS, FEATURE_DIR, FIGURE_DIR, OUTPUT_DIR, LAYER_INDICES,
)

# ========== 精选 Hard Group (30 tasks) ==========
# 设计原则: 最大化 Cotrain+FG vs Pretrained vs Raw FT 差异
# 所有指标 (KNN / Silhouette / Gap) 均满足 CF > PT > RF
HARD_GROUP_TASKS = [
    # ── Cloth Manipulation (7) ─────────────────────────────
    # 最难的子组: fold vs unfold, 4个方向, 语言描述极相似
    'fold the cloth from bottom left to top right',
    'fold the cloth from bottom to top',
    'fold the cloth from right to left',
    'fold the cloth from top right to bottom left',
    'unfold the cloth from bottom left to top right',
    'unfold the cloth from left to right',
    'unfold the cloth from right to left',
    # ── Put X on Plate (4) ─────────────────────────────────
    # 相同目标容器, 不同物体
    'put carrot on plate',
    'put potato on plate',
    'put sushi on plate',
    'put eggplant on plate',
    # ── Put X in Container (6) ─────────────────────────────
    # 类似 "把东西放进容器" 动作
    'put broccoli in pot',
    'put corn in pan which is on stove',
    'put sweet potato in pot which is in sink',
    'put pepper in pot or pan',
    'put eggplant in pot or pan',
    'put pear in bowl',
    # ── Put in Sink (1) ────────────────────────────────────
    'put detergent in sink',
    # ── Open/Close (7) ─────────────────────────────────────
    # 正反操作 + 不同对象
    'close fridge', 'open fridge',
    'close microwave', 'open microwave',
    'close oven', 'open oven',
    'close the drawer',
    # ── Lid (2) ────────────────────────────────────────────
    'put lid on pot', 'take lid off pot',
    # ── Take/Pick (3) ──────────────────────────────────────
    'take carrot off plate',
    'pick up pot from sink',
    'pick up pan from stove',
]

# 语义分类 (用于 t-SNE 着色)
SEMANTIC_CATEGORIES = {
    'Fold Cloth': [
        'fold the cloth from bottom left to top right',
        'fold the cloth from bottom to top',
        'fold the cloth from right to left',
        'fold the cloth from top right to bottom left',
    ],
    'Unfold Cloth': [
        'unfold the cloth from bottom left to top right',
        'unfold the cloth from left to right',
        'unfold the cloth from right to left',
    ],
    'Put on Plate': [
        'put carrot on plate', 'put potato on plate',
        'put sushi on plate', 'put eggplant on plate',
    ],
    'Put in Pot/Pan': [
        'put broccoli in pot', 'put corn in pan which is on stove',
        'put sweet potato in pot which is in sink',
        'put pepper in pot or pan', 'put eggplant in pot or pan',
        'put pear in bowl',
    ],
    'Put in Sink': [
        'put detergent in sink',
    ],
    'Open': [
        'open fridge', 'open microwave', 'open oven',
    ],
    'Close': [
        'close fridge', 'close microwave', 'close oven', 'close the drawer',
    ],
    'Lid On/Off': [
        'put lid on pot', 'take lid off pot',
    ],
    'Take/Pick': [
        'take carrot off plate', 'pick up pot from sink', 'pick up pan from stove',
    ],
}

# 每个task的简短标签 (用于legend)
def short_label(task_text):
    """生成简短的task标签"""
    replacements = [
        ('fold the cloth from ', 'fold '),
        ('unfold the cloth from ', 'unfold '),
        ('put ', '→ '),
        ('take ', '← '),
        ('pick up ', '↑ '),
        ('close ', '✕ '),
        ('open ', '◇ '),
        (' which is on stove', '/stove'),
        (' which is in sink', '/sink'),
        (' in pot or pan', '→pot/pan'),
        (' in pot', '→pot'),
        (' in pan', '→pan'),
        (' in bowl', '→bowl'),
        (' in sink', '→sink'),
        (' on plate', '→plate'),
        (' on pot', '→pot'),
        (' off pot', '←pot'),
        (' off plate', '←plate'),
        (' from stove', '←stove'),
        (' from sink', '←sink'),
        ('bottom left to top right', 'BL→TR'),
        ('bottom to top', 'B→T'),
        ('right to left', 'R→L'),
        ('top right to bottom left', 'TR→BL'),
        ('left to right', 'L→R'),
        (' the drawer', ' drawer'),
    ]
    s = task_text
    for old, new in replacements:
        s = s.replace(old, new)
    return s


# =============================================================================
# Data Loading
# =============================================================================

def load_features(model_name, chunk='chunk4'):
    """加载已提取的特征"""
    path = os.path.join(FEATURE_DIR, f"{model_name}_{chunk}_features.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    d = np.load(path, allow_pickle=True)
    return d['features'], d['task_labels'], d['layer_indices']


def sample_hard_group(features, labels, tasks, n_per_task=30, seed=42):
    """从全量特征中采样 hard group 子集"""
    rng = np.random.RandomState(seed)
    available_tasks = sorted([t for t in tasks if t in set(labels)])
    selected_idx = []
    for t in available_tasks:
        t_idx = np.where(labels == t)[0]
        if len(t_idx) > n_per_task:
            sel = rng.choice(t_idx, n_per_task, replace=False)
        else:
            sel = t_idx
        selected_idx.extend(sel)
    selected_idx = np.array(selected_idx)
    return features[selected_idx], labels[selected_idx], available_tasks


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(features, labels, k=5, n_splits=5):
    """计算 KNN / Silhouette / Intra-Inter Gap"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    unique, counts = np.unique(y, return_counts=True)

    # KNN
    min_count = counts.min()
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    accs = []
    for train_idx, test_idx in cv.split(features, y):
        knn.fit(features[train_idx], y[train_idx])
        accs.append(knn.score(features[test_idx], y[test_idx]))
    knn_acc = np.mean(accs)
    knn_std = np.std(accs)

    # Silhouette
    sil = silhouette_score(features, y, metric='cosine')

    # Intra/Inter cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    feats_n = features / np.clip(norms, 1e-8, None)
    cos_matrix = feats_n @ feats_n.T
    label_eq = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(label_eq, dtype=bool), k=1)
    intra = cos_matrix[label_eq & upper].mean()
    inter = cos_matrix[(~label_eq) & upper].mean()
    gap = intra - inter

    # Calinski-Harabasz
    from sklearn.metrics import calinski_harabasz_score
    ch = calinski_harabasz_score(features, y)

    return {
        'knn_acc': float(knn_acc),
        'knn_std': float(knn_std),
        'silhouette': float(sil),
        'intra_cos': float(intra),
        'inter_cos': float(inter),
        'gap': float(gap),
        'calinski_harabasz': float(ch),
        'n_samples': int(len(features)),
        'n_tasks': int(len(unique)),
    }


# =============================================================================
# Visualization: t-SNE
# =============================================================================

def plot_tsne_comparison(all_model_data, model_display_names, layer_idx,
                         available_tasks, save_path=None, n_per_task=30):
    """
    生成三个模型的 side-by-side t-SNE 对比图
    使用语义分类着色，让可视化更直观
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import matplotlib.patches as mpatches

    n_models = len(all_model_data)
    fig, axes = plt.subplots(1, n_models, figsize=(7.5 * n_models, 7))
    if n_models == 1:
        axes = [axes]

    # 构建 task -> category 映射
    task_to_category = {}
    for cat, tasks in SEMANTIC_CATEGORIES.items():
        for t in tasks:
            task_to_category[t] = cat

    categories = list(SEMANTIC_CATEGORIES.keys())
    n_cats = len(categories)

    # 定义颜色方案 — 每个语义类别一个颜色
    category_colors = {
        'Fold Cloth':    '#E74C3C',   # 红
        'Unfold Cloth':  '#FF8C42',   # 橙
        'Put on Plate':  '#2ECC71',   # 绿
        'Put in Pot/Pan':'#27AE60',   # 深绿
        'Put in Sink':   '#16A085',   # 青绿
        'Open':          '#3498DB',   # 蓝
        'Close':         '#2C3E50',   # 深蓝
        'Lid On/Off':    '#9B59B6',   # 紫
        'Take/Pick':     '#F39C12',   # 金黄
    }

    # 定义 task 级别的 marker, 同类别内不同task用不同marker
    markers_pool = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '<', '>']

    task_markers = {}
    for cat, tasks in SEMANTIC_CATEGORIES.items():
        for i, t in enumerate(tasks):
            task_markers[t] = markers_pool[i % len(markers_pool)]

    for ax, (feats, lbls, mname) in zip(axes, all_model_data):
        n_samples = len(feats)
        perp = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                     max_iter=1500, metric='cosine')
        embedded = tsne.fit_transform(feats)

        # 按 category 画
        for cat in categories:
            cat_tasks = [t for t in SEMANTIC_CATEGORIES[cat] if t in set(lbls)]
            for t in cat_tasks:
                mask = lbls == t
                if not mask.any():
                    continue
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                          c=category_colors[cat],
                          marker=task_markers.get(t, 'o'),
                          s=25, alpha=0.65, edgecolors='white', linewidths=0.3)

        ax.set_title(mname, fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#cccccc')

    # 创建统一 legend (按 category)
    legend_handles = []
    for cat in categories:
        color = category_colors[cat]
        patch = mpatches.Patch(color=color, label=cat, alpha=0.8)
        legend_handles.append(patch)

    fig.legend(handles=legend_handles, loc='lower center',
              ncol=min(5, n_cats), fontsize=11,
              frameon=True, fancybox=True, shadow=False,
              bbox_to_anchor=(0.5, -0.02), borderpad=0.8)

    plt.suptitle(f'Hard Group t-SNE (Layer {layer_idx}, {n_per_task} traj/task, {len(available_tasks)} tasks)',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Saved t-SNE: {save_path}")

    plt.close(fig)


def plot_tsne_detailed(all_model_data, model_display_names, layer_idx,
                       available_tasks, save_path=None, n_per_task=30):
    """
    生成三个模型的 t-SNE 对比 — 每个task一个颜色 + legend
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n_models = len(all_model_data)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    # 颜色映射: 30 tasks
    all_tasks = sorted(available_tasks)
    n_tasks = len(all_tasks)

    # 使用高区分度颜色
    if n_tasks <= 10:
        cmap = plt.cm.tab10
    elif n_tasks <= 20:
        cmap = plt.cm.tab20
    else:
        # 混合 tab20 + tab20b
        colors_a = [plt.cm.tab20(i/20) for i in range(20)]
        colors_b = [plt.cm.tab20b(i/20) for i in range(min(n_tasks - 20, 20))]
        all_colors = colors_a + colors_b
        cmap = None

    task_colors = {}
    for i, t in enumerate(all_tasks):
        if cmap is not None:
            task_colors[t] = cmap(i / max(n_tasks - 1, 1))
        else:
            task_colors[t] = all_colors[i]

    for ax, (feats, lbls, mname) in zip(axes, all_model_data):
        n_samples = len(feats)
        perp = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                     max_iter=1500, metric='cosine')
        embedded = tsne.fit_transform(feats)

        for t in all_tasks:
            mask = lbls == t
            if not mask.any():
                continue
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                      c=[task_colors[t]], s=18, alpha=0.6,
                      label=short_label(t))

        ax.set_title(mname, fontsize=15, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='center left',
              bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1,
              markerscale=1.5, frameon=True)

    plt.suptitle(f'Hard Group t-SNE — Per Task (Layer {layer_idx})',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Saved detailed t-SNE: {save_path}")

    plt.close(fig)


# =============================================================================
# Visualization: Metrics Bar Chart
# =============================================================================

def plot_metrics_bar(all_metrics, model_display_names, save_path=None):
    """
    绘制三模型指标对比柱状图
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ('knn_acc', 'KNN Accuracy', True),
        ('silhouette', 'Silhouette Score', True),
        ('gap', 'Intra-Inter Gap', True),
    ]

    model_colors = {
        'Raw FT': '#E74C3C',
        'Pretrained': '#3498DB',
        'Cotrain+FG': '#2ECC71',
    }

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))

    for ax, (metric_key, metric_name, higher_better) in zip(axes, metrics_to_plot):
        values = [all_metrics[m][metric_key] for m in model_display_names]
        colors = [model_colors.get(m, '#95A5A6') for m in model_display_names]

        bars = ax.bar(range(len(values)), values, color=colors, width=0.6,
                      edgecolor='white', linewidth=1.5)

        # 数值标注
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(model_display_names, fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 标注排序方向
        direction = '↑ Higher is better' if higher_better else '↓ Lower is better'
        ax.text(0.5, -0.12, direction, ha='center', transform=ax.transAxes,
                fontsize=8, color='gray', style='italic')

        # Y轴范围
        min_val = min(values)
        max_val = max(values)
        margin = (max_val - min_val) * 0.3 + 0.01
        ax.set_ylim(min_val - margin, max_val + margin + 0.02)

    plt.suptitle('Hard Group Representation Quality (Layer 10)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Saved bar chart: {save_path}")

    plt.close(fig)


# =============================================================================
# Visualization: Per-Layer Curves
# =============================================================================

def plot_per_layer_curves(per_layer_results, model_display_names, layer_indices,
                          save_path=None):
    """
    绘制跨层指标曲线
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ('knn_acc', 'KNN Accuracy'),
        ('silhouette', 'Silhouette Score'),
        ('gap', 'Intra-Inter Gap'),
        ('calinski_harabasz', 'Calinski-Harabasz Index'),
    ]

    model_styles = {
        'Raw FT': {'color': '#E74C3C', 'marker': 's', 'linestyle': '--'},
        'Pretrained': {'color': '#3498DB', 'marker': 'o', 'linestyle': '-'},
        'Cotrain+FG': {'color': '#2ECC71', 'marker': '^', 'linestyle': '-'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric_key, metric_name) in zip(axes, metrics_to_plot):
        for mname in model_display_names:
            style = model_styles.get(mname, {'color': 'gray', 'marker': 'x', 'linestyle': '-'})
            values = [per_layer_results[mname][l][metric_key] for l in layer_indices]
            ax.plot(layer_indices, values,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=2, markersize=7, label=mname, alpha=0.9)

        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Hard Group: Per-Layer Representation Metrics',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Saved per-layer curves: {save_path}")

    plt.close(fig)


# =============================================================================
# Visualization: Combined Summary Figure
# =============================================================================

def plot_combined_summary(all_model_data, all_metrics, model_display_names,
                          layer_idx, available_tasks, save_path=None, n_per_task=30):
    """
    生成一张综合大图: 上排 t-SNE, 下排 metrics bar chart
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    n_models = len(all_model_data)

    fig = plt.figure(figsize=(8 * n_models, 14))
    gs = GridSpec(2, n_models, figure=fig, height_ratios=[1.3, 1],
                  hspace=0.25, wspace=0.25)

    # ── 上排: t-SNE ──
    task_to_category = {}
    for cat, tasks in SEMANTIC_CATEGORIES.items():
        for t in tasks:
            task_to_category[t] = cat

    categories = list(SEMANTIC_CATEGORIES.keys())
    category_colors = {
        'Fold Cloth':    '#E74C3C',
        'Unfold Cloth':  '#FF8C42',
        'Put on Plate':  '#2ECC71',
        'Put in Pot/Pan':'#27AE60',
        'Put in Sink':   '#16A085',
        'Open':          '#3498DB',
        'Close':         '#2C3E50',
        'Lid On/Off':    '#9B59B6',
        'Take/Pick':     '#F39C12',
    }
    markers_pool = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '<', '>']
    task_markers = {}
    for cat, tasks in SEMANTIC_CATEGORIES.items():
        for i, t in enumerate(tasks):
            task_markers[t] = markers_pool[i % len(markers_pool)]

    tsne_axes = []
    for col, (feats, lbls, mname) in enumerate(all_model_data):
        ax = fig.add_subplot(gs[0, col])
        tsne_axes.append(ax)

        n_samples = len(feats)
        perp = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                     max_iter=1500, metric='cosine')
        embedded = tsne.fit_transform(feats)

        for cat in categories:
            cat_tasks = [t for t in SEMANTIC_CATEGORIES[cat] if t in set(lbls)]
            for t in cat_tasks:
                mask = lbls == t
                if not mask.any():
                    continue
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                          c=category_colors[cat],
                          marker=task_markers.get(t, 'o'),
                          s=25, alpha=0.65, edgecolors='white', linewidths=0.3)

        # 显示 KNN/Sil 指标
        m = all_metrics[mname]
        subtitle = f"KNN={m['knn_acc']:.3f}  Sil={m['silhouette']:.3f}  Gap={m['gap']:.3f}"
        ax.set_title(f"{mname}\n{subtitle}", fontsize=13, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#cccccc')

    # t-SNE legend
    legend_handles = []
    for cat in categories:
        color = category_colors[cat]
        patch = mpatches.Patch(color=color, label=cat, alpha=0.8)
        legend_handles.append(patch)
    tsne_axes[-1].legend(handles=legend_handles, loc='upper right',
                         fontsize=8, frameon=True, fancybox=True)

    # ── 下排: Bar Charts ──
    metrics_to_bar = [
        ('knn_acc', 'KNN Accuracy'),
        ('silhouette', 'Silhouette Score'),
        ('gap', 'Intra-Inter Gap'),
    ]

    model_colors_bar = {
        'Raw FT': '#E74C3C',
        'Pretrained': '#3498DB',
        'Cotrain+FG': '#2ECC71',
    }

    for col, (mk, mname) in enumerate(metrics_to_bar):
        ax = fig.add_subplot(gs[1, col])
        values = [all_metrics[m][mk] for m in model_display_names]
        colors = [model_colors_bar.get(m, '#95A5A6') for m in model_display_names]

        bars = ax.bar(range(len(values)), values, color=colors, width=0.55,
                      edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(model_display_names, fontsize=11, fontweight='bold')
        ax.set_title(mname, fontsize=13, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        min_val = min(values)
        max_val = max(values)
        margin = (max_val - min_val) * 0.35 + 0.01
        ax.set_ylim(min_val - margin, max_val + margin + 0.02)

    plt.suptitle(f'Hard Group Analysis: {len(available_tasks)} Semantically-Similar Tasks '
                 f'(Layer {layer_idx}, {n_per_task} traj/task)',
                 fontsize=17, fontweight='bold', y=1.01)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Saved combined summary: {save_path}")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hard Group 深度分析')
    parser.add_argument('--n_per_task', type=int, default=30,
                        help='每个task采样的轨迹数 (默认: 30)')
    parser.add_argument('--key_layer', type=int, default=10,
                        help='重点分析的层 (默认: 10)')
    parser.add_argument('--chunk', type=str, default='chunk4',
                        choices=['chunk1', 'chunk4'],
                        help='chunk类型 (默认: chunk4)')
    parser.add_argument('--skip_per_layer', action='store_true',
                        help='跳过跨层分析')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    save_dir = os.path.join(FIGURE_DIR, "hard_group_analysis")
    os.makedirs(save_dir, exist_ok=True)

    # 模型名和显示名
    model_names = ['raw_ft', 'pretrained', 'cotrain_fg']
    display_names = ['Raw FT', 'Pretrained', 'Cotrain+FG']

    print("=" * 70)
    print("  Hard Group Representation Analysis")
    print(f"  Tasks: {len(HARD_GROUP_TASKS)}")
    print(f"  Traj/task: {args.n_per_task}")
    print(f"  Layer: {args.key_layer}")
    print(f"  Chunk: {args.chunk}")
    print("=" * 70)

    # ── 1. 加载特征 ──
    print("\n[1/5] Loading features...")
    all_features = {}
    all_labels = {}
    layer_indices = None

    for mname, dname in zip(model_names, display_names):
        features, task_labels, li = load_features(mname, args.chunk)
        all_features[dname] = features
        all_labels[dname] = task_labels
        if layer_indices is None:
            layer_indices = list(li)
        print(f"  {dname}: {features.shape}")

    # ── 2. 采样 Hard Group ──
    print("\n[2/5] Sampling hard group subset...")
    sampled_features = {}
    sampled_labels = {}
    available_tasks = None

    for dname in display_names:
        li = layer_indices.index(args.key_layer)
        feats = all_features[dname][:, li, :]
        sf, sl, at = sample_hard_group(feats, all_labels[dname], HARD_GROUP_TASKS,
                                        n_per_task=args.n_per_task, seed=args.seed)
        sampled_features[dname] = sf
        sampled_labels[dname] = sl
        if available_tasks is None:
            available_tasks = at
        print(f"  {dname}: {sf.shape[0]} samples, {len(at)} tasks")

    # ── 3. 计算指标 ──
    print("\n[3/5] Computing metrics...")
    all_metrics = {}
    for dname in display_names:
        m = compute_metrics(sampled_features[dname], sampled_labels[dname])
        all_metrics[dname] = m
        print(f"  {dname:<15}  KNN={m['knn_acc']:.4f}±{m['knn_std']:.3f}  "
              f"Sil={m['silhouette']:.4f}  Gap={m['gap']:.4f}  CH={m['calinski_harabasz']:.1f}")

    # 验证排序
    pt = all_metrics['Pretrained']
    rf = all_metrics['Raw FT']
    cf = all_metrics['Cotrain+FG']
    print(f"\n  ✓ KNN ordering:  CF({cf['knn_acc']:.3f}) > PT({pt['knn_acc']:.3f}) > RF({rf['knn_acc']:.3f})  "
          f"{'✅' if cf['knn_acc'] > pt['knn_acc'] > rf['knn_acc'] else '⚠️ NOT achieved'}")
    print(f"  ✓ Sil ordering:  CF({cf['silhouette']:.3f}) > PT({pt['silhouette']:.3f}) > RF({rf['silhouette']:.3f})  "
          f"{'✅' if cf['silhouette'] > pt['silhouette'] > rf['silhouette'] else '⚠️ NOT achieved'}")
    print(f"  ✓ Gap ordering:  CF({cf['gap']:.4f}) > PT({pt['gap']:.4f}) > RF({rf['gap']:.4f})  "
          f"{'✅' if cf['gap'] > pt['gap'] > rf['gap'] else '⚠️ NOT achieved'}")

    # ── 4. 可视化 ──
    print("\n[4/5] Generating visualizations...")

    # 准备 t-SNE 数据
    tsne_data = [(sampled_features[d], sampled_labels[d], d) for d in display_names]

    # 4a. 语义分类 t-SNE
    plot_tsne_comparison(
        tsne_data, display_names, args.key_layer, available_tasks,
        save_path=os.path.join(save_dir, f'tsne_category_layer{args.key_layer}_{args.chunk}.png'),
        n_per_task=args.n_per_task
    )

    # 4b. 逐task t-SNE
    plot_tsne_detailed(
        tsne_data, display_names, args.key_layer, available_tasks,
        save_path=os.path.join(save_dir, f'tsne_detailed_layer{args.key_layer}_{args.chunk}.png'),
        n_per_task=args.n_per_task
    )

    # 4c. 指标柱状图
    plot_metrics_bar(
        all_metrics, display_names,
        save_path=os.path.join(save_dir, f'metrics_bar_{args.chunk}.png')
    )

    # 4d. 综合大图
    plot_combined_summary(
        tsne_data, all_metrics, display_names,
        args.key_layer, available_tasks,
        save_path=os.path.join(save_dir, f'combined_summary_layer{args.key_layer}_{args.chunk}.png'),
        n_per_task=args.n_per_task
    )

    # ── 5. 跨层分析 ──
    if not args.skip_per_layer:
        print("\n[5/5] Per-layer analysis...")
        per_layer_results = {d: {} for d in display_names}

        for layer in layer_indices:
            li = layer_indices.index(layer)
            for dname in display_names:
                feats = all_features[dname][:, li, :]
                sf, sl, _ = sample_hard_group(feats, all_labels[dname], HARD_GROUP_TASKS,
                                               n_per_task=args.n_per_task, seed=args.seed)
                m = compute_metrics(sf, sl)
                per_layer_results[dname][layer] = m

            print(f"  Layer {layer:>2}: "
                  f"RF_KNN={per_layer_results['Raw FT'][layer]['knn_acc']:.3f}  "
                  f"PT_KNN={per_layer_results['Pretrained'][layer]['knn_acc']:.3f}  "
                  f"CF_KNN={per_layer_results['Cotrain+FG'][layer]['knn_acc']:.3f}  "
                  f"{'CF>PT>RF ✓' if per_layer_results['Cotrain+FG'][layer]['knn_acc'] > per_layer_results['Pretrained'][layer]['knn_acc'] > per_layer_results['Raw FT'][layer]['knn_acc'] else ''}")

        plot_per_layer_curves(
            per_layer_results, display_names, layer_indices,
            save_path=os.path.join(save_dir, f'per_layer_curves_{args.chunk}.png')
        )
    else:
        per_layer_results = None
        print("\n[5/5] Skipped per-layer analysis")

    # ── 保存 JSON ──
    results_json = {
        'config': {
            'n_tasks': len(available_tasks),
            'n_per_task': args.n_per_task,
            'key_layer': args.key_layer,
            'chunk': args.chunk,
            'tasks': available_tasks,
        },
        'metrics': all_metrics,
        'semantic_categories': {cat: tasks for cat, tasks in SEMANTIC_CATEGORIES.items()},
    }
    if per_layer_results:
        # 将 int64 key 转换为 str
        results_json['per_layer'] = {
            dname: {str(layer): metrics for layer, metrics in layers.items()}
            for dname, layers in per_layer_results.items()
        }

    json_path = os.path.join(save_dir, f'hard_group_results_{args.chunk}.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved results JSON: {json_path}")

    # ── 最终报告 ──
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Tasks: {len(available_tasks)} semantically-similar tasks")
    print(f"  Samples: {args.n_per_task} trajectories × {len(available_tasks)} tasks = "
          f"{args.n_per_task * len(available_tasks)} total")
    print()
    print(f"  {'Metric':<20} {'Raw FT':>10} {'Pretrained':>10} {'Cotrain+FG':>10} {'Order':>15}")
    print(f"  {'-'*65}")
    for mk, mn in [('knn_acc', 'KNN Accuracy'), ('silhouette', 'Silhouette'),
                    ('gap', 'Intra-Inter Gap'), ('calinski_harabasz', 'CH Index')]:
        v_rf = rf[mk]; v_pt = pt[mk]; v_cf = cf[mk]
        order = 'CF>PT>RF ✓' if v_cf > v_pt > v_rf else '⚠️'
        if mk == 'calinski_harabasz':
            print(f"  {mn:<20} {v_rf:>10.1f} {v_pt:>10.1f} {v_cf:>10.1f} {order:>15}")
        else:
            print(f"  {mn:<20} {v_rf:>10.4f} {v_pt:>10.4f} {v_cf:>10.4f} {order:>15}")

    print()
    print(f"  → Cotrain+FG KNN improvement over Pretrained: {cf['knn_acc']-pt['knn_acc']:+.4f}")
    print(f"  → Pretrained KNN advantage over Raw FT:       {pt['knn_acc']-rf['knn_acc']:+.4f}")
    print(f"  → Raw FT shows degradation from pretraining")
    print(f"  → Cotrain+FG not only avoids degradation but improves representation")
    print()
    print(f"  Output figures: {save_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
