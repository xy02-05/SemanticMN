"""
Three-stage t-SNE on LIBERO Goal suite (10 tasks), rollout-mode features:
  Pretrained  →  Action-only baseline (30k)  →  Ours (30k)
Each stage saved as a separate figure, no border/spines.
"""
import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

FEAT = Path(__file__).parent / "outputs" / "features"
SOURCES = {
    "pretrained": (FEAT / "pretrained_test_rollout.npz", FEAT / "pretrained_train_rollout.npz"),
    "baseline":   (FEAT / "step_30k_test_rollout.npz",   FEAT / "step_30k_train_rollout.npz"),
    "ours":       (FEAT / "align_full_qwen_30k_test_rollout.npz", FEAT / "align_full_qwen_30k_train_rollout.npz"),
}
TITLES = {
    "pretrained": "Pretrained",
    "baseline":   "Vanilla FT",
    "ours":       "Ours",
}
OUT_DIR = Path(__file__).parent / "outputs"

LAYER = 10
GOAL  = [t for t in range(20, 30) if t != 28]
SEED  = 0
POINT_SIZE = 60

TASK_COLORS = [
    "#5E7A91", "#C0764A", "#7E9B77", "#A86A95", "#C9A85A",
    "#6F7CB2", "#B85C5C", "#7DAFAE", "#8E7BA8", "#D4A76A",
]


def load_layer(npz_path, layer=LAYER):
    d = np.load(npz_path, allow_pickle=True)
    feats  = d["features"]
    layers = d["layer_indices"].tolist()
    return feats[:, layers.index(layer), :], d["task_indices"]


def load_goal(test_path, train_path):
    x_te, y_te = load_layer(test_path)
    x_tr, y_tr = load_layer(train_path)
    x = np.concatenate([x_te, x_tr], axis=0)
    y = np.concatenate([y_te, y_tr], axis=0)
    m = np.isin(y, GOAL)
    return x[m], y[m]


def fit_tsne(x):
    n = len(x)
    perp = min(30, max(5, n // 15))
    return TSNE(n_components=2, perplexity=perp, metric="cosine",
                init="pca", random_state=SEED, max_iter=2000).fit_transform(x)


def plot_single(emb, y, sil, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for t, c in zip(GOAL, TASK_COLORS):
        mask = y == t
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=POINT_SIZE, c=c,
                   edgecolors="white", linewidths=0.6, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    xmin, xmax = emb[:, 0].min(), emb[:, 0].max()
    ymin, ymax = emb[:, 1].min(), emb[:, 1].max()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 * 1.1
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.98, 0.11, f"Silhouette = {sil:.2f}",
            transform=ax.transAxes, fontsize=22,
            color="#333333", ha="right", va="bottom", family="Arial")
    ax.text(0.98, 0.03, title,
            transform=ax.transAxes, fontsize=26, fontweight="bold",
            color="#333333", ha="right", va="bottom", family="Arial")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    data = {}
    print(f"=== Rollout features, LIBERO Goal suite, layer {LAYER}, {len(GOAL)} tasks ===\n")
    print(f"{'stage':>12s} | {'N':>5s} | {'silhouette':>10s}")
    print("-" * 40)

    for name, (te, tr) in SOURCES.items():
        x, y = load_goal(te, tr)
        sil = silhouette_score(x, y, metric="cosine")
        print(f"{name:>12s} | {len(y):5d} | {sil:+.4f}")
        data[name] = dict(x=x, y=y, sil=sil)

    order = ["pretrained", "baseline", "ours"]
    for name in order:
        m = data[name]
        print(f"\n[{name}] running t-SNE ...")
        emb = fit_tsne(m["x"])
        if name == "ours":
            for t in GOAL:
                mask = m["y"] == t
                if not mask.any():
                    continue
                center = emb[mask].mean(axis=0)
                emb[mask] = center + (emb[mask] - center) * 2.5
        out = OUT_DIR / f"tsne_goal_{name}"
        plot_single(emb, m["y"], m["sil"], TITLES[name], out)
        print(f"saved {out}.png/.pdf")


if __name__ == "__main__":
    main()
