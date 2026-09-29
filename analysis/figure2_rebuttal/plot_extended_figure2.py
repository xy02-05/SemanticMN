"""绘制 rebuttal 使用的双面板 extended Figure 2(a) 草案。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

import matplotlib.pyplot as plt
import numpy as np


# 只展示 10k 间隔的 checkpoint，不包含 pretrained reference。
STEPS = np.asarray([10, 20, 30, 40, 50])

# Baseline：沿用旧版 extended Figure 2 中相同步数的数据。
BASELINE_ALIGNMENT = np.asarray([45.16, 51.42, 47.60, 47.40, 46.50])
BASELINE_OOD = np.asarray([28.50, 35.80, 33.20, 34.40, 32.30])
BASELINE_ID = np.asarray([62.40, 84.40, 88.60, 89.30, 88.90])

# Ours：rebuttal 双面板排版草案值，集中保存在这里以便替换实测结果。
OURS_ALIGNMENT = np.asarray([76.80, 79.00, 80.00, 79.80, 79.50])
OURS_OOD = np.asarray([31.70, 38.50, 40.50, 40.20, 39.90])
OURS_ID = np.asarray([64.00, 85.70, 91.40, 92.30, 91.00])

# 与旧版单面板成品完全相同的总画布尺寸（878.209 × 315.094 PDF points，
# 即 300 dpi 下 3659 × 1313 px）；两个子图共享该尺寸。
FIGSIZE = (878.209140625 / 72, 315.0941076812 / 72)
# Matplotlib truncates raster dimensions; this tiny adjustment reproduces the old
# 3659 × 1313 px PNG while the vector outputs keep the exact original MediaBox.
PNG_DPI = 300.025


def parse_args() -> argparse.Namespace:
    mirror_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=mirror_root / "doc" / "figures",
        help="PNG/PDF/SVG 输出目录",
    )
    parser.add_argument(
        "--stem",
        default="figure2a_extended_split_draft",
        help="输出文件名（不含扩展名）",
    )
    return parser.parse_args()


def _draw_panel(
    ax: plt.Axes,
    *,
    title: str,
    alignment: np.ndarray,
    ood: np.ndarray,
    id_success: np.ndarray,
) -> list[plt.Line2D]:
    colors = {
        "alignment": "#3A6FB0",
        "ood": "#C44E52",
        "id": "#4DA86A",
    }

    alignment_line, = ax.plot(
        STEPS,
        alignment,
        "o-",
        color=colors["alignment"],
        linewidth=2.7,
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Alignment",
        zorder=5,
    )
    ood_line, = ax.plot(
        STEPS,
        ood,
        "D-",
        color=colors["ood"],
        linewidth=2.7,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="OOD success",
        zorder=5,
    )
    id_line, = ax.plot(
        STEPS,
        id_success,
        "s-",
        color=colors["id"],
        linewidth=2.7,
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="ID success",
        zorder=5,
    )

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_facecolor("#FAFBFC")
    ax.set_xlim(7, 53)
    ax.set_ylim(20, 100)
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{step}k" for step in STEPS])
    ax.set_yticks(np.arange(20, 101, 20))
    ax.grid(True, color="#D5D8DC", linewidth=0.75, alpha=0.52)
    ax.tick_params(direction="out", length=4.5, width=1.0)
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")

    return [alignment_line, ood_line, id_line]


def build_figure() -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
        facecolor="white",
        gridspec_kw={"wspace": 0.10},
    )

    handles = _draw_panel(
        axes[0],
        title="(a) Baseline",
        alignment=BASELINE_ALIGNMENT,
        ood=BASELINE_OOD,
        id_success=BASELINE_ID,
    )
    _draw_panel(
        axes[1],
        title="(b) Ours",
        alignment=OURS_ALIGNMENT,
        ood=OURS_OOD,
        id_success=OURS_ID,
    )

    axes[0].set_ylabel("Score (%)")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.supxlabel("Training steps", fontsize=16, y=0.045)
    fig.legend(
        handles,
        [line.get_label() for line in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.80,
        wspace=0.10,
    )
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = args.output_dir / args.stem
    # 不使用 bbox_inches="tight"，避免双面板内容改变最终画布尺寸。
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
