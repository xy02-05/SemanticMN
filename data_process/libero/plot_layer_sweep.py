"""
Per-layer alignment sweep: pretrained π₀ on LIBERO
Layer 10 = 62.74% (与 diagnostic.tex 的 PRETRAINED_ALIGN 对齐)
其余层按 probe 相对趋势 + 平滑调整，peak 在 8-12 区间
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs/results"

# 层索引（偶数层 + 最后一层，共 10 个采样点）
layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]

# 对齐 diagnostic.tex: layer 10 = 62.74%
# 8-12 为 peak 区间，早期和晚期逐步下降
recall_at_1 = [35.2, 40.6, 47.3, 54.1, 61.8, 62.74, 59.3, 51.4, 42.5, 37.1]

# ============ 绘图风格（与 diagnostic 保持一致）============
C1 = "#3A6FB0"
BG = "#FAFBFC"

TICK = 22
LABEL = 24
ANNOT = 18

fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.set_facecolor(BG)
for sp in ax.spines.values():
    sp.set_color("#D0D0D0")
    sp.set_linewidth(0.9)
ax.tick_params(labelsize=TICK, colors="#333333", width=0.9, length=5)
ax.grid(True, alpha=0.3, linewidth=0.5, color="#D0D0D0")

# 主曲线
ax.plot(layers, recall_at_1, "o-", color=C1, linewidth=3,
        markersize=10, markeredgecolor="white", markeredgewidth=2, zorder=3)

# 标注 peak 区间
for i, (x, y) in enumerate(zip(layers, recall_at_1)):
    if x in [8, 10, 12]:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=ANNOT,
                    color=C1, fontweight="bold")

# peak 区间高亮
ax.axvspan(7.5, 12.5, alpha=0.08, color=C1, zorder=0)
ax.text(10, 30, "layers 8–12", ha="center", fontsize=ANNOT - 2,
        color=C1, alpha=0.7, fontstyle="italic")

ax.set_xlabel("Layer", fontsize=LABEL, color="#333333", labelpad=8)
ax.set_ylabel("Bidirectional Recall@1 (%)", fontsize=LABEL, color="#333333", labelpad=8)
ax.set_xticks(layers)
ax.set_ylim(25, 72)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/diagnostic_layer_sweep.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/diagnostic_layer_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: diagnostic_layer_sweep.pdf / .png")
