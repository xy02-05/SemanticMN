"""
Combined 3-panel diagnostic figure: alignment erosion over fine-tuning
  (a) Average Retrieval Accuracy
  (b) In-Distribution Success (LIBERO)
  (c) Alignment vs OOD Success scatter (all checkpoints + Pearson r)
All font sizes doubled compared to original single-panel versions.
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT_DIR = "outputs/results"

# ============ data ============
FT_STEPS = [10, 15, 20, 25, 30]

PRETRAINED_ALIGN = 62.74
align_vals = [45.16, 45.61, 51.42, 50.33, 47.60]
id_vals    = [62.4, 80.0, 84.4, 88.0, 88.6]

# Fig 3 extended data (all checkpoints, LIBERO_Pro OOD)
baseline_labels = ["5k", "10k", "15k", "20k", "25k", "30k"]
baseline_align  = [41.68, 45.16, 45.61, 51.42, 50.33, 47.60]
baseline_ood    = [ 9.9,  28.5,  34.5,  35.8,  35.0,  33.2]

align_labels = ["Ours\n30k"]
align_align  = [75.00]
align_ood    = [40.5]

# ============ palette ============
C1 = "#3A6FB0"
C2 = "#4DA86A"
C3 = "#C44E52"
BG = "#FAFBFC"

# ============ 2x font sizes ============
TICK_SIZE = 28
LABEL_SIZE = 32
ANNOT_SIZE = 28
LEGEND_SIZE = 28
TITLE_SIZE = 34


def style_ax(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=TICK_SIZE, colors="#333333", width=1.2, length=6)
    ax.grid(True, alpha=0.3, linewidth=0.6, color="#D0D0D0")


fig, axes = plt.subplots(1, 3, figsize=(30, 9), facecolor="white")

# ========== (a) Alignment ==========
ax = axes[0]
ax.plot(FT_STEPS, align_vals, "o-", color=C1, linewidth=3.5,
        markersize=16, markeredgecolor="white", markeredgewidth=3, zorder=3)
for x, y in zip(FT_STEPS, align_vals):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, 20), ha="center", fontsize=ANNOT_SIZE,
                color=C1, fontweight="bold")
ax.axhline(y=PRETRAINED_ALIGN, color="#AAAAAA", linestyle="--",
           linewidth=2, alpha=0.55, zorder=1)
ax.text(0.97, 0.95, f"Pretrained ({PRETRAINED_ALIGN}%)",
        fontsize=LEGEND_SIZE, color="#888888", fontweight="bold",
        transform=ax.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#D0D0D0", alpha=0.85))
ax.set_xlabel("Training Steps (k)", fontsize=LABEL_SIZE, color="#333333", labelpad=10)
ax.set_ylabel("Average Retrieval Accuracy (%)", fontsize=LABEL_SIZE, color="#333333", labelpad=10)
ax.set_xticks(FT_STEPS)
ax.set_xticklabels([f"{s}k" for s in FT_STEPS])
ax.set_title("(a)", fontsize=TITLE_SIZE, fontweight="bold", color="#333333", pad=12)
style_ax(ax)

# ========== (b) ID success ==========
ax = axes[1]
ax.plot(FT_STEPS, id_vals, "o-", color=C2, linewidth=3.5,
        markersize=16, markeredgecolor="white", markeredgewidth=3, zorder=3)
for i, (x, y) in enumerate(zip(FT_STEPS, id_vals)):
    if i in {0, len(id_vals) - 1}:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 20), ha="center", fontsize=ANNOT_SIZE,
                    color=C2, fontweight="bold")
ax.set_xlabel("Training Steps (k)", fontsize=LABEL_SIZE, color="#333333", labelpad=10)
ax.set_ylabel("ID Success Rate (%)", fontsize=LABEL_SIZE, color="#333333", labelpad=10)
ax.set_xticks(FT_STEPS)
ax.set_xticklabels([f"{s}k" for s in FT_STEPS])
ax.set_title("(b)", fontsize=TITLE_SIZE, fontweight="bold", color="#333333", pad=12)
style_ax(ax)

# ========== (c) Scatter – Alignment vs OOD ==========
ax = axes[2]
all_x = baseline_align + align_align
all_y = baseline_ood + align_ood
rho_all, p_sp = stats.spearmanr(all_x, all_y)
r_all, p_pr = stats.pearsonr(all_x, all_y)

ax.scatter(baseline_align, baseline_ood, s=500, c=C3, marker="h",
           edgecolors="white", linewidths=3, zorder=4)
ax.scatter(align_align, align_ood, s=600, c=C1, marker="*",
           edgecolors="white", linewidths=2, zorder=5)

slope, intercept = np.polyfit(all_x, all_y, 1)
xs = np.linspace(min(all_x) - 3, max(all_x) + 3, 100)
ax.plot(xs, slope * xs + intercept, "--", color="#999999", linewidth=2,
        alpha=0.6, zorder=2)

bl_nudge = {
    0: (0, -30),    # 5k
    1: (-42, 0),    # 10k
    2: (-42, 0),    # 15k
    3: (0, 28),     # 20k
    4: (38, 0),     # 25k
    5: (0, -30),    # 30k
}
for i, (x, y) in enumerate(zip(baseline_align, baseline_ood)):
    nx, ny = bl_nudge[i]
    ax.annotate(baseline_labels[i], (x, y), textcoords="offset points",
                xytext=(nx, ny), ha="center", va="center", fontsize=ANNOT_SIZE,
                color=C3, fontweight="bold")

for i, (x, y) in enumerate(zip(align_align, align_ood)):
    ax.annotate(align_labels[i], (x, y), textcoords="offset points",
                xytext=(0, -36), ha="center", va="center", fontsize=ANNOT_SIZE,
                color=C1, fontweight="bold")

ax.text(0.03, 0.97,
        f"$\\rho$ = {rho_all:.3f}",
        fontsize=LEGEND_SIZE, color="#333333", fontweight="bold",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#D0D0D0", alpha=0.85))

x_margin = 4
y_margin = 3
ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

ax.set_xlabel("Average Retrieval Accuracy (%)", fontsize=LABEL_SIZE, color="#333333", labelpad=10)
ax.set_ylabel("OOD Success Rate (%)", fontsize=LABEL_SIZE, color="#333333", labelpad=10)
ax.set_title("(c)", fontsize=TITLE_SIZE, fontweight="bold", color="#333333", pad=12)
style_ax(ax)

fig.tight_layout(w_pad=4)
fig.savefig(f"{OUT_DIR}/diagnostic_combined.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/diagnostic_combined.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: diagnostic_combined (Spearman rho={rho_all:.3f}, Pearson r={r_all:.3f})")
