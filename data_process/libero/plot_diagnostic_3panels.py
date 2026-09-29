"""
Diagnostic figures (3 separate): alignment erosion over fine-tuning
  Fig 1: Average Retrieval Accuracy
  Fig 2: In-Distribution Success (LIBERO)
  Fig 3: Alignment vs OOD Success scatter (all checkpoints + Pearson r)
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT_DIR = "outputs/results"

# ============ data (no 5k) ============
# Test tasks: {4, 7, 12, 17, 20, 22, 36, 38} (2 per suite, qwen3 probe)
FT_STEPS = [10, 15, 20, 25, 30]

PRETRAINED_ALIGN = 62.74
align_vals = [45.16, 45.61, 51.42, 50.33, 47.60]
id_vals    = [62.4, 80.0, 84.4, 88.0, 88.6]
ood_vals   = [28.5, 34.5, 35.8, 35.0, 34.2]

# ============ Fig 3 extended data (all checkpoints, LIBERO_Pro OOD) ============
# Baseline: retrieval on new test split + LIBERO_Pro OOD (paper values)
baseline_labels = ["5k", "10k", "15k", "20k", "25k", "30k"]
baseline_align  = [41.68, 45.16, 45.61, 51.42, 50.33, 47.60]
baseline_ood    = [ 9.9,  28.5,  34.5,  35.8,  35.0,  33.2]

# Alignment: Ours 30k (paper LIBERO_Pro: 40.5%)
align_labels = ["Ours\n30k"]
align_align  = [75.00]
align_ood    = [40.5]

# ============ palette ============
C1 = "#3A6FB0"
C2 = "#4DA86A"
C3 = "#C44E52"

BG = "#FAFBFC"


def style_ax(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.9)
    ax.tick_params(labelsize=22, colors="#333333", width=0.9, length=5)
    ax.grid(True, alpha=0.3, linewidth=0.5, color="#D0D0D0")


# ========== Fig 1: Alignment ==========
fig1, ax = plt.subplots(figsize=(7, 6.875), facecolor="white")
ax.plot(FT_STEPS, align_vals, "o-", color=C1, linewidth=3,
        markersize=12, markeredgecolor="white", markeredgewidth=2.5, zorder=3)
for x, y in zip(FT_STEPS, align_vals):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=20,
                color=C1, fontweight="bold")
ax.axhline(y=PRETRAINED_ALIGN, color="#AAAAAA", linestyle="--",
           linewidth=1.5, alpha=0.55, zorder=1)
ax.text(0.97, 0.95, f"Pretrained ({PRETRAINED_ALIGN}%)",
        fontsize=20, color="#888888", fontweight="bold",
        transform=ax.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#D0D0D0", alpha=0.85))
ax.set_xlabel("Training Steps (k)", fontsize=24, color="#333333", labelpad=8)
ax.set_ylabel("Average Retrieval Accuracy (%)", fontsize=22, color="#333333", labelpad=8)
ax.set_xticks(FT_STEPS)
ax.set_xticklabels([f"{s}k" for s in FT_STEPS])
style_ax(ax)
fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}/diagnostic_align.pdf", bbox_inches="tight")
fig1.savefig(f"{OUT_DIR}/diagnostic_align.png", dpi=200, bbox_inches="tight")
plt.close(fig1)
print("Saved: diagnostic_align")


# ========== Fig 2: ID success ==========
fig2, ax = plt.subplots(figsize=(7, 6.875), facecolor="white")
ax.plot(FT_STEPS, id_vals, "o-", color=C2, linewidth=3,
        markersize=12, markeredgecolor="white", markeredgewidth=2.5, zorder=3)
for i, (x, y) in enumerate(zip(FT_STEPS, id_vals)):
    if i in {0, len(id_vals) - 1}:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=20,
                    color=C2, fontweight="bold")
ax.set_xlabel("Training Steps (k)", fontsize=24, color="#333333", labelpad=8)
ax.set_ylabel("ID Success Rate (%)", fontsize=24, color="#333333", labelpad=8)
ax.set_xticks(FT_STEPS)
ax.set_xticklabels([f"{s}k" for s in FT_STEPS])
style_ax(ax)
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/diagnostic_id.pdf", bbox_inches="tight")
fig2.savefig(f"{OUT_DIR}/diagnostic_id.png", dpi=200, bbox_inches="tight")
plt.close(fig2)
print("Saved: diagnostic_id")


# ========== Fig 3: Scatter – Alignment vs OOD (all points + Pearson r) ==========
all_x = baseline_align + align_align
all_y = baseline_ood + align_ood
all_labels = baseline_labels + align_labels
rho_all, p_all = stats.spearmanr(all_x, all_y)
rho_bl, p_bl = stats.spearmanr(baseline_align, baseline_ood)

fig3, ax = plt.subplots(figsize=(7, 6.875), facecolor="white")

ax.scatter(baseline_align, baseline_ood, s=400, c=C3, marker="h",
           edgecolors="white", linewidths=2.5, zorder=4)
ax.scatter(align_align, align_ood, s=500, c=C1, marker="*",
           edgecolors="white", linewidths=1.5, zorder=5)

slope, intercept = np.polyfit(all_x, all_y, 1)
xs = np.linspace(min(all_x) - 3, max(all_x) + 3, 100)
ax.plot(xs, slope * xs + intercept, "--", color="#999999", linewidth=1.5,
        alpha=0.6, zorder=2)

bl_nudge = {
    0: (0, -26),    # 5k   – bottom
    1: (-36, 0),    # 10k  – left
    2: (-36, 0),    # 15k  – left
    3: (0, 24),     # 20k  – top
    4: (32, 0),     # 25k  – right
    5: (0, -26),    # 30k  – bottom
}
for i, (x, y) in enumerate(zip(baseline_align, baseline_ood)):
    nx, ny = bl_nudge[i]
    ax.annotate(baseline_labels[i], (x, y), textcoords="offset points",
                xytext=(nx, ny), ha="center", va="center", fontsize=20,
                color=C3, fontweight="bold")

for i, (x, y) in enumerate(zip(align_align, align_ood)):
    ax.annotate(align_labels[i], (x, y), textcoords="offset points",
                xytext=(0, -30), ha="center", va="center", fontsize=20,
                color=C1, fontweight="bold")

ax.text(0.03, 0.97,
        f"$\\rho$ = {rho_all:.3f}",
        fontsize=20, color="#333333", fontweight="bold",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#D0D0D0", alpha=0.85))

x_margin = 4
y_margin = 3
ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

ax.set_xlabel("Average Retrieval Accuracy (%)", fontsize=24, color="#333333", labelpad=8)
ax.set_ylabel("OOD Success Rate (%)", fontsize=24, color="#333333", labelpad=8)
style_ax(ax)
fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}/diagnostic_scatter.pdf", bbox_inches="tight")
fig3.savefig(f"{OUT_DIR}/diagnostic_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: diagnostic_scatter (Spearman rho_all={rho_all:.3f} p={p_all:.4f} n={len(all_x)}, rho_baseline={rho_bl:.3f})")
