"""
Single-panel diagnostic figure: 3 curves on one chart
  - Average Retrieval Accuracy
  - ID Success Rate (LIBERO)
  - OOD Success Rate (LIBERO-Pro)
All font sizes doubled.
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs/results"

# ============ data ============
FT_STEPS_ALL = [5, 10, 15, 20, 25, 30]

align_vals = [41.68, 45.61, 45.61, 51.42, 50.33, 47.60]
ood_vals   = [9.9, 28.5, 34.5, 35.8, 35.0, 33.2]

FT_STEPS_ID = FT_STEPS_ALL
id_vals     = [31.75, 62.4, 80.0, 84.4, 88.0, 88.6]

PRETRAINED_ALIGN = 62.74

# ============ palette ============
C1 = "#3A6FB0"   # alignment
C2 = "#4DA86A"   # ID
C3 = "#C44E52"   # OOD
BG = "#FAFBFC"

# ============ 2x font sizes ============
TICK_SIZE = 40
LABEL_SIZE = 44
ANNOT_SIZE = 36
LEGEND_SIZE = 38
TITLE_SIZE = 48

fig, ax = plt.subplots(figsize=(16, 14), facecolor="white")
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_color("#D0D0D0")
    spine.set_linewidth(1.2)
ax.tick_params(labelsize=TICK_SIZE, colors="#333333", width=1.2, length=6)
ax.grid(True, alpha=0.3, linewidth=0.6, color="#D0D0D0")

# --- Retrieval Accuracy ---
ax.plot(FT_STEPS_ALL, align_vals, "o-", color=C1, linewidth=3.5,
        markersize=16, markeredgecolor="white", markeredgewidth=2.5,
        zorder=3, label="Retrieval Accuracy")
for i, (x, y) in enumerate(zip(FT_STEPS_ALL, align_vals)):
    if i == 0 or i == len(align_vals) - 1:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 18), ha="center", fontsize=ANNOT_SIZE,
                    color=C1, fontweight="bold")

# --- ID Success ---
ax.plot(FT_STEPS_ID, id_vals, "s-", color=C2, linewidth=3.5,
        markersize=16, markeredgecolor="white", markeredgewidth=2.5,
        zorder=3, label="ID Success Rate")
for i, (x, y) in enumerate(zip(FT_STEPS_ID, id_vals)):
    if i == 0 or i == len(id_vals) - 1:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 18), ha="center", fontsize=ANNOT_SIZE,
                    color=C2, fontweight="bold")

# --- OOD Success ---
ax.plot(FT_STEPS_ALL, ood_vals, "D-", color=C3, linewidth=3.5,
        markersize=16, markeredgecolor="white", markeredgewidth=2.5,
        zorder=3, label="OOD Success Rate")
for i, (x, y) in enumerate(zip(FT_STEPS_ALL, ood_vals)):
    if i == 0 or i == len(ood_vals) - 1:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, -24), ha="center", fontsize=ANNOT_SIZE,
                    color=C3, fontweight="bold")

# --- Pretrained baseline ---
ax.axhline(y=PRETRAINED_ALIGN, color="#AAAAAA", linestyle="--",
           linewidth=2, alpha=0.55, zorder=1)
ax.text(30.3, PRETRAINED_ALIGN + 1, f"Pretrained Align ({PRETRAINED_ALIGN}%)",
        fontsize=ANNOT_SIZE - 4, color="#888888", fontweight="bold",
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#D0D0D0", alpha=0.85))

ax.set_xlabel("Training Steps (k)", fontsize=LABEL_SIZE, color="#333333", labelpad=10)
ax.set_ylabel("Rate (%)", fontsize=LABEL_SIZE, color="#333333", labelpad=10)
ax.set_xticks(FT_STEPS_ALL)
ax.set_xticklabels([f"{s}k" for s in FT_STEPS_ALL])
ax.set_ylim(5, 95)
ax.set_yticks([20, 30, 40, 50, 60, 70, 80, 90])

ax.legend(fontsize=LEGEND_SIZE, loc="upper left",
          frameon=True, fancybox=True, framealpha=0.9,
          edgecolor="#D0D0D0")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/diagnostic_single.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/diagnostic_single.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: diagnostic_single.pdf / .png")
