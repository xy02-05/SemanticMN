"""
Three-panel ablation figure:
  (a) Method components - 3 bars (Baseline → +Align → +DSN)
  (b) Alignment target - 6 bars, multi-color by encoder type, EgoHOD rightmost
  (c) Alignment layer - line chart, thick
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs/results"

# ============ (a) Method components ============
comp_names = ["Baseline", "+ Align", "+ DSN\n(full)"]
comp_avg   = [43.8, 49.0, 51.0]

# ============ (b) Alignment target — EgoHOD rightmost ============
target_names = ["Baseline", "Self", "Qwen3\nEmb.", "CLIP\nLarge", "Qwen3-VL\n Emb.", "EgoHOD"]
target_avg   = [43.8, 46.8, 45.8, 46.9, 49.0, 51.0]

# ============ (c) Alignment layer ============
layer_labels = ["5", "10", "15", "20", "26"]
layer_avg    = [43.8, 51.0, 50.0, 45.8, 44.8]

# ============ palette ============
BG = "#FAFBFC"

C_COMP = ["#B8B8B8", "#7A9DB5", "#4E85A8"]

C_TARGET = [
    "#D0D0D0",   # Baseline — light gray
    "#A0A0A0",   # Self — medium gray (no external encoder)
    "#E8B87C",   # Qwen3 Emb. — golden (language-only)
    "#E09090",   # CLIP Large — soft coral (image-text)
    "#80C4A4",   # Qwen3-VL Emb. — soft green (video, general)
    "#7AAFD4",   # EgoHOD — soft blue (video, domain-specific, best)
]

C_LINE = "#4E85A8"

# ============ font sizes ============
TICK  = 44
LABEL = 48
SUB   = 50
BAR_LABEL = 40

def style_ax(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color("#D0D0D0")
        sp.set_linewidth(1.2)
    ax.tick_params(labelsize=TICK, colors="#333333", width=1.2, length=6)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6, color="#D0D0D0")
    ax.grid(False, axis="x")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(38, 12),
                                     gridspec_kw={"width_ratios": [2.5, 6, 3.5]},
                                     facecolor="white")

# ==================== (a) Method components ====================
style_ax(ax1)
x1 = np.arange(len(comp_names))
bars1 = ax1.bar(x1, comp_avg, width=0.58, color=C_COMP,
                edgecolor="white", linewidth=2.5, zorder=3)

for bar, val in zip(bars1, comp_avg):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", va="bottom", fontsize=BAR_LABEL,
             fontweight="bold", color="#444444")

ax1.set_xticks(x1)
ax1.set_xticklabels(comp_names, fontsize=TICK - 4, fontweight="bold")
ax1.set_ylabel("Avg Success Rate (%)", fontsize=LABEL, color="#333333",
               fontweight="bold", labelpad=10)
ax1.set_ylim(38, 54)

# ==================== (b) Alignment target ====================
style_ax(ax2)
x2 = np.arange(len(target_names))
bars2 = ax2.bar(x2, target_avg, width=0.55, color=C_TARGET,
                edgecolor="white", linewidth=2.5, zorder=3)

for bar, val in zip(bars2, target_avg):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{val:.1f}", ha="center", va="bottom", fontsize=BAR_LABEL,
             fontweight="bold", color="#444444")

ax2.set_xticks(x2)
ax2.set_xticklabels(target_names, fontsize=TICK - 6, fontweight="bold")
ax2.set_ylim(38, 54)

# ==================== (c) Alignment layer ====================
style_ax(ax3)
x3 = np.arange(len(layer_labels))

ax3.plot(x3, layer_avg, "o-", color=C_LINE, linewidth=8,
         markersize=28, markeredgecolor="white", markeredgewidth=3.5,
         zorder=3)

for i, (x, y) in enumerate(zip(x3, layer_avg)):
    offset = 18 if i in (1, 2) else -26
    ax3.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                 xytext=(0, offset), ha="center", fontsize=BAR_LABEL,
                 fontweight="bold", color="#444444")

ax3.set_xticks(x3)
ax3.set_xticklabels([f"k={l}" for l in layer_labels], fontsize=TICK - 4, fontweight="bold")
ax3.set_ylim(38, 54)

# ==================== subplot labels at bottom, aligned ====================
label_y = -0.18
ax1.text(0.5, label_y, "(a) Method Components", transform=ax1.transAxes,
         fontsize=SUB, fontweight="bold", color="#333333", ha="center", va="top")
ax2.text(0.5, label_y, "(b) Alignment Target", transform=ax2.transAxes,
         fontsize=SUB, fontweight="bold", color="#333333", ha="center", va="top")
ax3.text(0.5, label_y, "(c) Alignment Layer", transform=ax3.transAxes,
         fontsize=SUB, fontweight="bold", color="#333333", ha="center", va="top")

fig.tight_layout(w_pad=5)
fig.subplots_adjust(bottom=0.22)
fig.savefig(f"{OUT_DIR}/ablation_target_layer.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/ablation_target_layer.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: ablation_target_layer.pdf / .png")
