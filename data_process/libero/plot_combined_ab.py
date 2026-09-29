"""
Combined 2-panel figure:
  (a) 3-line diagnostic chart (Retrieval Acc., ID Succ., OOD Succ.)
  (b) Success vs Failure bar chart (Cosine Similarity + Retrieval Acc.)
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs/results"

# ============ (a) line chart data ============
FT_STEPS = [5, 10, 15, 20, 25, 30]
align_vals = [41.68, 45.61, 45.61, 51.42, 50.33, 47.60]
id_vals    = [31.75, 62.4, 80.0, 84.4, 88.0, 88.6]
ood_vals   = [9.9, 28.5, 34.5, 35.8, 35.0, 33.2]
PRETRAINED_ALIGN = 62.74

# ============ (b) bar chart data ============
cos_success, cos_failure = 0.625, 0.518
ret_success, ret_failure = 86.7, 76.5

# ============ palette ============
C_ALIGN = "#3A6FB0"
C_ID    = "#4DA86A"
C_OOD   = "#C44E52"
C_SUC   = "#5B8DBE"
C_FAIL  = "#E8945A"
BG      = "#FAFBFC"

# ============ font sizes — unified across all axes ============
TICK      = 52
YLABEL_FS = 50          # same for ALL y-axis labels
XLABEL_FS = 56
ANNOT     = 48
LEG       = 44
SUB       = 62

def style_ax(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color("#D0D0D0")
        sp.set_linewidth(1.2)
    ax.tick_params(labelsize=TICK, colors="#333333", width=1.2, length=6)
    ax.grid(True, alpha=0.25, linewidth=0.6, color="#D0D0D0")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 16.8), facecolor="white",
                                gridspec_kw={"width_ratios": [1, 1]})

# ==================== (a) dual-axis line chart ====================
style_ax(ax1)

# Left axis: Retrieval Acc. + OOD Succ.
ax1.plot(FT_STEPS, align_vals, "o-", color=C_ALIGN, linewidth=6,
         markersize=22, markeredgecolor="white", markeredgewidth=2,
         zorder=4, label="Retrieval Acc.")
ax1.plot(FT_STEPS, ood_vals, "D-", color=C_OOD, linewidth=6,
         markersize=22, markeredgecolor="white", markeredgewidth=2,
         zorder=4, label="OOD Succ.")

ax1.annotate(f"{align_vals[-1]:.1f}", (FT_STEPS[-1], align_vals[-1]),
             textcoords="offset points", xytext=(-4, -24),
             ha="center", fontsize=ANNOT, color=C_ALIGN, fontweight="bold")
ax1.annotate(f"{ood_vals[-1]:.1f}", (FT_STEPS[-1], ood_vals[-1]),
             textcoords="offset points", xytext=(-4, 14),
             ha="center", fontsize=ANNOT, color=C_OOD, fontweight="bold")

ax1.set_xlabel("Training Steps (k)", fontsize=XLABEL_FS, color="#333333",
               fontweight="bold", labelpad=10)

# Multi-color y-label
ax1.set_ylabel("")
lx = -0.105
ax1.text(lx, 0.57, "Retrieval Acc. (%)", transform=ax1.transAxes,
         fontsize=YLABEL_FS, color=C_ALIGN, fontweight="bold",
         ha="center", va="bottom", rotation=90)
ax1.text(lx, 0.47, " / ", transform=ax1.transAxes,
         fontsize=YLABEL_FS, color="#333333", fontweight="bold",
         ha="center", va="bottom", rotation=90)
ax1.text(lx, 0.08, "OOD Succ. (%)", transform=ax1.transAxes,
         fontsize=YLABEL_FS, color=C_OOD, fontweight="bold",
         ha="center", va="bottom", rotation=90)

ax1.set_xticks(FT_STEPS)
ax1.set_xticklabels([f"{s}k" for s in FT_STEPS])
ax1.set_xlim(3, 32)
ax1.set_ylim(0, 60)
ax1.set_yticks([0, 10, 20, 30, 40, 50, 60])

# Right axis: ID Succ.
ax1r = ax1.twinx()
ax1r.plot(FT_STEPS, id_vals, "s-", color=C_ID, linewidth=6,
          markersize=22, markeredgecolor="white", markeredgewidth=2,
          zorder=3, label="ID Succ.")
ax1r.annotate(f"{id_vals[-1]:.1f}", (FT_STEPS[-1], id_vals[-1]),
              textcoords="offset points", xytext=(-4, 14),
              ha="center", fontsize=ANNOT, color=C_ID, fontweight="bold")
ax1r.set_ylabel("ID Succ. (%)", fontsize=YLABEL_FS, color=C_ID,
                fontweight="bold", labelpad=14)
ax1r.set_ylim(0, 100)
ax1r.set_yticks([0, 20, 40, 60, 80, 100])
ax1r.tick_params(labelsize=TICK, colors="#333333", width=1.2, length=6)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines1r, labels1r = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines1r, labels1 + labels1r,
           fontsize=LEG, loc="upper left", frameon=True, fancybox=True,
           framealpha=0.92, edgecolor="#D0D0D0")

# ==================== (b) bar chart ====================
style_ax(ax2)
ax2.grid(True, axis="y", alpha=0.25, linewidth=0.6, color="#D0D0D0")
ax2.grid(False, axis="x")

YMAX = 0.75
SCALE = 100 / YMAX

x_pos = np.array([0, 1.2])
w = 0.35

bars1 = ax2.bar(x_pos - w/2, [cos_success, ret_success / SCALE], w,
                color=C_SUC, edgecolor="white", linewidth=1.5,
                label="Success", zorder=3)
bars2 = ax2.bar(x_pos + w/2, [cos_failure, ret_failure / SCALE], w,
                color=C_FAIL, edgecolor="white", linewidth=1.5,
                label="Failure", zorder=3)

ax2.set_ylabel("Cosine Similarity", fontsize=YLABEL_FS, color="#333333",
               fontweight="bold", labelpad=14)
ax2.set_ylim(0, YMAX + 0.08)
ax2.set_yticks([0.0, 0.2, 0.4, 0.6])

ax2r = ax2.twinx()
ax2r.set_ylabel("Retrieval Acc. (%)", fontsize=YLABEL_FS, color="#333333",
                fontweight="bold", labelpad=14)
ax2r.set_ylim(0, (YMAX + 0.08) * SCALE)
ax2r.set_yticks([0, 25, 50, 75, 100])
ax2r.tick_params(labelsize=TICK, colors="#333333", width=1.2, length=6)
for sp in ax2r.spines.values():
    sp.set_color("#D0D0D0")
    sp.set_linewidth(1.2)

orig_suc = [cos_success, ret_success]
orig_fail = [cos_failure, ret_failure]
for bar, val in zip(bars1, orig_suc):
    label = f"{val:.3f}" if val < 1 else f"{val:.1f}"
    ax2.annotate(label, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                 textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=ANNOT, fontweight="bold", color=C_SUC)
for bar, val in zip(bars2, orig_fail):
    label = f"{val:.3f}" if val < 1 else f"{val:.1f}"
    ax2.annotate(label, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                 textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=ANNOT, fontweight="bold", color=C_FAIL)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(["Cosine\nSimilarity", "Retrieval\nAcc. (%)"],
                     fontsize=TICK, fontweight="bold")
ax2.legend(fontsize=LEG, loc="upper left", frameon=True, fancybox=True,
           framealpha=0.92, edgecolor="#D0D0D0")

label_y = -0.13
ax1.text(0.5, label_y, "(a)", transform=ax1.transAxes,
         fontsize=SUB, fontweight="bold", color="#333333", ha="center", va="top")
ax2.text(0.5, label_y, "(b)", transform=ax2.transAxes,
         fontsize=SUB, fontweight="bold", color="#333333", ha="center", va="top")

fig.tight_layout(w_pad=6)
fig.subplots_adjust(bottom=0.15)
fig.savefig(f"{OUT_DIR}/diagnostic_ab.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/diagnostic_ab.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: diagnostic_ab.pdf / .png")
