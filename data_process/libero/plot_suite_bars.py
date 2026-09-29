"""
Finding 2: Success vs Failure cosine similarity bar chart
Data source: sim rollout (400 traj, layer 10, step_30k baseline)
"""
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs/results"

# sim rollout layer 10, all 40 tasks
# from eval_rollout_libero_to_sim.json + eval_offline_probe_on_sim_rollout.json
succ_cos = 0.6255
fail_cos = 0.5185
n_succ = 364
n_fail = 36
succ_std = 0.1469
fail_std = 0.2232
p_val = 9.9e-5

C_SUCC = "#5B8DB8"
C_FAIL = "#E07B54"
BG = "#FAFBFC"

labels = ["Success", "Failure"]
vals = [succ_cos, fail_cos]
stds = [succ_std, fail_std]
colors = [C_SUCC, C_FAIL]
ns = [n_succ, n_fail]

x = np.arange(len(labels))
w = 0.45

fig, ax = plt.subplots(figsize=(5.5, 6.875), facecolor="white")
ax.set_facecolor(BG)

bars = ax.bar(x, vals, w, color=colors, edgecolor="white",
              linewidth=2, zorder=3,
              yerr=[s / np.sqrt(n) for s, n in zip(stds, ns)],
              capsize=8, error_kw=dict(lw=2, capthick=2, color="#555555"))

for i, (v, n) in enumerate(zip(vals, ns)):
    ax.text(x[i], v + 0.035, f"{v:.3f}",
            ha="center", va="bottom", fontsize=22, color=colors[i], fontweight="bold")
    ax.text(x[i], v + 0.01, f"(n={n})",
            ha="center", va="bottom", fontsize=14, color="#888888")

sig_y = max(vals) + 0.08
ax.plot([0, 0, 1, 1], [sig_y - 0.005, sig_y, sig_y, sig_y - 0.005],
        lw=1.5, color="#333333")
ax.text(0.5, sig_y + 0.005, f"p < 0.001 ***",
        ha="center", va="bottom", fontsize=18, color="#333333", fontweight="bold")

ax.set_ylabel("Cosine Similarity", fontsize=24, color="#333333", labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=24)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelsize=20, colors="#333333")
ax.set_ylim(0, max(vals) + 0.14)

for spine in ax.spines.values():
    spine.set_color("#D0D0D0")
    spine.set_linewidth(0.9)
ax.grid(True, axis="y", alpha=0.3, linewidth=0.5, color="#D0D0D0", zorder=0)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/succ_fail_bars.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/succ_fail_bars.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: succ_fail_bars")
