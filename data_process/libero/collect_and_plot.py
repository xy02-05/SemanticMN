"""
收集所有 probe 和 CKA 结果，生成 4 张表 + 4 张曲线图
- 表1: 轨迹级 32/8 probe top-1
- 表2: chunk级 32/8 probe top-1
- 表3: 轨迹级 30/Goal10 suite probe top-1
- 表4: 轨迹级 40-task CKA
每张表对应一张曲线图: ours (蓝) + baseline (灰) + pretrained (虚线)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt

PROBE_DIR = "outputs/probes"
RESULT_DIR = "outputs/results"
os.makedirs(RESULT_DIR, exist_ok=True)

STEPS = [5, 10, 15, 20, 25, 30]  # 千步

# checkpoint 命名映射
BASELINE_NAMES = {s: f"step_{s}k" for s in STEPS}
OURS_NAMES = {s: f"new_dsn_new_{s}k" for s in STEPS}

BLUE = "#6BAED6"
GRAY = "#888888"


def get_probe_top1(ckpt_name, probe_type, layer=10):
    """从 probe_results.json 读取 layer 10 的 test top-1
    probe_type: 'rollout' | 'rollout_chunk' | 'suite_goal_rollout'
    """
    if probe_type == "suite_goal_rollout":
        path = os.path.join(PROBE_DIR, f"{ckpt_name}_qwen3_suite_goal_rollout/probe_results.json")
    elif probe_type == "rollout_chunk":
        path = os.path.join(PROBE_DIR, f"{ckpt_name}_qwen3_rollout_chunk/probe_results.json")
    else:
        path = os.path.join(PROBE_DIR, f"{ckpt_name}_qwen3_rollout/probe_results.json")

    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    l = d.get("per_layer_results", {}).get(str(layer), {})
    tm = l.get("test_metrics", {})
    return tm.get("top1")


def get_cka(ckpt_name):
    """从 metrics_multi_subset.json 读取 all_40 trajectory qwen3 CKA"""
    path = os.path.join(RESULT_DIR, "metrics_multi_subset.json")
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    for r in data:
        if (r["checkpoint"] == ckpt_name and r["subset"] == "all_40"
                and r["granularity"] == "trajectory" and r["text_type"] == "qwen3"):
            return r["cka"]
    return None


def collect_table(value_fn):
    """收集一张表的数据: pretrained + baseline 5k-30k + ours 5k-30k"""
    pretrained = value_fn("pretrained")
    baseline = {s: value_fn(BASELINE_NAMES[s]) for s in STEPS}
    ours = {s: value_fn(OURS_NAMES[s]) for s in STEPS}
    return pretrained, baseline, ours


def plot_curve(pretrained, baseline, ours, title, ylabel, out_name):
    """绘制曲线图: 2条线 + pretrained 虚线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # pretrained 水平虚线
    if pretrained is not None:
        ax.axhline(y=pretrained, color="black", linestyle="--", linewidth=1.5,
                   label=f"Pretrained ({pretrained:.3f})", alpha=0.6)

    # baseline 曲线
    bx = [s for s in STEPS if baseline[s] is not None]
    by = [baseline[s] for s in bx]
    if bx:
        ax.plot(bx, by, "o-", color=GRAY, linewidth=2.5, markersize=8,
                label="Vanilla FT", zorder=3)
        for x, y in zip(bx, by):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=GRAY)

    # ours 曲线
    ox = [s for s in STEPS if ours[s] is not None]
    oy = [ours[s] for s in ox]
    if ox:
        ax.plot(ox, oy, "s-", color=BLUE, linewidth=2.5, markersize=8,
                label="Ours", zorder=3)
        for x, y in zip(ox, oy):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=BLUE)

    ax.set_xlabel("Training Steps (k)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{s}k" for s in STEPS])
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, out_name)
    plt.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}.png/.pdf")


def format_val(v):
    """格式化数值: None → '—', float → 百分比或小数"""
    if v is None:
        return "—"
    return f"{v:.4f}"


def format_pct(v):
    if v is None:
        return "—"
    return f"{v*100:.2f}%"


def main():
    # 定义4张表的数据获取函数
    tables = [
        ("Trajectory 32/8 Probe Top-1",
         lambda name: get_probe_top1(name, "rollout"),
         "Probe Top-1", "traj_32_8_probe", True),
        ("Chunk 32/8 Probe Top-1",
         lambda name: get_probe_top1(name, "rollout_chunk"),
         "Probe Top-1", "chunk_32_8_probe", True),
        ("Trajectory Suite (30/Goal10) Probe Top-1",
         lambda name: get_probe_top1(name, "suite_goal_rollout"),
         "Probe Top-1", "suite_goal_probe", True),
        ("Trajectory 40-task CKA",
         lambda name: get_cka(name),
         "CKA", "traj_40task_cka", False),
    ]

    md_lines = ["# 四表汇总: Probe Top-1 + CKA (Layer 10, Rollout, Qwen3)\n"]
    md_lines.append("> Ours = new_dsn_new (alignment + LoRA)\n")
    md_lines.append("> Baseline = step (vanilla FT)\n\n---\n")

    for title, value_fn, ylabel, out_name, is_pct in tables:
        pretrained, baseline, ours = collect_table(value_fn)
        fmt = format_pct if is_pct else format_val

        md_lines.append(f"\n## {title}\n")
        md_lines.append("| Step | Baseline | Ours |")
        md_lines.append("|------|----------|------|")
        md_lines.append(f"| pretrained | {fmt(pretrained)} | — |")
        for s in STEPS:
            md_lines.append(f"| {s}k | {fmt(baseline[s])} | {fmt(ours[s])} |")
        md_lines.append("")

        # 绘制曲线图
        print(f"\n{title}")
        plot_curve(pretrained, baseline, ours, title, ylabel, out_name)

        # 统计缺失
        missing_b = [s for s in STEPS if baseline[s] is None]
        missing_o = [s for s in STEPS if ours[s] is None]
        if missing_b or missing_o:
            note = f"> 缺失: baseline {missing_b}, ours {missing_o}\n"
            md_lines.append(note)

    # 写 markdown
    md_path = os.path.join(RESULT_DIR, "four_tables.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
