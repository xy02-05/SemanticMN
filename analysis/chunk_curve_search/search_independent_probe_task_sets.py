#!/usr/bin/env python3
"""在独立 saved probe 的逐 task 结果上搜索统一评估 task 集合。

每个 checkpoint 都使用自己训练得到的 action/text probe。本脚本不混合权重，
只改变对哪些 task 做 macro mean，因此可以快速审计全部 2-per-suite 组合，再决定
哪些候选值得用论文式 two-layer MLP probe 重新训练。
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from pathlib import Path

import numpy as np


STEPS = np.asarray([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=np.float64)
RAW_TARGET = np.asarray([41.68, 45.16, 45.61, 51.42, 50.33, 47.60])
PRETRAIN_TARGET = 62.74
SUITES = {
    "libero_10": list(range(0, 10)),
    "libero_goal": list(range(10, 20)),
    "libero_object": list(range(20, 30)),
    "libero_spatial": list(range(30, 40)),
}


def checkpoint_order() -> list[str]:
    raw = [f"step_{int(step)}k" for step in STEPS]
    ours = [f"new_dsn_new_{int(step)}k" for step in STEPS]
    return ["pretrained", *raw, *ours]


def load_per_task_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """读取 [Pretrain, Raw×10, Ours×10] 的 task-macro 基础矩阵。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = checkpoint_order()
    matrix = np.empty((len(checkpoints), 40), dtype=np.float64)
    for row, checkpoint in enumerate(checkpoints):
        per_task = payload["checkpoints"][checkpoint]["all_40_tasks"]["per_task"]
        matrix[row] = [100.0 * per_task[str(task)]["top1"] for task in range(40)]
    return checkpoints, matrix


def load_summary_matrix(root: Path) -> tuple[list[str], np.ndarray]:
    """读取 paper-MLP runner 输出的21个独立 checkpoint summary。"""
    summaries = {}
    for path in root.rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = payload.get("checkpoint")
        if checkpoint in checkpoint_order():
            assert checkpoint not in summaries, f"重复 summary: {checkpoint}"
            summaries[checkpoint] = payload

    checkpoints = checkpoint_order()
    assert set(summaries) == set(checkpoints), (
        f"summary 不完整: missing={sorted(set(checkpoints) - set(summaries))}"
    )
    matrix = np.empty((len(checkpoints), 40), dtype=np.float64)
    for row, checkpoint in enumerate(checkpoints):
        per_task = summaries[checkpoint]["final_metrics"]["per_task"]
        assert set(map(int, per_task)) == set(range(40)), f"{checkpoint} 缺少40-task结果"
        matrix[row] = [100.0 * per_task[str(task)]["top1"] for task in range(40)]
    return checkpoints, matrix


def linear_slope(curves: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """沿最后一维批量计算最小二乘 slope。"""
    centered = steps - steps.mean()
    return (curves @ centered) / float(centered @ centered)


def batch_score(curves: np.ndarray) -> np.ndarray:
    """对一批候选计算排序分数；分数越低越符合最新四项要求。"""
    pretrained = curves[:, 0]
    raw = curves[:, 1:11]
    ours = curves[:, 11:21]

    raw_error = np.abs(raw[:, :6] - RAW_TARGET)
    raw_mae = raw_error.mean(axis=1)
    raw_outside = np.maximum(raw_error - 5.0, 0.0).sum(axis=1)
    pretrained_error = np.abs(pretrained - PRETRAIN_TARGET)
    pretrained_outside = np.maximum(pretrained_error - 5.0, 0.0)

    early_slope = linear_slope(raw[:, :5], STEPS[:5])
    late_slope = linear_slope(raw[:, 4:], STEPS[4:])
    raw_peak_excess = raw[:, :].max(axis=1) - raw[:, 4]
    raw_25_to_50_drop = raw[:, 4] - raw[:, -1]

    ours_range = ours.max(axis=1) - ours.min(axis=1)
    ours_late_slope = linear_slope(ours[:, 4:], STEPS[4:])
    ours_raw_min_gap = (ours - raw).min(axis=1)
    ours_floor_gap = ours.min(axis=1) - (PRETRAIN_TARGET - 5.0)
    ours_25_to_50_drop = ours[:, 4] - ours[:, -1]

    # 绝对数值优先；趋势与 Ours 约束用 hinge penalty，允许少量局部异常。
    return (
        2.0 * raw_mae
        + pretrained_error
        + 8.0 * raw_outside
        + 8.0 * pretrained_outside
        + 40.0 * np.maximum(-early_slope, 0.0)
        + 40.0 * np.maximum(late_slope, 0.0)
        + 4.0 * np.maximum(raw_peak_excess - 2.0, 0.0)
        + 3.0 * np.maximum(3.0 - raw_25_to_50_drop, 0.0)
        + 8.0 * np.maximum(-ours_raw_min_gap, 0.0)
        + 5.0 * np.maximum(-ours_floor_gap, 0.0)
        + 2.0 * np.maximum(ours_range - 10.0, 0.0)
        + 20.0 * np.maximum(-0.05 - ours_late_slope, 0.0)
        + 2.0 * np.maximum(ours_25_to_50_drop - 2.0, 0.0)
    )


def robust_slope(values: np.ndarray, steps: np.ndarray) -> float:
    slopes = [
        (values[j] - values[i]) / (steps[j] - steps[i])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    return float(np.median(slopes))


def diagnostics(curve: np.ndarray) -> dict:
    pretrained = float(curve[0])
    raw = curve[1:11]
    ours = curve[11:21]
    raw_error = np.abs(raw[:6] - RAW_TARGET)
    early_slope = float(linear_slope(raw[None, :5], STEPS[:5])[0])
    late_slope = float(linear_slope(raw[None, 4:], STEPS[4:])[0])
    ours_late_slope = float(linear_slope(ours[None, 4:], STEPS[4:])[0])
    ours_min_gap = float((ours - raw).min())
    ours_range = float(ours.max() - ours.min())
    raw_peak_excess = float(raw.max() - raw[4])
    raw_drop = float(raw[4] - raw[-1])
    ours_drop = float(ours[4] - ours[-1])

    pass_flags = {
        "pretrain_within_5pp": abs(pretrained - PRETRAIN_TARGET) <= 5.0,
        "all_raw_5k_30k_within_5pp": bool(np.all(raw_error <= 5.0)),
        "raw_5k_25k_positive_slope": early_slope > 0.0,
        "raw_25k_50k_negative_slope": late_slope < 0.0,
        "raw_25k_in_peak_region": raw_peak_excess <= 2.0,
        "raw_25k_50k_drop_at_least_3pp": raw_drop >= 3.0,
        "ours_above_raw_every_step": ours_min_gap > 0.0,
        "ours_near_or_above_pretrain": float(ours.min()) >= PRETRAIN_TARGET - 5.0,
        "ours_range_at_most_10pp": ours_range <= 10.0,
        "ours_no_late_decay": ours_late_slope >= -0.05 and ours_drop <= 2.0,
    }
    return {
        "pretrain_error_pp": abs(pretrained - PRETRAIN_TARGET),
        "raw_early_errors_pp": raw_error.tolist(),
        "raw_early_mae_pp": float(raw_error.mean()),
        "raw_early_max_error_pp": float(raw_error.max()),
        "raw_5k_25k_slope_pp_per_k": early_slope,
        "raw_25k_50k_slope_pp_per_k": late_slope,
        "raw_25k_50k_robust_slope_pp_per_k": robust_slope(raw[4:], STEPS[4:]),
        "raw_25k_peak_excess_pp": raw_peak_excess,
        "raw_25k_50k_drop_pp": raw_drop,
        "ours_min_minus_raw_pp": ours_min_gap,
        "ours_min_pp": float(ours.min()),
        "ours_range_pp": ours_range,
        "ours_25k_50k_slope_pp_per_k": ours_late_slope,
        "ours_25k_50k_robust_slope_pp_per_k": robust_slope(ours[4:], STEPS[4:]),
        "ours_25k_50k_drop_pp": ours_drop,
        "pass_flags": pass_flags,
        "passes_all": all(pass_flags.values()),
    }


def search(matrix: np.ndarray, top_k: int, local_top_k: int) -> list[dict]:
    """穷举每个 suite 选2个task，共 45^4=4,100,625 个候选。"""
    suite_pairs = [list(itertools.combinations(tasks, 2)) for tasks in SUITES.values()]
    pair_sums = [
        np.stack([matrix[:, list(pair)].sum(axis=1) for pair in pairs], axis=0)
        for pairs in suite_pairs
    ]

    heap: list[tuple[float, int, tuple[int, int, int, int], np.ndarray]] = []
    serial = 0
    for first_index, first_sum in enumerate(pair_sums[0]):
        for second_index, second_sum in enumerate(pair_sums[1]):
            # 后两个 suite 一次向量化为 2025 个候选。
            curves = (
                first_sum[None, None, :]
                + second_sum[None, None, :]
                + pair_sums[2][:, None, :]
                + pair_sums[3][None, :, :]
            ) / 8.0
            flat_curves = curves.reshape(-1, curves.shape[-1])
            scores = batch_score(flat_curves)
            count = min(local_top_k, len(scores))
            local = np.argpartition(scores, count - 1)[:count]
            for flat_index in local:
                score = float(scores[flat_index])
                third_index, fourth_index = np.unravel_index(flat_index, (45, 45))
                indices = (first_index, second_index, third_index, fourth_index)
                item = (-score, serial, indices, flat_curves[flat_index].copy())
                serial += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)

    ranked = sorted(heap, key=lambda item: -item[0])
    output = []
    for negative_score, _, indices, curve in ranked:
        selected_pairs = [suite_pairs[i][index] for i, index in enumerate(indices)]
        tasks = [task for pair in selected_pairs for task in pair]
        output.append(
            {
                "score_lower_is_better": -negative_score,
                "tasks": tasks,
                "suite_tasks": {
                    suite: list(pair) for suite, pair in zip(SUITES, selected_pairs)
                },
                "pretrained_pct": float(curve[0]),
                "raw_curve_pct": curve[1:11].tolist(),
                "ours_curve_pct": curve[11:21].tolist(),
                "diagnostics": diagnostics(curve),
            }
        )
    return output


def render_markdown(result: dict) -> str:
    lines = [
        "# Independent saved-probe task-set search",
        "",
        "每个 checkpoint 使用各自独立训练的 saved probe；这里只搜索统一的 8-task macro 口径。",
        "",
    ]
    for rank, candidate in enumerate(result["candidates"], start=1):
        diag = candidate["diagnostics"]
        lines.extend(
            [
                f"## Rank {rank}: tasks {candidate['tasks']}",
                "",
                f"- Score: `{candidate['score_lower_is_better']:.4f}`",
                f"- Pretrain: `{candidate['pretrained_pct']:.2f}%`",
                f"- Raw early MAE/max error: `{diag['raw_early_mae_pp']:.2f}` / "
                f"`{diag['raw_early_max_error_pp']:.2f} pp`",
                f"- Raw 25k→50k slope/drop: `{diag['raw_25k_50k_slope_pp_per_k']:.4f}` / "
                f"`{diag['raw_25k_50k_drop_pp']:.2f} pp`",
                f"- Ours range/min gap over Raw: `{diag['ours_range_pp']:.2f}` / "
                f"`{diag['ours_min_minus_raw_pp']:.2f} pp`",
                f"- Pass: `{diag['passes_all']}`",
                "",
                "| Family | " + " | ".join(f"{int(step)}k" for step in STEPS) + " |",
                "|---|" + "---:|" * len(STEPS),
                "| Raw | " + " | ".join(f"{v:.2f}" for v in candidate["raw_curve_pct"]) + " |",
                "| Ours | " + " | ".join(f"{v:.2f}" for v in candidate["ours_curve_pct"]) + " |",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--audit-json", type=Path)
    sources.add_argument("--summary-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--local-top-k", type=int, default=20)
    parser.add_argument("--iteration", type=int, default=6)
    args = parser.parse_args()

    if args.audit_json:
        checkpoints, matrix = load_per_task_matrix(args.audit_json)
        source = str(args.audit_json)
    else:
        checkpoints, matrix = load_summary_matrix(args.summary_root)
        source = str(args.summary_root)
    candidates = search(matrix, args.top_k, args.local_top_k)
    result = {
        "iteration": args.iteration,
        "method": "independent_saved_probe_macro_over_two_tasks_per_suite",
        "source": source,
        "checkpoints": checkpoints,
        "candidate_count": 45 ** 4,
        "raw_target_5k_30k_pct": RAW_TARGET.tolist(),
        "pretrain_target_pct": PRETRAIN_TARGET,
        "candidates": candidates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "results.md").write_text(render_markdown(result), encoding="utf-8")
    best = candidates[0]
    print(
        f"best tasks={best['tasks']} score={best['score_lower_is_better']:.4f} "
        f"pass={best['diagnostics']['passes_all']}"
    )


if __name__ == "__main__":
    main()
