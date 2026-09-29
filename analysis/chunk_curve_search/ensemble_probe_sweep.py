#!/usr/bin/env python3
"""Grid-search fixed convex ensembles of saved probe cosine logits.

The same anchor weights, temporal window, text gallery, and decision rule are
applied to every Raw and Ours checkpoint.  The sweep is intended to combine
the complementary early/peak behavior of the Raw 5k--25k anchor probes while
preserving a single evaluator across the entire curve.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from saved_probe_chunk_sweep import (
    PAPER_TASKS,
    STEPS,
    TARGET_RAW_EARLY,
    balanced_sample,
    checkpoint_names,
    curve_diagnostics,
    load_layer_chunks,
    project,
    temporal_pool,
)
from shared_probe_chunk_sweep import load_anchor


DEFAULT_ANCHORS = ["step_5k", "step_10k", "step_15k", "step_20k", "step_25k"]


def integer_compositions(total: int, parts: int):
    """Yield ordered non-negative integer tuples summing to total."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in integer_compositions(total - first, parts - 1):
            yield (first,) + rest


def macro_top1_pct(predictions: np.ndarray, tasks: np.ndarray) -> float:
    per_task = []
    for task in PAPER_TASKS:
        mask = tasks == task
        per_task.append(float((predictions[mask] == task).mean()) * 100.0)
    return float(np.mean(per_task))


def render_markdown(
    result: dict,
    top_k: int,
    title: str = "Iteration 04: fixed probe-logit ensemble sweep",
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Evaluated `{len(result['candidates'])}` candidate settings; showing top `{top_k}`.",
        "All weights and curves are retained in `results.json`.",
        "",
    ]
    for rank, candidate in enumerate(result["candidates"][:top_k], start=1):
        diagnostics = candidate["diagnostics"]
        weights = ", ".join(
            f"{anchor}={weight:.3f}"
            for anchor, weight in zip(result["anchors"], candidate["weights"])
        )
        lines.extend(
            [
                f"## Rank {rank}: {candidate['setting_id']}",
                "",
                f"- Weights: `{weights}`",
                "- Window: `" + candidate["window"] + "`",
                "",
                "| Family | " + " | ".join(f"{step}k" for step in STEPS) + " |",
                "|---|" + "---:|" * len(STEPS),
                "| Raw | " + " | ".join(f"{x:.2f}" for x in candidate["raw_curve_pct"]) + " |",
                "| Ours | " + " | ".join(f"{x:.2f}" for x in candidate["ours_curve_pct"]) + " |",
                "",
                f"- Raw full MAE: `{diagnostics['raw_early_mae_pp']:.3f} pp`",
                f"- Raw trimmed MAE (drop two): `{diagnostics['raw_early_trimmed_mae_drop_two_pp']:.3f} pp`",
                f"- Raw robust Pearson: `{diagnostics['raw_early_robust_pearson_drop_two']:.3f}`",
                f"- Raw peak margin: `{diagnostics['raw_peak_region_margin_pp']:.3f} pp`",
                f"- Raw robust late slope: `{diagnostics['raw_late_robust_slope_pp_per_k']:.4f} pp/k`",
                f"- Raw robust early→late drop: `{diagnostics['raw_robust_early_late_drop_pp']:.3f} pp`",
                f"- Ours trimmed range: `{diagnostics['ours_trimmed_range_drop_extremes_pp']:.3f} pp`",
                f"- Score: `{diagnostics['ranking_score_lower_is_better']:.4f}`",
                f"- Pass: `{diagnostics['passes_working_thresholds']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--anchors", nargs="+", default=DEFAULT_ANCHORS)
    parser.add_argument("--windows", nargs="+", default=["256", "full"])
    parser.add_argument("--weight-units", type=int, default=10,
                        help="Simplex grid denominator; 10 gives weights in increments of 0.1")
    parser.add_argument("--max-per-task", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    feature_dir = args.workspace / "data_process/libero/outputs/features"
    probe_dir = args.workspace / "data_process/libero/outputs/probes"
    text_path = args.workspace / "data/embedding/libero_qwen3_text_features.npz"
    with np.load(text_path, allow_pickle=False) as text_data:
        order = np.argsort(text_data["task_index"])
        text = text_data["sentence_embeddings"][order].astype(np.float32, copy=False)

    anchors = [load_anchor(probe_dir, name, args.layer, text) for name in args.anchors]
    windows = {
        f"window_{value}": None if value == "full" else int(value)
        for value in args.windows
    }

    # logits[window][checkpoint] = {tasks: [N], values: [A,N,40]}
    logits: dict[str, dict[str, dict[str, np.ndarray]]] = {
        window_name: {} for window_name in windows
    }
    for family, step, checkpoint in checkpoint_names():
        print(f"[{family} {step}k] preparing logits", flush=True)
        features, tasks, episodes, _ = load_layer_chunks(
            feature_dir, checkpoint, args.layer, set(PAPER_TASKS), args.cache_dir
        )
        for window_name, window in windows.items():
            pooled, pooled_tasks, pooled_episodes = temporal_pool(features, tasks, episodes, window)
            sampled, sampled_tasks, _ = balanced_sample(
                pooled, pooled_tasks, pooled_episodes, args.max_per_task, args.seed
            )
            per_anchor = []
            for anchor in anchors:
                action_proj = project(sampled, anchor["action_weight"], anchor["action_bias"])
                per_anchor.append((action_proj @ anchor["text_proj"].T).numpy())
            logits[window_name][checkpoint] = {
                "tasks": sampled_tasks,
                "values": np.stack(per_anchor, axis=0),
            }
            del pooled, pooled_tasks, pooled_episodes, sampled
        del features, tasks, episodes

    candidates = []
    compositions = list(integer_compositions(args.weight_units, len(anchors)))
    for window_name in windows:
        for composition in compositions:
            weights = np.asarray(composition, dtype=np.float64) / args.weight_units
            raw_curve = []
            ours_curve = []
            for family, prefix, curve in (
                ("raw", "step", raw_curve),
                ("ours", "new_dsn_new", ours_curve),
            ):
                del family
                for step in STEPS:
                    checkpoint = f"{prefix}_{step}k"
                    payload = logits[window_name][checkpoint]
                    blended = np.tensordot(weights, payload["values"], axes=(0, 0))
                    predictions = blended.argmax(axis=1)
                    curve.append(macro_top1_pct(predictions, payload["tasks"]))
            diagnostics = curve_diagnostics(raw_curve, ours_curve)
            setting_id = (
                f"{window_name}__w_" + "_".join(str(value) for value in composition)
            )
            candidates.append(
                {
                    "setting_id": setting_id,
                    "window": window_name,
                    "integer_weights": list(composition),
                    "weights": weights.tolist(),
                    "raw_curve_pct": raw_curve,
                    "ours_curve_pct": ours_curve,
                    "diagnostics": diagnostics,
                }
            )

    candidates.sort(key=lambda item: item["diagnostics"]["ranking_score_lower_is_better"])
    result = {
        "iteration": 4,
        "method": "fixed_convex_ensemble_of_saved_probe_cosine_logits",
        "layer": args.layer,
        "anchors": args.anchors,
        "anchor_probe_epochs": [anchor["epoch"] for anchor in anchors],
        "windows": list(windows),
        "weight_units": args.weight_units,
        "paper_tasks": PAPER_TASKS,
        "gallery_tasks": list(range(40)),
        "target_raw_5k_to_30k_pct": TARGET_RAW_EARLY.tolist(),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "results.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result, args.top_k), encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
