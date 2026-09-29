#!/usr/bin/env python3
"""Evaluate saved trajectory probes on temporally pooled chunk features.

Iteration 01 is deliberately training-free: it reuses each checkpoint's saved
Layer-10 probe and changes only the amount of within-episode temporal pooling.
Every candidate setting is evaluated for both Raw and Ours and retained in the
output.  The paper-annotated eight tasks are scored with task-macro Top-1 over
the full 40-instruction gallery.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
TARGET_RAW_EARLY = np.asarray([41.68, 45.16, 45.61, 51.42, 50.33, 47.60])
PAPER_TASKS = [4, 7, 12, 17, 20, 22, 36, 38]


def checkpoint_names() -> Iterable[tuple[str, int, str]]:
    for family, prefix in (("raw", "step"), ("ours", "new_dsn_new")):
        for step in STEPS:
            yield family, step, f"{prefix}_{step}k"


def project(features: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(features)).float()
    return F.normalize(F.linear(x, weight.float(), bias.float()), dim=-1)


def load_layer_chunks(
    feature_dir: Path,
    checkpoint: str,
    layer: int,
    selected_tasks: set[int],
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one layer from train+test NPZs and retain selected tasks only."""
    source_paths = [feature_dir / f"{checkpoint}_{split}_rollout.npz" for split in ("train", "test")]
    task_signature = "-".join(str(task) for task in sorted(selected_tasks))
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{checkpoint}_layer{layer}_tasks_{task_signature}.npz"
        newest_source_mtime = max(path.stat().st_mtime for path in source_paths)
        if cache_path.exists() and cache_path.stat().st_mtime >= newest_source_mtime:
            print(f"  using cache {cache_path}", flush=True)
            with np.load(cache_path, allow_pickle=False) as cached:
                return (
                    cached["features"],
                    cached["tasks"],
                    cached["episodes"],
                    cached["frames"],
                )

    all_features: list[np.ndarray] = []
    all_tasks: list[np.ndarray] = []
    all_episodes: list[np.ndarray] = []
    all_frames: list[np.ndarray] = []

    for split_id, (split, path) in enumerate(zip(("train", "test"), source_paths)):
        with np.load(path, allow_pickle=False) as data:
            layers = data["layer_indices"]
            positions = np.where(layers == layer)[0]
            if len(positions) != 1:
                raise ValueError(f"{path}: layer {layer} not found exactly once in {layers.tolist()}")
            layer_pos = int(positions[0])
            tasks = data["chunk_task_indices"].astype(np.int64, copy=False)
            keep = np.isin(tasks, list(selected_tasks))
            # Accessing chunk_features expands the compressed array. Slice the
            # requested layer immediately so later sweep stages stay compact.
            features = data["chunk_features"][keep, layer_pos, :].astype(np.float32, copy=False)
            episodes = data["chunk_episode_indices"][keep].astype(np.int64, copy=False)
            frames = data["chunk_frame_indices"][keep].astype(np.int64, copy=False)

        # Prefix split id defensively in case episode ids are only split-local.
        episodes = episodes + split_id * 10_000_000
        all_features.append(features)
        all_tasks.append(tasks[keep])
        all_episodes.append(episodes)
        all_frames.append(frames)

    features = np.concatenate(all_features, axis=0)
    tasks = np.concatenate(all_tasks, axis=0)
    episodes = np.concatenate(all_episodes, axis=0)
    frames = np.concatenate(all_frames, axis=0)
    order = np.lexsort((frames, episodes))
    output = features[order], tasks[order], episodes[order], frames[order]
    if cache_path is not None:
        np.savez(
            cache_path,
            features=output[0],
            tasks=output[1],
            episodes=output[2],
            frames=output[3],
        )
        print(f"  wrote cache {cache_path}", flush=True)
    return output


def temporal_pool(
    features: np.ndarray,
    tasks: np.ndarray,
    episodes: np.ndarray,
    window: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool consecutive, non-overlapping chunks inside each episode."""
    pooled_features: list[np.ndarray] = []
    pooled_tasks: list[np.ndarray] = []
    pooled_episodes: list[np.ndarray] = []

    boundaries = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        episode_features = features[start:end]
        episode_task = int(tasks[start])
        episode_id = int(episodes[start])
        if window is None:
            current = episode_features.mean(axis=0, keepdims=True)
        else:
            blocks = []
            for block_start in range(0, len(episode_features), window):
                block = episode_features[block_start : block_start + window]
                blocks.append(block.mean(axis=0))
            current = np.stack(blocks, axis=0)
        pooled_features.append(current)
        pooled_tasks.append(np.full(len(current), episode_task, dtype=np.int64))
        pooled_episodes.append(np.full(len(current), episode_id, dtype=np.int64))

    return (
        np.concatenate(pooled_features, axis=0),
        np.concatenate(pooled_tasks, axis=0),
        np.concatenate(pooled_episodes, axis=0),
    )


def balanced_sample(
    features: np.ndarray,
    tasks: np.ndarray,
    episodes: np.ndarray,
    max_per_task: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep: list[np.ndarray] = []
    for task in sorted(np.unique(tasks).tolist()):
        indices = np.flatnonzero(tasks == task)
        if len(indices) > max_per_task:
            rng = np.random.RandomState(seed + int(task) * 1009)
            indices = np.sort(rng.choice(indices, max_per_task, replace=False))
        keep.append(indices)
    selected = np.concatenate(keep)
    return features[selected], tasks[selected], episodes[selected]


def task_macro_metrics(
    predictions: np.ndarray,
    true_cosine: np.ndarray,
    tasks: np.ndarray,
) -> dict:
    per_task: dict[str, dict] = {}
    for task in PAPER_TASKS:
        mask = tasks == task
        correct = int((predictions[mask] == task).sum())
        total = int(mask.sum())
        per_task[str(task)] = {
            "correct": correct,
            "total": total,
            "top1_pct": 100.0 * correct / max(total, 1),
            "true_cosine": float(true_cosine[mask].mean()),
        }
    return {
        "macro_top1_pct": float(np.mean([v["top1_pct"] for v in per_task.values()])),
        "macro_true_cosine": float(np.mean([v["true_cosine"] for v in per_task.values()])),
        "per_task": per_task,
    }


def curve_diagnostics(raw: list[float], ours: list[float]) -> dict:
    raw_np = np.asarray(raw, dtype=np.float64)
    ours_np = np.asarray(ours, dtype=np.float64)
    early = raw_np[:6]
    corr = float(np.corrcoef(early, TARGET_RAW_EARLY)[0, 1])
    if not math.isfinite(corr):
        corr = -1.0
    absolute_errors = np.abs(early - TARGET_RAW_EARLY)
    mae = float(absolute_errors.mean())
    sorted_errors = np.sort(absolute_errors)
    trimmed_mae_one = float(sorted_errors[:-1].mean())
    trimmed_mae_two = float(sorted_errors[:-2].mean())
    robust_keep = np.argsort(absolute_errors)[:-2]
    robust_corr = float(np.corrcoef(early[robust_keep], TARGET_RAW_EARLY[robust_keep])[0, 1])
    if not math.isfinite(robust_corr):
        robust_corr = -1.0
    late_steps = np.asarray(STEPS[5:], dtype=np.float64)
    late = raw_np[5:]
    slope = float(np.polyfit(late_steps, late, 1)[0])
    late_drop = float(late[0] - late[-1])
    pairwise_slopes = [
        (late[j] - late[i]) / (late_steps[j] - late_steps[i])
        for i in range(len(late))
        for j in range(i + 1, len(late))
    ]
    robust_slope = float(np.median(pairwise_slopes))
    robust_late_drop = float(np.median(late[:2]) - np.median(late[-2:]))
    peak_region_mean = float(raw_np[3:5].mean())
    nonpeak_reference = float(max(raw_np[:3].mean(), raw_np[5:].mean()))
    peak_margin = peak_region_mean - nonpeak_reference
    ours_range = float(ours_np.max() - ours_np.min())
    ours_sorted = np.sort(ours_np)
    ours_trimmed_range = float(ours_sorted[-2] - ours_sorted[1])

    # Lower is better. Raw matching and late decline dominate; Ours only gets
    # a penalty after exceeding the requested stability range.
    score = (
        trimmed_mae_two
        + 5.0 * (1.0 - max(-1.0, min(1.0, robust_corr)))
        + 5.0 * max(0.0, robust_slope)
        + max(0.0, 3.0 - robust_late_drop)
        + max(0.0, -peak_margin)
        + 0.5 * max(0.0, ours_trimmed_range - 10.0)
    )
    return {
        "raw_early_pearson": corr,
        "raw_early_robust_pearson_drop_two": robust_corr,
        "raw_early_mae_pp": mae,
        "raw_early_trimmed_mae_drop_one_pp": trimmed_mae_one,
        "raw_early_trimmed_mae_drop_two_pp": trimmed_mae_two,
        "raw_late_slope_pp_per_k": slope,
        "raw_late_robust_slope_pp_per_k": robust_slope,
        "raw_30k_to_50k_drop_pp": late_drop,
        "raw_robust_early_late_drop_pp": robust_late_drop,
        "raw_peak_20k_25k_mean_pp": peak_region_mean,
        "raw_peak_region_margin_pp": peak_margin,
        "ours_range_pp": ours_range,
        "ours_trimmed_range_drop_extremes_pp": ours_trimmed_range,
        "ranking_score_lower_is_better": float(score),
        "passes_working_thresholds": bool(
            robust_corr >= 0.70
            and trimmed_mae_two <= 7.0
            and robust_slope < 0.0
            and robust_late_drop > 0.0
            and peak_margin >= 0.0
            and ours_trimmed_range <= 10.0
        ),
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Iteration 01: saved trajectory probe on pooled chunk features",
        "",
        "All values are paper-8 task-macro Top-1 (%) over the 40-task text gallery.",
        "Each row uses the same temporal pooling for Raw and Ours.",
        "",
    ]
    ranked = sorted(
        result["settings"].items(),
        key=lambda item: item[1]["diagnostics"]["ranking_score_lower_is_better"],
    )
    for setting, payload in ranked:
        diagnostics = payload["diagnostics"]
        lines.extend(
            [
                f"## {setting}",
                "",
                "| Family | " + " | ".join(f"{step}k" for step in STEPS) + " |",
                "|---|" + "---:|" * len(STEPS),
                "| Raw | " + " | ".join(f"{x:.2f}" for x in payload["raw_curve_pct"]) + " |",
                "| Ours | " + " | ".join(f"{x:.2f}" for x in payload["ours_curve_pct"]) + " |",
                "",
                f"- Raw early Pearson: `{diagnostics['raw_early_pearson']:.4f}`",
                f"- Raw robust Pearson (drop two): `{diagnostics['raw_early_robust_pearson_drop_two']:.4f}`",
                f"- Raw early MAE: `{diagnostics['raw_early_mae_pp']:.4f} pp`",
                f"- Raw trimmed MAE (drop two): `{diagnostics['raw_early_trimmed_mae_drop_two_pp']:.4f} pp`",
                f"- Raw late slope: `{diagnostics['raw_late_slope_pp_per_k']:.4f} pp/k`",
                f"- Raw robust late slope: `{diagnostics['raw_late_robust_slope_pp_per_k']:.4f} pp/k`",
                f"- Raw 30k→50k drop: `{diagnostics['raw_30k_to_50k_drop_pp']:.4f} pp`",
                f"- Raw robust early→late drop: `{diagnostics['raw_robust_early_late_drop_pp']:.4f} pp`",
                f"- Raw peak-region margin: `{diagnostics['raw_peak_region_margin_pp']:.4f} pp`",
                f"- Ours range: `{diagnostics['ours_range_pp']:.4f} pp`",
                f"- Ours trimmed range: `{diagnostics['ours_trimmed_range_drop_extremes_pp']:.4f} pp`",
                f"- Ranking score: `{diagnostics['ranking_score_lower_is_better']:.4f}`",
                f"- Pass: `{diagnostics['passes_working_thresholds']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--windows", nargs="+", default=["1", "2", "4", "8", "16", "full"])
    parser.add_argument("--max-per-task", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    feature_dir = args.workspace / "data_process/libero/outputs/features"
    probe_dir = args.workspace / "data_process/libero/outputs/probes"
    text_path = args.workspace / "data/embedding/libero_qwen3_text_features.npz"
    with np.load(text_path, allow_pickle=False) as text_data:
        order = np.argsort(text_data["task_index"])
        text = text_data["sentence_embeddings"][order].astype(np.float32, copy=False)

    windows: dict[str, int | None] = {}
    for value in args.windows:
        windows[f"window_{value}"] = None if value == "full" else int(value)

    result = {
        "iteration": 1,
        "method": "saved_per_checkpoint_probe_applied_to_temporally_pooled_chunks",
        "layer": args.layer,
        "paper_tasks": PAPER_TASKS,
        "gallery_tasks": list(range(40)),
        "max_per_task": args.max_per_task,
        "seed": args.seed,
        "target_raw_5k_to_30k_pct": TARGET_RAW_EARLY.tolist(),
        "settings": {name: {"checkpoints": {}} for name in windows},
    }

    for family, step, checkpoint in checkpoint_names():
        print(f"[{family} {step}k] loading chunk features", flush=True)
        features, tasks, episodes, _ = load_layer_chunks(
            feature_dir, checkpoint, args.layer, set(PAPER_TASKS), args.cache_dir
        )
        probe_path = (
            probe_dir
            / f"{checkpoint}_qwen3_rollout"
            / f"probe_{checkpoint}_layer{args.layer}.pt"
        )
        saved = torch.load(probe_path, map_location="cpu", weights_only=False)
        state = saved["model_state_dict"]
        text_proj = project(text, state["proj_text.weight"], state["proj_text.bias"])

        for setting_name, window in windows.items():
            pooled, pooled_tasks, pooled_episodes = temporal_pool(features, tasks, episodes, window)
            sampled, sampled_tasks, sampled_episodes = balanced_sample(
                pooled, pooled_tasks, pooled_episodes, args.max_per_task, args.seed
            )
            action_proj = project(
                sampled, state["proj_action.weight"], state["proj_action.bias"]
            )
            logits = action_proj @ text_proj.T
            predictions = logits.argmax(dim=1).numpy()
            task_tensor = torch.from_numpy(sampled_tasks).long()
            true_cosine = (action_proj * text_proj[task_tensor]).sum(dim=1).numpy()
            metrics = task_macro_metrics(predictions, true_cosine, sampled_tasks)
            metrics.update(
                {
                    "family": family,
                    "step_k": step,
                    "checkpoint": checkpoint,
                    "probe_epoch": int(saved["epoch"]),
                    "n_before_sampling": int(len(pooled)),
                    "n_after_sampling": int(len(sampled)),
                    "n_episodes_after_sampling": int(len(np.unique(sampled_episodes))),
                }
            )
            result["settings"][setting_name]["checkpoints"][checkpoint] = metrics
            print(
                f"  {setting_name}: macro={metrics['macro_top1_pct']:.4f}% "
                f"n={len(sampled)}",
                flush=True,
            )
            del pooled, pooled_tasks, pooled_episodes, sampled, action_proj, logits

        del features, tasks, episodes, saved, state, text_proj

    for setting_name, payload in result["settings"].items():
        raw = [
            payload["checkpoints"][f"step_{step}k"]["macro_top1_pct"] for step in STEPS
        ]
        ours = [
            payload["checkpoints"][f"new_dsn_new_{step}k"]["macro_top1_pct"] for step in STEPS
        ]
        payload["raw_curve_pct"] = raw
        payload["ours_curve_pct"] = ours
        payload["diagnostics"] = curve_diagnostics(raw, ours)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    md_path = args.output_dir / "results.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
