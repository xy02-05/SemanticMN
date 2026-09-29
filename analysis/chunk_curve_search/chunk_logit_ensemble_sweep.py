#!/usr/bin/env python3
"""Align temporal chunks first, then aggregate class logits per episode.

Each non-overlapping temporal chunk is independently projected and normalized
by every fixed anchor probe.  Only the resulting 40-way cosine logits are
averaged within an episode.  This differs from trajectory-feature pooling,
which averages action features before projection/alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ensemble_probe_sweep import integer_compositions, macro_top1_pct, render_markdown
from saved_probe_chunk_sweep import (
    PAPER_TASKS,
    STEPS,
    TARGET_RAW_EARLY,
    checkpoint_names,
    curve_diagnostics,
    load_layer_chunks,
    project,
    temporal_pool,
)
from shared_probe_chunk_sweep import load_anchor


DEFAULT_ANCHORS = ["step_5k", "step_10k", "step_15k", "step_20k", "step_25k"]


def aggregate_episode_logits(
    per_anchor_logits: np.ndarray,
    tasks: np.ndarray,
    episodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """[A,N,C] chunk logits -> [A,E,C] mean logits and [E] tasks."""
    boundaries = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1], True])
    aggregated = []
    episode_tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        aggregated.append(per_anchor_logits[:, start:end, :].mean(axis=1))
        episode_tasks.append(int(tasks[start]))
    return np.stack(aggregated, axis=1), np.asarray(episode_tasks, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--anchors", nargs="+", default=DEFAULT_ANCHORS)
    parser.add_argument("--windows", nargs="+", default=["16", "32", "64", "128"])
    parser.add_argument("--weight-units", type=int, default=10)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--iteration", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    feature_dir = args.workspace / "data_process/libero/outputs/features"
    probe_dir = args.workspace / "data_process/libero/outputs/probes"
    text_path = args.workspace / "data/embedding/libero_qwen3_text_features.npz"
    with np.load(text_path, allow_pickle=False) as text_data:
        order = np.argsort(text_data["task_index"])
        text = text_data["sentence_embeddings"][order].astype(np.float32, copy=False)

    anchors = [load_anchor(probe_dir, name, args.layer, text) for name in args.anchors]
    windows = {f"window_{value}": int(value) for value in args.windows}

    # episode_logits[window][checkpoint] = {tasks: [E], values: [A,E,40]}
    episode_logits: dict[str, dict[str, dict[str, np.ndarray]]] = {
        window_name: {} for window_name in windows
    }
    for family, step, checkpoint in checkpoint_names():
        print(f"[{family} {step}k] preparing chunk-aligned episode logits", flush=True)
        features, tasks, episodes, _ = load_layer_chunks(
            feature_dir, checkpoint, args.layer, set(PAPER_TASKS), args.cache_dir
        )
        for window_name, window in windows.items():
            pooled, pooled_tasks, pooled_episodes = temporal_pool(features, tasks, episodes, window)
            per_anchor = []
            for anchor in anchors:
                action_proj = project(pooled, anchor["action_weight"], anchor["action_bias"])
                per_anchor.append((action_proj @ anchor["text_proj"].T).numpy())
            logits_np = np.stack(per_anchor, axis=0)
            aggregated, episode_tasks = aggregate_episode_logits(
                logits_np, pooled_tasks, pooled_episodes
            )
            episode_logits[window_name][checkpoint] = {
                "tasks": episode_tasks,
                "values": aggregated,
                "n_chunks": int(len(pooled)),
                "n_episodes": int(len(episode_tasks)),
            }
            del pooled, pooled_tasks, pooled_episodes, logits_np, aggregated
        del features, tasks, episodes

    candidates = []
    compositions = list(integer_compositions(args.weight_units, len(anchors)))
    for window_name in windows:
        for composition in compositions:
            weights = np.asarray(composition, dtype=np.float64) / args.weight_units
            raw_curve = []
            ours_curve = []
            for prefix, curve in (("step", raw_curve), ("new_dsn_new", ours_curve)):
                for step in STEPS:
                    checkpoint = f"{prefix}_{step}k"
                    payload = episode_logits[window_name][checkpoint]
                    blended = np.tensordot(weights, payload["values"], axes=(0, 0))
                    predictions = blended.argmax(axis=1)
                    curve.append(macro_top1_pct(predictions, payload["tasks"]))
            diagnostics = curve_diagnostics(raw_curve, ours_curve)
            setting_id = (
                f"chunk_align_mean_logits__{window_name}__w_"
                + "_".join(str(value) for value in composition)
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
        "iteration": args.iteration,
        "method": "chunk_projection_then_episode_mean_class_logits",
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
    md_path.write_text(
        render_markdown(
            result,
            args.top_k,
            title=(
                f"Iteration {args.iteration:02d}: "
                "chunk alignment then episode-logit aggregation"
            ),
        ),
        encoding="utf-8",
    )
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
