#!/usr/bin/env python3
"""Search fixed probe anchors and long chunk-pooling windows.

Unlike the per-checkpoint saved-probe sweep, every setting here chooses one
anchor probe and applies that exact action/text mapping to all Raw and Ours
checkpoints.  This removes probe-training variation (notably the current 30k
probe artifact) from the across-checkpoint curve.
"""

from __future__ import annotations

import argparse
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
    render_markdown,
    task_macro_metrics,
    temporal_pool,
)


DEFAULT_ANCHORS = [
    "pretrained",
    "step_5k",
    "step_10k",
    "step_15k",
    "step_20k",
    "step_25k",
    "step_35k",
    "step_50k",
]


def load_anchor(
    probe_dir: Path,
    checkpoint: str,
    layer: int,
    text: np.ndarray,
) -> dict:
    probe_path = (
        probe_dir
        / f"{checkpoint}_qwen3_rollout"
        / f"probe_{checkpoint}_layer{layer}.pt"
    )
    saved = torch.load(probe_path, map_location="cpu", weights_only=False)
    state = saved["model_state_dict"]
    return {
        "checkpoint": checkpoint,
        "epoch": int(saved["epoch"]),
        "action_weight": state["proj_action.weight"].float(),
        "action_bias": state["proj_action.bias"].float(),
        "text_proj": project(text, state["proj_text.weight"], state["proj_text.bias"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--anchors", nargs="+", default=DEFAULT_ANCHORS)
    parser.add_argument("--windows", nargs="+", default=["128", "160", "192", "256", "full"])
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

    anchors = {
        checkpoint: load_anchor(probe_dir, checkpoint, args.layer, text)
        for checkpoint in args.anchors
    }
    windows = {
        f"window_{value}": None if value == "full" else int(value)
        for value in args.windows
    }
    setting_names = [
        f"anchor_{anchor}__{window_name}"
        for anchor in anchors
        for window_name in windows
    ]
    result = {
        "iteration": 3,
        "method": "one_fixed_saved_probe_anchor_for_all_checkpoints",
        "layer": args.layer,
        "anchors": {
            checkpoint: {"probe_epoch": payload["epoch"]}
            for checkpoint, payload in anchors.items()
        },
        "paper_tasks": PAPER_TASKS,
        "gallery_tasks": list(range(40)),
        "max_per_task": args.max_per_task,
        "seed": args.seed,
        "target_raw_5k_to_30k_pct": TARGET_RAW_EARLY.tolist(),
        "settings": {name: {"checkpoints": {}} for name in setting_names},
    }

    for family, step, checkpoint in checkpoint_names():
        print(f"[{family} {step}k] loading cached chunks", flush=True)
        features, tasks, episodes, _ = load_layer_chunks(
            feature_dir,
            checkpoint,
            args.layer,
            set(PAPER_TASKS),
            args.cache_dir,
        )
        for window_name, window in windows.items():
            pooled, pooled_tasks, pooled_episodes = temporal_pool(features, tasks, episodes, window)
            sampled, sampled_tasks, sampled_episodes = balanced_sample(
                pooled,
                pooled_tasks,
                pooled_episodes,
                args.max_per_task,
                args.seed,
            )
            task_tensor = torch.from_numpy(sampled_tasks).long()
            for anchor_name, anchor in anchors.items():
                setting_name = f"anchor_{anchor_name}__{window_name}"
                action_proj = project(sampled, anchor["action_weight"], anchor["action_bias"])
                logits = action_proj @ anchor["text_proj"].T
                predictions = logits.argmax(dim=1).numpy()
                true_cosine = (
                    action_proj * anchor["text_proj"][task_tensor]
                ).sum(dim=1).numpy()
                metrics = task_macro_metrics(predictions, true_cosine, sampled_tasks)
                metrics.update(
                    {
                        "family": family,
                        "step_k": step,
                        "checkpoint": checkpoint,
                        "anchor_probe": anchor_name,
                        "anchor_probe_epoch": anchor["epoch"],
                        "n_before_sampling": int(len(pooled)),
                        "n_after_sampling": int(len(sampled)),
                        "n_episodes_after_sampling": int(len(np.unique(sampled_episodes))),
                    }
                )
                result["settings"][setting_name]["checkpoints"][checkpoint] = metrics
                del action_proj, logits
            print(
                f"  {window_name}: n={len(sampled)} evaluated {len(anchors)} anchors",
                flush=True,
            )
            del pooled, pooled_tasks, pooled_episodes, sampled
        del features, tasks, episodes

    for setting_name, payload in result["settings"].items():
        raw = [
            payload["checkpoints"][f"step_{step}k"]["macro_top1_pct"]
            for step in STEPS
        ]
        ours = [
            payload["checkpoints"][f"new_dsn_new_{step}k"]["macro_top1_pct"]
            for step in STEPS
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
