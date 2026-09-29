#!/usr/bin/env python3
"""按论文附录协议独立训练 action/text 两个两层投影头。

默认把原有 train/test 特征重新合并，再按给定的 8 个 task 做 task-disjoint
划分。每个 checkpoint 都从相同种子重新初始化，固定训练 optimizer steps，
并报告最后一步结果，不根据测试集挑选 best checkpoint。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FEATURE_DIRS = [
    Path(
        "/mnt/bn/2d-videos/xy/work/mirror_neuron/"
        "data_process/libero/outputs/features"
    ),
    Path("/opt/tiger/rh2/rh2/init/rebuttal_50k_rollout_features"),
]
DEFAULT_TEXT_PATH = Path(
    "/mnt/bn/2d-videos/xy/work/mirror_neuron/"
    "data/embedding/libero_qwen3_text_features.npz"
)
DEFAULT_EVAL_TASKS = [4, 7, 12, 17, 20, 22, 36, 38]


class ProjectionHead(nn.Module):
    """论文所述 two-layer MLP projector；中间维度沿用训练配置的 1024。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class PaperMLPProbe(nn.Module):
    """Raw/Ours 各自独立实例化，action 与 text 投影头也没有共享参数。"""

    def __init__(
        self,
        action_dim: int,
        text_dim: int,
        hidden_dim: int,
        output_dim: int,
        initial_temperature: float,
    ):
        super().__init__()
        self.action_projector = ProjectionHead(action_dim, hidden_dim, output_dim)
        self.text_projector = ProjectionHead(text_dim, hidden_dim, output_dim)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / initial_temperature), dtype=torch.float32)
        )

    def encode_action(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.action_projector(inputs), dim=-1)

    def encode_text(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projector(inputs), dim=-1)

    def temperature(self) -> torch.Tensor:
        return 1.0 / self.logit_scale.exp().clamp(max=100.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_feature_path(
    checkpoint: str,
    split: str,
    suffix: str,
    feature_dirs: list[Path],
) -> Path:
    filename = f"{checkpoint}_{split}{suffix}.npz"
    matches = [directory / filename for directory in feature_dirs if (directory / filename).is_file()]
    assert matches, f"找不到特征文件: {filename}; searched={feature_dirs}"
    return matches[0]


def load_layer(path: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """只读取轨迹特征，避免把同一 npz 内的大型 chunk 数组载入内存。"""
    with np.load(path) as payload:
        layer_indices = payload["layer_indices"].astype(np.int64)
        positions = np.flatnonzero(layer_indices == layer)
        assert len(positions) == 1, f"{path} 中 Layer {layer} 不唯一或不存在"
        features = payload["features"][:, int(positions[0]), :].astype(np.float32)
        tasks = payload["task_indices"].astype(np.int64)
    assert len(features) == len(tasks), f"feature/task 数量不一致: {path}"
    return features, tasks


def load_probe_data(
    checkpoint: str,
    feature_dirs: list[Path],
    suffix: str,
    layer: int,
    eval_tasks: list[int],
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    train_path = resolve_feature_path(checkpoint, "train", suffix, feature_dirs)
    test_path = resolve_feature_path(checkpoint, "test", suffix, feature_dirs)
    old_train_features, old_train_tasks = load_layer(train_path, layer)
    old_test_features, old_test_tasks = load_layer(test_path, layer)

    all_features = np.concatenate([old_train_features, old_test_features], axis=0)
    all_tasks = np.concatenate([old_train_tasks, old_test_tasks], axis=0)
    eval_mask = np.isin(all_tasks, np.asarray(eval_tasks, dtype=np.int64))
    assert eval_mask.any(), "评估划分为空"
    assert set(np.unique(all_tasks[eval_mask])) == set(eval_tasks), "评估 task 特征不完整"

    if split_mode == "task_disjoint":
        assert (~eval_mask).any(), "task-disjoint 训练划分为空"
        train_features = all_features[~eval_mask]
        train_tasks = all_tasks[~eval_mask]
    else:
        # 复现旧数值时保留原始32-task训练集；评估仍取合并后的统一8-task macro。
        train_features = old_train_features
        train_tasks = old_train_tasks

    return (
        train_features,
        train_tasks,
        all_features[eval_mask],
        all_tasks[eval_mask],
        {"original_train": str(train_path), "original_test": str(test_path)},
    )


def load_text_gallery(path: Path, key: str) -> np.ndarray:
    with np.load(path) as payload:
        gallery = payload[key].astype(np.float32)
    assert gallery.ndim == 2 and len(gallery) == 40, f"text gallery 形状异常: {gallery.shape}"
    return gallery


def symmetric_infonce(
    model: PaperMLPProbe,
    action: torch.Tensor,
    text: torch.Tensor,
) -> torch.Tensor:
    action_embedding = model.encode_action(action)
    text_embedding = model.encode_text(text)
    scale = model.logit_scale.exp().clamp(max=100.0)
    logits = scale * action_embedding @ text_embedding.T
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


@torch.no_grad()
def evaluate(
    model: PaperMLPProbe,
    action: torch.Tensor,
    tasks: torch.Tensor,
    gallery: torch.Tensor,
    eval_tasks: list[int],
    batch_size: int,
) -> dict:
    """使用完整 40-task text gallery，完整遍历测试集后同时报告 macro/micro。"""
    model.eval()
    text_embedding = model.encode_text(gallery)
    predictions: list[torch.Tensor] = []
    cosines: list[torch.Tensor] = []
    for start in range(0, len(action), batch_size):
        action_batch = model.encode_action(action[start : start + batch_size])
        task_batch = tasks[start : start + batch_size]
        similarities = action_batch @ text_embedding.T
        predictions.append(similarities.argmax(dim=-1))
        cosines.append((action_batch * text_embedding[task_batch]).sum(dim=-1))

    prediction = torch.cat(predictions)
    cosine = torch.cat(cosines)
    correct = prediction.eq(tasks)
    per_task = {}
    for task in eval_tasks:
        mask = tasks.eq(task)
        per_task[str(task)] = {
            "correct": int(correct[mask].sum().item()),
            "total": int(mask.sum().item()),
            "top1": float(correct[mask].float().mean().item()),
            "mean_cosine": float(cosine[mask].mean().item()),
        }
    return {
        "macro_top1": float(np.mean([item["top1"] for item in per_task.values()])),
        "micro_top1": float(correct.float().mean().item()),
        "macro_cosine": float(
            np.mean([item["mean_cosine"] for item in per_task.values()])
        ),
        "micro_cosine": float(cosine.mean().item()),
        "per_task": per_task,
    }


def train_checkpoint(
    checkpoint: str,
    args: argparse.Namespace,
    gallery_numpy: np.ndarray,
) -> dict:
    """固定步数训练一个 checkpoint；函数调用之间不复用模型或 optimizer。"""
    set_seed(args.seed)
    train_features, train_tasks, eval_features, eval_tasks, sources = load_probe_data(
        checkpoint,
        args.feature_dirs,
        args.feature_suffix,
        args.layer,
        args.eval_tasks,
        args.split_mode,
    )
    device = torch.device(args.device)
    train_action = torch.from_numpy(train_features).to(device)
    train_target = torch.from_numpy(train_tasks).long().to(device)
    eval_action = torch.from_numpy(eval_features).to(device)
    eval_target = torch.from_numpy(eval_tasks).long().to(device)
    gallery = torch.from_numpy(gallery_numpy).to(device)

    model = PaperMLPProbe(
        action_dim=train_action.shape[-1],
        text_dim=gallery.shape[-1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        initial_temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    order = torch.randperm(len(train_action), device=device)
    cursor = 0
    loss_sum = 0.0
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(train_action), device=device)
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        action_batch = train_action[indices]
        text_batch = gallery[train_target[indices]]

        model.train()
        loss = symmetric_infonce(model, action_batch, text_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                eval_action,
                eval_target,
                gallery,
                args.eval_tasks,
                args.eval_batch_size,
            )
            interval = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
            history.append(
                {
                    "step": step,
                    "train_loss": loss_sum / interval,
                    "temperature": float(model.temperature().item()),
                    **{key: value for key, value in metrics.items() if key != "per_task"},
                }
            )
            loss_sum = 0.0

    final_metrics = evaluate(
        model,
        eval_action,
        eval_target,
        gallery,
        args.eval_tasks,
        args.eval_batch_size,
    )
    output_dir = args.output_dir / checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "checkpoint": checkpoint,
            "steps": args.steps,
            "final_metrics": final_metrics,
        },
        output_dir / "probe_final.pt",
    )
    result = {
        "checkpoint": checkpoint,
        "protocol": f"{args.split_mode}_two_layer_mlp_fixed_optimizer_steps",
        "feature_sources": sources,
        "layer": args.layer,
        "train_tasks": sorted(np.unique(train_tasks).astype(int).tolist()),
        "eval_tasks": args.eval_tasks,
        "train_eval_task_overlap": sorted(
            set(np.unique(train_tasks).astype(int).tolist()) & set(args.eval_tasks)
        ),
        "n_train": int(len(train_tasks)),
        "n_eval": int(len(eval_tasks)),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "initial_temperature": args.temperature,
        "learnable_temperature": True,
        "selection": "final_step_without_test_selection",
        "history": history,
        "final_metrics": final_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{checkpoint}: macro={100 * final_metrics['macro_top1']:.2f}% "
        f"micro={100 * final_metrics['micro_top1']:.2f}%"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--feature-dirs", nargs="+", type=Path, default=DEFAULT_FEATURE_DIRS)
    parser.add_argument("--feature-suffix", default="_rollout")
    parser.add_argument("--text-path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--text-key", default="sentence_embeddings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-tasks", nargs="+", type=int, default=DEFAULT_EVAL_TASKS)
    parser.add_argument(
        "--split-mode",
        choices=["task_disjoint", "original_train"],
        default="task_disjoint",
    )
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert len(set(args.eval_tasks)) == len(args.eval_tasks), "eval tasks 不得重复"
    assert args.steps > 0 and args.eval_every > 0
    return args


def main() -> None:
    args = parse_args()
    gallery = load_text_gallery(args.text_path, args.text_key)
    results = [train_checkpoint(checkpoint, args, gallery) for checkpoint in args.checkpoints]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve = {
        "checkpoints": args.checkpoints,
        "macro_top1_pct": [100 * item["final_metrics"]["macro_top1"] for item in results],
        "micro_top1_pct": [100 * item["final_metrics"]["micro_top1"] for item in results],
    }
    (args.output_dir / "curve.json").write_text(
        json.dumps(curve, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
