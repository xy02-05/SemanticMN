"""
compute_gaussian_statistic.py

基于 SpatialVLA 官方数据读取与标准化逻辑，计算 ADAPT 所需的 gaussian_statistic。

实现原则：
1. 复用官方的 dataset config 与 standardize_fn。
2. 统计位置严格放在「标准化之后、动作归一化之前」。
3. 只读取 action，不解码图像，尽量减少 bridge 全量统计的额外开销。
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 当前环境没有通过 pip 安装 dlimp，这里优先复用本机已有源码目录。
DLIMP_REPO_ROOT_CANDIDATES = [
    Path("/data/xuyuan/UniVLA_env/project/INT-ACT"),
    Path("/data/xuyuan/UniVLA_env/project/INT-ACT-tmp"),
]
for candidate in reversed(DLIMP_REPO_ROOT_CANDIDATES):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

DLIMP_PARENT_CANDIDATES = [
    Path("/data/xuyuan/UniVLA_env/project/INT-ACT/src/data"),
    Path("/data/xuyuan/UniVLA_env/project/INT-ACT-tmp/src/data"),
]
for candidate in reversed(DLIMP_PARENT_CANDIDATES):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dlimp as dl

from data.oxe import make_oxe_dataset_kwargs
from data.utils.data_utils import (
    NormalizationType,
    cartesian_to_spherical,
    get_dataset_statistics,
)


tf.config.set_visible_devices([], "GPU")

GAUSSIAN_KEYS = ("x", "y", "z", "theta", "phi", "r", "roll", "pitch", "yaw")


class GaussianAccumulator:
    """
    流式统计九个空间维度的均值和标准差。

    这里只保留 sum / sum_sq / count，避免在 bridge 全量数据上堆积巨大内存。
    """

    def __init__(self):
        self.sum = np.zeros(len(GAUSSIAN_KEYS), dtype=np.float64)
        self.sum_sq = np.zeros(len(GAUSSIAN_KEYS), dtype=np.float64)
        self.count = 0
        self.num_transitions = 0
        self.num_trajectories = 0

    def update(self, actions: np.ndarray):
        if actions.ndim != 2 or actions.shape[1] < 6:
            raise ValueError(f"动作维度不正确，期望形状为 (N, >=6)，实际得到 {actions.shape}")
        theta, phi, radius = cartesian_to_spherical(
            actions[:, 0],
            actions[:, 1],
            actions[:, 2],
        )
        values = np.stack(
            [
                actions[:, 0],
                actions[:, 1],
                actions[:, 2],
                theta,
                phi,
                radius,
                actions[:, 3],
                actions[:, 4],
                actions[:, 5],
            ],
            axis=1,
        ).astype(np.float64)

        self.sum += values.sum(axis=0)
        self.sum_sq += np.square(values).sum(axis=0)
        self.count += values.shape[0]
        self.num_transitions += values.shape[0]
        self.num_trajectories += 1

    def to_dict(self):
        mean = self.sum / self.count
        var = self.sum_sq / self.count - np.square(mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

        return {
            key: {
                "mu": float(mean[idx]),
                "sigma": float(std[idx]),
            }
            for idx, key in enumerate(GAUSSIAN_KEYS)
        }


def build_action_only_dataset(
    dataset_name: str,
    data_root_dir: str,
    split: str,
    num_parallel_reads: int,
    num_parallel_calls: int,
):
    """
    复用官方 make_oxe_dataset_kwargs 与 standardize_fn。

    这里刻意不走 make_dataset_from_rlds 的后半段，因为那个函数会继续做动作归一化。
    我们只保留官方的：
    - builder_from_directory
    - standardize_fn
    - action dtype / shape 约定
    """
    dataset_kwargs = make_oxe_dataset_kwargs(
        dataset_name=dataset_name,
        data_root_dir=Path(data_root_dir),
        load_camera_views=(),
        load_depth=False,
        load_proprio=False,
        load_language=False,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    standardize_fn = dataset_kwargs["standardize_fn"]
    builder = tfds.builder_from_directory(os.path.join(dataset_kwargs["data_dir"], dataset_kwargs["name"]))

    def restructure_action_only(traj):
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if "observation" not in traj or "action" not in traj:
            raise ValueError("轨迹缺少 observation/action，官方 standardize_fn 输出不符合预期。")

        # observation 占位 proprio：下游 get_dataset_statistics 的 traj_map 内部
        # 会做 `'proprio' in traj['observation']` 检查，剥光 observation 会撞 KeyError；
        # 我们只需要 action 的 q01/q99/min/max（用于 BOUNDS_Q99 归一化），不统计 proprio，
        # 所以给一个与 action 同 shape 的零张量即可走 if_exp 的 else（zeros_like）分支。
        action_f32 = tf.cast(traj["action"], tf.float32)
        return {
            "action": action_f32,
            "observation": {"proprio": tf.zeros_like(action_f32)},
        }

    dataset = dl.DLataset.from_rlds(
        builder,
        split=split,
        shuffle=False,
        num_parallel_reads=num_parallel_reads,
    ).traj_map(restructure_action_only, num_parallel_calls)
    return dataset, builder, standardize_fn, dataset_kwargs


def normalize_actions(actions: np.ndarray, dataset_statistics: dict, action_normalization_mask):
    """
    复用官方 BOUNDS_Q99 归一化公式。

    bridge 对齐实验已经证明：官方 gs_bridge.json 不是基于 raw action，
    而是基于这一步归一化后的动作分布。
    """
    low = np.asarray(dataset_statistics["action"]["q01"], dtype=np.float64)
    high = np.asarray(dataset_statistics["action"]["q99"], dtype=np.float64)
    mins = np.asarray(dataset_statistics["action"]["min"], dtype=np.float64)
    maxs = np.asarray(dataset_statistics["action"]["max"], dtype=np.float64)
    mask = np.asarray(action_normalization_mask, dtype=bool)

    actions = np.where(
        mask,
        np.clip(2 * (actions - low) / (high - low + 1e-8) - 1, -1, 1),
        actions,
    )
    actions = np.where(mins == maxs, 0.0, actions)
    return actions


def compute_gaussian_statistic(
    dataset_name: str,
    data_root_dir: str,
    split: str,
    num_parallel_reads: int,
    num_parallel_calls: int,
    action_space: str,
):
    dataset, builder, standardize_fn, dataset_kwargs = build_action_only_dataset(
        dataset_name=dataset_name,
        data_root_dir=data_root_dir,
        split=split,
        num_parallel_reads=num_parallel_reads,
        num_parallel_calls=num_parallel_calls,
    )
    dataset_statistics = None
    if action_space == "normalized":
        dataset_statistics = get_dataset_statistics(
            dataset,
            hash_dependencies=(
                str(builder.info),
                str(()),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
            ),
            save_dir=builder.data_dir,
        )

    cardinality = dataset.cardinality().numpy()
    accumulator = GaussianAccumulator()

    for traj in tqdm(
        dataset.iterator(),
        total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
        desc=f"compute {dataset_name}",
    ):
        actions = np.asarray(traj["action"], dtype=np.float64)
        if action_space == "normalized":
            actions = normalize_actions(
                actions,
                dataset_statistics=dataset_statistics,
                action_normalization_mask=dataset_kwargs["action_normalization_mask"],
            )
        accumulator.update(actions)

    return {
        "dataset_name": dataset_name,
        "split": split,
        "action_space": action_space,
        "num_transitions": accumulator.num_transitions,
        "num_trajectories": accumulator.num_trajectories,
        "gaussian_statistic": accumulator.to_dict(),
    }


def compare_gaussian_statistics(pred: dict, gt: dict):
    report = {}
    max_abs_diff = 0.0
    mean_abs_diff = []

    for key in GAUSSIAN_KEYS:
        mu_diff = pred[key]["mu"] - gt[key]["mu"]
        sigma_diff = pred[key]["sigma"] - gt[key]["sigma"]
        report[key] = {
            "mu_pred": pred[key]["mu"],
            "mu_gt": gt[key]["mu"],
            "mu_abs_diff": abs(mu_diff),
            "sigma_pred": pred[key]["sigma"],
            "sigma_gt": gt[key]["sigma"],
            "sigma_abs_diff": abs(sigma_diff),
        }
        max_abs_diff = max(
            max_abs_diff,
            report[key]["mu_abs_diff"],
            report[key]["sigma_abs_diff"],
        )
        mean_abs_diff.extend([report[key]["mu_abs_diff"], report[key]["sigma_abs_diff"]])

    report["_summary"] = {
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(np.mean(mean_abs_diff)),
    }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SpatialVLA gaussian_statistic from official data pipeline")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name with version, e.g. bridge_orig/1.0.0")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Dataset root dir passed to official OXE loader")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save gaussian_statistic json")
    parser.add_argument("--split", type=str, default="all", help="TFDS split, default uses all trajectories")
    parser.add_argument("--action_space", type=str, default="normalized", choices=["normalized", "raw"],
                        help="Compute gaussian on normalized or raw actions; official gs_bridge matches normalized")
    parser.add_argument("--compare_to", type=str, default=None, help="Optional reference gaussian_statistic json")
    parser.add_argument("--compare_report_path", type=str, default=None, help="Optional path to save compare report")
    parser.add_argument("--num_parallel_reads", type=int, default=tf.data.AUTOTUNE, help="Parallel reads for RLDS builder")
    parser.add_argument("--num_parallel_calls", type=int, default=tf.data.AUTOTUNE, help="Parallel traj_map calls")
    return parser.parse_args()


def main():
    args = parse_args()
    result = compute_gaussian_statistic(
        dataset_name=args.dataset_name,
        data_root_dir=args.data_root_dir,
        split=args.split,
        num_parallel_reads=args.num_parallel_reads,
        num_parallel_calls=args.num_parallel_calls,
        action_space=args.action_space,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result["gaussian_statistic"], f, indent=2)

    print(f"saved gaussian_statistic to {output_path}")
    print(f"dataset={result['dataset_name']} split={result['split']} action_space={result['action_space']}")
    print(f"num_trajectories={result['num_trajectories']} num_transitions={result['num_transitions']}")

    if args.compare_to is not None:
        with open(args.compare_to, "r") as f:
            gt = json.load(f)
        report = compare_gaussian_statistics(result["gaussian_statistic"], gt)
        print(f"compare_to={args.compare_to}")
        print(json.dumps(report["_summary"], indent=2))

        if args.compare_report_path is not None:
            compare_report_path = Path(args.compare_report_path)
            compare_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(compare_report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"saved compare report to {compare_report_path}")


if __name__ == "__main__":
    main()

