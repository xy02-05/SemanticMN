"""
确定 LIBERO 训练/测试集划分

支持两种模式:
  1. task 级别划分（默认）: 留出完整的 task 作为测试集
     - 每个 suite 随机留出 n_test_per_suite 个 task
     - 测试集 task 的所有 episode 归入 test，其余归入 train
     - 测量模型对未见 task 的泛化能力

  2. episode 级别划分（旧模式）: 每个 task 内 80/20 切分 episode
     - 所有 40 个 task 同时出现在 train 和 test 中
     - 仅测量同 task 内的插值能力，区分度虚高

输出:
  outputs/split_task.json  (task 级别) 或 outputs/split_episode.json
  {
    "mode": "task" / "episode",
    "seed": 42,
    "train_tasks": [task_idx, ...],
    "test_tasks": [task_idx, ...],
    "all_train": [ep_idx, ...],
    "all_test": [ep_idx, ...],
    ...
  }

用法:
    python build_split.py                    # 默认 task 级别
    python build_split.py --mode episode     # 旧的 episode 级别
    python build_split.py --n_test_per_suite 2
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import LIBERO_DATA_DIR, TASKS_WITH_ID_PATH, OUTPUT_DIR, LIBERO_SUITES


def _load_task_episodes():
    """读取 LIBERO 的 task→episode 映射"""
    task_text_to_idx = {}
    with open(TASKS_WITH_ID_PATH) as f:
        for line in f:
            d = json.loads(line)
            task_text_to_idx[d["task"]] = d["task_index"]

    episodes_path = os.path.join(LIBERO_DATA_DIR, "meta/episodes.jsonl")
    task_episodes = defaultdict(list)
    with open(episodes_path) as f:
        for line in f:
            e = json.loads(line)
            task_text = e["tasks"][0]
            task_idx = task_text_to_idx.get(task_text, -1)
            task_episodes[task_idx].append(e["episode_index"])

    # 同时读 task_index → task_text（用于输出可读信息）
    idx_to_text = {v: k for k, v in task_text_to_idx.items()}
    return task_episodes, idx_to_text


def build_split_by_task(seed=42, n_test_per_suite=1, explicit_test_tasks=None):
    """
    按 task 级别划分: 每个 suite 留出 n_test_per_suite 个完整 task。
    留出的 task 的全部 episode 归入 test，其余归入 train。

    Args:
        explicit_test_tasks: list of int 或 None。如果给定，直接用作测试 task 集合
            （忽略 n_test_per_suite 的随机采样）。用于挑选物体在训练集出现过的 task。
    """
    rng = np.random.RandomState(seed)
    task_episodes, idx_to_text = _load_task_episodes()

    train_tasks, test_tasks = [], []

    if explicit_test_tasks is not None:
        # 指定模式：直接使用给定的 test task 列表
        test_set_explicit = set(explicit_test_tasks)
        for suite_name, suite_info in LIBERO_SUITES.items():
            for ti in suite_info["task_indices"]:
                if ti in test_set_explicit:
                    test_tasks.append(ti)
                else:
                    train_tasks.append(ti)
    else:
        # 随机模式：每个 suite 独立采样
        for suite_name, suite_info in LIBERO_SUITES.items():
            suite_indices = list(suite_info["task_indices"])
            rng.shuffle(suite_indices)
            n_test = min(n_test_per_suite, len(suite_indices))
            test_tasks.extend(suite_indices[:n_test])
            train_tasks.extend(suite_indices[n_test:])

    train_tasks = sorted(train_tasks)
    test_tasks = sorted(test_tasks)
    test_set = set(test_tasks)

    # 分配 episode
    all_train, all_test = [], []
    per_task = {}

    for task_idx in sorted(task_episodes.keys()):
        eps = sorted(task_episodes[task_idx])
        is_test = task_idx in test_set
        per_task[str(task_idx)] = {
            "split": "test" if is_test else "train",
            "episodes": eps,
            "n_episodes": len(eps),
        }
        if is_test:
            all_test.extend(eps)
        else:
            all_train.extend(eps)

    # 统计
    stats_per_suite = {}
    for suite_name, suite_info in LIBERO_SUITES.items():
        s_train = [ti for ti in suite_info["task_indices"] if ti not in test_set]
        s_test = [ti for ti in suite_info["task_indices"] if ti in test_set]
        stats_per_suite[suite_name] = {
            "train_tasks": s_train,
            "test_tasks": s_test,
            "n_train_tasks": len(s_train),
            "n_test_tasks": len(s_test),
            "n_train_episodes": sum(len(task_episodes[t]) for t in s_train),
            "n_test_episodes": sum(len(task_episodes[t]) for t in s_test),
        }

    split = {
        "mode": "task",
        "seed": seed,
        "n_test_per_suite": n_test_per_suite,
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "n_train_tasks": len(train_tasks),
        "n_test_tasks": len(test_tasks),
        "per_task": per_task,
        "all_train": sorted(all_train),
        "all_test": sorted(all_test),
        "stats": {
            "total_episodes": len(all_train) + len(all_test),
            "total_train": len(all_train),
            "total_test": len(all_test),
            "n_tasks": len(per_task),
            "per_suite": stats_per_suite,
        },
    }

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "split_task.json")
    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)

    # 同时保存为默认 split.json（后续脚本默认读这个）
    default_path = os.path.join(OUTPUT_DIR, "split.json")
    with open(default_path, "w") as f:
        json.dump(split, f, indent=2)

    # 打印
    print(f"LIBERO Task-Level Split (seed={seed}, {n_test_per_suite} test tasks/suite)")
    print(f"  Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")
    print(f"  Train episodes: {len(all_train)}, Test episodes: {len(all_test)}")
    print()

    for suite_name, suite_info in LIBERO_SUITES.items():
        ss = stats_per_suite[suite_name]
        print(f"  {suite_name}:")
        print(f"    Train: {ss['n_train_tasks']} tasks ({ss['n_train_episodes']} eps)")
        print(f"    Test:  {ss['n_test_tasks']} tasks ({ss['n_test_episodes']} eps)")
        for ti in ss["test_tasks"]:
            text = idx_to_text.get(ti, "?")
            print(f"      [TEST] task {ti:>2d}: {text[:60]}  ({per_task[str(ti)]['n_episodes']} eps)")
        print()

    print(f"保存: {output_path}")
    print(f"默认: {default_path}")
    return split


def build_split_by_episode(seed=42, train_ratio=0.8):
    """旧模式: 每个 task 内按 episode 切分 80/20"""
    rng = np.random.RandomState(seed)
    task_episodes, idx_to_text = _load_task_episodes()

    per_task = {}
    all_train, all_test = [], []

    for task_idx in sorted(task_episodes.keys()):
        eps = sorted(task_episodes[task_idx])
        n = len(eps)
        eps_arr = np.array(eps)
        rng.shuffle(eps_arr)
        n_train = max(1, int(n * train_ratio))
        if (n - n_train) < 5 and n > 5:
            n_train = n - 5
        train_eps = sorted(eps_arr[:n_train].tolist())
        test_eps = sorted(eps_arr[n_train:].tolist())
        per_task[str(task_idx)] = {
            "train": train_eps, "test": test_eps,
            "n_train": len(train_eps), "n_test": len(test_eps),
        }
        all_train.extend(train_eps)
        all_test.extend(test_eps)

    split = {
        "mode": "episode",
        "seed": seed,
        "train_ratio": train_ratio,
        "train_tasks": sorted(task_episodes.keys()),
        "test_tasks": sorted(task_episodes.keys()),
        "per_task": per_task,
        "all_train": sorted(all_train),
        "all_test": sorted(all_test),
        "stats": {
            "total_episodes": len(all_train) + len(all_test),
            "total_train": len(all_train),
            "total_test": len(all_test),
            "n_tasks": len(per_task),
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "split_episode.json")
    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"LIBERO Episode-Level Split (seed={seed}, ratio={train_ratio})")
    print(f"  Train: {len(all_train)}, Test: {len(all_test)}")
    print(f"保存: {output_path}")
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO train/test 划分")
    parser.add_argument("--mode", type=str, default="task",
                        choices=["task", "episode"],
                        help="划分模式: task=留出整个task, episode=task内切分")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test_per_suite", type=int, default=1,
                        help="task 模式下每个 suite 留出的 test task 数")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="episode 模式下的训练比例")
    parser.add_argument("--test_tasks", type=str, default=None,
                        help="指定测试 task 索引（逗号分隔），覆盖 n_test_per_suite 的随机采样")
    args = parser.parse_args()

    if args.mode == "task":
        explicit = None
        if args.test_tasks:
            explicit = [int(x) for x in args.test_tasks.split(",")]
        build_split_by_task(args.seed, args.n_test_per_suite, explicit_test_tasks=explicit)
    else:
        build_split_by_episode(args.seed, args.train_ratio)
