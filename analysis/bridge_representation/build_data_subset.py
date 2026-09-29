"""
一次性扫描 RLDS 数据集，提取目标 task 的轨迹原始数据，保存为独立子集。

核心保证:
  1. 数据只采集一次，后续所有模型从同一个子集加载 → 输入 100% 一致
  2. 图像经 decode + resize 到 224×224，但 **不做任何数据增强**（无 random crop/brightness 等）
  3. 每条轨迹保存为一个 .npz 文件（images + actions + timesteps）
  4. manifest.json 记录全部元数据

输出目录:
  outputs/data_subset/          (train)
  outputs/val/data_subset/      (val)

用法:
  cd /root/data/xuyuan1/Codes/analysis/SpatialVLA
  python .../build_data_subset.py                  # 默认 train
  python .../build_data_subset.py --split val      # val 集
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ===================== 路径配置 =====================
ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
SPATIALVLA_DIR = os.path.join(ANALYSIS_DIR, "SpatialVLA")

sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, SPATIALVLA_DIR)

from bridge_representation.config import (
    RLDS_DATA_ROOT, TARGET_TASK_LANGS, TRAJECTORIES_PER_TASK,
    OUTPUT_DIR, SELECTED_TASKS, get_canonical_task_label,
    SUBSET_DIR,
    # val 配置
    VAL_SELECTED_TASKS, VAL_TARGET_TASK_LANGS,
    VAL_TRAJECTORIES_PER_TASK, VAL_SUBSET_DIR,
    get_val_canonical_task_label,
)


def build_rlds_pipeline(is_train=True, target_task_langs=None):
    """
    构建 RLDS pipeline: 单遍扫描，带 task_lang 过滤，decode+resize，无数据增强。

    Args:
        is_train: True→train split, False→val split
        target_task_langs: 需要过滤的 task 文本集合 (lowercase)
    """
    import tensorflow as tf
    from data.rlds import (
        make_dataset_from_rlds,
        apply_trajectory_transforms,
        apply_frame_transforms,
    )
    from data.oxe import get_oxe_dataset_kwargs_and_weights
    from data.utils.data_utils import NormalizationType

    mixture_spec = [("bridge_orig/1.0.0", 1.0)]
    per_dataset_kwargs, _ = get_oxe_dataset_kwargs_and_weights(
        RLDS_DATA_ROOT,
        mixture_spec,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=False,
        load_language=True,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )

    dataset_kwargs = per_dataset_kwargs[0].copy()
    dataset_kwargs.pop("dataset_frame_transform_kwargs", None)

    split_name = "train" if is_train else "val"
    print(f"Loading RLDS dataset (split={split_name})...")
    dataset, _ = make_dataset_from_rlds(
        **dataset_kwargs,
        shuffle_seed=42,
        train=is_train,         # ★ 控制 train/val split
        shuffle=is_train,       # val 不打乱
        num_parallel_calls=tf.data.AUTOTUNE,
        num_parallel_reads=2,
    )

    # ★ 轨迹级 task_lang 过滤
    if target_task_langs is None:
        target_task_langs = TARGET_TASK_LANGS
    target_langs_lower = [t.lower().encode('utf-8') for t in target_task_langs]
    target_tensor = tf.constant(target_langs_lower, dtype=tf.string)

    def _is_target_task(traj):
        lang = traj["task"]["language_instruction"][0]
        lang_lower = tf.strings.lower(lang)
        return tf.reduce_any(tf.equal(lang_lower, target_tensor))

    dataset = dataset.filter(_is_target_task)
    print(f"[Filter] Keeping {len(target_task_langs)} target task variants ({split_name})")

    # 轨迹级变换: action chunking (forward=3 → chunk_size=4)
    print("Applying trajectory transforms (forward_window_size=3, action_chunk_size=4)...")
    dataset = apply_trajectory_transforms(
        dataset,
        train=True,             # ★ 始终 True, 保证 goal_relabeling 正常
        skip_unlabeled=True,
        goal_relabeling_strategy="uniform",
        backward_windows_size=0,
        backward_delta=1,
        forward_window_size=3,
    ).flatten(num_parallel_calls=4)

    # 帧级变换: decode + resize → 224×224 uint8, 无增强
    print("Applying frame transforms (decode+resize, NO augmentation)...")
    dataset = apply_frame_transforms(
        dataset,
        train=False,
        resize_size=(224, 224),
        image_augment_kwargs={},
        num_parallel_calls=16,
    )

    return dataset


def build_subset(split="train"):
    """
    主函数: 扫描 RLDS → 收集 → 保存

    Args:
        split: "train" 或 "val"
    """
    # 根据 split 选择配置
    if split == "val":
        subset_dir = VAL_SUBSET_DIR
        # val canonical = lowercase text, 无 task_index 映射
        canonical_labels = set(t.lower() for t in VAL_SELECTED_TASKS)
        target_task_langs = VAL_TARGET_TASK_LANGS
        traj_per_task = VAL_TRAJECTORIES_PER_TASK
        canonical_fn = get_val_canonical_task_label
    else:
        subset_dir = SUBSET_DIR
        canonical_labels = set(t["text"] for t in SELECTED_TASKS)
        target_task_langs = TARGET_TASK_LANGS
        traj_per_task = TRAJECTORIES_PER_TASK
        canonical_fn = get_canonical_task_label

    os.makedirs(subset_dir, exist_ok=True)
    is_train = (split == "train")

    dataset = build_rlds_pipeline(is_train=is_train, target_task_langs=target_task_langs)

    # ========== 收集帧数据 ==========
    traj_frames = defaultdict(list)     # (canonical, rlds_traj_idx) → [frame_dict]
    task_traj_set = defaultdict(set)    # canonical → {rlds_traj_idx, ...}
    total_frames = 0
    total_skipped = 0

    n_tasks = len(canonical_labels)
    target_trajs = n_tasks * traj_per_task

    print(f"\nCollection target:")
    print(f"  Tasks: {n_tasks}")
    print(f"  Trajectories/task: {traj_per_task}")
    print(f"  Total trajectories: {target_trajs}")
    print(f"  Target task langs (all text variants): {len(target_task_langs)}")
    print()

    pbar = tqdm(desc="Scanning RLDS")
    for frame in dataset.as_numpy_iterator():
        # 提取 language
        lang_raw = frame["task"]["language_instruction"]
        if isinstance(lang_raw, np.ndarray):
            lang_raw = lang_raw.flat[0]
        if isinstance(lang_raw, bytes):
            lang = lang_raw.decode()
        else:
            lang = str(lang_raw)

        canonical = canonical_fn(lang)

        if canonical not in canonical_labels:
            total_skipped += 1
            pbar.update(1)
            continue

        # 提取 traj_index
        traj_idx_raw = frame.get("traj_index", -1)
        if isinstance(traj_idx_raw, np.ndarray):
            traj_idx = int(traj_idx_raw.flat[0]) if traj_idx_raw.size > 0 else -1
        else:
            traj_idx = int(traj_idx_raw)

        # 该 task 已够 traj_per_task 条轨迹，且这是新轨迹 → 跳过
        if (len(task_traj_set[canonical]) >= traj_per_task
                and traj_idx not in task_traj_set[canonical]):
            total_skipped += 1
            pbar.update(1)
            continue

        task_traj_set[canonical].add(traj_idx)

        # 提取 image (已 decode+resize)
        # backward_windows_size=0 → image shape = [1, H, W, C]
        image = frame["observation"]["image_primary"]
        if image.ndim == 4:
            image = image[0]  # [1, H, W, C] → [H, W, C]

        # 提取 action (forward_window_size=3 → action shape = [4, D])
        # ★ 不做 squeeze! 保留完整的 action chunk
        action = frame["action"]  # [4, D] float32

        # 提取 timestep
        obs_ts = frame["observation"]["timestep"]
        if isinstance(obs_ts, np.ndarray):
            timestep = int(obs_ts.flat[-1])
        else:
            timestep = int(obs_ts)

        traj_frames[(canonical, traj_idx)].append({
            'image': image,     # [224, 224, 3] uint8
            'action': action,   # [4, D] float32 (action chunk: current + 3 future)
            'timestep': timestep,
            'lang': lang,
        })
        total_frames += 1
        pbar.update(1)

        collected = sum(len(s) for s in task_traj_set.values())
        pbar.set_postfix(frames=total_frames, trajs=f"{collected}/{target_trajs}", skip=total_skipped)

        # 所有 task 收满 → 提前终止
        if all(len(task_traj_set[c]) >= traj_per_task for c in canonical_labels):
            print(f"\n✅ All {target_trajs} trajectories collected!")
            break

    pbar.close()

    # 检查未收满的 task (val 集中部分 task 本身就很少, 仅在 train 时提醒)
    incomplete = {c: len(task_traj_set[c]) for c in canonical_labels
                  if len(task_traj_set[c]) < traj_per_task}
    if incomplete:
        print(f"\n⚠️  Incomplete tasks ({len(incomplete)}):")
        for c, n in sorted(incomplete.items()):
            print(f"    {c}: {n}/{traj_per_task}")

    # ========== 保存轨迹 ==========
    print(f"\nSaving {len(traj_frames)} trajectories to {subset_dir}...")

    manifest = {
        'split': split,
        'n_tasks': n_tasks,
        'trajectories_per_task': traj_per_task,
        'total_trajectories': 0,
        'total_frames': 0,
        'trajectories': [],
    }

    local_id = 0
    for (canonical, rlds_traj_idx), frames in sorted(traj_frames.items()):
        # 按 timestep 排序
        frames.sort(key=lambda f: f['timestep'])

        # 去重 timestep（同轨迹的同 timestep 可能因 chunking 重复）
        seen_ts = set()
        unique_frames = []
        for f in frames:
            if f['timestep'] not in seen_ts:
                seen_ts.add(f['timestep'])
                unique_frames.append(f)
        frames = unique_frames

        images = np.stack([f['image'] for f in frames])    # [T, 224, 224, 3]
        actions = np.stack([f['action'] for f in frames])  # [T, 4, D] (action chunk)
        timesteps = np.array([f['timestep'] for f in frames], dtype=np.int32)  # [T]

        filename = f"traj_{local_id:04d}.npz"
        np.savez_compressed(
            os.path.join(subset_dir, filename),
            images=images,
            actions=actions,
            timesteps=timesteps,
        )

        manifest['trajectories'].append({
            'file': filename,
            'canonical_label': canonical,
            'lang': frames[0]['lang'],
            'local_traj_id': local_id,
            'n_timesteps': len(frames),
        })
        manifest['total_frames'] += len(frames)
        local_id += 1

    manifest['total_trajectories'] = local_id

    # 保存 manifest
    manifest_path = os.path.join(subset_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ========== 打印汇总 ==========
    print(f"\n{'=' * 60}")
    print(f"Data subset built successfully! (split={split})")
    print(f"  Directory: {subset_dir}")
    print(f"  Trajectories: {manifest['total_trajectories']}")
    print(f"  Total frames: {manifest['total_frames']}")
    print(f"  Avg frames/traj: {manifest['total_frames'] / max(manifest['total_trajectories'], 1):.1f}")
    print(f"\n  Per-task stats:")
    task_stats = defaultdict(lambda: {'trajs': 0, 'frames': 0})
    for t in manifest['trajectories']:
        task_stats[t['canonical_label']]['trajs'] += 1
        task_stats[t['canonical_label']]['frames'] += t['n_timesteps']
    for label, stats in sorted(task_stats.items()):
        avg_f = stats['frames'] / max(stats['trajs'], 1)
        print(f"    {label:<55} trajs={stats['trajs']:>3}  frames={stats['frames']:>5}  avg={avg_f:.0f}")

    # 磁盘占用
    total_bytes = sum(
        os.path.getsize(os.path.join(subset_dir, t['file']))
        for t in manifest['trajectories']
    )
    print(f"\n  Disk usage: {total_bytes / 1e9:.2f} GB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="数据 split: train(默认) 或 val")
    args = parser.parse_args()

    os.chdir(SPATIALVLA_DIR)
    build_subset(split=args.split)
