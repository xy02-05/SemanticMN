#!/usr/bin/env python3
"""
将 Agilex HDF5 数据集转换为 RLDS (TFRecord) 格式。

转换后的数据集可以直接被 openpi 的 RobotwinRldsDataset 加载器使用，
从而可以使用 compute_norm_stats.py 计算归一化统计量，
以及使用 train_pytorch_xy.py 进行训练。

HDF5 源数据结构（每个文件 = 一个 episode）:
  - action: (T, 14)  float32
  - base_action: (T, 2) float32
  - observations/effort: (T, 14) float32
  - observations/images/cam_high: (T, 480, 640, 3) uint8
  - observations/images/cam_left_wrist: (T, 480, 640, 3) uint8
  - observations/images/cam_right_wrist: (T, 480, 640, 3) uint8
  - observations/qpos: (T, 14) float32
  - observations/qvel: (T, 14) float32

RLDS 目标数据结构（与 RobotwinRldsDataset 对齐）:
  steps/:
    - action: (14,) float32     -> 展平存储为 T*14 个 float
    - observation/cam_high: jpeg-encoded image  -> T 个 jpeg bytes
    - observation/cam_left_wrist: jpeg-encoded image
    - observation/cam_right_wrist: jpeg-encoded image
    - observation/state: (14,) float32  -> 展平存储为 T*14 个 float
    - language_instruction: string  -> T 个 string
    - language_embedding: (512,) float32  -> T*512 个 float (占位符)
    - is_first, is_last, is_terminal: bool  -> T 个 int64
    - reward: float32  -> T 个 float
    - discount: float32  -> T 个 float
  episode_metadata/:
    - episode_id: int32
    - file_path: string

用法:
  # 单个任务目录
  python scripts/convert_hdf5_to_rlds.py \\
    --input_dir /root/data/xuyuan1/dataset/agilex/data_20260211/data_20260210/banana_to_white_plate \\
    --output_dir /root/data/xuyuan1/dataset/agilex_rlds \\
    --dataset_name agilex_dataset

  # 多个任务目录（自动扫描子目录）
  python scripts/convert_hdf5_to_rlds.py \\
    --input_dir /root/data/xuyuan1/dataset/agilex/data_20260211/data_20260210 \\
    --output_dir /root/data/xuyuan1/dataset/agilex_rlds \\
    --dataset_name agilex_dataset

  # 指定图像大小
  python scripts/convert_hdf5_to_rlds.py \\
    --input_dir /root/data/xuyuan1/dataset/agilex/data_20260211/data_20260210 \\
    --output_dir /root/data/xuyuan1/dataset/agilex_rlds \\
    --dataset_name agilex_dataset \\
    --image_size 480 640
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def encode_image_jpeg(img_array: np.ndarray, quality: int = 95) -> bytes:
    """将 numpy 图像数组编码为 JPEG bytes"""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def resize_image(img_array: np.ndarray, height: int, width: int) -> np.ndarray:
    """将图像调整到指定大小"""
    img = Image.fromarray(img_array)
    img = img.resize((width, height), Image.LANCZOS)
    return np.array(img)


def load_hdf5_episode(hdf5_path: str) -> dict:
    """加载单个 HDF5 文件的数据"""
    with h5py.File(hdf5_path, "r") as f:
        data = {
            "action": f["action"][:].astype(np.float32),  # (T, 14)
            "qpos": f["observations/qpos"][:].astype(np.float32),  # (T, 14)
            "cam_high": f["observations/images/cam_high"][:],  # (T, H, W, 3) uint8
            "cam_left_wrist": f["observations/images/cam_left_wrist"][:],  # (T, H, W, 3) uint8
            "cam_right_wrist": f["observations/images/cam_right_wrist"][:],  # (T, H, W, 3) uint8
        }
    return data


def find_hdf5_tasks(input_dir: str) -> list[dict]:
    """
    扫描输入目录，找到所有 HDF5 数据。

    支持两种目录结构:
    1. 直接包含 .hdf5 文件的目录（单任务）
    2. 包含多个子目录的目录（多任务），每个子目录包含 .hdf5 文件

    Returns:
        list of dict: [{"task_name": str, "instruction": str, "hdf5_files": [str]}]
    """
    input_path = Path(input_dir)
    tasks = []

    # 检查是否直接包含 HDF5 文件
    hdf5_files = sorted(input_path.glob("*.hdf5"))
    if hdf5_files:
        # 单任务目录
        instruction = _load_instruction(input_path)
        task_name = input_path.name
        tasks.append({
            "task_name": task_name,
            "instruction": instruction,
            "hdf5_files": [str(f) for f in hdf5_files],
        })
    else:
        # 多任务目录 - 扫描子目录
        for sub_dir in sorted(input_path.iterdir()):
            if not sub_dir.is_dir():
                continue
            sub_hdf5_files = sorted(sub_dir.glob("*.hdf5"))
            if not sub_hdf5_files:
                continue
            instruction = _load_instruction(sub_dir)
            task_name = sub_dir.name
            tasks.append({
                "task_name": task_name,
                "instruction": instruction,
                "hdf5_files": [str(f) for f in sub_hdf5_files],
            })

    return tasks


def _load_instruction(task_dir: Path) -> str:
    """从 task_info.json 中加载指令"""
    task_info_path = task_dir / "task_info.json"
    if task_info_path.exists():
        with open(task_info_path, "r") as f:
            info = json.load(f)
        if isinstance(info, list) and len(info) > 0:
            instruction = info[0].get("instruction", "")
            # 清理转义字符
            instruction = instruction.replace("\\", "").strip()
            return instruction
    # 如果没有 task_info.json，使用目录名作为指令
    return task_dir.name.replace("_", " ")


def episode_to_tf_example(
    episode_data: dict,
    episode_id: int,
    file_path: str,
    instruction: str,
    image_height: int,
    image_width: int,
) -> bytes:
    """
    将一个 episode 序列化为 tf.train.Example 的 bytes。

    RLDS/TFDS 的序列化格式：
    - 标量 features: 单个值
    - 时间序列张量 features: 所有时间步展平为一维列表
    - 时间序列图像 features: 所有时间步的 JPEG bytes 列表
    - 时间序列文本 features: 所有时间步的 string 列表
    """
    import tensorflow as tf

    num_steps = episode_data["action"].shape[0]

    # ===== 展平 actions 和 state =====
    # action: (T, 14) -> T*14 个 float
    actions_flat = episode_data["action"].flatten().tolist()
    # state: (T, 14) -> T*14 个 float
    state_flat = episode_data["qpos"].flatten().tolist()

    # ===== 编码图像为 JPEG =====
    cam_high_jpegs = []
    cam_left_jpegs = []
    cam_right_jpegs = []

    for t in range(num_steps):
        cam_h = episode_data["cam_high"][t]
        cam_l = episode_data["cam_left_wrist"][t]
        cam_r = episode_data["cam_right_wrist"][t]

        # 可选 resize
        if image_height != cam_h.shape[0] or image_width != cam_h.shape[1]:
            cam_h = resize_image(cam_h, image_height, image_width)
            cam_l = resize_image(cam_l, image_height, image_width)
            cam_r = resize_image(cam_r, image_height, image_width)

        cam_high_jpegs.append(encode_image_jpeg(cam_h))
        cam_left_jpegs.append(encode_image_jpeg(cam_l))
        cam_right_jpegs.append(encode_image_jpeg(cam_r))

    # ===== 构建 is_first / is_last / is_terminal =====
    is_first = [1 if t == 0 else 0 for t in range(num_steps)]
    is_last = [1 if t == num_steps - 1 else 0 for t in range(num_steps)]
    is_terminal = is_last.copy()

    # ===== reward / discount =====
    reward = [1.0 if t == num_steps - 1 else 0.0 for t in range(num_steps)]
    discount = [1.0] * num_steps

    # ===== language_instruction: T 个重复的 string =====
    lang_bytes = [instruction.encode("utf-8")] * num_steps

    # ===== language_embedding: T * 512 个 float (占位符) =====
    lang_embed_flat = [0.0] * (num_steps * 512)

    # ===== 构建 tf.train.Example =====
    feature_dict = {
        # ========== episode_metadata (标量) ==========
        "episode_metadata/episode_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[episode_id])
        ),
        "episode_metadata/file_path": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[file_path.encode("utf-8")])
        ),
        # ========== steps (序列，展平存储) ==========
        "steps/action": tf.train.Feature(
            float_list=tf.train.FloatList(value=actions_flat)
        ),
        "steps/observation/state": tf.train.Feature(
            float_list=tf.train.FloatList(value=state_flat)
        ),
        "steps/observation/cam_high": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=cam_high_jpegs)
        ),
        "steps/observation/cam_left_wrist": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=cam_left_jpegs)
        ),
        "steps/observation/cam_right_wrist": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=cam_right_jpegs)
        ),
        "steps/language_instruction": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=lang_bytes)
        ),
        "steps/language_embedding": tf.train.Feature(
            float_list=tf.train.FloatList(value=lang_embed_flat)
        ),
        "steps/is_first": tf.train.Feature(
            int64_list=tf.train.Int64List(value=is_first)
        ),
        "steps/is_last": tf.train.Feature(
            int64_list=tf.train.Int64List(value=is_last)
        ),
        "steps/is_terminal": tf.train.Feature(
            int64_list=tf.train.Int64List(value=is_terminal)
        ),
        "steps/reward": tf.train.Feature(
            float_list=tf.train.FloatList(value=reward)
        ),
        "steps/discount": tf.train.Feature(
            float_list=tf.train.FloatList(value=discount)
        ),
    }

    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    )
    return example.SerializeToString()


def build_rlds_dataset(
    tasks: list[dict],
    output_dir: str,
    dataset_name: str,
    image_height: int = 480,
    image_width: int = 640,
    num_shards: int = 64,
):
    """
    构建 RLDS 数据集。

    生成格式：
      output_dir/dataset_name/1.0.0/
        features.json
        dataset_info.json
        dataset_name-train.tfrecord-XXXXX-of-YYYYY
    """
    import tensorflow as tf

    # 禁用 GPU
    tf.config.set_visible_devices([], "GPU")

    output_path = Path(output_dir) / dataset_name / "1.0.0"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_path}")
    print(f"数据集名称: {dataset_name}")
    print(f"图像大小: {image_height}x{image_width}")

    # 收集所有 episodes
    all_episodes = []
    for task in tasks:
        for hdf5_file in task["hdf5_files"]:
            all_episodes.append({
                "hdf5_file": hdf5_file,
                "instruction": task["instruction"],
                "task_name": task["task_name"],
            })

    total_episodes = len(all_episodes)
    print(f"总共 {total_episodes} 个 episodes")

    # 计算 shard 分配
    actual_num_shards = min(num_shards, total_episodes)
    episodes_per_shard = max(1, (total_episodes + actual_num_shards - 1) // actual_num_shards)

    shard_lengths = []
    total_steps = 0
    total_bytes = 0
    episode_id = 0

    for shard_idx in range(actual_num_shards):
        shard_start = shard_idx * episodes_per_shard
        shard_end = min(shard_start + episodes_per_shard, total_episodes)
        shard_episodes = all_episodes[shard_start:shard_end]

        if not shard_episodes:
            break

        shard_filename = f"{dataset_name}-train.tfrecord-{shard_idx:05d}-of-{actual_num_shards:05d}"
        shard_path = output_path / shard_filename

        shard_episode_count = 0

        writer = tf.io.TFRecordWriter(str(shard_path))

        for ep_info in tqdm(
            shard_episodes,
            desc=f"Shard {shard_idx+1}/{actual_num_shards}",
            leave=False,
        ):
            try:
                data = load_hdf5_episode(ep_info["hdf5_file"])
            except Exception as e:
                print(f"⚠️ 跳过损坏的文件 {ep_info['hdf5_file']}: {e}")
                continue

            num_steps = data["action"].shape[0]

            serialized = episode_to_tf_example(
                data,
                episode_id=episode_id,
                file_path=ep_info["hdf5_file"],
                instruction=ep_info["instruction"],
                image_height=image_height,
                image_width=image_width,
            )

            writer.write(serialized)

            shard_episode_count += 1
            total_steps += num_steps
            total_bytes += len(serialized)
            episode_id += 1

            # 释放内存
            del data, serialized

        writer.close()
        shard_lengths.append(str(shard_episode_count))

    # 生成 metadata 文件
    _generate_features_json(output_path, image_height, image_width)
    _generate_dataset_info_json(
        output_path, dataset_name, actual_num_shards, shard_lengths, total_bytes, total_steps
    )

    print(f"\n✅ 转换完成!")
    print(f"   总 episodes: {episode_id}")
    print(f"   总 steps: {total_steps}")
    print(f"   总大小: {total_bytes / 1e9:.2f} GB")
    print(f"   输出目录: {output_path}")

    return total_steps


def _generate_features_json(output_path: Path, image_height: int, image_width: int):
    """
    生成 features.json，描述数据集的 schema。
    与 RobotwinRldsDataset 的期望格式完全对齐。
    """

    def make_image_feature(desc: str):
        return {
            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
            "image": {
                "shape": {
                    "dimensions": [str(image_height), str(image_width), "3"]
                },
                "dtype": "uint8",
                "encodingFormat": "jpeg",
            },
            "description": desc,
        }

    def make_tensor_feature(shape: list[str], dtype: str, desc: str):
        return {
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
            "tensor": {
                "shape": {"dimensions": shape},
                "dtype": dtype,
                "encoding": "none",
            },
            "description": desc,
        }

    def make_scalar_feature(dtype: str, desc: str):
        return {
            "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
            "tensor": {
                "shape": {},
                "dtype": dtype,
                "encoding": "none",
            },
            "description": desc,
        }

    def make_text_feature(desc: str):
        return {
            "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
            "text": {},
            "description": desc,
        }

    features_json = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {
            "features": {
                "steps": {
                    "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                    "sequence": {
                        "feature": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {
                                "features": {
                                    "action": make_tensor_feature(
                                        ["14"], "float32",
                                        "Robot action, 14-dim joint positions."
                                    ),
                                    "language_embedding": make_tensor_feature(
                                        ["512"], "float32",
                                        "Language embedding placeholder."
                                    ),
                                    "is_terminal": make_scalar_feature(
                                        "bool",
                                        "True on last step of the episode if it is a terminal step."
                                    ),
                                    "is_last": make_scalar_feature(
                                        "bool",
                                        "True on last step of the episode."
                                    ),
                                    "language_instruction": make_text_feature(
                                        "Language Instruction."
                                    ),
                                    "observation": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "cam_high": make_image_feature(
                                                    "High camera RGB observation."
                                                ),
                                                "state": make_tensor_feature(
                                                    ["14"], "float32",
                                                    "Robot state, 14-dim joint positions (qpos)."
                                                ),
                                                "cam_left_wrist": make_image_feature(
                                                    "Left wrist camera RGB observation."
                                                ),
                                                "cam_right_wrist": make_image_feature(
                                                    "Right wrist camera RGB observation."
                                                ),
                                            }
                                        },
                                    },
                                    "is_first": make_scalar_feature(
                                        "bool",
                                        "True on first step of the episode."
                                    ),
                                    "discount": make_scalar_feature(
                                        "float32",
                                        "Discount if provided, default to 1."
                                    ),
                                    "reward": make_scalar_feature(
                                        "float32",
                                        "Reward if provided, 1 on final step for demos."
                                    ),
                                }
                            },
                        },
                        "length": "-1",
                    },
                },
                "episode_metadata": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {
                        "features": {
                            "file_path": make_text_feature(
                                "Path to the original data file."
                            ),
                            "episode_id": make_scalar_feature(
                                "int32",
                                "ID of episode."
                            ),
                        }
                    },
                },
            }
        },
    }

    features_path = output_path / "features.json"
    with open(features_path, "w") as f:
        json.dump(features_json, f, indent=4)
    print(f"已生成: {features_path}")


def _generate_dataset_info_json(
    output_path: Path,
    dataset_name: str,
    num_shards: int,
    shard_lengths: list[str],
    total_bytes: int,
    total_steps: int,
):
    """生成 dataset_info.json"""

    dataset_info = {
        "citation": "",
        "description": f"Agilex robot dataset converted from HDF5 to RLDS format. Total steps: {total_steps}.",
        "fileFormat": "tfrecord",
        "moduleName": f"{dataset_name}.{dataset_name}_dataset_builder",
        "name": dataset_name,
        "releaseNotes": {
            "1.0.0": "Initial release."
        },
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "train",
                "numBytes": str(total_bytes),
                "shardLengths": shard_lengths,
            }
        ],
        "version": "1.0.0",
    }

    info_path = output_path / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"已生成: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将 Agilex HDF5 数据集转换为 RLDS (TFRecord) 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入 HDF5 数据目录。可以是单个任务目录（包含 .hdf5 文件），或包含多个任务子目录的父目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出 RLDS 数据集的根目录",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="agilex_dataset",
        help="数据集名称（用于 tfds.builder 加载），默认 agilex_dataset",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("HEIGHT", "WIDTH"),
        help="输出图像大小 (height width)，默认 256 256（与 bridge RLDS 对齐）",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=64,
        help="TFRecord shard 数量，默认 64",
    )

    args = parser.parse_args()

    # 扫描任务
    tasks = find_hdf5_tasks(args.input_dir)
    if not tasks:
        print(f"❌ 在 {args.input_dir} 中未找到任何 HDF5 数据")
        sys.exit(1)

    total_hdf5 = sum(len(t["hdf5_files"]) for t in tasks)
    print(f"找到 {len(tasks)} 个任务, 共 {total_hdf5} 个 episodes:")
    for task in tasks:
        print(f"  - {task['task_name']}: {len(task['hdf5_files'])} episodes")
        print(f"    instruction: '{task['instruction']}'")

    # 执行转换
    total_steps = build_rlds_dataset(
        tasks,
        args.output_dir,
        args.dataset_name,
        image_height=args.image_size[0],
        image_width=args.image_size[1],
        num_shards=args.num_shards,
    )

    print(f"\n📋 后续使用说明:")
    print(f"   数据集路径: {args.output_dir}")
    print(f"   数据集名称: {args.dataset_name}")
    print(f"")
    print(f"   在 config.py 中使用 RobotwinRLDSDataConfig 配置:")
    print(f"      rlds_data_dir='{args.output_dir}'")
    print(f"      dataset_name='{args.dataset_name}'")
    print(f"")
    print(f"   然后运行:")
    print(f"      python scripts/compute_norm_stats.py <config_name>")
    print(f"      python scripts/train_pytorch_xy.py <config_name>")


if __name__ == "__main__":
    main()
