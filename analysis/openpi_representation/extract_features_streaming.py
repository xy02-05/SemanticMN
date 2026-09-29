"""
OpenPI (PI0) 全量流式特征提取脚本

核心设计:
  1. 从 Bridge RLDS 数据集按轨迹迭代（不展平、不shuffle）
  2. 每条轨迹：解码图像 → 构建 action chunk → apply OpenPI transforms → 模型 forward
  3. 两级 mean pool:
     Level 1: 每个 timestep → mean_pool(action_tokens) → [L, D]
     Level 2: 每条轨迹 → mean_pool(timestep_features) → [L, D]
  4. 每条轨迹关联 prompt、task_index、task_id
  5. 每处理 save_interval 条轨迹保存一次 checkpoint

复用:
  - 模型加载/transforms/forward 全部复用 openpi_representation/extract_features.py
  - RLDS 加载参考 openpi/training/droid_rlds_dataset.py 的 Bridgev2RldsDataset
  - 任务映射复用 load_task_mapping_for_rlds

用法:
    python extract_features_streaming.py --model_name pretrained
    python extract_features_streaming.py --model_name raw_ft --save_interval 500
"""
import os
import sys
import io
import argparse
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

# ===================== 路径配置 =====================
ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
OPENPI_DIR = os.path.join(ANALYSIS_DIR, "openpi")
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, os.path.join(OPENPI_DIR, "src"))

# 复用现有模块
from openpi_representation.config import (
    MODEL_CONFIGS, LAYER_INDICES, PI0_MODEL_CONFIG,
    OUTPUT_DIR, NORM_STATS_PATH, TASKS_WITH_ID_PATH,
)
from openpi_representation.extract_features import (
    load_pi0_model, build_transforms, apply_transforms, load_norm_stats,
)

# ===================== 流式输出路径 =====================
STREAMING_DIR = os.path.join(OUTPUT_DIR, "streaming")
STREAMING_FEATURE_DIR = os.path.join(STREAMING_DIR, "features")

# ===================== Bridge RLDS 数据路径 =====================
RLDS_DATA_DIR = "/root/data/xuyuan1/Codes/mirror_neuron/data"
RLDS_DATASET_NAME = "bridge_orig"


# ===================== 轨迹级 RLDS 迭代器 =====================
class BridgeTrajectoryIterator:
    """
    从 Bridge RLDS 数据集按轨迹迭代（不展平、不 repeat）

    与 Bridgev2RldsDataset 的区别:
    - 不展平(flatten): 保留轨迹边界, 每次 yield 一整条轨迹
    - 不 repeat: 遍历一次后结束
    - 不 shuffle: 保持原始顺序
    - 包含 task_index/task_id 映射
    """

    def __init__(
        self,
        data_dir=RLDS_DATA_DIR,
        dataset_name=RLDS_DATASET_NAME,
        action_chunk_size=5,
        task_mapping_path=TASKS_WITH_ID_PATH,
    ):
        # 延迟导入 TF (避免非 RLDS 用户的依赖)
        import tensorflow as tf
        import tensorflow_datasets as tfds
        import dlimp as dl

        tf.config.set_visible_devices([], "GPU")

        # 加载任务映射: prompt → task_index, task_index → task_id
        from openpi.training.droid_rlds_dataset import load_task_mapping_for_rlds
        self.lang_to_task_index, self.task_index_to_task_id = \
            load_task_mapping_for_rlds(task_mapping_path)

        self.action_chunk_size = action_chunk_size

        # 构建 RLDS 数据集 (轨迹级, 不 flatten/repeat/shuffle)
        print(f"Loading Bridge RLDS: {dataset_name} from {data_dir}")
        builder = tfds.builder(dataset_name, data_dir=data_dir, version="1.0.0")
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False)

        # 过滤空 prompt 轨迹
        dataset = dataset.filter(lambda traj: tf.reduce_any(traj["language_instruction"] != b""))
        self.dataset = dataset
        print("Bridge RLDS dataset ready (trajectory-level, no flatten)")

    def __iter__(self):
        """
        每次 yield 一条轨迹 dict:
          images: list[np.ndarray] [H,W,3] uint8 (已解码)
          actions: np.ndarray [T, chunk_size, 7]
          states: np.ndarray [T, 7]
          prompt: str
          task_index: int
          task_id: int
          traj_len: int
        """
        for traj in self.dataset.as_numpy_iterator():
            # --- 解析原始字段 ---
            raw_actions = traj["action"]                   # [T, 7]
            raw_images = traj["observation"]["image_0"]    # [T, ...] 编码的 JPEG bytes
            raw_states = traj["observation"]["state"]      # [T, 7]
            raw_prompt = traj["language_instruction"]       # [T] bytes (全部相同)

            traj_len = len(raw_actions)
            if traj_len == 0:
                continue

            # prompt: 取第一个, bytes → str
            prompt_bytes = raw_prompt[0] if isinstance(raw_prompt, np.ndarray) else raw_prompt
            prompt = prompt_bytes.decode("utf-8") if isinstance(prompt_bytes, bytes) else str(prompt_bytes)

            # 查找 task_index 和 task_id
            prompt_norm = prompt.strip().lower()
            task_index = self.lang_to_task_index.get(prompt_norm, -1)
            task_id = self.task_index_to_task_id.get(task_index, task_index)

            # --- 解码图像 ---
            # RLDS 中图像是 JPEG 编码的 bytes, 需要逐帧解码
            images = []
            for enc in raw_images:
                img = np.array(Image.open(io.BytesIO(enc)))  # [H, W, 3] uint8
                images.append(img)

            # --- 构建 action chunks ---
            # 每个时间步 t 的 action chunk: [action[t], ..., action[t+chunk-1]]
            # 超出轨迹末尾的部分用最后一帧补齐 (与训练一致)
            chunk_size = self.action_chunk_size
            action_chunks = np.zeros((traj_len, chunk_size, raw_actions.shape[-1]), dtype=np.float32)
            for t in range(traj_len):
                for c in range(chunk_size):
                    idx = min(t + c, traj_len - 1)
                    action_chunks[t, c] = raw_actions[idx]

            yield {
                "images": images,             # list of [H,W,3] uint8
                "actions": action_chunks,     # [T, chunk_size, 7]
                "states": raw_states,         # [T, 7]
                "prompt": prompt,
                "task_index": task_index,
                "task_id": task_id,
                "traj_len": traj_len,
            }


# ===================== 单条轨迹处理 =====================
def process_trajectory(traj_data, transforms, state_fill, model, pi0_config,
                       layer_indices, device, batch_size=16):
    """
    对一条轨迹的所有帧提取特征, 返回 Level 2 mean pool 后的轨迹级特征 [L, D]

    流程:
      1. 每帧构建 OpenPI 输入 (images/state/actions/prompt) → apply transforms
      2. 分批 forward PI0 模型 (time=0, 无噪声)
      3. 从 suffix_hidden_states 提取 action token 特征
      4. Level 1: 每帧 mean(action_tokens) → [L, D]
      5. Level 2: mean(所有帧) → [L, D]
    """
    from openpi.models.model import Observation

    images = traj_data["images"]
    actions = traj_data["actions"]       # [T, 5, 7]
    prompt = traj_data["prompt"]
    traj_len = traj_data["traj_len"]
    action_horizon = pi0_config.action_horizon  # 5

    # --- 1. 对每帧 apply transforms ---
    processed_frames = []
    for t in range(traj_len):
        raw_sample = {
            "images": {"cam_high": images[t]},     # [H, W, 3] uint8
            "state": state_fill.copy(),              # [7] float32
            "actions": actions[t],                   # [5, 7] float32
            "prompt": prompt,
        }
        processed = apply_transforms(raw_sample, transforms)
        processed_frames.append({
            "image": {k: np.asarray(v) for k, v in processed["image"].items()},
            "image_mask": {k: np.asarray(v) for k, v in processed["image_mask"].items()},
            "state": np.asarray(processed["state"]),
            "actions": np.asarray(processed["actions"]),
            "tokenized_prompt": np.asarray(processed["tokenized_prompt"]),
            "tokenized_prompt_mask": np.asarray(processed["tokenized_prompt_mask"]),
        })

    # --- 2. 分批 forward, 收集每帧特征 ---
    all_frame_feats = []  # 每个元素: [L, D]

    for start in range(0, traj_len, batch_size):
        end = min(start + batch_size, traj_len)
        batch_frames = processed_frames[start:end]
        cur_bs = len(batch_frames)

        # 手动 collate: stack array, 嵌套 dict 也 stack
        batch = {}
        for key in batch_frames[0]:
            if isinstance(batch_frames[0][key], dict):
                batch[key] = {
                    k: np.stack([f[key][k] for f in batch_frames], axis=0)
                    for k in batch_frames[0][key]
                }
            else:
                batch[key] = np.stack([f[key] for f in batch_frames], axis=0)

        # 构造 Observation, 移到 GPU
        obs_dict = {
            "image": {k: torch.from_numpy(v).to(device) for k, v in batch["image"].items()},
            "image_mask": {k: torch.from_numpy(v).to(device) for k, v in batch["image_mask"].items()},
            "state": torch.from_numpy(batch["state"]).to(device),
            "tokenized_prompt": torch.from_numpy(batch["tokenized_prompt"]).to(device),
            "tokenized_prompt_mask": torch.from_numpy(batch["tokenized_prompt_mask"]).to(device),
        }
        observation = Observation.from_dict(obs_dict)
        actions_tensor = torch.from_numpy(batch["actions"]).to(device)

        # Forward: time=0 → 无噪声, 确定性特征
        time_zero = torch.zeros(cur_bs, device=device, dtype=torch.float32)
        noise_zero = torch.zeros_like(actions_tensor)

        _, suffix_hidden_states, _ = model.forward(
            observation, actions_tensor,
            noise=noise_zero, time=time_zero,
            output_hidden_states=True,
        )

        # suffix_hidden_states: tuple of [B, suffix_len, D] for each layer
        # 提取 action token 位置 (最后 action_horizon 个位置)
        # 选取指定层, stack → [L, B, action_horizon, D]
        selected_layers = torch.stack(
            [suffix_hidden_states[li][:, -action_horizon:, :] for li in layer_indices],
            dim=0,
        )  # [L, B, 5, D]

        # Level 1: mean across action tokens → [L, B, D]
        frame_feats = selected_layers.mean(dim=2)  # [L, B, D]
        frame_feats = frame_feats.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]

        for j in range(cur_bs):
            all_frame_feats.append(frame_feats[j])  # [L, D]

    # --- 3. Level 2: mean across all frames → [L, D] ---
    traj_feature = np.mean(np.stack(all_frame_feats, axis=0), axis=0)  # [L, D]
    return traj_feature


# ===================== 保存功能 =====================
def save_checkpoint(results, model_name, chunk_id, final=False):
    """
    保存特征 checkpoint

    results: list of dict, 每个 dict 包含:
      feature: [L, D]
      prompt: str
      task_index: int
      task_id: int
      n_timesteps: int
    """
    os.makedirs(STREAMING_FEATURE_DIR, exist_ok=True)

    features = np.stack([r["feature"] for r in results], axis=0)      # [N, L, D]
    task_labels = np.array([r["prompt"] for r in results])             # [N]
    task_indices = np.array([r["task_index"] for r in results], dtype=np.int64)
    task_ids = np.array([r["task_id"] for r in results], dtype=np.int64)
    n_timesteps = np.array([r["n_timesteps"] for r in results], dtype=np.int64)

    if final:
        fname = f"{model_name}_streaming_features.npz"
    else:
        fname = f"{model_name}_streaming_chunk{chunk_id:04d}.npz"

    output_path = os.path.join(STREAMING_FEATURE_DIR, fname)
    np.savez_compressed(
        output_path,
        features=features,
        task_labels=task_labels,
        task_indices=task_indices,
        task_ids=task_ids,
        n_timesteps=n_timesteps,
        layer_indices=np.array(LAYER_INDICES),
    )

    tag = "FINAL" if final else f"CHUNK-{chunk_id}"
    print(f"  [{tag}] Saved: {output_path}")
    print(f"    N={len(results)}, shape={features.shape}, "
          f"unique_tasks={len(set(r['prompt'] for r in results))}")
    return output_path


# ===================== 主提取流程 =====================
def extract_streaming(model_name, device="cuda", batch_size=16,
                      save_interval=500, max_traj_per_task=None):
    """
    全量流式提取: 遍历 Bridge RLDS 全部轨迹, 每 save_interval 条保存

    Args:
        model_name: 模型名 (pretrained / raw_ft / aligned)
        batch_size: 每条轨迹内的帧批量大小
        save_interval: 每处理多少条轨迹保存一次
        max_traj_per_task: 每个 task 最多处理多少条轨迹 (None=不限)
    """
    # 1. 加载模型
    model, pi0_config = load_pi0_model(model_name, device)
    action_horizon = pi0_config.action_horizon

    # 2. 构建 transforms
    transforms_chain = build_transforms(pi0_config)

    # 3. 预计算 state_fill: 用 norm_stats.mean 填充 (归一化后为 0, 所有轨迹共用)
    raw_norm = load_norm_stats()
    state_fill = np.array(raw_norm["state"]["mean"], dtype=np.float32)

    # 4. 创建轨迹迭代器
    traj_iter = BridgeTrajectoryIterator(
        action_chunk_size=action_horizon,
    )

    # 5. 提取循环
    results = []          # 当前 chunk 的结果
    all_saved_paths = []  # 所有已保存的文件路径
    chunk_id = 0
    total_traj = 0
    task_traj_count = {}  # task_index → 已处理轨迹数 (用于 per-task 限制)

    print(f"\n{'=' * 60}")
    print(f"Streaming extraction: {MODEL_CONFIGS[model_name]['label']}")
    print(f"  batch_size={batch_size}, save_interval={save_interval}")
    print(f"  max_traj_per_task={max_traj_per_task}")
    print(f"  layers={LAYER_INDICES}")
    print(f"  action_horizon={action_horizon}")
    print(f"  output_dir={STREAMING_FEATURE_DIR}")
    print(f"{'=' * 60}\n")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for traj_data in tqdm(traj_iter, desc=f"[{model_name}] trajectories"):
            task_idx = traj_data["task_index"]

            # per-task 轨迹数限制
            if max_traj_per_task is not None:
                cnt = task_traj_count.get(task_idx, 0)
                if cnt >= max_traj_per_task:
                    continue
                task_traj_count[task_idx] = cnt + 1

            # 处理单条轨迹 → [L, D]
            traj_feat = process_trajectory(
                traj_data, transforms_chain, state_fill,
                model, pi0_config, LAYER_INDICES,
                device, batch_size=batch_size,
            )

            results.append({
                "feature": traj_feat,
                "prompt": traj_data["prompt"],
                "task_index": traj_data["task_index"],
                "task_id": traj_data["task_id"],
                "n_timesteps": traj_data["traj_len"],
            })
            total_traj += 1

            # 定期保存
            if len(results) >= save_interval:
                path = save_checkpoint(results, model_name, chunk_id)
                all_saved_paths.append(path)
                results = []
                chunk_id += 1

    # 保存剩余
    if results:
        path = save_checkpoint(results, model_name, chunk_id)
        all_saved_paths.append(path)

    # 合并所有 chunk 为最终文件
    print(f"\n{'=' * 60}")
    print(f"合并 {len(all_saved_paths)} 个 chunk 文件...")
    _merge_chunks(all_saved_paths, model_name)
    print(f"总轨迹数: {total_traj}")
    print(f"{'=' * 60}")


def _merge_chunks(chunk_paths, model_name):
    """合并所有 chunk npz 文件为一个最终文件"""
    all_features, all_labels, all_indices, all_ids, all_nts = [], [], [], [], []

    for p in chunk_paths:
        data = np.load(p, allow_pickle=True)
        all_features.append(data["features"])
        all_labels.append(data["task_labels"])
        all_indices.append(data["task_indices"])
        all_ids.append(data["task_ids"])
        all_nts.append(data["n_timesteps"])

    merged = {
        "features": np.concatenate(all_features, axis=0),
        "task_labels": np.concatenate(all_labels, axis=0),
        "task_indices": np.concatenate(all_indices, axis=0),
        "task_ids": np.concatenate(all_ids, axis=0),
        "n_timesteps": np.concatenate(all_nts, axis=0),
        "layer_indices": np.array(LAYER_INDICES),
    }

    final_path = os.path.join(STREAMING_FEATURE_DIR, f"{model_name}_streaming_features.npz")
    np.savez_compressed(final_path, **merged)

    N = merged["features"].shape[0]
    unique_tasks = len(set(merged["task_labels"]))
    print(f"  FINAL: {final_path}")
    print(f"    N={N}, shape={merged['features'].shape}, unique_tasks={unique_tasks}")

    # 删除 chunk 文件
    for p in chunk_paths:
        if os.path.exists(p) and p != final_path:
            os.remove(p)
    print(f"    已清理 {len(chunk_paths)} 个 chunk 文件")


# ===================== 入口 =====================
def main():
    parser = argparse.ArgumentParser(description="OpenPI Bridge 全量流式特征提取")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每条轨迹内的帧批量大小")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="每处理多少条轨迹保存一次")
    parser.add_argument("--max_traj_per_task", type=int, default=None,
                        help="每个 task 最多处理多少条轨迹 (默认不限)")
    args = parser.parse_args()

    extract_streaming(
        args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        max_traj_per_task=args.max_traj_per_task,
    )


if __name__ == "__main__":
    main()
