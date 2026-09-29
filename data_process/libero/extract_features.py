"""
LIBERO Action Representation 提取

支持两种 feature_mode:
  - clean (默认):    forward 一次,time=0, noise=0 → x_t = GT actions
                     用 LAYER_INDICES = [0,5,10,17] (4 层)
  - rollout:         走 sample_actions 10 步去噪,在 t=0.1 提取 (与 main.py 一致)
                     用 ACTION_FEATURE_LAYER_INDICES = [0,2,4,6,8,10,12,14,16,17] (10 层)
                     与推理时 rollout 保存的 chunk feature 完全同分布,消除 train/rollout
                     的 (time, x_t) 不匹配问题

原理 (clean):
  Pi0 使用 flow matching 训练。forward 中，suffix（action expert）每层输出 hidden states（1024维）。
  固定 time=0, noise=0（x_t = actions，干净动作），提取指定层的 suffix hidden states，
  对 action token 位置做 mean pool，得到每个时间步的 action representation。再按 episode 聚合。

  为什么 time=0: flow matching 中 x_t = t*noise + (1-t)*actions。
  t=0 时 x_t=actions，模型看到真实动作，hidden states 语义信息最丰富。

原理 (rollout):
  推理时实际只在 t=0.1 提取 (sample_actions 的最后一步去噪)。clean 模式下 train/rollout
  的 t 和 x_t 都不一致 → probe 学到的是 "GT action → task" 的捷径,迁移到 rollout 时浅层退化。
  rollout 模式让模型在 train 集上跑完整推理 (从纯噪声 denoise 10 步),提取相同 t=0.1
  时刻的 hidden state → 与 main.py 的 chunk feature 同分布。

数据划分:
  使用 build_split.py 产生的 split.json，对每个 task 的 episodes 做 80/20 划分。
  --split train 提取训练集特征（用于训练线性探针）
  --split test  提取测试集特征（用于评估和相似度计算）
  --split all   提取全部（向后兼容）

存储格式:
  每个 (checkpoint, split) 产出一个 .npz 文件:
    features:        [N_episodes, L_layers, D=1024]  轨迹级特征（所有时间步 mean pool）
    task_indices:    [N_episodes]                      每条轨迹的 task_index
    episode_indices: [N_episodes]                      episode_index
    n_timesteps:     [N_episodes]                      每条轨迹的时间步数
    layer_indices:   [L]                               提取的层号

用法:
    python extract_features.py --checkpoint pretrained --split test
    python extract_features.py --checkpoint step_5k --split train
    python extract_features.py --checkpoint all --split all
"""
import os
import sys

# HuggingFace datasets 缓存重定向到数据盘，避免撑爆根分区。
# 远端保持原默认值，本机通过环境变量指定共享缓存。
_HF_CACHE = os.environ.get(
    "MIRROR_HF_CACHE",
    "/root/data/xuyuan1/.cache/huggingface",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_HF_CACHE, "datasets"))

import json
import argparse
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===================== 路径设置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENPI_DIR = os.environ.get(
    "OPENPI_ROOT",
    "/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi",
)
sys.path.insert(0, os.path.join(OPENPI_DIR, "src"))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    CHECKPOINTS, ALIGN_CHECKPOINTS, PI05_CHECKPOINTS, ALL_CHECKPOINTS,
    PI0_MODEL_CONFIG, PI05_MODEL_CONFIG, LAYER_INDICES,
    LIBERO_DATA_DIR, NORM_STATS_PATH, PI05_NORM_STATS_PATH, TASKS_WITH_ID_PATH,
    FEATURE_DIR, SPLIT_PATH,
)


def _resolve_model_variant(checkpoint_name: str) -> str:
    """ckpt entry 里的 model_variant 字段决定走 pi0 还是 pi05；缺省=pi0。"""
    return ALL_CHECKPOINTS[checkpoint_name].get("model_variant", "pi0")


def _model_config_for(variant: str) -> dict:
    if variant == "pi05":
        return PI05_MODEL_CONFIG
    return PI0_MODEL_CONFIG


def _norm_stats_path_for(variant: str) -> str:
    if variant == "pi05":
        return PI05_NORM_STATS_PATH
    return NORM_STATS_PATH


# ===================== 模型加载 =====================
def load_pi0_model(checkpoint_name: str, device="cuda"):
    """
    加载 Pi0 / Pi0.5 模型。支持 raw / alignment / pi05 三种 checkpoint，均为完整 safetensors。
    alignment checkpoint 中 LoRA 已合并，加载方式与 raw 完全一致。
    pi05 由 ckpt entry 里 model_variant="pi05" 触发，使用 PI05_MODEL_CONFIG（pi05=True，
    max_token_len=200，state→discrete tokens，time→adaRMSNorm）。
    """
    import safetensors.torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    cfg = ALL_CHECKPOINTS[checkpoint_name]
    weight_path = os.path.join(cfg["path"], "model.safetensors")

    variant = _resolve_model_variant(checkpoint_name)
    pi0_config = Pi0Config(**_model_config_for(variant))
    model = PI0Pytorch(pi0_config)
    safetensors.torch.load_model(model, weight_path, device=str(device))

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    print(f"模型加载完成: {checkpoint_name} (step={cfg['step']})")
    print(f"  路径: {weight_path}")
    print(f"  action_horizon={pi0_config.action_horizon}, action_dim={pi0_config.action_dim}")
    return model, pi0_config


# ===================== 数据加载 =====================
def load_norm_stats(path: str = None):
    """加载 LIBERO 的 norm_stats（state/actions 归一化参数）。
    path=None 时使用默认 NORM_STATS_PATH（pi0 路径）；pi05 调用方需显式传 PI05_NORM_STATS_PATH。
    """
    target = path or NORM_STATS_PATH
    with open(target) as f:
        raw = json.load(f)
    return raw["norm_stats"]


def build_transforms(pi0_config):
    """
    构建 LIBERO → Pi0 / Pi0.5 的数据变换链。
    复用 OpenPI 自带的 transform 类: LiberoInputs → Normalize → ResizeImages → TokenizePrompt → Pad

    pi0.5 差异（由 pi0_config.pi05 / discrete_state_input 自动驱动）：
      - PaligemmaTokenizer 的 max_len 由 pi0_config.max_token_len 决定（pi0=48 / pi05=200）
      - TokenizePrompt(discrete_state_input=True) 时把离散化 state 拼进 prompt token
      - norm_stats 从 pi05 ckpt 自带的 PI05_NORM_STATS_PATH 读
    """
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.transforms import Normalize, TokenizePrompt, PadStatesAndActions, ResizeImages
    from openpi.shared.normalize import NormStats
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models.model import ModelType

    is_pi05 = bool(getattr(pi0_config, "pi05", False))
    norm_stats_path = PI05_NORM_STATS_PATH if is_pi05 else NORM_STATS_PATH
    raw = load_norm_stats(norm_stats_path)
    norm_stats = {}
    for key in ["state", "actions"]:
        ns = raw[key]
        norm_stats[key] = NormStats(
            mean=np.array(ns["mean"], dtype=np.float32),
            std=np.array(ns["std"], dtype=np.float32),
            q01=np.array(ns["q01"], dtype=np.float32) if ns.get("q01") else None,
            q99=np.array(ns["q99"], dtype=np.float32) if ns.get("q99") else None,
        )

    transforms = [
        LiberoInputs(model_type=ModelType.PI05 if is_pi05 else ModelType.PI0),
        Normalize(norm_stats, use_quantiles=True),
        ResizeImages(224, 224),
        TokenizePrompt(
            PaligemmaTokenizer(pi0_config.max_token_len),
            discrete_state_input=is_pi05,
        ),
        PadStatesAndActions(pi0_config.action_dim),
    ]
    return transforms


def apply_transforms(sample, transforms):
    for t in transforms:
        sample = t(sample)
    return sample


def load_split(split_name="all"):
    """
    加载 split.json，返回需要提取的 episode_index 集合。
    兼容 task 级别和 episode 级别两种划分格式。

    split_name: "train" / "test" / "all"
    返回: set of episode_index 或 None (all 模式不过滤)
    """
    if split_name == "all":
        return None

    assert os.path.exists(SPLIT_PATH), (
        f"split.json 不存在: {SPLIT_PATH}\n请先运行 python build_split.py"
    )
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    mode = split.get("mode", "episode")
    print(f"  Split mode: {mode}, requested: {split_name}")
    if mode == "task":
        print(f"  Train tasks: {split['train_tasks']}")
        print(f"  Test tasks: {split['test_tasks']}")

    if split_name == "train":
        return set(split["all_train"])
    elif split_name == "test":
        return set(split["all_test"])
    else:
        raise ValueError(f"未知 split: {split_name}, 可选: train / test / all")


class LiberoFeatureDataset(Dataset):
    """
    LIBERO LeRobot 数据集，按帧加载，记录 episode_index 用于轨迹级聚合。

    关键: 保留 episode_index 信息，确保同一 episode 的所有 frame 聚合到一条轨迹。
    支持通过 episode_filter 过滤，只保留指定 episode 的帧。
    """

    def __init__(self, transforms, norm_stats_raw, episode_filter=None):
        """
        Args:
            episode_filter: set of int 或 None。只保留这些 episode 的帧。None=不过滤。
        """
        from openpi.training.data_loader import CompatLeRobotDataset
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        meta = lerobot_dataset.LeRobotDatasetMetadata(LIBERO_DATA_DIR)
        action_horizon = PI0_MODEL_CONFIG["action_horizon"]

        full_dataset = CompatLeRobotDataset(
            LIBERO_DATA_DIR,
            delta_timestamps={
                "actions": [t / meta.fps for t in range(action_horizon)],
            },
        )

        # 如果有 episode_filter，批量获取 episode_index 列后过滤
        if episode_filter is not None:
            all_ep_indices = np.array(full_dataset.hf_dataset["episode_index"])
            mask = np.isin(all_ep_indices, np.array(list(episode_filter)))
            self.valid_indices = np.where(mask)[0].tolist()
            print(f"  Episode 过滤: {len(full_dataset)} 帧 → {len(self.valid_indices)} 帧")
        else:
            self.valid_indices = list(range(len(full_dataset)))

        self.dataset = full_dataset
        self.tasks = meta.tasks
        self.transforms = transforms

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.dataset[real_idx]

        task_index = int(item["task_index"])
        episode_index = int(item["episode_index"])
        frame_index = int(item["frame_index"])
        prompt = self.tasks[task_index]

        raw_sample = {
            "observation/image": np.array(item["image"]),
            "observation/wrist_image": np.array(item["wrist_image"]),
            "observation/state": np.array(item["state"]),
            "actions": np.array(item["actions"]),
            "prompt": prompt,
            "task_index": task_index,
        }

        processed = apply_transforms(raw_sample, self.transforms)

        return {
            "image": {k: np.asarray(v) for k, v in processed["image"].items()},
            "image_mask": {k: np.asarray(v) for k, v in processed["image_mask"].items()},
            "state": np.asarray(processed["state"]),
            "actions": np.asarray(processed["actions"]),
            "tokenized_prompt": np.asarray(processed["tokenized_prompt"]),
            "tokenized_prompt_mask": np.asarray(processed["tokenized_prompt_mask"]),
            "task_index": task_index,
            "episode_index": episode_index,
            "frame_index": frame_index,
        }


def collate_fn(features):
    """Collator: string/scalar 字段单独处理，array 字段 stack"""
    meta_keys = ["task_index", "episode_index", "frame_index"]
    meta = {k: [f.pop(k) for f in features] for k in meta_keys}

    batch = {}
    for key in features[0]:
        if isinstance(features[0][key], dict):
            batch[key] = {
                k: np.stack([f[key][k] for f in features], axis=0)
                for k in features[0][key]
            }
        else:
            batch[key] = np.stack([f[key] for f in features], axis=0)

    for k, v in meta.items():
        batch[k] = v
    return batch


# ===================== 特征提取核心 =====================
ROLLOUT_LAYER_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]


@torch.no_grad()
def extract_features(checkpoint_name: str, split_name="all",
                     device="cuda", batch_size=8, max_frames=None,
                     feature_mode: str = "clean", num_steps: int = 10,
                     noise_seed: int = 0):
    """
    从指定 checkpoint 提取 action hidden states，按 episode 聚合为轨迹级特征。

    Args:
        checkpoint_name: CHECKPOINTS 中的 key
        split_name: "train" / "test" / "all"
        max_frames: 最多处理多少帧（None=全部，调试用）
        feature_mode: "clean" (forward time=0, noise=0) 或
                      "rollout" (sample_actions 10 步去噪,t=0.1 提取,与推理一致)
        num_steps: rollout 模式去噪步数,默认 10
        noise_seed: rollout 模式的噪声种子,保证可复现

    Returns:
        output_path: 保存的 .npz 路径
    """
    from openpi.models.model import Observation

    assert feature_mode in ("clean", "rollout", "noisy_gt"), f"unknown feature_mode: {feature_mode}"

    # 1. 加载模型
    model, pi0_config = load_pi0_model(checkpoint_name, device)
    action_horizon = pi0_config.action_horizon  # 50

    # 2. 构建变换 + 数据集（按 split 过滤 episode）
    norm_stats_raw = load_norm_stats()
    transforms = build_transforms(pi0_config)
    episode_filter = load_split(split_name)
    dataset = LiberoFeatureDataset(transforms, norm_stats_raw, episode_filter=episode_filter)

    if max_frames and max_frames < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(max_frames)))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, shuffle=False,
    )

    # 3. 按 episode 收集特征
    episode_data = defaultdict(lambda: {"features": [], "task_index": -1})

    if feature_mode == "clean":
        layer_indices = LAYER_INDICES
    else:
        # rollout 与 noisy_gt 都在 t=0.1 截面，使用同一组 layer，保证两边可直接对比
        layer_indices = ROLLOUT_LAYER_INDICES

    print(f"\n提取配置:")
    print(f"  Checkpoint: {checkpoint_name} (step={ALL_CHECKPOINTS[checkpoint_name]['step']})")
    print(f"  Feature mode: {feature_mode}")
    if feature_mode == "rollout":
        print(f"  Num denoise steps: {num_steps}, noise_seed: {noise_seed}")
    print(f"  Split: {split_name}")
    print(f"  总帧数: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  提取层: {layer_indices}")
    print(f"  Action horizon: {action_horizon}")

    # rollout / noisy_gt 模式都需要采样 noise → 共享一个固定 generator 保证可复现
    rollout_gen = None
    if feature_mode in ("rollout", "noisy_gt"):
        rollout_gen = torch.Generator(device=device)
        rollout_gen.manual_seed(int(noise_seed))

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in tqdm(dataloader, desc=f"[{checkpoint_name}/{feature_mode}]"):
            cur_bs = batch["state"].shape[0]

            obs_dict = {
                "image": {k: torch.from_numpy(v).to(device) for k, v in batch["image"].items()},
                "image_mask": {k: torch.from_numpy(v).to(device) for k, v in batch["image_mask"].items()},
                "state": torch.from_numpy(batch["state"]).to(device),
                "tokenized_prompt": torch.from_numpy(batch["tokenized_prompt"]).to(device),
                "tokenized_prompt_mask": torch.from_numpy(batch["tokenized_prompt_mask"]).to(device),
            }
            observation = Observation.from_dict(obs_dict)

            if feature_mode == "clean":
                actions = torch.from_numpy(batch["actions"]).float().to(device)
                time_zero = torch.zeros(cur_bs, device=device, dtype=torch.float32)
                noise_zero = torch.zeros_like(actions)
                result = model.forward(
                    observation, actions,
                    noise=noise_zero, time=time_zero,
                    output_hidden_states=True,
                    train=False,
                )
                _, suffix_hidden_states, _ = result
                selected = torch.stack(
                    [suffix_hidden_states[li][:, -action_horizon:, :] for li in layer_indices],
                    dim=0,
                )  # [L, B, 50, 1024]
                frame_feats = selected.mean(dim=2)  # [L, B, 1024]
                frame_feats = frame_feats.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]
            elif feature_mode == "noisy_gt":
                # noisy_gt 模式：取 flow matching 训练分布在 t=0.1 处的 sample
                # x_t = 0.1 * noise + 0.9 * actions_gt
                # 与 rollout 共享同一个 (t, noise level) 截面，唯一差别是这里用 GT 而 rollout 用 model_pred
                # → probe 在 noisy_gt 上训练 = 学 "GT-anchored alignment"
                # → probe 应用到 rollout = 测 "模型预测是否仍然指向 GT 语义"
                actions = torch.from_numpy(batch["actions"]).float().to(device)
                noise = torch.randn(actions.shape, device=device, generator=rollout_gen)
                time_val = torch.full((cur_bs,), 0.1, device=device, dtype=torch.float32)
                # model.forward 内部会算 x_t = time * noise + (1-time) * actions
                result = model.forward(
                    observation, actions,
                    noise=noise, time=time_val,
                    output_hidden_states=True,
                    train=False,
                )
                _, suffix_hidden_states, _ = result
                selected = torch.stack(
                    [suffix_hidden_states[li][:, -action_horizon:, :] for li in layer_indices],
                    dim=0,
                )
                frame_feats = selected.mean(dim=2)
                frame_feats = frame_feats.permute(1, 0, 2).float().cpu().numpy()
            else:
                # rollout 模式: 走完整 sample_actions, 在 t=0.1 提取
                # output_action_features=True 时返回 (x_t_predicted, action_feature_np[B,L,D])
                # action_feature_np 已在模型内 mean-pool 50 个 action token, 转 cpu numpy
                _, frame_feats = model.sample_actions(
                    device, observation,
                    num_steps=num_steps,
                    generator=rollout_gen,
                    output_action_features=True,
                )
                # frame_feats: numpy [B, 10, 1024], 层固定为 ROLLOUT_LAYER_INDICES

            for j in range(cur_bs):
                ep_idx = batch["episode_index"][j]
                task_idx = batch["task_index"][j]
                episode_data[ep_idx]["features"].append(frame_feats[j])
                episode_data[ep_idx]["task_index"] = task_idx

    # 4. Level 2 聚合: 每条轨迹所有时间步 mean pool → [L, D]
    # 同时把 Level 1（per-frame / per-chunk）特征展平保存,便于 chunk 级 alignment
    print(f"\nLevel 2 聚合: {len(episode_data)} 条轨迹")

    traj_features = []          # [N_ep, L, D] 轨迹级 mean
    traj_task_indices = []      # [N_ep]
    traj_episode_indices = []   # [N_ep]
    traj_n_timesteps = []       # [N_ep]

    chunk_features = []         # [N_total_chunks, L, D] 每帧（每个 action chunk）的 mean
    chunk_task_indices = []     # [N_total_chunks]
    chunk_episode_indices = []  # [N_total_chunks]
    chunk_frame_indices = []    # [N_total_chunks] 在轨迹内的 frame 位置（0,1,2,...）

    for ep_idx in sorted(episode_data.keys()):
        data = episode_data[ep_idx]
        if not data["features"]:
            continue
        feats = np.stack(data["features"], axis=0)  # [T, L, D]
        # 轨迹级 mean pool
        traj_feat = feats.mean(axis=0)  # [L, D]
        traj_features.append(traj_feat)
        traj_task_indices.append(data["task_index"])
        traj_episode_indices.append(ep_idx)
        traj_n_timesteps.append(feats.shape[0])

        # 展平 chunk 级特征
        chunk_features.append(feats)
        chunk_task_indices.extend([data["task_index"]] * feats.shape[0])
        chunk_episode_indices.extend([ep_idx] * feats.shape[0])
        chunk_frame_indices.extend(list(range(feats.shape[0])))

    traj_features = np.stack(traj_features, axis=0)             # [N_ep, L, D]
    traj_task_indices = np.array(traj_task_indices, dtype=np.int64)
    traj_episode_indices = np.array(traj_episode_indices, dtype=np.int64)
    traj_n_timesteps = np.array(traj_n_timesteps, dtype=np.int64)

    chunk_features = np.concatenate(chunk_features, axis=0)     # [N_chunks, L, D]
    chunk_task_indices = np.array(chunk_task_indices, dtype=np.int64)
    chunk_episode_indices = np.array(chunk_episode_indices, dtype=np.int64)
    chunk_frame_indices = np.array(chunk_frame_indices, dtype=np.int64)

    # 5. 保存: 文件名格式 {checkpoint}_{split}[_mode].npz
    # rollout / noisy_gt 模式加对应后缀，clean 不加
    os.makedirs(FEATURE_DIR, exist_ok=True)
    split_suffix = f"_{split_name}" if split_name != "all" else ""
    mode_suffix_map = {"clean": "", "rollout": "_rollout", "noisy_gt": "_noisy_gt"}
    mode_suffix = mode_suffix_map[feature_mode]
    output_path = os.path.join(FEATURE_DIR, f"{checkpoint_name}{split_suffix}{mode_suffix}.npz")
    np.savez_compressed(
        output_path,
        # 轨迹级（兼容旧 train_probe.py / eval_similarity.py）
        features=traj_features,
        task_indices=traj_task_indices,
        episode_indices=traj_episode_indices,
        n_timesteps=traj_n_timesteps,
        layer_indices=np.array(layer_indices),
        # chunk 级（新增,Part B 用）
        chunk_features=chunk_features,
        chunk_task_indices=chunk_task_indices,
        chunk_episode_indices=chunk_episode_indices,
        chunk_frame_indices=chunk_frame_indices,
    )

    print(f"\n保存: {output_path}")
    print(f"  traj features:  {traj_features.shape}  (N_ep={len(traj_features)})")
    print(f"  chunk features: {chunk_features.shape} (N_chunks={len(chunk_features)})")
    print(f"  唯一 task 数: {len(np.unique(traj_task_indices))}")
    print(f"  平均时间步/轨迹: {traj_n_timesteps.mean():.1f}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="LIBERO action representation 提取")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint 名称 (如 pretrained, step_5k, align_new_30k, 或 all)")
    parser.add_argument("--group", type=str, default="raw",
                        choices=["raw", "align", "pi05", "all"],
                        help="checkpoint 组: raw(全参数), align(alignment+LoRA), pi05(pi0.5 系列), all(全部)")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"],
                        help="数据划分: train(探针训练) / test(评估) / all(全部)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="最多处理帧数 (调试用)")
    parser.add_argument("--feature_mode", type=str, default="clean",
                        choices=["clean", "rollout", "noisy_gt"],
                        help="clean: forward(t=0,noise=0); rollout: sample_actions 10 步 t=0.1; "
                             "noisy_gt: forward(t=0.1, x_t=0.1*noise+0.9*GT) — canonical setup, "
                             "与 rollout 同截面但用 GT 替代 model_pred，用于训 alignment probe")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="rollout 模式去噪步数")
    parser.add_argument("--noise_seed", type=int, default=0,
                        help="rollout 模式噪声种子")
    args = parser.parse_args()

    # 根据 group 选择 checkpoint 字典
    ckpt_map = {
        "raw": CHECKPOINTS,
        "align": ALIGN_CHECKPOINTS,
        "pi05": PI05_CHECKPOINTS,
        "all": ALL_CHECKPOINTS,
    }
    target_ckpts = ckpt_map[args.group]

    if args.checkpoint == "all":
        for name in target_ckpts:
            print(f"\n{'='*60}")
            print(f"提取 checkpoint: {name} (split={args.split}, mode={args.feature_mode})")
            print(f"{'='*60}")
            extract_features(name, args.split, args.device, args.batch_size, args.max_frames,
                             feature_mode=args.feature_mode, num_steps=args.num_steps,
                             noise_seed=args.noise_seed)
    else:
        extract_features(args.checkpoint, args.split, args.device, args.batch_size, args.max_frames,
                         feature_mode=args.feature_mode, num_steps=args.num_steps,
                         noise_seed=args.noise_seed)


if __name__ == "__main__":
    main()
