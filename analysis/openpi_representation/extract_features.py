"""
OpenPI (PI0) 表征提取脚本

核心设计（对标 bridge_representation/extract_features.py）:
  1. 复用 SpatialVLA 的固定数据子集 → 所有模型输入 100% 一致
  2. CachedOpenPIDataset: 从 .npz 加载原始图像 → OpenPI transform → model-ready
  3. action_horizon=5, suffix = state(1) + action_tokens(5)
  4. 使用 time=0（无噪声）获得确定性特征
  5. 两级 mean pool:
     Level 1: 每个 timestep → mean_pool(action_tokens) → [L, D]
     Level 2: 每条轨迹 → mean_pool(timestep_features) → [L, D]
  6. 输出格式与 SpatialVLA 一致，可直接用 analyze_representation.py 分析

用法:
    python extract_features.py --model_name pretrained
    python extract_features.py --model_name raw_ft
    python extract_features.py --model_name aligned
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===================== 路径配置 =====================
ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
OPENPI_DIR = os.path.join(ANALYSIS_DIR, "openpi")
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, os.path.join(OPENPI_DIR, "src"))

from openpi_representation.config import (
    MODEL_CONFIGS, LAYER_INDICES, PI0_MODEL_CONFIG,
    FEATURE_DIR, OUTPUT_DIR, SUBSET_DIR, NORM_STATS_PATH,
)


# ===================== 模型加载 =====================
def load_pi0_model(model_name: str, device="cuda"):
    """
    加载 PI0Pytorch 模型

    三种检查点的 model.safetensors 都是干净的推理权重（LoRA 已合并），
    可以统一用 safetensors.torch.load_model 加载。
    """
    import safetensors.torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    cfg = MODEL_CONFIGS[model_name]
    weight_path = cfg["weight_path"]
    safetensors_path = os.path.join(weight_path, "model.safetensors")

    print(f"\n{'=' * 60}")
    print(f"Loading PI0 model: {cfg['label']}")
    print(f"  Weights: {safetensors_path}")

    # 构造 Pi0Config（与训练时一致）
    pi0_config = Pi0Config(**PI0_MODEL_CONFIG)

    # 创建模型并加载权重
    model = PI0Pytorch(pi0_config)
    safetensors.torch.load_model(model, safetensors_path, device=str(device))

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    print(f"  action_expert depth=18, width=1024")
    print(f"  action_horizon={pi0_config.action_horizon}, device={device} ✓")
    print(f"{'=' * 60}")
    return model, pi0_config


# ===================== 数据变换 =====================
def load_norm_stats():
    """
    加载 OpenPI 的 norm_stats（用于 state 和 actions 的归一化）
    返回原始 dict，供 CachedOpenPIDataset 用 state_mean 填充
    """
    with open(NORM_STATS_PATH) as f:
        raw = json.load(f)
    return raw["norm_stats"]


def build_transforms(pi0_config):
    """
    构建 OpenPI 数据变换流水线

    复用 OpenPI 自带的 transform 类：
    AlohaInputs → Normalize → ResizeImages → TokenizePrompt → PadStatesAndActions
    """
    from openpi.policies.aloha_policy import AlohaInputs
    from openpi.transforms import Normalize, TokenizePrompt, PadStatesAndActions, ResizeImages
    from openpi.shared.normalize import NormStats
    from openpi.models.tokenizer import PaligemmaTokenizer

    # 加载 norm_stats 并构造 NormStats 格式
    raw = load_norm_stats()
    norm_stats = {
        "state": NormStats(
            mean=np.array(raw["state"]["mean"], dtype=np.float32),
            std=np.array(raw["state"]["std"], dtype=np.float32),
            q01=np.array(raw["state"]["q01"], dtype=np.float32) if raw["state"].get("q01") else None,
            q99=np.array(raw["state"]["q99"], dtype=np.float32) if raw["state"].get("q99") else None,
        ),
        "actions": NormStats(
            mean=np.array(raw["actions"]["mean"], dtype=np.float32),
            std=np.array(raw["actions"]["std"], dtype=np.float32),
            q01=np.array(raw["actions"]["q01"], dtype=np.float32) if raw["actions"].get("q01") else None,
            q99=np.array(raw["actions"]["q99"], dtype=np.float32) if raw["actions"].get("q99") else None,
        ),
    }

    # 与训练时一致的变换链（adapt_to_pi=False, use_only_cam_high=True）
    transforms = [
        AlohaInputs(adapt_to_pi=False, use_only_cam_high=True),
        Normalize(norm_stats),
        ResizeImages(224, 224),
        TokenizePrompt(PaligemmaTokenizer(pi0_config.max_token_len)),
        PadStatesAndActions(pi0_config.action_dim),
    ]
    return transforms


def apply_transforms(sample, transforms):
    """依次应用变换链"""
    for t in transforms:
        sample = t(sample)
    return sample


# ===================== 缓存数据集 =====================
class CachedOpenPIDataset(Dataset):
    """
    从 SpatialVLA 预采集的数据子集加载，经 OpenPI 变换后输出 model-ready tensors。

    关键处理:
    - images: uint8 [224,224,3] → 保持不变，由 AlohaInputs 处理
    - actions: [4,7] → 扩展到 [5,7]（重复最后一步，模拟 action_horizon=5）
    - state: 使用 norm_stats.mean 作为输入（归一化后为 0，公平比较）
    - prompt: 原始文本，由 TokenizePrompt 分词
    """

    def __init__(self, subset_dir, transforms, norm_stats):
        self.transforms = transforms
        # state 用 mean 值填充，归一化后为 0（所有模型一致，公平比较）
        self.state_fill = np.array(norm_stats["state"]["mean"], dtype=np.float32)  # [7]

        # 加载 manifest
        manifest_path = os.path.join(subset_dir, "manifest.json")
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # 加载所有轨迹数据到内存
        self.items = []
        for traj_info in tqdm(self.manifest["trajectories"], desc="Loading data subset"):
            traj_path = os.path.join(subset_dir, traj_info["file"])
            data = np.load(traj_path)
            images = data["images"]       # [T, 224, 224, 3] uint8
            actions = data["actions"]     # [T, 4, 7] float32
            timesteps = data["timesteps"] # [T] int32

            for t_idx in range(len(timesteps)):
                self.items.append({
                    "image": images[t_idx],                     # [224, 224, 3] uint8
                    "action": actions[t_idx],                   # [4, 7] float32
                    "canonical_label": traj_info["canonical_label"],
                    "local_traj_id": traj_info["local_traj_id"],
                    "timestep": int(timesteps[t_idx]),
                    "lang": traj_info["lang"],
                })

        print(f"  Loaded {len(self.items)} frames from "
              f"{self.manifest['total_trajectories']} trajectories")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # 扩展 actions: [4,7] → [5,7]（重复最后一步）
        act = item["action"]  # [4, 7]
        act_ext = np.concatenate([act, act[-1:]], axis=0)  # [5, 7]

        # 构造 OpenPI 变换期望的输入格式（模拟 RepackTransform 后的结构）
        raw_sample = {
            "images": {"cam_high": item["image"]},    # uint8 [224,224,3]
            "state": self.state_fill.copy(),           # [7] float32
            "actions": act_ext,                        # [5, 7] float32
            "prompt": item["lang"],
        }

        # 应用变换链: AlohaInputs → Normalize → ResizeImages → TokenizePrompt → Pad
        processed = apply_transforms(raw_sample, self.transforms)

        return {
            "image": {k: np.asarray(v) for k, v in processed["image"].items()},
            "image_mask": {k: np.asarray(v) for k, v in processed["image_mask"].items()},
            "state": np.asarray(processed["state"]),
            "actions": np.asarray(processed["actions"]),
            "tokenized_prompt": np.asarray(processed["tokenized_prompt"]),
            "tokenized_prompt_mask": np.asarray(processed["tokenized_prompt_mask"]),
            # metadata（字符串字段在 collator 中特殊处理）
            "canonical_label": item["canonical_label"],
            "local_traj_id": item["local_traj_id"],
            "timestep": item["timestep"],
        }


def openpi_collator(features):
    """
    Collator: 将 list[dict] 合并为 batch dict。
    字符串/标量字段单独处理，array 字段 stack。
    """
    # 提取 metadata 字段
    canonical_labels = [f.pop("canonical_label") for f in features]
    local_traj_ids = [f.pop("local_traj_id") for f in features]
    timesteps = [f.pop("timestep") for f in features]

    # stack array 字段（支持嵌套 dict）
    batch = {}
    for key in features[0]:
        if isinstance(features[0][key], dict):
            batch[key] = {
                k: np.stack([f[key][k] for f in features], axis=0)
                for k in features[0][key]
            }
        else:
            batch[key] = np.stack([f[key] for f in features], axis=0)

    # 加回 metadata
    batch["canonical_label"] = canonical_labels
    batch["local_traj_id"] = local_traj_ids
    batch["timestep"] = timesteps
    return batch


# ===================== 聚合 + 保存 =====================
def _aggregate_and_save(traj_data, model_name, total_forward, final=False):
    """
    Level 2 聚合: mean pool 所有 timestep features → 轨迹级特征
    同时保存 chunk1（仅第 1 个 action token）和 chunk5（全部 5 个 action tokens）
    """
    results = {}

    for chunk_tag in ["chunk1", "chunk5"]:
        feat_key = f"timestep_features_{chunk_tag}"
        all_features, all_traj_ids = [], []
        all_n_timesteps, all_task_labels = [], []

        for (canonical_label, local_id), data in sorted(traj_data.items()):
            ts_feats = data[feat_key]
            if not ts_feats:
                continue
            # mean pool 所有 timestep 的特征 → [L, D]
            traj_feat = np.mean(np.stack(ts_feats, axis=0), axis=0)
            all_features.append(traj_feat)
            all_traj_ids.append(local_id)
            all_n_timesteps.append(len(ts_feats))
            all_task_labels.append(canonical_label)

        if not all_features:
            continue

        results[chunk_tag] = {
            "features": np.stack(all_features, axis=0),            # [N, L, D]
            "traj_indices": np.array(all_traj_ids, dtype=np.int64),
            "n_timesteps": np.array(all_n_timesteps, dtype=np.int64),
            "task_labels": np.array(all_task_labels),
        }

    if not results:
        return None

    os.makedirs(FEATURE_DIR, exist_ok=True)
    output_paths = []

    for chunk_tag, data in results.items():
        if final:
            output_path = os.path.join(FEATURE_DIR, f"{model_name}_{chunk_tag}_features.npz")
        else:
            output_path = os.path.join(FEATURE_DIR, f"{model_name}_{chunk_tag}_checkpoint.npz")

        np.savez_compressed(
            output_path,
            features=data["features"],
            traj_indices=data["traj_indices"],
            n_timesteps=data["n_timesteps"],
            layer_indices=np.array(LAYER_INDICES),
            task_labels=data["task_labels"],
        )
        output_paths.append(output_path)

        tag = "FINAL" if final else "CHECKPOINT"
        print(f"\n  [{tag}] Saved {chunk_tag}: {output_path}")
        print(f"    Trajectories: {len(data['features'])}, "
              f"Shape: {data['features'].shape}, GPU forwards: {total_forward}")

        if final:
            unique_labels = sorted(set(data["task_labels"]))
            for label in unique_labels:
                mask = data["task_labels"] == label
                n = mask.sum()
                avg_ts = data["n_timesteps"][mask].mean() if n > 0 else 0
                print(f"      {label:<55} trajs={n:>3} avg_ts={avg_ts:.1f}")
            # 删除 checkpoint
            ckpt_path = os.path.join(FEATURE_DIR, f"{model_name}_{chunk_tag}_checkpoint.npz")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    return output_paths


# ===================== 特征提取 =====================
def extract_features_for_model(model_name: str, device="cuda", batch_size=8):
    """
    从固定数据子集提取指定 PI0 模型的轨迹级特征。

    流程:
    1. 加载 PI0 模型
    2. 构建变换流水线
    3. 从 data_subset/ 加载数据（CachedOpenPIDataset）
    4. DataLoader 迭代 → GPU forward（time=0, 无噪声）→ suffix_hidden_states
    5. 提取 action token 位置的 hidden states
    6. 按 (canonical_label, local_traj_id) 分组
    7. 两级 mean pool → 保存
    """
    from openpi.models.model import Observation

    # Step 1: 加载模型
    model, pi0_config = load_pi0_model(model_name, device)
    action_horizon = pi0_config.action_horizon  # 5

    # Step 2: 构建变换 + 加载 norm_stats
    norm_stats = load_norm_stats()
    transforms = build_transforms(pi0_config)

    # Step 3: 加载数据子集
    print(f"\nLoading data subset from: {SUBSET_DIR}")
    dataset = CachedOpenPIDataset(SUBSET_DIR, transforms, norm_stats)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=openpi_collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    # Step 4: 提取特征
    # suffix_hidden_states: [num_layers=18, B, suffix_len, D=1024]
    # suffix 结构: state(pos 0) + action_tokens(pos 1..5)
    # action tokens 在最后 action_horizon=5 个位置
    traj_data = defaultdict(lambda: {
        "timestep_features_chunk1": [],   # 仅第 1 个 action token
        "timestep_features_chunk5": [],   # 全部 5 个 action tokens
        "seen_timesteps": set(),
    })

    total_forward = 0
    CHECKPOINT_INTERVAL = 20000

    print(f"\nExtraction config:")
    print(f"  Model: {MODEL_CONFIGS[model_name]['label']}")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Layers: {LAYER_INDICES}")
    print(f"  Action horizon: {action_horizon} tokens")
    print(f"  Output: chunk1 (1st action token) + chunk5 (all 5 tokens)")
    print(f"  Data source: {SUBSET_DIR}")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in tqdm(dataloader, desc=f"Extracting [{model_name}]",
                          total=len(dataloader)):
            cur_bs = batch["state"].shape[0]

            # 构造 Observation（to GPU）
            obs_dict = {
                "image": {k: torch.from_numpy(v).to(device) for k, v in batch["image"].items()},
                "image_mask": {k: torch.from_numpy(v).to(device) for k, v in batch["image_mask"].items()},
                "state": torch.from_numpy(batch["state"]).to(device),
                "tokenized_prompt": torch.from_numpy(batch["tokenized_prompt"]).to(device),
                "tokenized_prompt_mask": torch.from_numpy(batch["tokenized_prompt_mask"]).to(device),
            }
            observation = Observation.from_dict(obs_dict)
            actions = torch.from_numpy(batch["actions"]).to(device)

            # Forward pass: time=0 → x_t = actions（无噪声，确定性特征）
            # noise=0, time=0 → x_t = 0*noise + 1*actions = actions
            time_zero = torch.zeros(cur_bs, device=device, dtype=torch.float32)
            noise_zero = torch.zeros_like(actions)

            result = model.forward(
                observation, actions,
                noise=noise_zero, time=time_zero,
                output_hidden_states=True,
            )
            # result = (loss, suffix_hidden_states, learnable_token_hidden_states)
            _, suffix_hidden_states, _ = result

            # suffix_hidden_states: [num_layers, B, suffix_len, D]
            # 提取 action token 位置（最后 action_horizon 个位置）
            # 形状: [num_layers, B, action_horizon, D]
            action_hs = suffix_hidden_states[:, :, -action_horizon:, :]

            # 选取指定层
            # action_hs[layer_indices]: [L, B, action_horizon, D]
            selected_hs = action_hs[LAYER_INDICES]  # [L, B, 5, D]

            # Level 1 pooling:
            # chunk1: 只取第 1 个 action token → mean → [L, B, D]
            # chunk5: 取全部 5 个 action tokens → mean → [L, B, D]
            feats_c1 = selected_hs[:, :, :1, :].mean(dim=2)   # [L, B, D]
            feats_c5 = selected_hs.mean(dim=2)                 # [L, B, D]

            feats_c1 = feats_c1.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]
            feats_c5 = feats_c5.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]

            # 分发到各轨迹容器
            for j in range(cur_bs):
                canonical = batch["canonical_label"][j]
                local_id = batch["local_traj_id"][j]
                if isinstance(local_id, (torch.Tensor, np.ndarray)):
                    local_id = int(local_id)
                timestep = batch["timestep"][j]
                if isinstance(timestep, (torch.Tensor, np.ndarray)):
                    timestep = int(timestep)

                key = (canonical, local_id)
                if timestep not in traj_data[key]["seen_timesteps"]:
                    traj_data[key]["timestep_features_chunk1"].append(feats_c1[j])
                    traj_data[key]["timestep_features_chunk5"].append(feats_c5[j])
                    traj_data[key]["seen_timesteps"].add(timestep)

            total_forward += cur_bs

            # 定期 checkpoint
            if total_forward % CHECKPOINT_INTERVAL < batch_size:
                _aggregate_and_save(traj_data, model_name, total_forward, final=False)

    # 最终保存
    print(f"\n{'=' * 60}")
    print(f"Level 2 aggregation: mean pool ALL timesteps per trajectory...")
    print(f"  chunk1: first action token")
    print(f"  chunk5: all {action_horizon} action tokens")
    output_paths = _aggregate_and_save(traj_data, model_name, total_forward, final=True)
    print(f"{'=' * 60}")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="OpenPI (PI0) feature extraction")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    extract_features_for_model(args.model_name, args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
