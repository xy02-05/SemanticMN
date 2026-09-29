"""
OpenPI 表征提取管线测试脚本

测试数据变换 + 模型加载 + 单 batch forward，验证输出形状正确。
不涉及全量提取，仅验证管线能跑通。

用法:
    python test_pipeline.py
"""
import os
import sys
import json
import numpy as np

import torch

# ===================== 路径 =====================
ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
OPENPI_DIR = os.path.join(ANALYSIS_DIR, "openpi")
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, os.path.join(OPENPI_DIR, "src"))

from openpi_representation.config import (
    MODEL_CONFIGS, LAYER_INDICES, PI0_MODEL_CONFIG,
    SUBSET_DIR, NORM_STATS_PATH,
)
from openpi_representation.extract_features import (
    load_norm_stats, build_transforms, apply_transforms,
    CachedOpenPIDataset, openpi_collator, load_pi0_model,
)


def test_transforms():
    """测试数据变换流水线：单样本通过全部变换"""
    print("=" * 60)
    print("Test 1: 数据变换流水线")
    print("=" * 60)

    # 加载 norm_stats
    norm_stats = load_norm_stats()
    state_fill = np.array(norm_stats["state"]["mean"], dtype=np.float32)
    print(f"  state_fill shape: {state_fill.shape}, dtype: {state_fill.dtype}")

    # 构建 PI0Config（只用于获取参数）
    from openpi.models.pi0_config import Pi0Config
    pi0_config = Pi0Config(**PI0_MODEL_CONFIG)
    print(f"  action_dim={pi0_config.action_dim}, action_horizon={pi0_config.action_horizon}")

    # 构建变换链
    transforms = build_transforms(pi0_config)
    print(f"  变换链: {[type(t).__name__ for t in transforms]}")

    # 构造单样本（模拟 CachedOpenPIDataset.__getitem__）
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    actions_4 = np.random.randn(4, 7).astype(np.float32)
    actions_5 = np.concatenate([actions_4, actions_4[-1:]], axis=0)

    raw_sample = {
        "images": {"cam_high": image},
        "state": state_fill.copy(),
        "actions": actions_5,
        "prompt": "close the microwave",
    }

    processed = apply_transforms(raw_sample, transforms)

    print(f"\n  变换后字段:")
    for k, v in processed.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                vv = np.asarray(vv)
                print(f"    {k}/{kk}: shape={vv.shape}, dtype={vv.dtype}")
        elif isinstance(v, np.ndarray):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, str):
            print(f"    {k}: '{v[:40]}'")
        else:
            print(f"    {k}: {type(v).__name__} = {v}")

    # 关键断言
    assert "image" in processed
    assert "base_0_rgb" in processed["image"]
    assert processed["state"].shape[-1] == 32, f"state dim should be 32, got {processed['state'].shape[-1]}"
    assert processed["actions"].shape == (5, 32), f"actions shape should be (5,32), got {processed['actions'].shape}"
    assert processed["tokenized_prompt"].shape == (48,), f"tokens shape should be (48,), got {processed['tokenized_prompt'].shape}"
    print("\n  ✅ 变换管线通过!")
    return pi0_config, transforms, norm_stats


def test_dataset(pi0_config, transforms, norm_stats):
    """测试 CachedOpenPIDataset 加载 + collator"""
    print("\n" + "=" * 60)
    print("Test 2: CachedOpenPIDataset")
    print("=" * 60)

    dataset = CachedOpenPIDataset(SUBSET_DIR, transforms, norm_stats)
    print(f"  Dataset size: {len(dataset)} frames")

    # 取前 4 个样本，测试 collator
    samples = [dataset[i] for i in range(4)]
    batch = openpi_collator(samples)

    print(f"\n  Batch 字段:")
    for k, v in batch.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"    {k}/{kk}: shape={vv.shape}, dtype={vv.dtype}")
        elif isinstance(v, np.ndarray):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"    {k}: list[{len(v)}] = {v}")
        else:
            print(f"    {k}: {v}")

    # 关键断言
    assert batch["state"].shape == (4, 32)
    assert batch["actions"].shape == (4, 5, 32)
    assert batch["tokenized_prompt"].shape == (4, 48)
    assert len(batch["canonical_label"]) == 4
    print("\n  ✅ Dataset + Collator 通过!")
    return batch


def test_model_forward(batch, device="cuda"):
    """测试模型加载 + forward"""
    print("\n" + "=" * 60)
    print("Test 3: 模型加载 + Forward")
    print("=" * 60)

    from openpi.models.model import Observation

    # 用 pretrained 模型测试（最小风险）
    model, pi0_config = load_pi0_model("pretrained", device)
    action_horizon = pi0_config.action_horizon

    # 准备 GPU tensors
    obs_dict = {
        "image": {k: torch.from_numpy(v).to(device) for k, v in batch["image"].items()},
        "image_mask": {k: torch.from_numpy(v).to(device) for k, v in batch["image_mask"].items()},
        "state": torch.from_numpy(batch["state"]).to(device),
        "tokenized_prompt": torch.from_numpy(batch["tokenized_prompt"]).to(device),
        "tokenized_prompt_mask": torch.from_numpy(batch["tokenized_prompt_mask"]).to(device),
    }
    observation = Observation.from_dict(obs_dict)
    actions = torch.from_numpy(batch["actions"]).to(device)
    cur_bs = actions.shape[0]

    print(f"\n  Input shapes:")
    print(f"    images: {list(obs_dict['image'].keys())}")
    for k, v in obs_dict["image"].items():
        print(f"      {k}: {v.shape} {v.dtype}")
    print(f"    state: {obs_dict['state'].shape}")
    print(f"    actions: {actions.shape}")
    print(f"    tokenized_prompt: {obs_dict['tokenized_prompt'].shape}")

    # Forward pass
    time_zero = torch.zeros(cur_bs, device=device, dtype=torch.float32)
    noise_zero = torch.zeros_like(actions)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        result = model.forward(
            observation, actions,
            noise=noise_zero, time=time_zero,
            output_hidden_states=True,
        )

    loss, suffix_hidden_states, learnable_token_hidden_states = result

    print(f"\n  Output shapes:")
    print(f"    loss: {loss.shape}")
    print(f"    suffix_hidden_states: {suffix_hidden_states.shape}")
    print(f"    learnable_token_hidden_states: {learnable_token_hidden_states}")

    # 提取 action tokens（最后 action_horizon 个位置）
    action_hs = suffix_hidden_states[:, :, -action_horizon:, :]
    print(f"    action_hidden_states: {action_hs.shape}")
    print(f"      - dim 0: num_layers = {action_hs.shape[0]}")
    print(f"      - dim 1: batch = {action_hs.shape[1]}")
    print(f"      - dim 2: action_horizon = {action_hs.shape[2]}")
    print(f"      - dim 3: hidden_dim = {action_hs.shape[3]}")

    # 选取指定层
    selected = action_hs[LAYER_INDICES]
    print(f"    selected layers {LAYER_INDICES}: {selected.shape}")

    # Level 1 pooling
    feats_c1 = selected[:, :, :1, :].mean(dim=2)   # [L, B, D]
    feats_c5 = selected.mean(dim=2)                  # [L, B, D]
    feats_c1 = feats_c1.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]
    feats_c5 = feats_c5.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]
    print(f"    chunk1 features: {feats_c1.shape}")
    print(f"    chunk5 features: {feats_c5.shape}")

    # 关键断言
    assert suffix_hidden_states.shape[0] == 18, f"Expected 18 layers, got {suffix_hidden_states.shape[0]}"
    assert action_hs.shape[2] == 5, f"Expected 5 action tokens, got {action_hs.shape[2]}"
    assert feats_c1.shape == (4, len(LAYER_INDICES), 1024), f"chunk1 shape mismatch: {feats_c1.shape}"
    assert feats_c5.shape == (4, len(LAYER_INDICES), 1024), f"chunk5 shape mismatch: {feats_c5.shape}"

    print("\n  ✅ 模型 Forward + 特征提取通过!")

    # 清理 GPU 内存
    del model, result, suffix_hidden_states, action_hs, selected
    torch.cuda.empty_cache()


def main():
    print("OpenPI 表征提取管线测试")
    print("=" * 60)

    # Test 1: 变换
    pi0_config, transforms, norm_stats = test_transforms()

    # Test 2: 数据集
    batch = test_dataset(pi0_config, transforms, norm_stats)

    # Test 3: 模型 forward
    test_model_forward(batch)

    print("\n" + "=" * 60)
    print("全部测试通过! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
