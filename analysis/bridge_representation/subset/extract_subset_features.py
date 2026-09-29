"""
SpatialVLA 表征提取脚本 v2 — 基于固定数据子集

核心设计:
  1. 从 outputs/data_subset/ 加载预采集的固定子集（build_data_subset.py 生成）
     → 所有模型的输入 100% 一致，消除数据差异的混杂因素
  2. CachedBridgeDataset: 从 .npz 加载原始图像 → VLA processor → model-ready tensors
  3. action_chunk_size=4（与训练时 action_forward_steps=3 一致）
     每个 timestep 有 12 个 action tokens（4步 × 3分量: translation/rotation/gripper）
  4. 两级 mean pool:
     Level 1: 每个 timestep → mean_pool(action_tokens) → [L, D]
     Level 2: 每条轨迹 → mean_pool(timestep_features) → [L, D]
  5. 支持 Pretrained / LoRA FT 模型

输出: npz 文件，每条轨迹一个 [L, D] 特征向量

用法:
    python extract_features.py --model_name pretrained
    python extract_features.py --model_name raw_ft
    python extract_features.py --model_name cotrain_fg
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# ===================== 路径配置 =====================
ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
sys.path.insert(0, ANALYSIS_DIR)

from bridge_representation.config import (
    SPATIALVLA_DIR, MODEL_CONFIGS, LAYER_INDICES,
    FEATURE_DIR, OUTPUT_DIR,
)

SUBSET_DIR = os.path.join(OUTPUT_DIR, "data_subset")

sys.path.insert(0, SPATIALVLA_DIR)


# ===================== 模型加载 =====================
def load_model(model_name: str, device="cuda"):
    """加载 SpatialVLA 模型（支持 LoRA adapter）"""
    from model import (
        SpatialVLAConfig,
        SpatialVLAForConditionalGeneration,
        SpatialVLAProcessor,
        SpatialActionTokenizer,
    )

    cfg = MODEL_CONFIGS[model_name]
    base_model_path = cfg["base_model"]
    adapter_path = cfg["adapter"]

    print(f"\n{'=' * 60}")
    print(f"Loading model: {cfg['label']}")
    print(f"  Base: {base_model_path}")
    print(f"  Adapter: {adapter_path or 'None'}")
    print(f"{'=' * 60}")

    processor = SpatialVLAProcessor.from_pretrained(base_model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    torch_dtype = torch.bfloat16
    config = SpatialVLAConfig.from_pretrained(
        base_model_path, torch_dtype=torch_dtype, local_files_only=True
    )
    model = SpatialVLAForConditionalGeneration.from_pretrained(
        base_model_path, config=config, torch_dtype=torch_dtype, local_files_only=True
    )

    if adapter_path is not None:
        from peft import PeftModel
        print(f"  Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"  LoRA merged ✓")

    model.language_model.config._attn_implementation = "flash_attention_2"
    model.vision_tower.config._attn_implementation = "flash_attention_2"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    action_tokenizer = SpatialActionTokenizer(
        tokenizer,
        num_bins=processor.action_config["num_bins"],
        bin_policy=processor.action_tokenizer.bin_policy,
        use_spherical=processor.action_config["use_spherical"],
        min_sigma=processor.action_config.get("min_sigma", 0.0),
    )
    model.action_token_begin_idx = action_tokenizer.action_token_begin_idx

    print(f"  hidden_size={config.text_config.hidden_size}, device={device} ✓")
    return model, processor, tokenizer, action_tokenizer, config


# ===================== VLA Processor 构建 =====================
def build_vla_processor(processor, tokenizer, action_tokenizer):
    """构建完整的 VLA processor（与训练时一致）"""
    from model import SpatialVLAProcessor
    from data.dataset import OpenXIterableDataset, build_datasets
    from data.utils.data_utils import NormalizationType, save_dataset_statistics
    from data.oxe import get_oxe_dataset_kwargs_and_weights
    from pathlib import Path

    # 需要 dataset_statistics 来设置 processor.statistics
    # 用一个轻量方式获取统计量（不实际加载数据）
    rlds_data_root = "/root/data/xuyuan1/Codes/mirror_neuron/data"
    mixture_spec = [("bridge_orig/1.0.0", 1.0)]
    per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        rlds_data_root,
        mixture_spec,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=False,
        load_language=True,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )

    # 获取 dataset statistics
    from data.rlds import dataset_statistics as compute_dataset_statistics
    ds_stats_config = dict(
        dataset_kwargs_list=per_dataset_kwargs,
        sample_weights=weights,
        balance_weights=True,
        train=True,
        shuffle_seed=3407,
    )
    _, all_dataset_statistics, _ = compute_dataset_statistics(**ds_stats_config)

    tmp_dir = "/tmp/spatialvla_extract_v2"
    os.makedirs(tmp_dir, exist_ok=True)
    statistic = save_dataset_statistics(all_dataset_statistics, Path(tmp_dir) / "ds_stats.json")
    processor.statistics.update(statistic)

    full_processor = SpatialVLAProcessor(
        image_processor=processor.image_processor,
        tokenizer=tokenizer,
        statistics=processor.statistics,
        bin_policy=action_tokenizer.bin_policy,
        intrinsic_config=processor.intrinsic_config,
        action_config=processor.action_config,
        num_obs_steps=1,        # backward_steps=0 → single observation
        obs_delta=1,
        action_chunk_size=4,    # ★ forward_steps=3 → 4 action steps (与训练一致)
    )
    return full_processor


# ===================== 缓存数据集 =====================
class CachedBridgeDataset(Dataset):
    """
    从预采集的数据子集加载，经 VLA processor 处理后输出 model-ready tensors。
    保证所有模型收到完全相同的输入。
    """

    def __init__(self, subset_dir, vla_processor, max_length=2048):
        self.vla_processor = vla_processor
        self.max_length = max_length

        manifest_path = os.path.join(subset_dir, 'manifest.json')
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # 加载所有轨迹数据到内存（~4-5 GB uint8 images）
        self.items = []  # 每个元素: dict with image, action, metadata
        for traj_info in tqdm(self.manifest['trajectories'], desc="Loading data subset"):
            traj_path = os.path.join(subset_dir, traj_info['file'])
            data = np.load(traj_path)
            images = data['images']       # [T, 224, 224, 3] uint8
            actions = data['actions']     # [T, 4, D] float32 (action chunk)
            timesteps = data['timesteps'] # [T] int32

            for t_idx in range(len(timesteps)):
                self.items.append({
                    'image': images[t_idx],                 # [224, 224, 3] uint8
                    'action': actions[t_idx],               # [4, D] float32 (action chunk)
                    'canonical_label': traj_info['canonical_label'],
                    'local_traj_id': traj_info['local_traj_id'],
                    'timestep': int(timesteps[t_idx]),
                    'lang': traj_info['lang'],
                })

        print(f"  Loaded {len(self.items)} frames from "
              f"{self.manifest['total_trajectories']} trajectories")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # 图像: uint8 numpy → PIL → VLA processor
        pil_image = Image.fromarray(item['image'])
        # action chunk: [4, D] → torch tensor (与训练一致)
        actions_tensor = torch.from_numpy(item['action'])  # [4, D] (不需要 unsqueeze)
        lang = item['lang'].lower()

        ret = self.vla_processor(
            text=lang,
            images=[pil_image],
            suffix_actions=actions_tensor,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length,
            truncation=True,
            do_normalize=False,
        )

        return {
            'input_ids': ret['input_ids'][0],
            'labels': ret['labels'][0],
            'token_type_ids': ret['token_type_ids'][0],
            'attention_mask': ret['attention_mask'][0],
            'pixel_values': ret['pixel_values'],
            'intrinsic': ret['intrinsic'],
            'lang': lang,
            'canonical_label': item['canonical_label'],
            'local_traj_id': item['local_traj_id'],
            'timestep': item['timestep'],
        }


def analysis_collator(features, pad_id=0):
    """
    Collator wrapper: 提取 string metadata，再调用标准 concat_pad_data_collator。
    """
    from train.monkey_patch import concat_pad_data_collator

    # 提取 string 字段（标准 collator 会跳过 string）
    canonical_labels = [f.pop('canonical_label') for f in features]

    batch = concat_pad_data_collator(features, pad_id)

    # 加回 string 字段
    batch['canonical_label'] = canonical_labels
    return batch


# ===================== 聚合 + 保存 =====================
# 每个 action step = 3 tokens (translation/rotation/gripper)
TOKENS_PER_ACTION_STEP = 3

def _aggregate_and_save(traj_data, model_name, total_forward, final=False,
                        feature_dir=None):
    """
    Level 2 聚合: mean pool 所有 timestep features → 轨迹级特征
    同时保存 chunk1（仅当前步 3 tokens）和 chunk4（全部 12 tokens）两种特征。
    """
    if feature_dir is None:
        feature_dir = FEATURE_DIR

    results = {}  # chunk_tag → {features, traj_ids, n_timesteps, task_labels}

    for chunk_tag in ['chunk1', 'chunk4']:
        feat_key = f'timestep_features_{chunk_tag}'
        all_features, all_traj_ids = [], []
        all_n_timesteps, all_task_labels = [], []

        for (canonical_label, local_id), data in sorted(traj_data.items()):
            ts_feats = data[feat_key]
            if not ts_feats:
                continue
            traj_feat = np.mean(np.stack(ts_feats, axis=0), axis=0)  # [L, D]
            all_features.append(traj_feat)
            all_traj_ids.append(local_id)
            all_n_timesteps.append(len(ts_feats))
            all_task_labels.append(canonical_label)

        if not all_features:
            continue

        results[chunk_tag] = {
            'features': np.stack(all_features, axis=0),          # [N, L, D]
            'traj_indices': np.array(all_traj_ids, dtype=np.int64),
            'n_timesteps': np.array(all_n_timesteps, dtype=np.int64),
            'task_labels': np.array(all_task_labels),
        }

    if not results:
        return None

    os.makedirs(feature_dir, exist_ok=True)
    output_paths = []

    for chunk_tag, data in results.items():
        if final:
            output_path = os.path.join(feature_dir, f"{model_name}_{chunk_tag}_features.npz")
        else:
            output_path = os.path.join(feature_dir, f"{model_name}_{chunk_tag}_checkpoint.npz")

        np.savez_compressed(
            output_path,
            features=data['features'],         # [N, L, D]
            traj_indices=data['traj_indices'],  # [N]
            n_timesteps=data['n_timesteps'],    # [N]
            layer_indices=np.array(LAYER_INDICES),
            task_labels=data['task_labels'],    # [N] canonical task text
        )
        output_paths.append(output_path)

        tag = "FINAL" if final else "CHECKPOINT"
        print(f"\n  [{tag}] Saved {chunk_tag}: {output_path}")
        print(f"    Trajectories: {len(data['features'])}, "
              f"Shape: {data['features'].shape}, GPU forwards: {total_forward}")

        if final:
            unique_labels = sorted(set(data['task_labels']))
            for label in unique_labels:
                mask = data['task_labels'] == label
                n = mask.sum()
                avg_ts = data['n_timesteps'][mask].mean() if n > 0 else 0
                print(f"      {label:<55} trajs={n:>3} avg_ts={avg_ts:.1f}")
            # 删除 checkpoint
            ckpt_path = os.path.join(feature_dir, f"{model_name}_{chunk_tag}_checkpoint.npz")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    return output_paths


# ===================== 特征提取 =====================
def extract_features_for_model(model_name: str, device="cuda", batch_size=12,
                                subset_dir=None, feature_dir=None):
    """
    从固定数据子集提取指定模型的轨迹级特征。

    Args:
        subset_dir: 数据子集目录 (默认用 train subset)
        feature_dir: 特征输出目录 (默认用 train feature dir)
    """
    # 默认目录: 与原来兼容
    if subset_dir is None:
        subset_dir = SUBSET_DIR
    if feature_dir is None:
        feature_dir = FEATURE_DIR

    # Step 1: 加载模型
    model, processor, tokenizer, action_tokenizer, config = load_model(model_name, device)

    # Step 2: 构建 VLA processor
    print("\nBuilding VLA processor...")
    full_processor = build_vla_processor(processor, tokenizer, action_tokenizer)

    # Step 3: 加载数据子集
    print(f"\nLoading data subset from: {subset_dir}")
    dataset = CachedBridgeDataset(subset_dir, full_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=analysis_collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False,  # 保持顺序，方便对比
    )

    # Step 4: 提取特征 — 一次 forward 同时计算 chunk1 和 chunk4
    traj_data = defaultdict(lambda: {
        'timestep_features_chunk1': [],   # 仅当前步 (3 tokens) 的 mean pool
        'timestep_features_chunk4': [],   # 全部 4 步 (12 tokens) 的 mean pool
        'seen_timesteps': set(),
    })

    total_forward = 0
    CHECKPOINT_INTERVAL = 20000

    print(f"\nExtraction config:")
    print(f"  Model: {MODEL_CONFIGS[model_name]['label']}")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Layers: {LAYER_INDICES}")
    print(f"  Action chunk: 4 steps × 3 tokens = 12 action tokens")
    print(f"  Output: chunk1 (3 tokens, 当前步) + chunk4 (12 tokens, 全部步)")
    print(f"  Data source: {SUBSET_DIR} (固定子集, 所有模型共用)")

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in tqdm(dataloader, desc=f"Extracting [{model_name}]",
                          total=len(dataloader)):
            cur_bs = batch['input_ids'].shape[0]

            # GPU forward
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            vla_outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                intrinsic=batch['intrinsic'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                output_hidden_states=True,
                return_dict=True,
            )

            # 提取 action hidden states → [L, B, A, D]
            # A = 12 (4 steps × 3 tokens per step)
            if (hasattr(vla_outputs, 'action_hidden_states')
                    and vla_outputs.action_hidden_states is not None):
                from egovlpv2.utils.model_forward import get_layer_vla_features
                action_hs = get_layer_vla_features(
                    vla_outputs.action_hidden_states, LAYER_INDICES
                )  # [L, B, A, D]
            else:
                from egovlpv2.utils.model_forward import get_vla_features
                action_hs = get_vla_features(
                    hidden_states=vla_outputs.hidden_states,
                    vision_patches_num=256,
                    batch=batch,
                    action_token_begin_idx=action_tokenizer.action_token_begin_idx,
                    layer_indices=LAYER_INDICES,
                )  # [L, B, A, D]

            # ★ 一次 forward，两种 Level 1 pooling:
            # chunk1: 只 pool 前 3 个 token (当前步的 trans/rot/grip)
            # chunk4: pool 全部 A 个 token (12 action + 1 EOS, 与原始模型一致,
            #         modeling_spatialvla.py line 478 注释掉了 :-1 裁切)
            # 由于 causal attention，前 3 个 token 的 hidden states 完全一致
            n_tokens_step0 = TOKENS_PER_ACTION_STEP  # 3
            feats_c1 = action_hs[:, :, :n_tokens_step0, :].mean(dim=2)  # [L, B, D]
            feats_c4 = action_hs.mean(dim=2)                             # [L, B, D]

            feats_c1 = feats_c1.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]
            feats_c4 = feats_c4.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]

            # 分发到各轨迹容器
            for j in range(cur_bs):
                canonical = batch['canonical_label'][j]

                local_id = batch['local_traj_id'][j]
                if isinstance(local_id, torch.Tensor):
                    local_id = local_id.item()

                timestep = batch['timestep'][j]
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.item()

                key = (canonical, local_id)
                if timestep not in traj_data[key]['seen_timesteps']:
                    traj_data[key]['timestep_features_chunk1'].append(feats_c1[j])
                    traj_data[key]['timestep_features_chunk4'].append(feats_c4[j])
                    traj_data[key]['seen_timesteps'].add(timestep)

            total_forward += cur_bs

            # 定期 checkpoint
            if total_forward % CHECKPOINT_INTERVAL < batch_size:
                _aggregate_and_save(traj_data, model_name, total_forward,
                                    final=False, feature_dir=feature_dir)

    # 最终保存
    print(f"\n{'=' * 60}")
    print(f"Level 2 aggregation: mean pool ALL unique timesteps per trajectory...")
    print(f"  chunk1: pool over first {TOKENS_PER_ACTION_STEP} action tokens (current step)")
    print(f"  chunk4: pool over all action tokens (4 steps)")
    output_paths = _aggregate_and_save(traj_data, model_name, total_forward,
                                        final=True, feature_dir=feature_dir)
    print(f"{'=' * 60}")
    return output_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=12)
    # 可选: 指定自定义目录 (用于 val 集)
    parser.add_argument("--subset_dir", type=str, default=None,
                        help="数据子集目录 (默认: train subset)")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="特征输出目录 (默认: train features)")
    args = parser.parse_args()

    os.chdir(SPATIALVLA_DIR)
    extract_features_for_model(
        args.model_name, args.device, batch_size=args.batch_size,
        subset_dir=args.subset_dir, feature_dir=args.feature_dir,
    )


if __name__ == "__main__":
    main()
