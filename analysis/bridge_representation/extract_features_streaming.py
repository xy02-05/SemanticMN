"""
SpatialVLA 全量流式特征提取脚本

核心设计:
  1. 从 Bridge RLDS 数据集流式读取所有轨迹（不限 task，不限轨迹数）
  2. 每帧经 VLA processor 处理后 GPU forward，提取 action hidden states
  3. 两级 mean pool:
     Level 1: 每个 timestep → mean_pool(action_tokens) → [L, D]
     Level 2: 每条轨迹 → mean_pool(timestep_features) → [L, D]
  4. 每 save_interval 条轨迹保存一次 checkpoint npz
  5. 最终 merge 所有 chunk 为一个完整 npz

复用:
  - load_model, build_vla_processor: 从 extract_features.py 导入
  - analysis_collator, TOKENS_PER_ACTION_STEP: 从 extract_features.py 导入
  - RLDS pipeline: 从 extract_alignment_features.py 的 build_alignment_rlds_pipeline
  - 帧解析: 从 extract_alignment_features.py 的 parse_rlds_frame
  - VLA forward: 从 extract_alignment_features.py 的 forward_batch

输出格式: 标准 npz (与 OpenPI streaming 格式一致)
  features [N, L, D=2304], task_labels [N], task_indices [N],
  layer_indices [L], n_timesteps [N]

用法:
    cd /root/data/xuyuan1/Codes/analysis/SpatialVLA
    python .../extract_features_streaming.py --model_name pretrained
    python .../extract_features_streaming.py --model_name raw_ft --save_interval 500
"""
import os
import sys
import json
import argparse
import glob
import numpy as np
from collections import defaultdict

import torch
from tqdm import tqdm
from PIL import Image

# ===================== 路径配置 =====================
ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
SPATIALVLA_DIR = os.path.join(ANALYSIS_DIR, "SpatialVLA")
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, SPATIALVLA_DIR)

from bridge_representation.config import (
    RLDS_DATA_ROOT, TASK_RLDS_PATH, EGOHOD_EMB_PATH,
    MODEL_CONFIGS, LAYER_INDICES, OUTPUT_DIR,
)
# 复用 extract_features.py 的核心函数
from bridge_representation.extract_features import (
    load_model, build_vla_processor, analysis_collator, TOKENS_PER_ACTION_STEP,
)
# 复用 extract_alignment_features.py 的 RLDS pipeline 和帧解析
from bridge_representation.extract_alignment_features import (
    build_alignment_rlds_pipeline, parse_rlds_frame, process_frame_for_vla,
)

# ===================== 输出路径 =====================
STREAMING_DIR = os.path.join(OUTPUT_DIR, "streaming")
STREAMING_FEATURE_DIR = os.path.join(STREAMING_DIR, "features")


# ===================== GPU Forward (复用 extract_alignment_features 逻辑) =====================
def forward_batch_streaming(batch_tensors, batch_meta, model, action_tokenizer,
                            device, traj_accum):
    """
    GPU forward + 特征提取 + 累加到轨迹容器
    与 extract_alignment_features.forward_batch 相同逻辑
    """
    batch = analysis_collator(batch_tensors)
    cur_bs = batch['input_ids'].shape[0]

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

    # 提取多层 action hidden states → [L, B, A, D]
    if (hasattr(vla_outputs, 'action_hidden_states')
            and vla_outputs.action_hidden_states is not None):
        from egovlpv2.utils.model_forward import get_layer_vla_features
        action_hs = get_layer_vla_features(vla_outputs.action_hidden_states, LAYER_INDICES)
    else:
        from egovlpv2.utils.model_forward import get_vla_features
        action_hs = get_vla_features(
            hidden_states=vla_outputs.hidden_states,
            vision_patches_num=256, batch=batch,
            action_token_begin_idx=action_tokenizer.action_token_begin_idx,
            layer_indices=LAYER_INDICES,
        )

    # Level 1 pooling: chunk4 全部 action tokens mean
    feats = action_hs.mean(dim=2)  # [L, B, D]
    feats = feats.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]

    # 累加到轨迹容器 (running sum + count → 最终 mean)
    for j in range(cur_bs):
        meta = batch_meta[j]
        key = meta['traj_key']
        ts = meta['timestep']
        if ts in traj_accum[key]['seen_ts']:
            continue
        traj_accum[key]['seen_ts'].add(ts)
        if traj_accum[key]['count'] == 0:
            traj_accum[key]['sum_feat'] = feats[j].copy()  # [L, D]
        else:
            traj_accum[key]['sum_feat'] += feats[j]
        traj_accum[key]['count'] += 1

    return cur_bs


# ===================== Checkpoint 保存 =====================
def save_checkpoint(traj_accum, completed_keys, model_name, chunk_idx, feature_dir):
    """
    保存已完成轨迹的 checkpoint

    Args:
        traj_accum: 轨迹累加器
        completed_keys: 这批要保存的轨迹 key 列表
        model_name: 模型名
        chunk_idx: chunk 编号
        feature_dir: 输出目录
    """
    all_feats, all_labels, all_indices, all_counts = [], [], [], []

    for key in completed_keys:
        data = traj_accum[key]
        if data['count'] == 0:
            continue
        all_feats.append(data['sum_feat'] / data['count'])  # [L, D] mean
        all_labels.append(data['lang'])
        all_indices.append(data['task_index'])
        all_counts.append(data['count'])

    if not all_feats:
        return None

    features = np.stack(all_feats, axis=0)  # [N, L, D]
    out_path = os.path.join(
        feature_dir, f"{model_name}_streaming_chunk{chunk_idx:04d}.npz"
    )
    np.savez_compressed(
        out_path,
        features=features,
        task_labels=np.array(all_labels),
        task_indices=np.array(all_indices, dtype=np.int64),
        n_timesteps=np.array(all_counts, dtype=np.int64),
        layer_indices=np.array(LAYER_INDICES),
    )
    print(f"  [CHUNK-{chunk_idx}] Saved: {out_path}")
    print(f"    N={len(all_feats)}, shape={features.shape}, "
          f"unique_tasks={len(set(all_indices))}")
    sys.stdout.flush()

    # 清理已保存的轨迹数据，释放内存（用 pop 避免重复 key 导致 KeyError）
    for key in completed_keys:
        traj_accum.pop(key, None)

    return out_path


def merge_chunks(model_name, feature_dir):
    """合并所有 chunk 文件为一个完整的 npz"""
    pattern = os.path.join(feature_dir, f"{model_name}_streaming_chunk*.npz")
    chunk_files = sorted(glob.glob(pattern))
    if not chunk_files:
        print("  ⚠ 没有找到 chunk 文件")
        return None

    all_feats, all_labels, all_indices, all_counts = [], [], [], []
    layer_indices = None

    for cf in chunk_files:
        data = np.load(cf, allow_pickle=True)
        all_feats.append(data['features'])
        all_labels.append(data['task_labels'])
        all_indices.append(data['task_indices'])
        all_counts.append(data['n_timesteps'])
        if layer_indices is None:
            layer_indices = data['layer_indices']

    features = np.concatenate(all_feats, axis=0)
    task_labels = np.concatenate(all_labels)
    task_indices = np.concatenate(all_indices)
    n_timesteps = np.concatenate(all_counts)

    out_path = os.path.join(feature_dir, f"{model_name}_streaming_features.npz")
    np.savez_compressed(
        out_path,
        features=features,
        task_labels=task_labels,
        task_indices=task_indices,
        n_timesteps=n_timesteps,
        layer_indices=layer_indices,
    )
    print(f"  [MERGED] {out_path}")
    print(f"    shape={features.shape}, unique_tasks={len(np.unique(task_indices))}")
    return out_path


# ===================== 多卡合并 =====================
def merge_all_shards(model_name, feature_dir, num_shards):
    """
    合并所有分片的 chunk 文件为一个完整 npz
    分片文件命名: {model_name}_shard{i}_streaming_chunk*.npz
    """
    all_feats, all_labels, all_indices, all_counts = [], [], [], []
    layer_indices = None

    for shard_id in range(num_shards):
        pattern = os.path.join(
            feature_dir, f"{model_name}_shard{shard_id}_streaming_chunk*.npz")
        chunk_files = sorted(glob.glob(pattern))
        if not chunk_files:
            print(f"  ⚠ shard {shard_id}: 没有 chunk 文件")
            continue
        for cf in chunk_files:
            data = np.load(cf, allow_pickle=True)
            all_feats.append(data['features'])
            all_labels.append(data['task_labels'])
            all_indices.append(data['task_indices'])
            all_counts.append(data['n_timesteps'])
            if layer_indices is None:
                layer_indices = data['layer_indices']
        print(f"  shard {shard_id}: {len(chunk_files)} chunks loaded")

    if not all_feats:
        print("  没有任何数据可合并!")
        return None

    features = np.concatenate(all_feats, axis=0)
    task_labels = np.concatenate(all_labels)
    task_indices = np.concatenate(all_indices)
    n_timesteps = np.concatenate(all_counts)

    out_path = os.path.join(feature_dir, f"{model_name}_streaming_features.npz")
    np.savez_compressed(
        out_path,
        features=features, task_labels=task_labels,
        task_indices=task_indices, n_timesteps=n_timesteps,
        layer_indices=layer_indices,
    )
    print(f"  [MERGED] {out_path}")
    print(f"    shape={features.shape}, unique_tasks={len(np.unique(task_indices))}")
    return out_path


# ===================== 单分片提取 =====================
def run_shard(args, shard_id, num_shards):
    """
    单个 GPU 分片的提取逻辑
    使用 TFDS subsplit 在文件级别分片，每个 GPU 只读 1/N 的 TFRecord 文件，
    无需帧级跳过，零 I/O 浪费
    """
    tag = f"[shard {shard_id}/{num_shards}]"
    print(f"\n{'=' * 70}")
    print(f"{tag} SpatialVLA 流式特征提取")
    print(f"  model={args.model_name}, device={args.device}, batch_size={args.batch_size}")
    print(f"  save_interval={args.save_interval}")
    print(f"  layers={list(LAYER_INDICES)} (共{len(LAYER_INDICES)}层)")
    print(f"  output={STREAMING_FEATURE_DIR}")
    print(f"{'=' * 70}")

    # 预加载 text → task_index 映射
    text_to_task_index = {}
    with open(TASK_RLDS_PATH) as f:
        for line in f:
            d = json.loads(line)
            text_to_task_index[d["task"].lower()] = d["task_index"]
    n_egohod = np.load(EGOHOD_EMB_PATH)["embeddings"].shape[0]
    print(f"  {tag} text→task_index: {len(text_to_task_index)}, EgoHOD: {n_egohod}")

    # 加载模型到指定 device
    model, processor, tokenizer, action_tokenizer, config = load_model(
        args.model_name, args.device
    )
    full_processor = build_vla_processor(processor, tokenizer, action_tokenizer)

    # 构建 RLDS pipeline (TFDS subsplit: 每个 GPU 只读自己的 TFRecord 文件)
    dataset = build_alignment_rlds_pipeline(
        num_shards=num_shards, shard_id=shard_id)

    # 轨迹容器
    traj_accum = defaultdict(lambda: {
        'sum_feat': None, 'count': 0, 'seen_ts': set(),
        'task_index': -1, 'lang': '',
    })
    task_traj_set = defaultdict(set)

    batch_tensors, batch_meta = [], []
    total_frames, total_skipped = 0, 0
    completed_trajs, chunk_idx = 0, 0
    pending_keys, pending_keys_set = [], set()
    prev_traj_idx, prev_traj_key = None, None

    # 分片前缀: 保证不同 shard 的 chunk 文件名不冲突
    shard_prefix = f"{args.model_name}_shard{shard_id}"

    print(f"  {tag} 开始流式扫描 (file-level split, 只读 1/{num_shards} 文件)...")
    pbar = tqdm(desc=f"{tag} streaming")

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for frame in dataset.as_numpy_iterator():
            pbar.update(1)

            fd = parse_rlds_frame(frame)
            lang_lower = fd['lang'].lower()
            task_index = text_to_task_index.get(lang_lower, -1)

            if task_index < 0 or task_index >= n_egohod:
                total_skipped += 1
                continue

            traj_idx = fd['traj_idx']

            if args.max_traj_per_task is not None:
                if (len(task_traj_set[task_index]) >= args.max_traj_per_task
                        and traj_idx not in task_traj_set[task_index]):
                    total_skipped += 1
                    continue
            task_traj_set[task_index].add(traj_idx)

            traj_key = (task_index, traj_idx)
            if prev_traj_idx is not None and traj_idx != prev_traj_idx:
                if (prev_traj_key in traj_accum
                        and traj_accum[prev_traj_key]['count'] > 0
                        and prev_traj_key not in pending_keys_set):
                    pending_keys.append(prev_traj_key)
                    pending_keys_set.add(prev_traj_key)
                    completed_trajs += 1

                    if completed_trajs >= args.save_interval:
                        if batch_tensors:
                            n = forward_batch_streaming(
                                batch_tensors, batch_meta, model,
                                action_tokenizer, args.device, traj_accum)
                            total_frames += n
                            batch_tensors, batch_meta = [], []

                        save_checkpoint(
                            traj_accum, pending_keys,
                            shard_prefix, chunk_idx, STREAMING_FEATURE_DIR)
                        chunk_idx += 1
                        pending_keys, pending_keys_set = [], set()
                        completed_trajs = 0

            prev_traj_idx = traj_idx
            prev_traj_key = traj_key

            traj_accum[traj_key]['task_index'] = task_index
            traj_accum[traj_key]['lang'] = lang_lower

            processed = process_frame_for_vla(fd, full_processor)
            batch_tensors.append(processed)
            batch_meta.append({'traj_key': traj_key, 'timestep': fd['timestep']})

            if len(batch_tensors) >= args.batch_size:
                n = forward_batch_streaming(
                    batch_tensors, batch_meta, model,
                    action_tokenizer, args.device, traj_accum)
                total_frames += n
                batch_tensors, batch_meta = [], []
                pbar.set_postfix(
                    tasks=len(task_traj_set),
                    trajs=sum(len(v) for v in task_traj_set.values()),
                    frames=total_frames, chunk=chunk_idx)

        if batch_tensors:
            n = forward_batch_streaming(
                batch_tensors, batch_meta, model,
                action_tokenizer, args.device, traj_accum)
            total_frames += n

    pbar.close()

    remaining_keys = list(traj_accum.keys())
    if remaining_keys:
        save_checkpoint(
            traj_accum, remaining_keys,
            shard_prefix, chunk_idx, STREAMING_FEATURE_DIR)

    n_tasks = len(task_traj_set)
    n_trajs = sum(len(v) for v in task_traj_set.values())
    print(f"\n{tag} 完成: tasks={n_tasks}, trajs={n_trajs}, "
          f"frames={total_frames}, skipped={total_skipped}")

    # 单卡模式: 直接合并本 shard 的 chunks
    if num_shards == 1:
        print(f"\n合并所有 chunk...")
        merged_path = merge_chunks(args.model_name, STREAMING_FEATURE_DIR)
        print(f"  merged: {merged_path}")


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description="SpatialVLA 全量流式特征提取 (支持多卡并行)")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--save_interval", type=int, default=500,
                        help="每多少条轨迹保存一次 checkpoint")
    parser.add_argument("--max_traj_per_task", type=int, default=None,
                        help="每个 task 最多采集的轨迹数 (默认不限)")
    # 多卡并行参数
    parser.add_argument("--num_shards", type=int, default=1,
                        help="总分片数 (= GPU 数)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="当前分片 ID (0-indexed)")
    parser.add_argument("--merge_only", action="store_true",
                        help="仅合并所有分片结果，不做提取")
    args = parser.parse_args()

    os.chdir(SPATIALVLA_DIR)
    os.makedirs(STREAMING_FEATURE_DIR, exist_ok=True)

    # 仅合并模式
    if args.merge_only:
        print(f"合并 {args.num_shards} 个分片...")
        merge_all_shards(args.model_name, STREAMING_FEATURE_DIR, args.num_shards)
        return

    assert 0 <= args.shard_id < args.num_shards, \
        f"shard_id={args.shard_id} 超出范围 [0, {args.num_shards})"

    run_shard(args, args.shard_id, args.num_shards)


if __name__ == "__main__":
    main()
