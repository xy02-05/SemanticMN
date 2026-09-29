"""
全量 SpatialVLA Action Token 特征提取 (用于 Action ↔ Text 线性对齐)

与 extract_features.py 的区别:
  - 覆盖所有 bridge task (19000+), 不限于选定的 67 个
  - 每个 task_index 最多收集 max_traj_per_task 条轨迹
  - 流式处理: RLDS 迭代 → VLA forward → 特征累加, 不存储中间图像 (节省 ~70GB 磁盘)
  - 输出: 轨迹级特征 npz (与 extract_features.py 格式兼容, 额外含 task_indices)

复用的已有代码 (不修改任何原文件):
  - load_model, build_vla_processor: 从 extract_features.py 导入
  - analysis_collator, TOKENS_PER_ACTION_STEP: 从 extract_features.py 导入
  - RLDS pipeline 构建逻辑: 基于 build_data_subset.py, 内联实现, 去掉 task 过滤
  - 帧解析 + VLA processor 调用: 复用 CachedBridgeDataset 的处理逻辑
  - VLA forward + action hidden states 提取: 复用 extract_features.py 的提取逻辑

用法:
    cd /root/data/xuyuan1/Codes/analysis/SpatialVLA
    python .../extract_alignment_features.py --model_name cotrain_fg
    python .../extract_alignment_features.py --model_name pretrained --max_traj_per_task 3
"""
import os
import sys
import json
import argparse
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
# 复用 extract_features.py 的模型加载和 VLA processor
from bridge_representation.extract_features import (
    load_model, build_vla_processor, analysis_collator, TOKENS_PER_ACTION_STEP,
)

# 对齐特征输出目录 (独立于现有的 features/ 目录)
ALIGN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "alignment")
ALIGN_FEATURE_DIR = os.path.join(ALIGN_OUTPUT_DIR, "features")


# ===================== RLDS Pipeline (不带 task 过滤) =====================
def build_alignment_rlds_pipeline(num_shards=1, shard_id=0):
    """
    构建全量 RLDS pipeline: 遍历所有 task, 不做 task_lang 过滤

    多卡加速: 使用 TFDS subsplit (如 "train[0%:50%]") 在文件级别分片，
    每个 GPU 只读自己的 TFRecord 文件。TFDS subsplit 基于 example 绝对位置，
    是确定性的 (不受并行读取顺序影响)，保证各 shard 不重叠、不遗漏。
    """
    import tensorflow as tf
    # ★ TF 只用 CPU, GPU 留给 PyTorch 的 VLA 模型
    tf.config.set_visible_devices([], 'GPU')

    from data.rlds import (
        make_dataset_from_rlds,
        apply_trajectory_transforms,
        apply_frame_transforms,
    )
    from data.oxe import get_oxe_dataset_kwargs_and_weights
    from data.utils.data_utils import NormalizationType

    # TFDS subsplit: 每个 GPU 只读 1/N 的 TFRecord 文件
    if num_shards > 1:
        pct_start = (shard_id * 100) // num_shards
        pct_end = ((shard_id + 1) * 100) // num_shards
        split_str = f"train[{pct_start}%:{pct_end}%]"
        print(f"[SHARD] file-level split: {split_str} (shard {shard_id}/{num_shards})")
    else:
        split_str = None

    mixture_spec = [("bridge_orig/1.0.0", 1.0)]
    per_dataset_kwargs, _ = get_oxe_dataset_kwargs_and_weights(
        RLDS_DATA_ROOT, mixture_spec,
        load_camera_views=("primary",), load_depth=False,
        load_proprio=False, load_language=True,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    dataset_kwargs = per_dataset_kwargs[0].copy()
    dataset_kwargs.pop("dataset_frame_transform_kwargs", None)

    print("Loading RLDS dataset...")
    dataset, _ = make_dataset_from_rlds(
        **dataset_kwargs, shuffle_seed=42,
        train=True, shuffle=False,
        num_parallel_calls=tf.data.AUTOTUNE, num_parallel_reads=2,
        split_override=split_str,
    )
    print("[Alignment] No task filter — collecting ALL tasks")

    # 轨迹级变换: action chunking (forward_window_size=3 → 4 action steps, 与训练一致)
    dataset = apply_trajectory_transforms(
        dataset, train=True, skip_unlabeled=True,
        goal_relabeling_strategy="uniform",
        backward_windows_size=0, backward_delta=1, forward_window_size=3,
    ).flatten(num_parallel_calls=4)

    # 帧级变换: decode + resize → 224×224, 无增强
    dataset = apply_frame_transforms(
        dataset, train=False, resize_size=(224, 224),
        image_augment_kwargs={}, num_parallel_calls=16,
    )
    return dataset


# ===================== RLDS 帧解析 (复用 build_data_subset.py 逻辑) =====================
def parse_rlds_frame(frame):
    """从 RLDS frame 中提取 image/action/text/traj_idx/timestep"""
    lang_raw = frame["task"]["language_instruction"]
    if isinstance(lang_raw, np.ndarray):
        lang_raw = lang_raw.flat[0]
    lang = lang_raw.decode() if isinstance(lang_raw, bytes) else str(lang_raw)

    image = frame["observation"]["image_primary"]
    if image.ndim == 4:
        image = image[0]                    # [1,H,W,C] → [H,W,C]

    action = frame["action"]                # [4, D] float32 (action chunk)

    traj_idx_raw = frame.get("traj_index", -1)
    if isinstance(traj_idx_raw, np.ndarray):
        traj_idx = int(traj_idx_raw.flat[0]) if traj_idx_raw.size > 0 else -1
    else:
        traj_idx = int(traj_idx_raw)

    obs_ts = frame["observation"]["timestep"]
    timestep = int(obs_ts.flat[-1]) if isinstance(obs_ts, np.ndarray) else int(obs_ts)

    return {
        'image': image, 'action': action,
        'lang': lang, 'traj_idx': traj_idx, 'timestep': timestep,
    }


# ===================== VLA Processor 处理 (复用 CachedBridgeDataset 逻辑) =====================
def process_frame_for_vla(frame_data, vla_processor):
    """将原始帧数据处理为 VLA model-ready tensors"""
    pil_image = Image.fromarray(frame_data['image'])
    actions_tensor = torch.from_numpy(frame_data['action'])
    lang = frame_data['lang'].lower()

    ret = vla_processor(
        text=lang, images=[pil_image],
        suffix_actions=actions_tensor,
        return_tensors="pt", padding=False,
        max_length=2048, truncation=True, do_normalize=False,
    )
    return {
        'input_ids': ret['input_ids'][0],
        'labels': ret['labels'][0],
        'token_type_ids': ret['token_type_ids'][0],
        'attention_mask': ret['attention_mask'][0],
        'pixel_values': ret['pixel_values'],
        'intrinsic': ret['intrinsic'],
        'canonical_label': lang,        # analysis_collator 需要 (会被 pop)
    }


# ===================== GPU Batch 处理 (复用 extract_features.py 逻辑) =====================
def forward_batch(batch_tensors, batch_meta, model, action_tokenizer, device, traj_accum):
    """
    GPU forward + 特征提取 + 累加到轨迹容器
    特征提取逻辑与 extract_features.py 完全一致: chunk1 (前3 tokens) + chunk4 (全部)
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

    # Level 1 pooling: chunk1 (当前步前3 tokens) + chunk4 (全部 action tokens)
    n_step0 = TOKENS_PER_ACTION_STEP  # 3
    feats_c1 = action_hs[:, :, :n_step0, :].mean(dim=2)   # [L, B, D]
    feats_c4 = action_hs.mean(dim=2)                        # [L, B, D]
    feats_c1 = feats_c1.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]
    feats_c4 = feats_c4.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]

    # 累加到各轨迹容器 (running sum, 最终除以 count 得到 mean)
    for j in range(cur_bs):
        meta = batch_meta[j]
        key = meta['traj_key']
        ts = meta['timestep']
        if ts in traj_accum[key]['seen_ts']:
            continue
        traj_accum[key]['seen_ts'].add(ts)
        if traj_accum[key]['count'] == 0:
            traj_accum[key]['sum_c1'] = feats_c1[j].copy()   # [L, D]
            traj_accum[key]['sum_c4'] = feats_c4[j].copy()
        else:
            traj_accum[key]['sum_c1'] += feats_c1[j]
            traj_accum[key]['sum_c4'] += feats_c4[j]
        traj_accum[key]['count'] += 1

    return cur_bs


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description="全量 SpatialVLA 特征提取 (alignment)")
    parser.add_argument("--model_name", type=str, default="cotrain_fg",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tasks", type=int, default=15000,
                        help="最多收集的 task_index 数量")
    parser.add_argument("--max_traj_per_task", type=int, default=3,
                        help="每个 task_index 最多收集的轨迹数")
    args = parser.parse_args()

    os.chdir(SPATIALVLA_DIR)

    print("=" * 60)
    print("全量 SpatialVLA 特征提取 (Alignment 模式)")
    print(f"  model={args.model_name}, batch_size={args.batch_size}")
    print(f"  max_tasks={args.max_tasks}, max_traj_per_task={args.max_traj_per_task}")
    print(f"  layers={list(LAYER_INDICES)}")
    print("=" * 60)

    # ---- 1. 预加载 text→task_index 映射 ----
    text_to_task_index = {}
    with open(TASK_RLDS_PATH) as f:
        for line in f:
            d = json.loads(line)
            text_to_task_index[d["task"].lower()] = d["task_index"]
    # EgoHOD embedding 数量 (用于过滤无效 task_index)
    n_egohod = np.load(EGOHOD_EMB_PATH)["embeddings"].shape[0]
    print(f"  text→task_index 映射: {len(text_to_task_index)} texts")
    print(f"  EgoHOD embedding: {n_egohod} entries")

    # ---- 2. 加载 VLA 模型 (复用 extract_features.py) ----
    model, processor, tokenizer, action_tokenizer, config = load_model(
        args.model_name, args.device
    )
    full_processor = build_vla_processor(processor, tokenizer, action_tokenizer)

    # ---- 3. 构建 RLDS pipeline (无 task 过滤) ----
    dataset = build_alignment_rlds_pipeline()

    # ---- 4. 流式提取特征 ----
    # 轨迹容器: key=(task_index, rlds_traj_idx), 值为 running sum + count
    traj_accum = defaultdict(lambda: {
        'sum_c1': None, 'sum_c4': None, 'count': 0, 'seen_ts': set(),
        'task_index': -1, 'lang': '',
    })
    task_traj_set = defaultdict(set)   # task_index → {rlds_traj_idx, ...}

    batch_tensors = []   # VLA-processed tensors (待 collate)
    batch_meta = []      # 每个 sample 的轨迹元数据
    total_frames = 0
    total_skipped = 0

    print(f"\n开始流式扫描 RLDS → VLA forward → 特征累加...")
    pbar = tqdm(desc="Streaming extract")

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for frame in dataset.as_numpy_iterator():
            pbar.update(1)

            # 解析帧
            fd = parse_rlds_frame(frame)
            lang_lower = fd['lang'].lower()
            task_index = text_to_task_index.get(lang_lower, -1)

            # 跳过: 无 task_index 或无 EgoHOD embedding
            if task_index < 0 or task_index >= n_egohod:
                total_skipped += 1
                continue

            # 跳过: 已达到 max_tasks 上限且该 task 是新 task
            traj_idx = fd['traj_idx']
            if (task_index not in task_traj_set
                    and len(task_traj_set) >= args.max_tasks):
                total_skipped += 1
                continue

            # 跳过: 该 task 已收满轨迹
            if (len(task_traj_set[task_index]) >= args.max_traj_per_task
                    and traj_idx not in task_traj_set[task_index]):
                total_skipped += 1
                continue
            task_traj_set[task_index].add(traj_idx)

            # VLA processor 处理帧
            processed = process_frame_for_vla(fd, full_processor)
            traj_key = (task_index, traj_idx)
            traj_accum[traj_key]['task_index'] = task_index
            traj_accum[traj_key]['lang'] = lang_lower

            batch_tensors.append(processed)
            batch_meta.append({'traj_key': traj_key, 'timestep': fd['timestep']})

            # 攒满一个 batch → GPU forward
            if len(batch_tensors) >= args.batch_size:
                n = forward_batch(
                    batch_tensors, batch_meta, model, action_tokenizer,
                    args.device, traj_accum
                )
                total_frames += n
                batch_tensors = []
                batch_meta = []
                pbar.set_postfix(
                    tasks=len(task_traj_set), trajs=len(traj_accum),
                    frames=total_frames, skip=total_skipped
                )

        # 处理剩余帧
        if batch_tensors:
            n = forward_batch(
                batch_tensors, batch_meta, model, action_tokenizer,
                args.device, traj_accum
            )
            total_frames += n

    pbar.close()

    # ---- 5. 汇总: running sum / count → 轨迹级 mean 特征, 保存 ----
    print(f"\n扫描完成: {len(task_traj_set)} tasks, {len(traj_accum)} traj, {total_frames} frames")

    all_c1, all_c4 = [], []
    all_labels, all_indices, all_counts = [], [], []

    for key, data in sorted(traj_accum.items()):
        if data['count'] == 0:
            continue
        all_c1.append(data['sum_c1'] / data['count'])       # [L, D] mean
        all_c4.append(data['sum_c4'] / data['count'])
        all_labels.append(data['lang'])
        all_indices.append(data['task_index'])
        all_counts.append(data['count'])

    os.makedirs(ALIGN_FEATURE_DIR, exist_ok=True)

    if not all_c1 or not all_c4:
        raise RuntimeError(
            "未收集到有效轨迹特征（all_c1/all_c4 为空）。请检查 task_index 映射、EgoHOD embedding 或筛选参数。"
        )

    for chunk_tag, feats_list in [('chunk1', all_c1), ('chunk4', all_c4)]:
        features = np.stack(feats_list, axis=0)              # [N, L, D]
        out_path = os.path.join(
            ALIGN_FEATURE_DIR, f"align_{args.model_name}_{chunk_tag}_features.npz"
        )
        np.savez_compressed(
            out_path,
            features=features,                                # [N, L, D]
            task_labels=np.array(all_labels),                 # [N] str
            task_indices=np.array(all_indices, dtype=np.int64),  # [N] int
            n_timesteps=np.array(all_counts, dtype=np.int64),   # [N] int
            layer_indices=np.array(LAYER_INDICES),             # [L] int
        )
        print(f"  {chunk_tag}: {out_path}")
        print(f"    shape={features.shape}, tasks={len(set(all_indices))}")

    print(f"\n{'=' * 60}")
    print(f"全量特征提取完成!")
    print(f"  轨迹数: {len(all_labels)}, task 数: {len(set(all_indices))}")
    print(f"  输出: {ALIGN_FEATURE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
