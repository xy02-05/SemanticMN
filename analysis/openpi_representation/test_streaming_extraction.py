"""
流式特征提取 — 全量 RLDS 遍历验证

验证标准:
  Phase 1 (快速全量扫描): 遍历完整 Bridge RLDS 数据集的每一条轨迹
    - 不解码图像、不跑模型，只检查: prompt, action shape, state shape, task 映射
    - 统计: 总轨迹数、unique task 数、映射率、轨迹长度分布
    - 预期: 30000+ 轨迹, 20000+ unique tasks
  Phase 2 (少量模型验证): 从不同 task 取 5 条轨迹跑完整 pipeline
    - 图像解码 + transforms + forward + 两级 mean pool + save/load

用法:
    python test_streaming_extraction.py          # 全量扫描 + 模型测试
    python test_streaming_extraction.py --scan   # 仅全量扫描 (无 GPU)
"""
import os
import sys
import io
import json
import time
import argparse
import tempfile
import numpy as np
from collections import Counter

# ===================== 路径 =====================
ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
OPENPI_DIR = os.path.join(ANALYSIS_DIR, "openpi")
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, os.path.join(OPENPI_DIR, "src"))

from openpi_representation.config import (
    MODEL_CONFIGS, LAYER_INDICES, TASKS_WITH_ID_PATH,
    SELECTED_TASK_INDICES,
)
from openpi_representation.extract_features_streaming import (
    RLDS_DATA_DIR, RLDS_DATASET_NAME,
)

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
n_pass, n_fail = 0, 0


def check(condition, msg):
    global n_pass, n_fail
    if condition:
        print(f"  {PASS}  {msg}")
        n_pass += 1
    else:
        print(f"  {FAIL}  {msg}")
        n_fail += 1


# ================================================================
# Phase 1: 全量轻量扫描 (不解码图像, 不加载模型)
# ================================================================
def phase1_full_scan():
    """
    遍历完整 Bridge RLDS 数据集的每一条轨迹，仅检查元数据。
    跳过图像解码，速度快。
    """
    print("\n" + "=" * 60)
    print("PHASE 1: 全量 RLDS 轻量扫描 (不解码图像)")
    print("=" * 60)

    import tensorflow as tf
    import tensorflow_datasets as tfds
    import dlimp as dl

    tf.config.set_visible_devices([], "GPU")

    # 加载 task 映射 (与 BridgeTrajectoryIterator 完全一致)
    from openpi.training.droid_rlds_dataset import load_task_mapping_for_rlds
    lang_to_task_index, task_index_to_task_id = load_task_mapping_for_rlds(TASKS_WITH_ID_PATH)
    print(f"  task 映射: {len(lang_to_task_index)} prompts → task_index")

    # 加载 RLDS (与 BridgeTrajectoryIterator 完全一致)
    print(f"  Loading Bridge RLDS: {RLDS_DATASET_NAME} from {RLDS_DATA_DIR}")
    builder = tfds.builder(RLDS_DATASET_NAME, data_dir=RLDS_DATA_DIR, version="1.0.0")
    dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False)
    dataset = dataset.filter(lambda traj: tf.reduce_any(traj["language_instruction"] != b""))
    print("  RLDS ready, 开始遍历所有轨迹...\n")

    # --- 遍历 ---
    total_traj = 0
    total_frames = 0
    task_index_counter = Counter()
    task_id_counter = Counter()
    prompt_set = set()
    traj_lens = []
    bad_action_shape = 0
    bad_state_shape = 0
    bad_prompt = 0
    n_mapped = 0
    n_unmapped = 0
    prompt_mismatch = 0
    id_mismatch = 0

    # 用于交叉验证的 jsonl
    idx_to_text = {}
    idx_to_id = {}
    with open(TASKS_WITH_ID_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            ti = item.get("task_index", -1)
            text = item.get("task", "")
            tid = item.get("task_id", -1)
            if ti >= 0 and text:
                idx_to_text[ti] = text.strip().lower()
                if tid >= 0:
                    idx_to_id[ti] = tid

    t0 = time.time()
    for traj in dataset.as_numpy_iterator():
        raw_actions = traj["action"]              # [T, 7]
        raw_states = traj["observation"]["state"]  # [T, 7]
        raw_prompt = traj["language_instruction"]  # [T] bytes

        T = len(raw_actions)
        if T == 0:
            continue

        total_traj += 1
        total_frames += T
        traj_lens.append(T)

        # --- action shape ---
        if raw_actions.shape != (T, 7):
            bad_action_shape += 1

        # --- state shape ---
        if raw_states.shape != (T, 7):
            bad_state_shape += 1

        # --- prompt ---
        prompt_bytes = raw_prompt[0] if isinstance(raw_prompt, np.ndarray) else raw_prompt
        prompt = prompt_bytes.decode("utf-8") if isinstance(prompt_bytes, bytes) else str(prompt_bytes)
        if not prompt or len(prompt.strip()) == 0:
            bad_prompt += 1
            continue
        prompt_set.add(prompt)

        # --- task 映射 (与 BridgeTrajectoryIterator 一致) ---
        prompt_norm = prompt.strip().lower()
        task_index = lang_to_task_index.get(prompt_norm, -1)
        task_id = task_index_to_task_id.get(task_index, task_index)
        task_index_counter[task_index] += 1
        task_id_counter[task_id] += 1

        if task_index >= 0:
            n_mapped += 1
            # 交叉验证
            expected_text = idx_to_text.get(task_index)
            if expected_text and prompt_norm != expected_text:
                prompt_mismatch += 1
            expected_id = idx_to_id.get(task_index, task_index)
            if int(task_id) != int(expected_id):
                id_mismatch += 1
        else:
            n_unmapped += 1

        # 进度 (每 5000 条打印)
        if total_traj % 5000 == 0:
            elapsed = time.time() - t0
            speed = total_traj / elapsed
            print(f"    [{total_traj:>6} trajs] "
                  f"unique_tasks={len(task_index_counter):>5}, "
                  f"unique_prompts={len(prompt_set):>5}, "
                  f"speed={speed:.0f} traj/s")

    elapsed = time.time() - t0

    # --- 结果报告 ---
    print(f"\n  扫描完成: {elapsed:.1f}s ({total_traj / elapsed:.0f} traj/s)")
    print(f"  {'─' * 50}")

    unique_task_indices = set(task_index_counter.keys()) - {-1}
    unique_task_ids = set(task_id_counter.keys())
    subset_indices = set(SELECTED_TASK_INDICES)
    outside_subset = unique_task_indices - subset_indices

    # 核心断言
    check(total_traj >= 30000,
          f"总轨迹数: {total_traj} (>= 30000)")
    check(len(unique_task_indices) >= 15000,
          f"unique task_index (已映射): {len(unique_task_indices)} (>= 15000)")
    check(len(prompt_set) >= 15000,
          f"unique prompts: {len(prompt_set)} (>= 15000)")
    check(len(outside_subset) > 1000,
          f"子集外 task_index: {len(outside_subset)} (> 1000, 证明非子集)")
    check(total_frames >= 500000,
          f"总帧数: {total_frames} (>= 500000)")

    # 格式断言
    check(bad_action_shape == 0,
          f"action shape 异常: {bad_action_shape}/{total_traj}")
    check(bad_state_shape == 0,
          f"state shape 异常: {bad_state_shape}/{total_traj}")
    check(bad_prompt == 0,
          f"空 prompt: {bad_prompt}/{total_traj}")

    # 映射断言
    map_pct = n_mapped / total_traj * 100
    check(map_pct >= 95,
          f"task 映射率: {n_mapped}/{total_traj} ({map_pct:.1f}%, >= 95%)")
    check(prompt_mismatch == 0,
          f"prompt↔task_index 不匹配: {prompt_mismatch}/{n_mapped}")
    check(id_mismatch == 0,
          f"task_id↔task_index 不匹配: {id_mismatch}/{n_mapped}")

    # 统计摘要
    avg_len = np.mean(traj_lens)
    print(f"\n  统计摘要:")
    print(f"    总轨迹: {total_traj}")
    print(f"    总帧数: {total_frames}")
    print(f"    unique prompts: {len(prompt_set)}")
    print(f"    unique task_index (已映射): {len(unique_task_indices)}")
    print(f"    unique task_id: {len(unique_task_ids)}")
    print(f"    子集内 task_index: {len(unique_task_indices & subset_indices)}/{len(subset_indices)}")
    print(f"    子集外 task_index: {len(outside_subset)}")
    print(f"    映射率: {n_mapped}/{total_traj} ({map_pct:.1f}%), 未映射: {n_unmapped}")
    print(f"    轨迹长度: avg={avg_len:.1f}, min={min(traj_lens)}, max={max(traj_lens)}")

    # top 10 task
    print(f"\n  Top-10 task (按轨迹数):")
    for tidx, cnt in task_index_counter.most_common(10):
        in_sub = "★" if tidx in subset_indices else " "
        print(f"    {in_sub} task_index={tidx:>5} count={cnt:>4}")

    return total_traj


# ================================================================
# Phase 2: 少量模型验证 (需要 GPU)
# ================================================================
def phase2_model_test():
    """从不同 task 取 5 条轨迹, 跑完整 pipeline (图像解码 + forward + save)"""
    print("\n" + "=" * 60)
    print("PHASE 2: 模型 Pipeline 验证 (5 条不同 task 轨迹)")
    print("=" * 60)

    import torch
    from openpi_representation.extract_features import (
        load_pi0_model, build_transforms, apply_transforms, load_norm_stats,
    )
    from openpi_representation.extract_features_streaming import (
        BridgeTrajectoryIterator, process_trajectory, save_checkpoint,
    )

    device = "cuda"
    model, pi0_config = load_pi0_model("pretrained", device)
    transforms_chain = build_transforms(pi0_config)
    raw_norm = load_norm_stats()
    state_fill = np.array(raw_norm["state"]["mean"], dtype=np.float32)
    L = len(LAYER_INDICES)

    # 取 5 条不同 task 的轨迹
    traj_iter = BridgeTrajectoryIterator(action_chunk_size=pi0_config.action_horizon)
    seen_tasks = set()
    test_trajs = []
    for traj in traj_iter:
        if traj["task_index"] not in seen_tasks:
            seen_tasks.add(traj["task_index"])
            test_trajs.append(traj)
        if len(test_trajs) >= 5:
            break

    print(f"  选取 {len(test_trajs)} 条轨迹:")
    for t in test_trajs:
        print(f"    task_idx={t['task_index']}, task_id={t['task_id']}, "
              f"len={t['traj_len']}, prompt='{t['prompt'][:50]}...'")

    # 逐条 process_trajectory
    results = []
    for traj in test_trajs:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feat = process_trajectory(
                traj, transforms_chain, state_fill,
                model, pi0_config, LAYER_INDICES,
                device, batch_size=8,
            )
        ok = (feat.shape == (L, 1024) and not np.any(np.isnan(feat))
              and not np.all(feat == 0) and np.abs(feat).max() < 1000)
        check(ok, f"[task_idx={traj['task_index']}] shape={feat.shape}, "
              f"|max|={np.abs(feat).max():.1f}")
        results.append({
            "feature": feat,
            "prompt": traj["prompt"],
            "task_index": int(traj["task_index"]),
            "task_id": int(traj["task_id"]),
            "n_timesteps": traj["traj_len"],
        })

    # save / load
    import openpi_representation.extract_features_streaming as efs
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir = efs.STREAMING_FEATURE_DIR
        efs.STREAMING_FEATURE_DIR = tmpdir
        path = save_checkpoint(results, "test", chunk_id=0)
        data = np.load(path, allow_pickle=True)
        N = len(results)
        check(data["features"].shape == (N, L, 1024),
              f"save/load features shape: {data['features'].shape}")
        for i in range(N):
            check(int(data["task_indices"][i]) == results[i]["task_index"]
                  and int(data["task_ids"][i]) == results[i]["task_id"]
                  and str(data["task_labels"][i]) == results[i]["prompt"],
                  f"[{i}] metadata 一致: idx={results[i]['task_index']}, "
                  f"id={results[i]['task_id']}")
        efs.STREAMING_FEATURE_DIR = orig_dir


# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true",
                        help="仅全量扫描, 不需要 GPU")
    args = parser.parse_args()

    print("=" * 60)
    print(" OpenPI 流式特征提取 — 全量验证测试")
    print("=" * 60)

    total_traj = phase1_full_scan()

    if not args.scan:
        phase2_model_test()

    print("\n" + "=" * 60)
    total = n_pass + n_fail
    print(f" 测试完成: {n_pass}/{total} passed, {n_fail}/{total} failed")
    print(f" 总轨迹: {total_traj}")
    if n_fail == 0:
        print(f" {PASS}  全部通过!")
    else:
        print(f" {FAIL}  有 {n_fail} 个测试失败")
    print("=" * 60)
    sys.exit(0 if n_fail == 0 else 1)
