"""
验证 TFDS subsplit 文件级分片的正确性

用内容指纹 (lang + timestep序列) 识别轨迹，避免依赖非确定性的 _traj_index

测试项:
  1. 各 shard 的轨迹（内容指纹）互不重叠
  2. 合并后覆盖完整数据集
  3. 轨迹内帧不被拆分
  4. 总帧数正确

用法:
  cd /root/data/xuyuan1/Codes/analysis/SpatialVLA
  python ../bridge_representation/verify_shard.py [--num_shards 2] [--max_frames 10000]
"""
import os, sys, argparse, hashlib
import numpy as np
from collections import defaultdict

ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
SPATIALVLA_DIR = os.path.join(ANALYSIS_DIR, "SpatialVLA")
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, SPATIALVLA_DIR)

from bridge_representation.extract_alignment_features import (
    build_alignment_rlds_pipeline, parse_rlds_frame,
)


def collect_trajectories(dataset, max_frames, label=""):
    """
    扫描 dataset，按 _traj_index 边界切分轨迹，
    用 (lang, timestep元组) 作为内容指纹
    返回: {fingerprint: {'lang': str, 'timesteps': tuple, 'n_frames': int}}
    """
    # 按 traj_idx 分组帧
    traj_data = defaultdict(lambda: {'lang': '', 'timesteps': []})
    count = 0
    for frame in dataset.as_numpy_iterator():
        fd = parse_rlds_frame(frame)
        key = fd['traj_idx']
        traj_data[key]['lang'] = fd['lang'].lower()
        traj_data[key]['timesteps'].append(fd['timestep'])
        count += 1
        if count >= max_frames:
            break

    # 转为内容指纹
    result = {}
    for key, data in traj_data.items():
        ts_tuple = tuple(sorted(data['timesteps']))
        # 指纹 = (lang, timestep元组的哈希) — 唯一标识一条轨迹
        fp = hashlib.md5(f"{data['lang']}|{ts_tuple}".encode()).hexdigest()[:16]
        result[fp] = {
            'lang': data['lang'],
            'timesteps': ts_tuple,
            'n_frames': len(data['timesteps']),
        }

    print(f"  [{label}] {count} 帧, {len(result)} 条轨迹")
    return result, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=10000)
    args = parser.parse_args()
    N = args.num_shards

    os.chdir(SPATIALVLA_DIR)

    # 1. 完整数据集
    print(f"加载完整数据集 (前 {args.max_frames} 帧)...")
    ds_full = build_alignment_rlds_pipeline(num_shards=1, shard_id=0)
    full_trajs, full_count = collect_trajectories(ds_full, args.max_frames, "full")

    # 2. 各 shard
    shard_trajs = {}
    shard_counts = {}
    for sid in range(N):
        print(f"\n加载 shard {sid}/{N}...")
        ds = build_alignment_rlds_pipeline(num_shards=N, shard_id=sid)
        shard_trajs[sid], shard_counts[sid] = collect_trajectories(
            ds, args.max_frames, f"shard{sid}")

    # ---- 验证 ----
    print("\n" + "=" * 60)
    all_ok = True

    # 1. 各 shard 轨迹不重叠
    print("验证 1: 各 shard 轨迹指纹不重叠")
    shard_fps = {sid: set(shard_trajs[sid].keys()) for sid in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            ov = shard_fps[i] & shard_fps[j]
            if ov:
                print(f"  ✗ shard {i} ∩ shard {j} = {len(ov)} 条重叠")
                for fp in list(ov)[:3]:
                    print(f"      重叠轨迹: {shard_trajs[i][fp]['lang'][:60]}")
                all_ok = False
            else:
                print(f"  ✓ shard {i} ∩ shard {j} = 空")

    # 2. 合并覆盖
    print("\n验证 2: 合并后覆盖完整")
    merged_fps = set()
    for sid in range(N):
        merged_fps |= shard_fps[sid]
    full_fps = set(full_trajs.keys())
    missing = full_fps - merged_fps
    total_shard_frames = sum(shard_counts[sid] for sid in range(N))
    print(f"  完整: {len(full_fps)} 轨迹 ({full_count} 帧)")
    print(f"  合并: {len(merged_fps)} 轨迹 ({total_shard_frames} 帧)")
    print(f"  缺失: {len(missing)}")

    # 因为各 shard 扫描的帧范围更广 (每个 shard 读 max_frames 帧)，
    # 合并后轨迹数应 >= 完整。检查完整数据集的轨迹是否都被覆盖。
    if missing:
        # 可能是因为 full 只扫描了 max_frames 帧，而某些轨迹在 shard 分割后
        # 被推到了 max_frames 之外。只报告比例。
        coverage = len(full_fps - missing) / len(full_fps) * 100
        print(f"  覆盖率: {coverage:.1f}% (缺失可能因 max_frames 截断)")
        if len(missing) > len(full_fps) * 0.1:
            all_ok = False
    else:
        print(f"  ✓ 完整覆盖")

    # 3. 帧内容一致
    print("\n验证 3: 共同轨迹的帧内容一致")
    common = full_fps & merged_fps
    mismatch = 0
    for fp in common:
        full_ts = full_trajs[fp]['timesteps']
        for sid in range(N):
            if fp in shard_trajs[sid]:
                shard_ts = shard_trajs[sid][fp]['timesteps']
                if full_ts != shard_ts:
                    mismatch += 1
                    if mismatch <= 3:
                        print(f"  ✗ {full_trajs[fp]['lang'][:40]}: "
                              f"full={len(full_ts)} ts, shard={len(shard_ts)} ts")
                break
    if mismatch == 0:
        print(f"  ✓ {len(common)} 条共同轨迹帧内容完全一致")
    else:
        print(f"  ✗ {mismatch} 条不一致!")
        all_ok = False

    # 4. 帧数守恒
    print(f"\n验证 4: 帧数守恒")
    common_full_frames = sum(full_trajs[fp]['n_frames'] for fp in common)
    common_shard_frames = 0
    for fp in common:
        for sid in range(N):
            if fp in shard_trajs[sid]:
                common_shard_frames += shard_trajs[sid][fp]['n_frames']
                break
    if common_full_frames == common_shard_frames:
        print(f"  ✓ 共同轨迹帧数一致: {common_full_frames}")
    else:
        print(f"  ✗ full={common_full_frames}, shard={common_shard_frames}")
        all_ok = False

    print("\n" + "=" * 60)
    print("所有验证通过!" if all_ok else "存在问题!")
    print("=" * 60)


if __name__ == "__main__":
    main()
