#!/usr/bin/env bash

set -euo pipefail

# 等待某一步数的四个 rollout NPZ 完成，再复用原 train_probe.py 生成辅助 probe。
STEP_K="${1:?usage: run_rebuttal_50k_semantic_postprocess.sh <35|40|45|50> <gpu_id>}"
GPU_ID="${2:?usage: run_rebuttal_50k_semantic_postprocess.sh <35|40|45|50> <gpu_id>}"

case "$STEP_K" in
    35|40|45|50) ;;
    *)
        echo "STEP_K must be one of 35, 40, 45, 50" >&2
        exit 2
        ;;
esac

WORK_ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
SCRIPT_DIR="$WORK_ROOT/data_process/libero"
FEATURE_DIR="$SCRIPT_DIR/outputs/features"
PROBE_DIR="$SCRIPT_DIR/outputs/probes"
PYTHON=/mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_openpi/bin/python
CHECKPOINTS=("step_${STEP_K}k" "new_dsn_new_${STEP_K}k")

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MIRROR_PROJECT_ROOT=/mnt/bn/2d-videos/xy
export MIRROR_WORK_ROOT="$WORK_ROOT"
export MIRROR_DATA_ROOT="$WORK_ROOT/data"
export MIRROR_ANALYSIS_ROOT="$WORK_ROOT/analysis"
export OPENPI_ROOT="$WORK_ROOT/vlas/openpi"
export OPENPI_DATA_HOME="$WORK_ROOT/cache/openpi"
export PI0_PRETRAINED_PATH="$WORK_ROOT/weights/openpi/pi0_base_pytorch"
export PYTHONPATH="$WORK_ROOT/analysis:$OPENPI_ROOT/src"
export PYTHONNOUSERSITE=1

cd "$SCRIPT_DIR"

# 必须先等同一步数的 Raw/Ours 四个 NPZ 全部完成。
# 这样 probe 开始时，该 GPU 上的 extract_features.py 已经退出，不会互相抢显存。
for checkpoint in "${CHECKPOINTS[@]}"; do
    for split in train test; do
        path="$FEATURE_DIR/${checkpoint}_${split}_rollout.npz"
        process_pattern="[e]xtract_features.py --checkpoint $checkpoint --split $split"
        while [ ! -f "$path" ] || pgrep -f "$process_pattern" >/dev/null; do
            echo "[WAIT] $(date --iso-8601=seconds) $path"
            sleep 60
        done
    done
done

for checkpoint in "${CHECKPOINTS[@]}"; do
    # 只读取必要数组，确认 task-disjoint split 和层索引；错误直接退出。
    "$PYTHON" - "$checkpoint" <<'PY'
import os
import sys
import numpy as np

checkpoint = sys.argv[1]
feature_dir = "outputs/features"
expected = {
    "train": (1344, 32),
    "test": (349, 8),
}
for split, (expected_episodes, expected_tasks) in expected.items():
    path = os.path.join(feature_dir, f"{checkpoint}_{split}_rollout.npz")
    data = np.load(path, allow_pickle=False)
    tasks = data["task_indices"]
    layers = data["layer_indices"].tolist()
    if len(tasks) != expected_episodes:
        raise ValueError(f"{path}: expected {expected_episodes} episodes, got {len(tasks)}")
    if len(np.unique(tasks)) != expected_tasks:
        raise ValueError(f"{path}: expected {expected_tasks} tasks, got {len(np.unique(tasks))}")
    if layers != [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]:
        raise ValueError(f"{path}: unexpected layers {layers}")
print(f"[NPZ_OK] {checkpoint}")
PY

    result="$PROBE_DIR/${checkpoint}_qwen3_rollout/probe_results.json"
    if [ -f "$result" ]; then
        echo "[SKIP_PROBE] $result"
        continue
    fi

    echo "[PROBE_START] $(date --iso-8601=seconds) checkpoint=$checkpoint gpu=$GPU_ID"
    "$PYTHON" train_probe.py \
        --checkpoint "$checkpoint" \
        --text_type qwen3 \
        --probe_train_mode rollout \
        --probe_test_mode rollout \
        --epochs 200 \
        --min_best_epoch 50 \
        --layers 10 \
        --seed 42 \
        --device cuda
    echo "[PROBE_DONE] $(date --iso-8601=seconds) checkpoint=$checkpoint gpu=$GPU_ID"
done

echo "[POSTPROCESS_STEP_DONE] $(date --iso-8601=seconds) step=${STEP_K}k"
