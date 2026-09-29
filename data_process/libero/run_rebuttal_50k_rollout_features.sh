#!/usr/bin/env bash

set -euo pipefail

# 每张 GPU 负责同一步数的 Raw 与 Ours，依次提取 train/test rollout 特征。
# 特征计算完全复用 extract_features.py，本脚本只固定本机路径和任务顺序。
STEP_K="${1:?usage: run_rebuttal_50k_rollout_features.sh <35|40|45|50> <gpu_id>}"
GPU_ID="${2:?usage: run_rebuttal_50k_rollout_features.sh <35|40|45|50> <gpu_id>}"
TRAINING_LOCK="${OPENPI_50K_TRAINING_LOCK:-/tmp/openpi_50k_training.lock}"

case "$STEP_K" in
    35|40|45|50) ;;
    *)
        echo "STEP_K must be one of 35, 40, 45, 50" >&2
        exit 2
        ;;
esac

if [ -e "$TRAINING_LOCK" ]; then
    echo "OpenPI 50k formal training is active; refusing feature extraction: $TRAINING_LOCK" >&2
    exit 3
fi

WORK_ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
SCRIPT_DIR="$WORK_ROOT/data_process/libero"
FEATURE_DIR="$SCRIPT_DIR/outputs/features"
LOCAL_FEATURE_DIR="${REBUTTAL_LOCAL_FEATURE_DIR:-/opt/tiger/rh2/rh2/init/rebuttal_50k_rollout_features}"
PUBLISH_LOCK="${REBUTTAL_FEATURE_PUBLISH_LOCK:-/tmp/rebuttal_50k_feature_publish.lock}"
PYTHON=/mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_openpi/bin/python

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MIRROR_PROJECT_ROOT=/mnt/bn/2d-videos/xy
export MIRROR_WORK_ROOT="$WORK_ROOT"
export MIRROR_DATA_ROOT="$WORK_ROOT/data"
export OPENPI_ROOT="$WORK_ROOT/vlas/openpi"
export OPENPI_DATA_HOME="$WORK_ROOT/cache/openpi"
export PI0_PRETRAINED_PATH="$WORK_ROOT/weights/openpi/pi0_base_pytorch"
export MIRROR_HF_CACHE="$WORK_ROOT/data/physical-intelligence/libero/.cache/huggingface"
export MIRROR_FEATURE_DIR="$LOCAL_FEATURE_DIR"
export PYTHONPATH="$OPENPI_ROOT/src"
export PYTHONNOUSERSITE=1

cd "$SCRIPT_DIR"
mkdir -p "$FEATURE_DIR" "$LOCAL_FEATURE_DIR"

validate_npz() {
    local path=$1
    local split=$2
    "$PYTHON" - "$path" "$split" <<'PY'
import sys
import numpy as np

path, split = sys.argv[1:]
expected = {
    "train": (1344, 32),
    "test": (349, 8),
}
expected_episodes, expected_tasks = expected[split]
with np.load(path, allow_pickle=False) as data:
    if data["features"].shape != (expected_episodes, 10, 1024):
        raise ValueError(f"{path}: unexpected features shape {data['features'].shape}")
    if len(np.unique(data["task_indices"])) != expected_tasks:
        raise ValueError(f"{path}: unexpected task count")
    if data["layer_indices"].tolist() != [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]:
        raise ValueError(f"{path}: unexpected layer indices")
PY
}

publish_npz() {
    local local_path=$1
    local output_path=$2
    local split=$3
    local temp_path="${output_path}.publishing"

    validate_npz "$local_path" "$split"
    (
        flock -x 9
        rm -f "$temp_path"
        ln -s "$local_path" "$temp_path"
        mv -Tf "$temp_path" "$output_path"
    ) 9>"$PUBLISH_LOCK"
    validate_npz "$output_path" "$split"
}

for checkpoint in "step_${STEP_K}k" "new_dsn_new_${STEP_K}k"; do
    for split in train test; do
        output_path="$FEATURE_DIR/${checkpoint}_${split}_rollout.npz"
        if [ -f "$output_path" ]; then
            if validate_npz "$output_path" "$split"; then
                echo "[SKIP] $output_path"
                continue
            fi
            rm -f "$output_path"
        fi

        local_path="$LOCAL_FEATURE_DIR/${checkpoint}_${split}_rollout.npz"
        if [ -f "$local_path" ] && validate_npz "$local_path" "$split"; then
            publish_npz "$local_path" "$output_path" "$split"
            echo "[PUBLISHED] $(date --iso-8601=seconds) $output_path"
            continue
        fi

        rm -f "$local_path"
        echo "[START] $(date --iso-8601=seconds) checkpoint=$checkpoint split=$split gpu=$GPU_ID"
        "$PYTHON" extract_features.py \
            --checkpoint "$checkpoint" \
            --split "$split" \
            --feature_mode rollout \
            --batch_size 8 \
            --device cuda
        publish_npz "$local_path" "$output_path" "$split"
        echo "[DONE] $(date --iso-8601=seconds) checkpoint=$checkpoint split=$split gpu=$GPU_ID"
    done
done

echo "[QUEUE_DONE] $(date --iso-8601=seconds) step=${STEP_K}k gpu=$GPU_ID"
