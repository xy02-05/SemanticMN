#!/usr/bin/env bash

set -euo pipefail

ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
CONDA_ROOT=/mnt/bn/2d-videos/xy/tools/miniconda3
REPO="$ROOT/Isaac-GR00T"
BASE_MODEL="$ROOT/weights/gr00t/GR00T-N1.6-3B"
DATASET="$ROOT/data/simplerenv_bridge/bridge_orig_lerobot"
OUTPUT="${OUTPUT:-/tmp/gr00t_bridge_h20_4gpu_aligned}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gr00t_bridge_h20_4gpu_aligned}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LOG="${LOG:-$ROOT/logs/gr00t_bridge_h20_4gpu_aligned_local.log}"
RESOURCE_GUARD_DISABLE_FILE=/mnt/bn/2d-videos/xy/.disable_resource_guard

mkdir -p "$ROOT/logs" "$OUTPUT"

cleanup_memory_guard() {
    if [ -n "${TRAIN_MEMORY_GUARD_PID:-}" ] && kill -0 "$TRAIN_MEMORY_GUARD_PID" 2>/dev/null; then
        kill "$TRAIN_MEMORY_GUARD_PID" 2>/dev/null || true
    fi
}

on_exit() {
    cleanup_memory_guard
}

trap on_exit EXIT

# 训练期间持久禁用资源波和默认 160 GiB watchdog，避免新 shell 再次拉起它们。
printf '%s\n' 'GR00T Bridge finetune is active. Do not start resource guards.' \
    >"$RESOURCE_GUARD_DISABLE_FILE"

if tmux has-session -t memory-watchdog 2>/dev/null; then
    tmux kill-session -t memory-watchdog
fi
if tmux has-session -t gpu-utilization 2>/dev/null; then
    tmux kill-session -t gpu-utilization
fi

# 本机可用内存上限约 900 GiB，训练进程严格限制在 850 GiB 内。
python3 /mnt/bn/2d-videos/xy/memory_guard.py \
    --quota-gib 850 \
    --threshold-percent 100 \
    --interval-seconds 1 \
    --log-every-seconds 30 >>"$ROOT/logs/gr00t_bridge_h20_4gpu_aligned_memory_guard.log" 2>&1 &
TRAIN_MEMORY_GUARD_PID=$!

# Make sure the dataset subset that GR00T actually reads is present.
if [ -f "$ROOT/scripts_download_bridge_required.sh" ]; then
    bash "$ROOT/scripts_download_bridge_required.sh"
else
    for required_path in \
        meta/info.json \
        meta/stats.json \
        meta/episodes.jsonl \
        meta/tasks.jsonl \
        meta/modality.json \
        data \
        videos; do
        if [ ! -e "$DATASET/$required_path" ]; then
            echo "Missing required Bridge dataset path: $DATASET/$required_path" >&2
            exit 1
        fi
    done
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate mirror_gr00t_n16

cd "$REPO"

export LD_LIBRARY_PATH="$CONDA_ROOT/envs/mirror_gr00t_n16/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export USE_WANDB=0
export MASTER_PORT="${MASTER_PORT:-29661}"
export TOKENIZERS_PARALLELISM=false

# 官方 Bridge 配置使用 8 卡、20000 steps、全局 batch 1024 和 state dropout 0.8。
# 本机 4 卡保持每卡 micro batch 128，并累积 2 次，使每次参数更新仍使用 1024 个样本。
{
    echo "[$(date '+%F %T')] launching GR00T N1.6 Bridge finetune"
    echo "output_dir=$OUTPUT"
    echo "experiment_name=$EXPERIMENT_NAME"
    echo "effective_update_batch=1024"
    NUM_GPUS=4 \
    MAX_STEPS="$MAX_STEPS" \
    GLOBAL_BATCH_SIZE=512 \
    SAVE_STEPS="$SAVE_STEPS" \
    SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
    DATALOADER_NUM_WORKERS=4 \
    bash examples/finetune.sh \
        --base-model-path "$BASE_MODEL" \
        --dataset-path "$DATASET" \
        --embodiment-tag OXE_WIDOWX \
        --output-dir "$OUTPUT" \
        --experiment-name "$EXPERIMENT_NAME" \
        --state-dropout-prob 0.8 \
        -- --gradient-accumulation-steps 2
} 2>&1 | tee -a "$LOG"
