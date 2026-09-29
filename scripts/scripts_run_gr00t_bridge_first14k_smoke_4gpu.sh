#!/usr/bin/env bash

set -euo pipefail

ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
CONDA_ROOT=/mnt/bn/2d-videos/xy/tools/miniconda3
REPO="$ROOT/Isaac-GR00T"
BASE_MODEL="$ROOT/weights/gr00t/GR00T-N1.6-3B"
DATASET="$ROOT/data/simplerenv_bridge/bridge_image0_first14k_lerobot"
OUTPUT="$ROOT/runs/gr00t_bridge_first14k_smoke_4gpu"
LOG="$ROOT/logs/gr00t_bridge_first14k_smoke_4gpu.log"

mkdir -p "$ROOT/logs" "$ROOT/runs"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate mirror_gr00t_n16

cd "$REPO"

export LD_LIBRARY_PATH="$CONDA_ROOT/envs/mirror_gr00t_n16/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export USE_WANDB=0
export MASTER_PORT="${MASTER_PORT:-29671}"
export TOKENIZERS_PARALLELISM=false

{
    echo "[$(date '+%F %T')] launching GR00T N1.6 Bridge first14k 4-GPU smoke"
    echo "dataset=$DATASET"
    echo "output_dir=$OUTPUT"
    NUM_GPUS=4 \
    MAX_STEPS=1 \
    GLOBAL_BATCH_SIZE=4 \
    SAVE_STEPS=1 \
    DATALOADER_NUM_WORKERS=0 \
    bash examples/finetune.sh \
        --base-model-path "$BASE_MODEL" \
        --dataset-path "$DATASET" \
        --embodiment-tag OXE_WIDOWX \
        --output-dir "$OUTPUT" \
        --experiment-name gr00t_bridge_first14k_smoke_4gpu \
        --state-dropout-prob 0.8
} 2>&1 | tee -a "$LOG"
