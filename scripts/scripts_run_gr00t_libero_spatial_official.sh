#!/usr/bin/env bash

set -euo pipefail

ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
CONDA_ROOT=/mnt/bn/2d-videos/xy/tools/miniconda3
REPO="$ROOT/Isaac-GR00T"
BASE_MODEL="$ROOT/weights/gr00t/GR00T-N1.6-3B"
DATASET="$ROOT/data/libero/libero_spatial_no_noops_1.0.0_lerobot"
OUTPUT="$ROOT/runs/gr00t_libero_spatial_official"

# 正式训练需要 8 张卡；这里只停止 GPU 压测，不在训练退出后自动恢复。
if tmux has-session -t gpu-utilization 2>/dev/null; then
    tmux kill-session -t gpu-utilization
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate mirror_gr00t_n16

cd "$REPO"

export LD_LIBRARY_PATH="$CONDA_ROOT/envs/mirror_gr00t_n16/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_WANDB=0
export MASTER_PORT=29641

# 对齐 examples/LIBERO/README.md 的 LIBERO Spatial 复现参数。
NUM_GPUS=8 \
MAX_STEPS=20000 \
GLOBAL_BATCH_SIZE=640 \
SAVE_STEPS=1000 \
DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET" \
    --embodiment-tag LIBERO_PANDA \
    --output-dir "$OUTPUT"
