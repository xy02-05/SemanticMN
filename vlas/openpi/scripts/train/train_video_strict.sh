#!/bin/bash
# Video-strict alignment 训练脚本（双卡）
# - 去掉 ek100，仅保留 vitra/bridge/droid/fractal/libero
# - sigmoid_weighted strong=0.9, weak=0.9, neg=0.75, bias=0（hard binary + ignore 中间区）
# - DSN cosine_hinge
# - fg_alignment_model._forward_as2vs 不 detach（已在 fg_alignment_model.py 修改）

set -e

CONFIG_NAME=${1:-pi0_libero_full_qwen_video_strict}
EXP_NAME=${2:-video_strict_no_ek100}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HF_ENDPOINT=https://hf-mirror.com

cd /root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi
source ~/miniconda3/bin/activate openpi

CHECKPOINT_DIR="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
mkdir -p "$CHECKPOINT_DIR"
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$CHECKPOINT_DIR/"
echo "Script copied to $CHECKPOINT_DIR"

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_port 56773 \
  scripts/train_pytorch_xy.py "$CONFIG_NAME" \
  --exp_name="$EXP_NAME" \
  --save_interval 5000 \
  2>&1 | tee -a "$CHECKPOINT_DIR/training.log"
