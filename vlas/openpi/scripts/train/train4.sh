#!/bin/bash

# 用法: ./train4.sh <config_name> <exp_name>
# 例如: ./train4.sh pi0_libero_align_lora libero_align_lora
# DSN 优化版（2026-05-04）：
#   1) disentangle.py::MSE/SIMSE 内部加 LayerNorm，复刻 DSN 原文 [-1,1] input 条件，
#      让 alpha=0.01 重新合适，diff_loss 真起约束
#   2) hidden_dim 1024 → 256，复刻 DSN code_size 瓶颈（4:1）
export CUDA_VISIBLE_DEVICES=0,1
CONFIG_NAME=${1:-pi0_libero_full_qwen}
EXP_NAME=${2:-new_DSN_new}

export HF_ENDPOINT=https://hf-mirror.com
# export XDG_CACHE_HOME=/mnt/nvmepool/xuyuan/.cache

cd /root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi
source ~/miniconda3/bin/activate openpi
# 复制脚本自身到指定目录
CHECKPOINT_DIR="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
mkdir -p "$CHECKPOINT_DIR"
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$CHECKPOINT_DIR/"
echo "Script copied to $CHECKPOINT_DIR"

torchrun --nnodes=1 --nproc_per_node=2 --master_port 56773 scripts/train_pytorch_xy.py $CONFIG_NAME --exp_name=$EXP_NAME --save_interval 5000
