#!/bin/bash

# 用法: ./train3.sh <config_name> <exp_name>
# 例如: ./train3.sh pi0_libero_lora libero_lora
export CUDA_VISIBLE_DEVICES=0
CONFIG_NAME=${1:-pi0_libero_lora}
EXP_NAME=${2:-test}

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

torchrun --nnodes=1 --nproc_per_node=1 --master_port 56772 scripts/train_pytorch_xy.py $CONFIG_NAME --exp_name=$EXP_NAME --save_interval 5000
