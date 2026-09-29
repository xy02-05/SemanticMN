#!/bin/bash

# 用法: ./train_align.sh <config_name> <exp_name>
# 例如: ./train_align.sh robotwin_rlds_hard_48x50_align align_0.25
export CUDA_VISIBLE_DEVICES=0,1
CONFIG_NAME=${1:-bridgev2_rlds_train}
EXP_NAME=${2:-raw}

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

torchrun --nnodes=1 --nproc_per_node=2 --master_port 56772 scripts/train_pytorch_xy.py $CONFIG_NAME --exp_name=$EXP_NAME --save_interval 5000

