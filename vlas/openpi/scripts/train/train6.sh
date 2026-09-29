#!/bin/bash

# 用法: ./train6.sh <config_name> <exp_name>
# 默认基于 align_new 的setting + at2tt_soft seq 细粒度对齐（DRL-WTI 软聚合）
#                    + 'both' disentangle 双路 (z_shared 主路 + raw 旁路) + 多卡 allgather 负样本 merge
#                    + Qwen3-VL-Embedding-8B 离线 text features
export CUDA_VISIBLE_DEVICES=0
CONFIG_NAME=${1:-pi0_libero_full_qwen_seq}
EXP_NAME=${2:-align_new_seq_wti}

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
