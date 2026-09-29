#!/bin/bash
# pi0_libero_full_qwen: as2ts + as2tt 双层级对齐 (Qwen3 embedding)
# 2 卡 A800 80G，全局 batch_size=32（每卡 16）
# 用法: ./train_qwen_as2tt.sh <config_name> <exp_name>

export CUDA_VISIBLE_DEVICES=0,1
CONFIG_NAME=${1:-pi0_libero_full_qwen}
EXP_NAME=${2:-as2ts_as2tt}

export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_MODE=disabled

cd /root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi
source ~/miniconda3/bin/activate openpi

CHECKPOINT_DIR="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
mkdir -p "$CHECKPOINT_DIR"
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$CHECKPOINT_DIR/"
cp /root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3_embedding_libero.json "$CHECKPOINT_DIR/" 2>/dev/null
echo "Scripts copied to $CHECKPOINT_DIR"

torchrun --nnodes=1 --nproc_per_node=2 --master_port 56774 scripts/train_pytorch_xy.py $CONFIG_NAME --exp_name=$EXP_NAME --save_interval 5000
