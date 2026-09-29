#!/bin/bash

# Pi0-FAST LIBERO 全量微调（JAX 训练）
# 用法: ./train_pi0fast_libero.sh [exp_name] [extra_args...]
# 例如: ./train_pi0fast_libero.sh v1
#
# 从 pi0_fast_base 预训练权重开始，在 LIBERO 数据上微调 30k 步。
# 参考官方 openpi 的 pi0_fast_libero config。

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
CONFIG_NAME="pi0_fast_libero_train"
EXP_NAME=${1:-default}
PYTHON_BIN=${OPENPI_PYTHON_BIN:-/root/miniconda3/envs/openpi/bin/python}

cd /root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi

# 缓存目录，避免走远端 gs://
export OPENPI_DATA_HOME=/root/data/xuyuan1/dataset/openpi_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OPENPI_DATA_HOME"

# 复制脚本自身到 checkpoint 目录，方便回溯
CHECKPOINT_DIR="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
mkdir -p "$CHECKPOINT_DIR"
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$CHECKPOINT_DIR/"
echo "Script copied to $CHECKPOINT_DIR"

"$PYTHON_BIN" scripts/train.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}" "${@:2}"
