#!/bin/bash

# 用法: ./train3_jax.sh <config_name> <exp_name>
# 例如: ./train3_jax.sh pi0_libero libero_jax
# 说明:
# 1. 这个脚本走官方 JAX 训练入口 scripts/train.py。
# 2. 配置中的预训练权重已改为本地路径 /root/data/xuyuan1/dataset/pi0_base_official/params。
# 3. 目标是尽量和官方 openpi 的 LIBERO JAX 训练方式对齐。

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
CONFIG_NAME=${1:-pi0_libero}
EXP_NAME=${2:-raw_jax}
PYTHON_BIN=${OPENPI_PYTHON_BIN:-/root/miniconda3/envs/openpi/bin/python}

cd /root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi

# JAX / openpi 的下载缓存固定到本地 dataset 目录，避免再走远端 gs://。
export OPENPI_DATA_HOME=/root/data/xuyuan1/dataset/openpi_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OPENPI_DATA_HOME"

# 复制脚本自身到 checkpoint 目录，方便回溯。
CHECKPOINT_DIR="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
mkdir -p "$CHECKPOINT_DIR"
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$CHECKPOINT_DIR/"
echo "Script copied to $CHECKPOINT_DIR"

"$PYTHON_BIN" scripts/train.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}" "${@:3}"
