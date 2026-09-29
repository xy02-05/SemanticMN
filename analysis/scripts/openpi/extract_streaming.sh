#!/bin/bash
# OpenPI 全量流式特征提取
# 遍历整个 Bridge RLDS 数据集 (所有 task), 每 500 条轨迹保存
#
# 用法:
#   tmux new -d -s opi_stream 'bash .../run_extract_streaming.sh'
#   或直接: bash .../run_extract_streaming.sh pretrained
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
SPATIALVLA_DIR="$ANALYSIS_DIR/SpatialVLA"
OPENPI_SRC="$ANALYSIS_DIR/openpi/src"
EXTRACT_SCRIPT="$SCRIPT_DIR/extract_features_streaming.py"
PYTHON="/root/miniconda3/envs/openpi/bin/python"

# 参数
MODEL_NAME=${1:-pretrained}
BATCH_SIZE=${2:-16}
SAVE_INTERVAL=${3:-500}
MAX_TRAJ_PER_TASK=${4:-50}

# 环境
export PYTHONPATH="$ANALYSIS_DIR:$SPATIALVLA_DIR:$OPENPI_SRC:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 日志
LOG_DIR="$SCRIPT_DIR/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/streaming_${MODEL_NAME}.log"

echo "============================================================"
echo " OpenPI 全量流式特征提取"
echo "   model: $MODEL_NAME"
echo "   batch_size: $BATCH_SIZE"
echo "   save_interval: $SAVE_INTERVAL traj"
echo "   max_traj_per_task: $MAX_TRAJ_PER_TASK"
echo "   GPU: $CUDA_VISIBLE_DEVICES"
echo "   时间: $(date)"
echo "   日志: $LOG_FILE"
echo "============================================================"

cd "$SPATIALVLA_DIR"

$PYTHON "$EXTRACT_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --save_interval "$SAVE_INTERVAL" \
    --max_traj_per_task "$MAX_TRAJ_PER_TASK" \
    2>&1 | tee "$LOG_FILE"

echo "============================================================"
echo " 完成: $MODEL_NAME  时间: $(date)"
echo " 日志: $LOG_FILE"
echo "============================================================"
