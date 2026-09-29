#!/bin/bash
# OpenPI 全量流式特征提取 (Bridge RLDS)
# 直接从 Bridge RLDS 数据集提取轨迹级特征, 每 500 条保存
#
# 用法:
#   bash run_extract_bridge_streaming.sh pretrained
#   bash run_extract_bridge_streaming.sh raw_ft 16 500 50
#   bash run_extract_bridge_streaming.sh aligned 16 500
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
MAX_TRAJ_PER_TASK=${4:-}  # 默认不限

# 环境
export PYTHONPATH="$ANALYSIS_DIR:$SPATIALVLA_DIR:$OPENPI_SRC:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 日志
LOG_DIR="$SCRIPT_DIR/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bridge_streaming_${MODEL_NAME}.log"

echo "============================================================"
echo " OpenPI Bridge 全量流式特征提取"
echo "   model: $MODEL_NAME"
echo "   batch_size: $BATCH_SIZE"
echo "   save_interval: $SAVE_INTERVAL traj"
echo "   max_traj_per_task: ${MAX_TRAJ_PER_TASK:-unlimited}"
echo "   GPU: $CUDA_VISIBLE_DEVICES"
echo "   时间: $(date)"
echo "   日志: $LOG_FILE"
echo "============================================================"

# 构建命令
CMD="$PYTHON $EXTRACT_SCRIPT \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --save_interval $SAVE_INTERVAL"

if [ -n "$MAX_TRAJ_PER_TASK" ]; then
    CMD="$CMD --max_traj_per_task $MAX_TRAJ_PER_TASK"
fi

$CMD 2>&1 | tee "$LOG_FILE"

echo "============================================================"
echo " 完成: $MODEL_NAME  时间: $(date)"
echo " 日志: $LOG_FILE"
echo "============================================================"
