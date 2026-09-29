#!/bin/bash
# OpenPI 三模型特征提取脚本
# 用法: bash run_extract_all.sh [batch_size]
#
# 需要先 conda activate openpi_analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
OPENPI_SRC="$ANALYSIS_DIR/openpi/src"
EXTRACT_SCRIPT="$SCRIPT_DIR/extract_features.py"
PYTHON="/root/miniconda3/envs/openpi_analysis/bin/python"
BATCH_SIZE=${1:-8}

export PYTHONPATH="$ANALYSIS_DIR:$OPENPI_SRC:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# 输出日志目录
LOG_DIR="$SCRIPT_DIR/outputs/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "OpenPI Feature Extraction"
echo "  Batch size: $BATCH_SIZE"
echo "  Log dir: $LOG_DIR"
echo "========================================"

for MODEL in pretrained raw_ft aligned; do
    echo ""
    echo ">>> Extracting: $MODEL"
    LOG_FILE="$LOG_DIR/extract_${MODEL}.log"
    $PYTHON "$EXTRACT_SCRIPT" --model_name "$MODEL" --batch_size "$BATCH_SIZE" 2>&1 | tee "$LOG_FILE"
    echo ">>> Done: $MODEL (log: $LOG_FILE)"
done

echo ""
echo "========================================"
echo "All done!"
echo "========================================"
