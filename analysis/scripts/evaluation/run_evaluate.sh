#!/bin/bash
# 统一特征评估脚本
# 用法:
#   bash run_evaluate.sh FEATURE_PATH [LABEL] [OUTPUT_DIR]
# 示例:
#   bash run_evaluate.sh /path/to/raw_ft_streaming_features.npz openpi_raw_ft
#   bash run_evaluate.sh /path/to/features.npz
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/root/miniconda3/envs/openpi/bin/python"

FEATURE_PATH=${1:?请指定特征文件路径}
LABEL=${2:-}
OUTPUT_DIR=${3:-$SCRIPT_DIR/outputs}

export PYTHONPATH="$ANALYSIS_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1

CMD="$PYTHON $SCRIPT_DIR/evaluate_features.py \
    --feature_path $FEATURE_PATH \
    --output_dir $OUTPUT_DIR"

if [ -n "$LABEL" ]; then
    CMD="$CMD --label $LABEL"
fi

echo "============================================================"
echo " 统一特征评估"
echo "   特征: $FEATURE_PATH"
echo "   标签: ${LABEL:-auto}"
echo "   输出: $OUTPUT_DIR"
echo "   时间: $(date)"
echo "============================================================"

$CMD

echo "============================================================"
echo " 完成: $(date)"
echo "============================================================"
