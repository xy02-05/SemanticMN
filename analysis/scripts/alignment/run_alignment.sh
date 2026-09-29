#!/bin/bash
# 通用对齐训练脚本
# 用法:
#   bash run_alignment.sh FEATURE_PATH [LABEL] [EPOCHS] [LR]
# 示例:
#   bash run_alignment.sh /path/to/raw_ft_streaming_features.npz openpi_raw_ft 100 1e-3
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/root/miniconda3/envs/openpi/bin/python"

FEATURE_PATH=${1:?请指定特征文件路径}
LABEL=${2:-}
EPOCHS=${3:-100}
LR=${4:-1e-3}

export PYTHONPATH="$ANALYSIS_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CMD="$PYTHON $SCRIPT_DIR/train_alignment.py \
    --feature_path $FEATURE_PATH \
    --epochs $EPOCHS \
    --lr $LR"

if [ -n "$LABEL" ]; then
    CMD="$CMD --label $LABEL"
fi

echo "============================================================"
echo " 对齐训练"
echo "   特征: $FEATURE_PATH"
echo "   标签: ${LABEL:-auto}"
echo "   epochs=$EPOCHS, lr=$LR"
echo "   GPU: ${CUDA_VISIBLE_DEVICES}"
echo "   时间: $(date)"
echo "============================================================"

$CMD

echo "============================================================"
echo " 完成: $(date)"
echo "============================================================"
