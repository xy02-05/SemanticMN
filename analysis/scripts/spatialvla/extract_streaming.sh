#!/bin/bash
# SpatialVLA 全量流式特征提取 (pretrained + raw_ft)
# 用法: tmux new -s svla_stream 'bash scripts/spatialvla/extract_streaming.sh'
set -e

cd /root/data/xuyuan1/Codes/analysis/SpatialVLA

PYTHON="/root/miniconda3/envs/spatialvla/bin/python"
SCRIPT="/root/data/xuyuan1/Codes/analysis/bridge_representation/extract_features_streaming.py"
LOG_DIR="/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/streaming/logs"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1

# 提取模型名称 (可通过参数覆盖: bash extract_streaming.sh raw_ft)
MODEL_NAME="${1:-pretrained}"
BS="${2:-12}"

echo "============================================================"
echo " SpatialVLA 全量流式特征提取: $MODEL_NAME"
echo " batch_size=$BS, save_interval=500"
echo " 时间: $(date)"
echo "============================================================"

$PYTHON "$SCRIPT" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BS" \
    --save_interval 500 \
    2>&1 | tee "$LOG_DIR/${MODEL_NAME}_streaming.log"

echo ""
echo "============================================================"
echo " 完成: $MODEL_NAME | 时间: $(date)"
echo "============================================================"
