#!/bin/bash
# 运行表征分析 - 自动检测已有特征文件
set -e

PYTHON=/root/miniconda3/envs/spatialvla/bin/python
SCRIPT=/root/data/xuyuan1/Codes/analysis/bridge_representation/analyze_representation.py
LOG_DIR=/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/logs
export PYTHONUNBUFFERED=1

cd /root/data/xuyuan1/Codes/analysis/SpatialVLA

mkdir -p $LOG_DIR

echo "========================================"
echo "Representation Analysis"
echo "Time: $(date)"
echo "========================================"

# Chunk4 analysis (all available models)
echo ""
echo "---- chunk4 analysis ----"
$PYTHON $SCRIPT \
  --chunk chunk4 \
  --key_layer 10 \
  --tsne_layers 10 20 26 \
  2>&1 | tee $LOG_DIR/analyze_chunk4.log

# Chunk1 analysis (all available models)
echo ""
echo "---- chunk1 analysis ----"
$PYTHON $SCRIPT \
  --chunk chunk1 \
  --key_layer 10 \
  --tsne_layers 10 \
  2>&1 | tee $LOG_DIR/analyze_chunk1.log

echo ""
echo "========================================"
echo "All analysis done! Time: $(date)"
echo "========================================"
