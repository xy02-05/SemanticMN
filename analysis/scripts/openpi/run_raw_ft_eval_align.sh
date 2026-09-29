#!/bin/bash
# OpenPI raw_ft: 评估 + 对齐训练
# 1. 评估 CKA/SVCCA/PWCCA/KNN 等全部指标
# 2. 训练各层线性对齐
set -e

ANALYSIS_DIR="/root/data/xuyuan1/Codes/analysis"
PYTHON="/root/miniconda3/envs/openpi/bin/python"
export PYTHONPATH="$ANALYSIS_DIR:$ANALYSIS_DIR/openpi/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

FEAT_PATH="$ANALYSIS_DIR/openpi_representation/outputs/streaming/features/raw_ft_streaming_features.npz"
LABEL="openpi_raw_ft_streaming"
LOG_DIR="$ANALYSIS_DIR/doc"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " OpenPI raw_ft: 评估 + 对齐训练"
echo " 特征: $FEAT_PATH"
echo " 时间: $(date)"
echo "============================================================"

# ---- Step 1: 评估 ----
echo ""
echo ">>> [Step 1] 评估指标 (CKA/SVCCA/PWCCA/KNN/Cluster)..."
$PYTHON "$ANALYSIS_DIR/evaluation/evaluate_features.py" \
    --feature_path "$FEAT_PATH" \
    --label "$LABEL" \
    --metrics cka svcca knn cluster \
    --text_types ego \
    --strategies all task_mean \
    --output_dir "$ANALYSIS_DIR/evaluation/outputs" \
    2>&1
echo ">>> [Step 1] 评估完成: $(date)"

# ---- Step 2: 对齐训练 ----
echo ""
echo ">>> [Step 2] 对齐训练 (all layers)..."
$PYTHON "$ANALYSIS_DIR/alignment/train_alignment.py" \
    --feature_path "$FEAT_PATH" \
    --label "$LABEL" \
    --layer all \
    --epochs 100 \
    --lr 1e-3 \
    --batch_size 256 \
    --temperature 0.07 \
    --output_dir "$ANALYSIS_DIR/alignment/outputs" \
    2>&1
echo ">>> [Step 2] 对齐训练完成: $(date)"

echo ""
echo "============================================================"
echo " 全部完成: $(date)"
echo " 评估结果: $ANALYSIS_DIR/evaluation/outputs/"
echo " 对齐模型: $ANALYSIS_DIR/alignment/outputs/$LABEL/"
echo "============================================================"
