#!/bin/bash
# SpatialVLA Action ↔ EgoHOD Text 线性对齐: 完整 pipeline
#   Step 1: 全量特征提取 — 流式 RLDS → VLA forward → 轨迹级特征 (无中间图像存储)
#   Step 2: 线性对齐训练 — InfoNCE 对比学习, 按 task 80/20 划分
#
# 用法: bash run_alignment.sh
# 预计: Step1 ~1-2h (GPU), Step2 ~1min (CPU/GPU)
set -e

cd /root/data/xuyuan1/Codes/analysis/SpatialVLA

PYTHON=/root/miniconda3/envs/spatialvla/bin/python
EXTRACT=/root/data/xuyuan1/Codes/analysis/bridge_representation/extract_alignment_features.py
TRAIN=/root/data/xuyuan1/Codes/analysis/alignment/train_alignment.py
LOG_DIR=/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/alignment/logs
mkdir -p $LOG_DIR
export PYTHONUNBUFFERED=1

MODEL=${MODEL:-cotrain_fg}
LAYER=${LAYER:-all}       # "all" → 所有层各自独立训练; "10" → 仅 layer 10
CHUNK=${CHUNK:-chunk4}
# 完整数据集: 不限制 task 和每 task 轨迹数 (Bridge 共 21938 tasks, ~53k episodes)
MAX_TASKS=${MAX_TASKS:-99999}
MAX_TRAJ=${MAX_TRAJ:-99999}

echo "========================================"
echo "SpatialVLA ↔ EgoHOD Alignment Pipeline"
echo "  model=$MODEL, layer=$LAYER, chunk=$CHUNK"
echo "  max_tasks=$MAX_TASKS, max_traj_per_task=$MAX_TRAJ"
echo "  Time: $(date)"
echo "========================================"

# Step 1: 全量特征提取 (完整 Bridge 数据集, 不限制)
FEAT_FILE=/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/alignment/features/align_${MODEL}_${CHUNK}_features.npz
if [ -f "$FEAT_FILE" ]; then
    echo ""
    echo "[Step 1] 特征文件已存在: $FEAT_FILE, 跳过提取."
    echo "  (删除该文件可强制重新提取)"
else
    echo ""
    echo "[Step 1] Extracting alignment features (full dataset)..."
    $PYTHON $EXTRACT \
        --model_name $MODEL \
        --batch_size 12 \
        --max_tasks $MAX_TASKS \
        --max_traj_per_task $MAX_TRAJ \
        2>&1 | tee $LOG_DIR/extract_${MODEL}.log
    echo "[Step 1] Done. Time: $(date)"
fi

# Step 2: 线性对齐训练 (使用统一的 alignment 脚本)
echo ""
echo "[Step 2] Training linear alignment (layer $LAYER)..."
export PYTHONPATH="/root/data/xuyuan1/Codes/analysis:$PYTHONPATH"
$PYTHON $TRAIN \
    --feature_path $FEAT_FILE \
    --label spatialvla_${MODEL}_${CHUNK} \
    --layer $LAYER \
    --train_ratio 0.8 \
    --epochs 150 \
    --lr 1e-3 \
    --batch_size 256 \
    --temperature 0.07 \
    2>&1 | tee $LOG_DIR/train_${MODEL}_${CHUNK}_layer${LAYER}.log
echo "[Step 2] Done. Time: $(date)"

echo ""
echo "========================================"
echo "Pipeline complete! Time: $(date)"
echo "Output: /root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/alignment/"
echo "========================================"
