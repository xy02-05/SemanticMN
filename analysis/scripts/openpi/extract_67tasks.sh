#!/bin/bash
# OpenPI 67-task 特征提取 (v2: 从 35 tasks/1750 traj → 67 tasks/3350 traj)
#
# 用法: 在 tmux 中运行:
#   tmux new -s opi_feat 'bash /root/data/xuyuan1/Codes/analysis/openpi_representation/run_extract_67tasks.sh'
#
# 依赖: conda env openpi_analysis
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
OPENPI_SRC="$ANALYSIS_DIR/openpi/src"
EXTRACT_SCRIPT="$SCRIPT_DIR/extract_features.py"
PYTHON="/root/miniconda3/envs/openpi_analysis/bin/python"
BATCH_SIZE=${1:-8}
FEAT_DIR="$SCRIPT_DIR/outputs/features"
LOG_DIR="$SCRIPT_DIR/outputs/logs"

export PYTHONPATH="$ANALYSIS_DIR:$OPENPI_SRC:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$LOG_DIR" "$FEAT_DIR"

echo "============================================================"
echo " OpenPI 67-task Feature Extraction (v2)"
echo " 从 35 tasks/1750 traj → 67 tasks/3350 traj"
echo " Time: $(date)"
echo " GPU: $CUDA_VISIBLE_DEVICES"
echo " Batch size: $BATCH_SIZE"
echo "============================================================"

# Step 1: 备份旧的 35-task 特征
echo ""
echo ">>> 备份旧的 35-task 特征..."
for f in "$FEAT_DIR"/*_chunk*_features.npz; do
    if [ -f "$f" ]; then
        backup="${f%.npz}_35tasks_backup.npz"
        if [ ! -f "$backup" ]; then
            cp "$f" "$backup"
            echo "  备份: $(basename $f) → $(basename $backup)"
        else
            echo "  已存在备份: $(basename $backup), 跳过"
        fi
    fi
done

# Step 2: 逐模型提取
for MODEL in pretrained raw_ft aligned; do
    echo ""
    echo "============================================================"
    echo ">>> Extracting: $MODEL (67 tasks, ~3350 trajectories)"
    echo "  Time: $(date)"
    echo "============================================================"
    LOG_FILE="$LOG_DIR/extract_${MODEL}_67tasks.log"
    $PYTHON "$EXTRACT_SCRIPT" --model_name "$MODEL" --batch_size "$BATCH_SIZE" 2>&1 | tee "$LOG_FILE"
    echo ">>> Done: $MODEL (log: $LOG_FILE)"
    echo "  Time: $(date)"
done

echo ""
echo "============================================================"
echo " 全部完成! Time: $(date)"
echo " 输出: $FEAT_DIR"
echo "============================================================"
ls -lh "$FEAT_DIR"/*_features.npz 2>/dev/null
