#!/bin/bash
# 完整 pipeline: 先建数据子集，再按顺序提取3个模型特征
# 用法: tmux new -s extract 'bash run_extract_all.sh'

set -e

cd /root/data/xuyuan1/Codes/analysis/SpatialVLA

PYTHON="/root/miniconda3/envs/spatialvla/bin/python"
SUBSET_SCRIPT="/root/data/xuyuan1/Codes/analysis/bridge_representation/build_data_subset.py"
EXTRACT_SCRIPT="/root/data/xuyuan1/Codes/analysis/bridge_representation/extract_features.py"
LOG_DIR="/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/logs"
mkdir -p $LOG_DIR

export PYTHONUNBUFFERED=1

BS=16

echo "========================================"
echo "Starting feature extraction pipeline v2"
echo "Time: $(date)"
echo "========================================"

# Step 0: 建数据子集
# 使用 --force 参数强制重建，否则复用已有子集
SUBSET_DIR="/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/data_subset"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
if [ "$FORCE_REBUILD" = "1" ] || [ ! -f "$SUBSET_DIR/manifest.json" ]; then
    echo ""
    echo "[Step 0] Building data subset (67 tasks × 50 trajs)..."
    rm -rf "$SUBSET_DIR"
    $PYTHON $SUBSET_SCRIPT 2>&1 | tee $LOG_DIR/build_subset.log
    echo "[Step 0] DONE. Time: $(date)"
else
    echo ""
    echo "[Step 0] Data subset already exists at $SUBSET_DIR, skipping build."
    N_TRAJS=$(python3 -c "import json; m=json.load(open('$SUBSET_DIR/manifest.json')); print(m['total_trajectories'])")
    N_FRAMES=$(python3 -c "import json; m=json.load(open('$SUBSET_DIR/manifest.json')); print(m['total_frames'])")
    echo "  Trajectories: $N_TRAJS, Frames: $N_FRAMES"
    echo "  (set FORCE_REBUILD=1 to rebuild)"
fi

# 清理旧特征文件
FEATURE_DIR="/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/features"
echo "[Clean] Removing old feature files..."
rm -f $FEATURE_DIR/*_chunk*_features.npz $FEATURE_DIR/*_chunk*_checkpoint.npz

# Step 1: Pretrained
echo ""
echo "[1/3] Extracting pretrained features..."
$PYTHON $EXTRACT_SCRIPT --model_name pretrained --device cuda --batch_size $BS 2>&1 | tee $LOG_DIR/pretrained.log
echo "[1/3] DONE. Time: $(date)"

# Step 2: Raw FT
echo ""
echo "[2/3] Extracting raw_ft features..."
$PYTHON $EXTRACT_SCRIPT --model_name raw_ft --device cuda --batch_size $BS 2>&1 | tee $LOG_DIR/raw_ft.log
echo "[2/3] DONE. Time: $(date)"

# Step 3: Cotrain FG
echo ""
echo "[3/3] Extracting cotrain_fg features..."
$PYTHON $EXTRACT_SCRIPT --model_name cotrain_fg --device cuda --batch_size $BS 2>&1 | tee $LOG_DIR/cotrain_fg.log
echo "[3/3] DONE. Time: $(date)"

echo ""
echo "========================================"
echo "All 3 models extracted!"
echo "Output: /root/data/xuyuan1/Codes/analysis/bridge_representation/outputs/features/"
echo "========================================"
