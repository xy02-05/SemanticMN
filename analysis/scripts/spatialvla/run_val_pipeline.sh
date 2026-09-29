#!/bin/bash
# Val 集完整 pipeline: 构建数据子集 → 提取3个模型特征 → 分析
# 与 train pipeline 完全对称, 输出到 outputs/val/ 下, 不影响 train 结果
#
# 用法: bash run_val_pipeline.sh
# 或:   tmux new -s val_extract 'bash run_val_pipeline.sh'

set -e

cd /root/data/xuyuan1/Codes/analysis/SpatialVLA

PYTHON="/root/miniconda3/envs/spatialvla/bin/python"
BASE="/root/data/xuyuan1/Codes/analysis/bridge_representation"
SUBSET_SCRIPT="$BASE/build_data_subset.py"
EXTRACT_SCRIPT="$BASE/extract_features.py"
ANALYZE_SCRIPT="$BASE/analyze_representation.py"

# Val 输出目录
VAL_DIR="$BASE/outputs/val"
SUBSET_DIR="$VAL_DIR/data_subset"
FEATURE_DIR="$VAL_DIR/features"
FIGURE_DIR="$VAL_DIR/figures"
LOG_DIR="$VAL_DIR/logs"
mkdir -p $LOG_DIR

export PYTHONUNBUFFERED=1
BS=16

echo "========================================"
echo "Val Set Feature Extraction Pipeline"
echo "Time: $(date)"
echo "========================================"

# Step 0: 构建 val 数据子集
FORCE_REBUILD="${FORCE_REBUILD:-0}"
if [ "$FORCE_REBUILD" = "1" ] || [ ! -f "$SUBSET_DIR/manifest.json" ]; then
    echo ""
    echo "[Step 0] Building val data subset..."
    rm -rf "$SUBSET_DIR"
    $PYTHON $SUBSET_SCRIPT --split val 2>&1 | tee $LOG_DIR/build_subset.log
    echo "[Step 0] DONE. Time: $(date)"
else
    echo ""
    echo "[Step 0] Val data subset exists at $SUBSET_DIR, skipping."
    echo "  (set FORCE_REBUILD=1 to rebuild)"
fi

# Step 1: Pretrained
echo ""
echo "[1/3] Extracting pretrained features (val)..."
$PYTHON $EXTRACT_SCRIPT \
    --model_name pretrained --device cuda --batch_size $BS \
    --subset_dir $SUBSET_DIR --feature_dir $FEATURE_DIR \
    2>&1 | tee $LOG_DIR/pretrained.log
echo "[1/3] DONE. Time: $(date)"

# Step 2: Raw FT
echo ""
echo "[2/3] Extracting raw_ft features (val)..."
$PYTHON $EXTRACT_SCRIPT \
    --model_name raw_ft --device cuda --batch_size $BS \
    --subset_dir $SUBSET_DIR --feature_dir $FEATURE_DIR \
    2>&1 | tee $LOG_DIR/raw_ft.log
echo "[2/3] DONE. Time: $(date)"

# Step 3: Cotrain FG
echo ""
echo "[3/3] Extracting cotrain_fg features (val)..."
$PYTHON $EXTRACT_SCRIPT \
    --model_name cotrain_fg --device cuda --batch_size $BS \
    --subset_dir $SUBSET_DIR --feature_dir $FEATURE_DIR \
    2>&1 | tee $LOG_DIR/cotrain_fg.log
echo "[3/3] DONE. Time: $(date)"

# Step 4: 分析
echo ""
echo "[4/4] Running representation analysis (val)..."
# 生成 val hard groups JSON (供 analyze_representation.py 使用)
$PYTHON -c "
import json, sys
sys.path.insert(0, '/root/data/xuyuan1/Codes/analysis')
from bridge_representation.config import VAL_HARD_GROUPS
with open('$VAL_DIR/val_hard_groups.json', 'w') as f:
    json.dump(VAL_HARD_GROUPS, f, indent=2, ensure_ascii=False)
print('Saved val hard groups JSON')
"

$PYTHON $ANALYZE_SCRIPT \
    --chunk chunk4 --key_layer 10 --tsne_layers 10 20 \
    --feature_dir $FEATURE_DIR \
    --output_dir $VAL_DIR \
    --figure_dir $FIGURE_DIR \
    --hard_groups_json $VAL_DIR/val_hard_groups.json \
    2>&1 | tee $LOG_DIR/analyze_chunk4.log

echo ""
echo "========================================"
echo "Val pipeline complete! Time: $(date)"
echo "  Data subset: $SUBSET_DIR"
echo "  Features: $FEATURE_DIR"
echo "  Figures: $FIGURE_DIR"
echo "  Metrics: $VAL_DIR"
echo "========================================"
