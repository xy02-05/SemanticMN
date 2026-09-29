#!/bin/bash
# SpatialVLA 多卡并行流式特征提取
#
# 原理: 使用 TFDS subsplit 在 TFRecord 文件级别分片
#       每个 GPU 只读 1/N 的文件，零 I/O 浪费
#       TFDS subsplit 基于 example 绝对位置，确定性，各 shard 不重叠不遗漏
#
# 用法:
#   bash scripts/spatialvla/run_streaming_parallel.sh raw_ft 4
#   bash scripts/spatialvla/run_streaming_parallel.sh cotrain_fg 4
#   bash scripts/spatialvla/run_streaming_parallel.sh raw_ft 4 merge  # 仅合并
set -e

MODEL_NAME=${1:?用法: $0 <model_name> <num_gpus> [merge]}
NUM_GPUS=${2:-4}
MODE=${3:-extract}

ANALYSIS_DIR="/root/data/xuyuan1/Codes/analysis"
SCRIPT="$ANALYSIS_DIR/bridge_representation/extract_features_streaming.py"
SPATIALVLA_DIR="$ANALYSIS_DIR/SpatialVLA"
CONDA_ENV="spatialvla"
PYTHON="/root/miniconda3/envs/$CONDA_ENV/bin/python"

cd "$SPATIALVLA_DIR"

if [ "$MODE" = "merge" ]; then
    echo "=========================================="
    echo " 合并 $NUM_GPUS 个分片: $MODEL_NAME"
    echo "=========================================="
    $PYTHON "$SCRIPT" --model_name "$MODEL_NAME" --num_shards "$NUM_GPUS" --merge_only
    exit 0
fi

echo "=========================================="
echo " SpatialVLA 多卡并行特征提取"
echo " 模型: $MODEL_NAME"
echo " GPU数: $NUM_GPUS"
echo " 时间: $(date)"
echo "=========================================="

# 启动 N 个 shard 进程，每个在独立 tmux session 中
for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SESSION="svla_${MODEL_NAME}_s${SHARD_ID}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true

    CMD="cd $SPATIALVLA_DIR && CUDA_VISIBLE_DEVICES=$SHARD_ID $PYTHON $SCRIPT \
        --model_name $MODEL_NAME \
        --device cuda \
        --batch_size 12 \
        --save_interval 500 \
        --num_shards $NUM_GPUS \
        --shard_id $SHARD_ID \
        2>&1 | tee $ANALYSIS_DIR/bridge_representation/outputs/streaming/shard${SHARD_ID}_${MODEL_NAME}.log"

    tmux new-session -d -s "$SESSION" "$CMD"
    echo "  启动 shard $SHARD_ID → GPU $SHARD_ID (tmux: $SESSION)"
done

echo ""
echo "所有 $NUM_GPUS 个分片已启动!"
echo "监控: tmux ls"
echo "查看某个分片: tmux a -t svla_${MODEL_NAME}_s0"
echo "合并: bash $0 $MODEL_NAME $NUM_GPUS merge"
