#!/bin/bash
# 等待 align/feat_collect 完成后, 自动启动 OpenPI 全量流式特征提取
# 遍历整个 Bridge 数据集, 每 500 条轨迹保存 checkpoint
#
# 用法: tmux new -d -s opi_feat 'bash .../run_extract_after_align.sh'
set -e

ALIGN_PID=54723  # align 进程 PID
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/outputs/logs/streaming_all_models.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "============================================================" | tee "$LOG_FILE"
echo " OpenPI 全量流式提取: 等待 GPU 空闲" | tee -a "$LOG_FILE"
echo " align PID=$ALIGN_PID" | tee -a "$LOG_FILE"
echo " 时间: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 等 align 进程结束
while kill -0 $ALIGN_PID 2>/dev/null; do
    echo "$(date '+%H:%M:%S') align 运行中, 等待 60s..." | tee -a "$LOG_FILE"
    sleep 60
done
echo "✓ align 结束" | tee -a "$LOG_FILE"
sleep 30

# 等 feat_collect 结束 (如有)
while pgrep -f "feature_extracting_evaluator" > /dev/null 2>&1; do
    echo "$(date '+%H:%M:%S') feat_collect 运行中, 等待 60s..." | tee -a "$LOG_FILE"
    sleep 60
done
echo "✓ GPU 空闲, 开始 OpenPI 全量提取" | tee -a "$LOG_FILE"
sleep 10

# 逐模型: pretrained → raw_ft → aligned
for MODEL in pretrained raw_ft aligned; do
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo " 开始: $MODEL  时间: $(date)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    bash "$SCRIPT_DIR/run_extract_streaming.sh" "$MODEL" 16 500 50 2>&1 | tee -a "$LOG_FILE"

    echo " 完成: $MODEL  时间: $(date)" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 全部 3 个模型提取完成!  时间: $(date)" | tee -a "$LOG_FILE"
echo " 日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
