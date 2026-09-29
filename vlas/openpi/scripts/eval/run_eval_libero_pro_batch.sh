#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LIBERO-PRO 批量评测脚本
# 支持多个 checkpoint 依次在 LIBERO-PRO 上评测。
# 每个 checkpoint 启动 4 个 tmux session（4 个 base suite），等待全部完成后再跑下一个。
#
# 用法:
#   # demo 模式（1 task × 1 episode，快速验证链路）
#   bash scripts/eval/run_eval_libero_pro_batch.sh
#
#   # 正式评测（10 task × 50 episode）
#   NUM_TRIALS_PER_TASK=50 MAX_TASKS="" bash scripts/eval/run_eval_libero_pro_batch.sh
#
#   # 指定 suffix
#   PRO_SUFFIX=object bash scripts/eval/run_eval_libero_pro_batch.sh
# ============================================================================

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
INNER_SCRIPT="${OPENPI_ROOT}/scripts/eval/run_eval_libero_pro_4tmux.sh"

# --- 待评测的 checkpoint 列表（按顺序执行）---
CKPT_LIST=(
  "/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/pi0_libero_full/raw/30000"
  "/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/pi0_libero_align_lora_qwen/lora_sigmoid_st01_lq/30000"
)

# --- 评测参数（可通过环境变量覆盖）---
PRO_SUFFIX="${PRO_SUFFIX:-lan}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
MAX_TASKS="${MAX_TASKS:-1}"
SEED="${SEED:-1000}"
GPU_LAYOUT="${GPU_LAYOUT:-0 0 1 1}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

# 检查必备文件
if [ ! -f "${INNER_SCRIPT}" ]; then
  echo "错误: 找不到内部脚本 ${INNER_SCRIPT}" >&2
  exit 1
fi

# 生成唯一的运行时间标签
BATCH_TAG="$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "LIBERO-PRO 批量评测"
echo "  PRO_SUFFIX:          ${PRO_SUFFIX}"
echo "  NUM_TRIALS_PER_TASK: ${NUM_TRIALS_PER_TASK}"
echo "  MAX_TASKS:           ${MAX_TASKS:-全量}"
echo "  SEED:                ${SEED}"
echo "  GPU_LAYOUT:          ${GPU_LAYOUT}"
echo "  BATCH_TAG:           ${BATCH_TAG}"
echo "  Checkpoint 数量:     ${#CKPT_LIST[@]}"
echo "========================================"

# 等待指定前缀的 tmux session 全部退出
wait_tmux_sessions() {
  local prefix="$1"
  while true; do
    # 统计还存活的 session 数量
    local alive
    alive=$(tmux list-sessions 2>/dev/null | grep -c "^${prefix}" || true)
    if [ "${alive}" -eq 0 ]; then
      return 0
    fi
    echo "[$(date +%H:%M:%S)] 等待中... 还有 ${alive} 个 session 在运行 (prefix=${prefix})"
    sleep "${POLL_INTERVAL}"
  done
}

ckpt_idx=0
for CKPT_DIR in "${CKPT_LIST[@]}"; do
  ckpt_idx=$((ckpt_idx + 1))

  if [ ! -d "${CKPT_DIR}" ]; then
    echo "警告: checkpoint 不存在，跳过: ${CKPT_DIR}" >&2
    continue
  fi

  # 为每个 checkpoint 生成独立的 session 前缀和 RUN_TAG
  RUN_TAG="${BATCH_TAG}_ckpt${ckpt_idx}"
  SESSION_PREFIX="lpro${ckpt_idx}"

  echo ""
  echo "========================================"
  echo "[${ckpt_idx}/${#CKPT_LIST[@]}] 启动评测"
  echo "  CKPT: ${CKPT_DIR}"
  echo "  RUN_TAG: ${RUN_TAG}"
  echo "  SESSION_PREFIX: ${SESSION_PREFIX}"
  echo "========================================"

  # 调用已有的 4tmux 脚本
  CKPT_DIR="${CKPT_DIR}" \
  PRO_SUFFIX="${PRO_SUFFIX}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
  MAX_TASKS="${MAX_TASKS}" \
  SEED="${SEED}" \
  GPU_LAYOUT="${GPU_LAYOUT}" \
  RUN_TAG="${RUN_TAG}" \
  SESSION_PREFIX="${SESSION_PREFIX}" \
  CONFIG_NAME="auto" \
  bash "${INNER_SCRIPT}"

  echo "[${ckpt_idx}] 4 个 tmux session 已启动，等待评测完成..."
  wait_tmux_sessions "${SESSION_PREFIX}"
  echo "[${ckpt_idx}] 评测完成！"
done

echo ""
echo "========================================"
echo "全部 ${#CKPT_LIST[@]} 个 checkpoint 评测完成"
echo "结果目录: ${OPENPI_ROOT}/results/libero_pro/"
echo "========================================"
