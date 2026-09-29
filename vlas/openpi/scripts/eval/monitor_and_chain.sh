#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
CKPT_DIR="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/pi0_libero_full_qwen/new_DSN_new_egohod/30000"
CONFIG_NAME="pi0_libero_full_qwen_video"
RESULT_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/results/libero/pi0_libero_full_qwen/new_DSN_new_egohod/30000_step30000_seed42_ep20/20260528_024557"
RUN_TAG="20260528_024557"
SP="libero4"

LOG="${OPENPI_ROOT}/checkpoints/bridgev2_rlds_train_egohod/new_DSN_egohod/monitor.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
gpu() { nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "$LOG"; }

wait_sessions() {
  while true; do
    local alive=0
    for s in "$@"; do tmux has-session -t "$s" 2>/dev/null && alive=$((alive+1)); done
    [ "$alive" -eq 0 ] && break
    log "waiting ($alive sessions alive)"; gpu; sleep 300
  done
}

launch_suite() {
  local suite=$1 port=$2 gpu_id=$3
  local rdir="${RESULT_ROOT}/${suite}"
  mkdir -p "${rdir}/videos"
  tmux new-session -d -s "${SP}_${RUN_TAG}_${suite#libero_}" \
    "cd '${OPENPI_ROOT}' && \
     CONFIG_NAME='${CONFIG_NAME}' CKPT_DIR='${CKPT_DIR}' SUITE='${suite}' \
     RESULT_DIR='${rdir}' PORT='${port}' GPU_ID='${gpu_id}' \
     NUM_TRIALS_PER_TASK='20' RESIZE_SIZE='224' REPLAN_STEPS='5' \
     NUM_STEPS_WAIT='10' SEED='1000' POLICY_SEED='1000' \
     bash '${OPENPI_ROOT}/scripts/eval/run_eval_libero_single_suite.sh'"
}

# === Phase 1: spatial(GPU0) + object(GPU1) 已在运行，等完成 ===
log "=== Phase 1: spatial + object ==="
wait_sessions "${SP}_${RUN_TAG}_spatial" "${SP}_${RUN_TAG}_object"

for s in libero_spatial libero_object; do
  log "$s: $(grep 'Total success' "${RESULT_ROOT}/${s}/client.log" 2>/dev/null || echo 'not done')"
done

# === Phase 2: goal(GPU0) + 10(GPU1) ===
log "=== Phase 2: goal + 10 ==="; gpu
launch_suite libero_goal 24002 0
launch_suite libero_10   24003 1
wait_sessions "${SP}_${RUN_TAG}_goal" "${SP}_${RUN_TAG}_10"

for s in libero_goal libero_10; do
  log "$s: $(grep 'Total success' "${RESULT_ROOT}/${s}/client.log" 2>/dev/null || echo 'not done')"
done
log "=== LIBERO standard complete ==="

# === Phase 3: LIBERO-Pro, 分两组 ===
log "=== Phase 3: LIBERO-Pro ==="; gpu
cd "${OPENPI_ROOT}"

# 组1: spatial_pro(GPU0) + object_pro(GPU1)
GPU_LAYOUT="0 1" \
CKPT_DIR="${CKPT_DIR}" CONFIG_NAME="auto" \
PRO_SUFFIXES="task swap env object lan" \
bash -c '
  source scripts/eval/run_eval_libero_pro_4tmux.sh 2>/dev/null || true
' 2>&1 | head -5 | tee -a "$LOG"

# 上面脚本会启动4个session，杀掉goal和10
sleep 5
PRO_SESSIONS=$(tmux ls 2>/dev/null | grep "^pro5_" | cut -d: -f1)
for s in $PRO_SESSIONS; do
  case "$s" in *_goal|*_10) tmux kill-session -t "$s" 2>/dev/null; log "killed $s";; esac
done

# 等 pro spatial + object 完成
PRO_S1=$(echo "$PRO_SESSIONS" | grep "_spatial$" || true)
PRO_S2=$(echo "$PRO_SESSIONS" | grep "_object$" || true)
if [ -n "$PRO_S1" ] || [ -n "$PRO_S2" ]; then
  log "Pro group 1: $PRO_S1 + $PRO_S2"
  wait_sessions $PRO_S1 $PRO_S2
fi

# 组2: goal_pro(GPU0) + 10_pro(GPU1)
log "Pro group 2: goal + 10"; gpu
GPU_LAYOUT="0 1 0 1" \
CKPT_DIR="${CKPT_DIR}" CONFIG_NAME="auto" \
PRO_SUFFIXES="task swap env object lan" \
SESSION_PREFIX="pro5b" \
bash -c '
  source scripts/eval/run_eval_libero_pro_4tmux.sh 2>/dev/null || true
' 2>&1 | head -5 | tee -a "$LOG"

sleep 5
PRO2_SESSIONS=$(tmux ls 2>/dev/null | grep "^pro5b_" | cut -d: -f1)
for s in $PRO2_SESSIONS; do
  case "$s" in *_spatial|*_object) tmux kill-session -t "$s" 2>/dev/null; log "killed $s";; esac
done

PRO2_S1=$(echo "$PRO2_SESSIONS" | grep "_goal$" || true)
PRO2_S2=$(echo "$PRO2_SESSIONS" | grep "_10$" || true)
if [ -n "$PRO2_S1" ] || [ -n "$PRO2_S2" ]; then
  log "Pro group 2: $PRO2_S1 + $PRO2_S2"
  wait_sessions $PRO2_S1 $PRO2_S2
fi

log "=== ALL EVALUATIONS COMPLETE ==="
gpu
