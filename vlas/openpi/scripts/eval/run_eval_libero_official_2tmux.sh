#!/usr/bin/env bash
set -euo pipefail

# 一键启动 2 个 LIBERO 标准 suite 的 tmux 评测。
# 设计目标：
# 1. 尽量复用现有 run_eval_libero_single_suite.sh，不重复写评测逻辑。
# 2. 默认评测官方 JAX checkpoint：/root/data/xuyuan1/dataset/pi0_libero_official
# 3. 当前只先跑 spatial 和 object，避免一次性起 4 路。

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"

# ========== 你通常只需要改下面这两项 ==========
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_official_eval}"
CKPT_DIR="${CKPT_DIR:-/root/data/xuyuan1/dataset/pi0_libero_official}"

RESULT_BASE="${RESULT_BASE:-${OPENPI_ROOT}/results/libero}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SESSION_PREFIX="${SESSION_PREFIX:-libero2off}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-1000}"
POLICY_SEED="${POLICY_SEED:-}"

# 每次 launch 默认自动生成新的端口段，避免和旧评测冲突。
if [ -z "${PORT_BASE:-}" ]; then
  _tag_num="${RUN_TAG//[^0-9]/}"
  _tag_suffix="${_tag_num: -4}"
  PORT_BASE=$((24000 + 10#${_tag_suffix} * 2))
fi

# 当前只先跑两个 suite。
SUITES=(
  "libero_spatial"
  "libero_object"
)
PORTS=(
  "${PORT_BASE}"
  "$((PORT_BASE + 1))"
)
# 两个 suite 分别放到两张卡上，避免和 4 路脚本一样在单卡上叠两路。
GPU_IDS=(0 1)

resolve_ckpt_dir() {
  if [ ! -d "${CKPT_DIR}" ]; then
    echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
    return 1
  fi
  echo "${CKPT_DIR}"
}

launch_one() {
  local suite="$1"
  local port="$2"
  local gpu_id="$3"
  local ckpt_dir="$4"
  local result_root="$5"

  local suite_short="${suite#libero_}"
  local session_name="${SESSION_PREFIX}_${RUN_TAG}_${suite_short}"
  local result_dir="${result_root}/${suite}"

  mkdir -p "${result_dir}"

  tmux new-session -d -s "${session_name}" \
    "cd '${OPENPI_ROOT}' && \
     CONFIG_NAME='${CONFIG_NAME}' \
     CKPT_DIR='${ckpt_dir}' \
     SUITE='${suite}' \
     RESULT_DIR='${result_dir}' \
     PORT='${port}' \
     GPU_ID='${gpu_id}' \
     NUM_TRIALS_PER_TASK='${NUM_TRIALS_PER_TASK}' \
     RESIZE_SIZE='${RESIZE_SIZE}' \
     REPLAN_STEPS='${REPLAN_STEPS}' \
     NUM_STEPS_WAIT='${NUM_STEPS_WAIT}' \
     SEED='${SEED}' \
     POLICY_SEED='${POLICY_SEED}' \
     bash '${OPENPI_ROOT}/scripts/eval/run_eval_libero_single_suite.sh'"
}

main() {
  local ckpt_dir
  ckpt_dir="$(resolve_ckpt_dir)"

  mkdir -p "${RESULT_BASE}"

  local ckpt_name
  ckpt_name="$(basename "${ckpt_dir}")"
  local result_root="${RESULT_BASE}/libero_${ckpt_name}_2tmux_${RUN_TAG}"
  mkdir -p "${result_root}"

  {
    echo "CKPT_DIR=${ckpt_dir}"
    echo "CONFIG_NAME=${CONFIG_NAME}"
    echo "RESULT_ROOT=${result_root}"
    echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
    echo "RESIZE_SIZE=${RESIZE_SIZE}"
    echo "REPLAN_STEPS=${REPLAN_STEPS}"
    echo "NUM_STEPS_WAIT=${NUM_STEPS_WAIT}"
    echo "SEED=${SEED}"
    echo "POLICY_SEED=${POLICY_SEED}"
    echo "PORT_BASE=${PORT_BASE}"
    echo "PORTS=${PORTS[*]}"
    echo "START=$(date --iso-8601=seconds)"
  } > "${result_root}/launch_meta.txt"

  local idx
  for idx in "${!SUITES[@]}"; do
    launch_one "${SUITES[$idx]}" "${PORTS[$idx]}" "${GPU_IDS[$idx]}" "${ckpt_dir}" "${result_root}"
  done

  {
    echo "Launched 2 tmux sessions:"
    for idx in "${!SUITES[@]}"; do
      local suite_short="${SUITES[$idx]#libero_}"
      echo "  - ${SESSION_PREFIX}_${RUN_TAG}_${suite_short}"
    done
    echo
    echo "Result root: ${result_root}"
  } | tee "${result_root}/attach_hint.txt"
}

main "$@"
