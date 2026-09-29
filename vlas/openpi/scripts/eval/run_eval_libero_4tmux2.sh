#!/usr/bin/env bash
set -euo pipefail

# 多 checkpoint 排队评测：对每个 checkpoint 启动 4 个 tmux session（每个 suite 一个），
# 等待 4 个 session 全部结束后，再启动下一个 checkpoint 的评测。
# 复用 run_eval_libero_single_suite.sh，不重复写评测逻辑。

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
OPENPI_PYTHON="${OPENPI_PYTHON:-/root/miniconda3/envs/openpi/bin/python}"

# ========== 你通常只需要改下面这两项 ==========
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_lora}"

# 支持多个 checkpoint 目录，空格分隔。会按顺序逐个评测，每个 checkpoint 并行跑 4 个 suite。
# 例: CKPT_DIRS="/path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3"
CKPT_DIRS="${CKPT_DIRS:-${CKPT_DIR:-/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/pi0_libero_lora/new/40000}}"

RESULT_BASE="${RESULT_BASE:-${OPENPI_ROOT}/results/libero}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SESSION_PREFIX="${SESSION_PREFIX:-libero4}"

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
  PORT_BASE=$((20000 + 10#${_tag_suffix} * 4))
fi

SUITES=(
  "libero_spatial"
  "libero_object"
  "libero_goal"
  "libero_10"
)
PORTS=(
  "${PORT_BASE}"
  "$((PORT_BASE + 1))"
  "$((PORT_BASE + 2))"
  "$((PORT_BASE + 3))"
)
GPU_LAYOUT="${GPU_LAYOUT:-0 0 0 0}"
read -r -a GPU_IDS <<< "${GPU_LAYOUT}"

if [ "${#GPU_IDS[@]}" -ne 4 ]; then
  echo "GPU_LAYOUT must provide exactly 4 gpu ids, got: ${GPU_LAYOUT}" >&2
  exit 1
fi

resolve_ckpt_dir() {
  local ckpt_dir="${1%/}"

  if [ -z "${ckpt_dir}" ]; then
    echo "Empty checkpoint directory in CKPT_DIRS" >&2
    return 1
  fi

  if [ ! -d "${ckpt_dir}" ]; then
    echo "Checkpoint directory not found: ${ckpt_dir}" >&2
    return 1
  fi

  echo "${ckpt_dir}"
}

path_tail_two() {
  local path="${1%/}"
  local last="${path##*/}"
  local parent="${path%/*}"

  if [ -z "${last}" ] || [ "${last}" = "${path}" ]; then
    echo "${path}"
    return 0
  fi

  if [ -z "${parent}" ] || [ "${parent}" = "${path}" ]; then
    echo "${last}"
    return 0
  fi

  local second_last="${parent##*/}"
  if [ -z "${second_last}" ] || [ "${second_last}" = "." ] || [ "${second_last}" = "/" ]; then
    echo "${last}"
    return 0
  fi

  echo "${second_last}/${last}"
}

build_result_root() {
  local ckpt_dir="${1%/}"
  local ckpt_tag
  ckpt_tag="$(basename "${ckpt_dir}")"
  local ckpt_parent
  ckpt_parent="$(dirname "${ckpt_dir}")"
  local ckpt_layout
  ckpt_layout="$(path_tail_two "${ckpt_parent}")"
  local train_step
  local train_seed

  # 结果目录里直接带上训练元信息，方便同一 setting 下横向比较。
  read -r train_step train_seed < <(read_ckpt_train_info "${ckpt_dir}")
  echo "${RESULT_BASE}/${ckpt_layout}/${ckpt_tag}_step${train_step}_seed${train_seed}/${RUN_TAG}"
}

read_ckpt_train_info() {
  local ckpt_dir="${1%/}"
  local meta_path="${ckpt_dir}/metadata.pt"

  "${OPENPI_PYTHON}" - "${meta_path}" <<'PY'
import sys
import torch

meta_path = sys.argv[1]
meta = torch.load(meta_path, map_location="cpu", weights_only=False)
config = meta.get("config", {}) if isinstance(meta, dict) else {}
global_step = meta.get("global_step", "unknown")
seed = config.get("seed", "unknown") if isinstance(config, dict) else "unknown"
print(f"{global_step} {seed}")
PY
}

build_session_name() {
  local ckpt_index="$1"
  local suite="$2"
  local suite_short="${suite#libero_}"

  echo "${SESSION_PREFIX}_${RUN_TAG}_ckpt${ckpt_index}_${suite_short}"
}

launch_one() {
  local suite="$1"
  local port="$2"
  local gpu_id="$3"
  local ckpt_dir="$4"
  local result_root="$5"
  local ckpt_index="$6"

  local session_name
  session_name="$(build_session_name "${ckpt_index}" "${suite}")"
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

  echo "${session_name}"
}

wait_for_sessions_exit() {
  local session_names=("$@")

  # 这里只做最简单的轮询：4 个 tmux 都退出后，再继续下一个 checkpoint。
  while true; do
    local alive_count=0
    local session_name

    for session_name in "${session_names[@]}"; do
      if tmux has-session -t "${session_name}" 2>/dev/null; then
        alive_count=$((alive_count + 1))
      fi
    done

    if [ "${alive_count}" -eq 0 ]; then
      return 0
    fi

    sleep 10
  done
}

run_one_ckpt() {
  local ckpt_dir="$1"
  local ckpt_index="$2"

  local result_root
  result_root="$(build_result_root "${ckpt_dir}")"
  mkdir -p "${result_root}"

  local train_step
  local train_seed
  read -r train_step train_seed < <(read_ckpt_train_info "${ckpt_dir}")

  {
    echo "CKPT_DIR=${ckpt_dir}"
    echo "CKPT_INDEX=${ckpt_index}"
    echo "CKPT_DIRS=${CKPT_DIRS}"
    echo "CONFIG_NAME=${CONFIG_NAME}"
    echo "RESULT_ROOT=${result_root}"
    echo "TRAIN_STEP=${train_step}"
    echo "TRAIN_SEED=${train_seed}"
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
  local -a session_names=()
  for idx in "${!SUITES[@]}"; do
    session_names+=("$(launch_one "${SUITES[$idx]}" "${PORTS[$idx]}" "${GPU_IDS[$idx]}" "${ckpt_dir}" "${result_root}" "${ckpt_index}")")
  done

  {
    echo "Launched 4 tmux sessions:"
    for idx in "${!session_names[@]}"; do
      echo "  - ${session_names[$idx]}"
    done
    echo
    echo "Result root: ${result_root}"
  } | tee "${result_root}/attach_hint.txt"

  wait_for_sessions_exit "${session_names[@]}"

  echo "END=$(date --iso-8601=seconds)" >> "${result_root}/launch_meta.txt"
}

main() {
  mkdir -p "${RESULT_BASE}"

  local -a ckpt_dir_list=()
  read -r -a ckpt_dir_list <<< "${CKPT_DIRS}"

  if [ "${#ckpt_dir_list[@]}" -eq 0 ]; then
    echo "CKPT_DIRS is empty" >&2
    exit 1
  fi

  local ckpt_index=0
  local raw_ckpt_dir
  for raw_ckpt_dir in "${ckpt_dir_list[@]}"; do
    ckpt_index=$((ckpt_index + 1))

    local ckpt_dir
    ckpt_dir="$(resolve_ckpt_dir "${raw_ckpt_dir}")"
    run_one_ckpt "${ckpt_dir}" "${ckpt_index}"
  done
}

main "$@"
