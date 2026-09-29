#!/usr/bin/env bash
set -euo pipefail

# 一键启动 4 个 LIBERO 标准 suite 的 tmux 评测。
# 设计目标：
# 1. 复用现有 run_eval_libero_single_suite.sh，不重复写评测逻辑。
# 2. 使用 4 个 tmux session，各 suite 互不影响。
# 3. 用户只需要改 checkpoint 相关信息就能复用。

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
OPENPI_PYTHON="${OPENPI_PYTHON:-/root/miniconda3/envs/openpi/bin/python}"
read -r -a TMUX_CMD <<< "${TMUX_CMD:-tmux}"

# ========== 你通常只需要改下面这两项 ==========
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_full_qwen_video}"
CKPT_DIR="${CKPT_DIR:-/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/pi0_libero_full_qwen/new_DSN_new_egohod/30000}"

# 可选：如果不直接传 CKPT_DIR，也可以传 CHECKPOINT_ROOT + CKPT_TAG 组合。
CHECKPOINT_ROOT=""
CKPT_TAG=""

RESULT_BASE="${RESULT_BASE:-${OPENPI_ROOT}/results/libero}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SESSION_PREFIX="${SESSION_PREFIX:-libero4}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-20}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-1000}"
# 默认与环境seed相同，确保扩散推理可复现；设为空可恢复随机行为
POLICY_SEED="${POLICY_SEED:-${SEED}}"
# 是否在推理时提取并保存 action feature（设为 1 开启，如 SAVE_ACTION_FEATURES=1）
SAVE_ACTION_FEATURES="${SAVE_ACTION_FEATURES:-}"
# 可选覆盖：rebuttal ID 使用历史入口和 official LIBERO package。
EVAL_MAIN="${EVAL_MAIN:-examples/libero/main.py}"
LIBERO_PACKAGE_ROOT="${LIBERO_PACKAGE_ROOT:-}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-}"
EVAL_PYTHONPATH="${EVAL_PYTHONPATH:-${PYTHONPATH:-}}"

# 每次 launch 默认自动生成新的端口段，避免和旧评测冲突。
# 如果用户显式传 PORT_BASE，则优先使用传入值。
if [ -z "${PORT_BASE:-}" ]; then
  _tag_num="${RUN_TAG//[^0-9]/}"
  _tag_suffix="${_tag_num: -4}"
  PORT_BASE=$((20000 + 10#${_tag_suffix} * 4))
fi

# 默认把 4 个 suite 分到 4 个 server。
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
# 默认分配为 0 0 1 1；如果需要单卡同时跑 4 个 tmux，可传 GPU_LAYOUT="0 0 0 0"。
GPU_LAYOUT="${GPU_LAYOUT:-0 0 1 1}"
read -r -a GPU_IDS <<< "${GPU_LAYOUT}"

if [ "${#GPU_IDS[@]}" -ne 4 ]; then
  echo "GPU_LAYOUT must provide exactly 4 gpu ids, got: ${GPU_LAYOUT}" >&2
  exit 1
fi

resolve_ckpt_dir() {
  if [ -n "${CKPT_DIR}" ]; then
    echo "${CKPT_DIR}"
    return 0
  fi

  if [ -n "${CKPT_TAG}" ]; then
    echo "${CHECKPOINT_ROOT}/${CKPT_TAG}"
    return 0
  fi

  echo "Need CKPT_DIR or CKPT_TAG" >&2
  return 1
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
  echo "${RESULT_BASE}/${ckpt_layout}/${ckpt_tag}_step${train_step}_seed${train_seed}_ep${NUM_TRIALS_PER_TASK}/${RUN_TAG}"
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

  "${TMUX_CMD[@]}" new-session -d -s "${session_name}" \
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
     SAVE_ACTION_FEATURES='${SAVE_ACTION_FEATURES}' \
     EVAL_MAIN='${EVAL_MAIN}' \
     LIBERO_PACKAGE_ROOT='${LIBERO_PACKAGE_ROOT}' \
     LIBERO_CONFIG_PATH='${LIBERO_CONFIG_PATH}' \
     PYTHONPATH='${EVAL_PYTHONPATH}' \
     bash '${OPENPI_ROOT}/scripts/eval/run_eval_libero_single_suite.sh'"
}

main() {
  local ckpt_dir
  ckpt_dir="$(resolve_ckpt_dir)"

  if [ ! -d "${ckpt_dir}" ]; then
    echo "Checkpoint directory not found: ${ckpt_dir}" >&2
    exit 1
  fi

  mkdir -p "${RESULT_BASE}"

  local result_root
  result_root="$(build_result_root "${ckpt_dir}")"
  mkdir -p "${result_root}"

  local train_step
  local train_seed
  read -r train_step train_seed < <(read_ckpt_train_info "${ckpt_dir}")

  {
    echo "CKPT_DIR=${ckpt_dir}"
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
  for idx in "${!SUITES[@]}"; do
    launch_one "${SUITES[$idx]}" "${PORTS[$idx]}" "${GPU_IDS[$idx]}" "${ckpt_dir}" "${result_root}"
  done

  {
    echo "Launched 4 tmux sessions:"
    for idx in "${!SUITES[@]}"; do
      local suite_short="${SUITES[$idx]#libero_}"
      echo "  - ${SESSION_PREFIX}_${RUN_TAG}_${suite_short}"
    done
    echo
    echo "Result root: ${result_root}"
  } | tee "${result_root}/attach_hint.txt"
}

main "$@"
