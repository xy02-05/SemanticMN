#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LIBERO-PRO 全量评测启动脚本（单卡版）
#
# 4 个 base suite 各占 1 个 tmux session，同一张 GPU 上并行。
# 每个 session 内依次跑 5 种扰动（lan/object/swap/task/env）。
# 某个 suite 跑完当前扰动后立刻启动下一个，不等其他 suite。
#
# 使用方式：
#   bash scripts/eval/launch_pro_eval.sh
#
# 自定义示例：
#   GPU_ID=0 NUM_TRIALS_PER_TASK=20 bash scripts/eval/launch_pro_eval.sh
#   PRO_SUFFIXES="lan swap" bash scripts/eval/launch_pro_eval.sh  # 只跑部分扰动
# ============================================================

# ─── 基础路径 ──────────────────────────────────────────────
OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
OPENPI_PYTHON="${OPENPI_PYTHON:-/root/miniconda3/envs/openpi/bin/python}"
LIBERO_PRO_ROOT="${LIBERO_PRO_ROOT:-/root/data/xuyuan1/Codes/mirror_neuron/vlas/LIBERO-PRO}"

# ─── Checkpoint（改这里）──────────────────────────────────
CKPT_DIR="${CKPT_DIR:-${OPENPI_ROOT}/checkpoints/pi0_libero_full/raw/30000}"
CONFIG_NAME="${CONFIG_NAME:-auto}"

# ─── GPU（单卡，4 个 session 都放同一张卡）─────────────────
GPU_ID="${GPU_ID:-0}"

# ─── 评测参数 ──────────────────────────────────────────────
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-1000}"
POLICY_SEED="${POLICY_SEED:-${SEED}}"

# ─── 扰动类型 ─────────────────────────────────────────────
read -r -a PRO_SUFFIXES <<< "${PRO_SUFFIXES:-lan object swap task env}"
BASE_SUITES=(libero_spatial libero_object libero_goal libero_10)

# ─── 输出路径 & 会话名 ────────────────────────────────────
RESULT_BASE="${RESULT_BASE:-${OPENPI_ROOT}/results/libero_pro}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SESSION_PREFIX="${SESSION_PREFIX:-pro}"
# SESSION_PREFIX 也可外部传入，用于多 GPU 队列互不干扰

# ─── 端口自动分配（4 个 session 各占 1 个端口）────────────
if [ -z "${PORT_BASE:-}" ]; then
  _tag_num="${RUN_TAG//[^0-9]/}"
  _tag_suffix="${_tag_num: -4}"
  PORT_BASE=$((24000 + 10#${_tag_suffix} * 4))
fi

# ─── 前置检查 ──────────────────────────────────────────────
[ -d "${LIBERO_PRO_ROOT}" ] || { echo "LIBERO_PRO_ROOT not found: ${LIBERO_PRO_ROOT}" >&2; exit 1; }
[ -d "${CKPT_DIR}" ]        || { echo "CKPT_DIR not found: ${CKPT_DIR}" >&2; exit 1; }

# ─── 工具函数 ──────────────────────────────────────────────
path_tail_two() {
  local path="${1%/}"
  local last="${path##*/}"
  local parent="${path%/*}"
  if [ -z "${last}" ] || [ "${last}" = "${path}" ]; then echo "${path}"; return; fi
  if [ -z "${parent}" ] || [ "${parent}" = "${path}" ]; then echo "${last}"; return; fi
  local second="${parent##*/}"
  if [ -z "${second}" ] || [ "${second}" = "." ] || [ "${second}" = "/" ]; then echo "${last}"; return; fi
  echo "${second}/${last}"
}

read_ckpt_meta() {
  "${OPENPI_PYTHON}" - "${1%/}/metadata.pt" <<'PY'
import sys, torch
meta = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = meta.get("config", {}) if isinstance(meta, dict) else {}
print(f"{meta.get('global_step','unknown')} "
      f"{cfg.get('seed','unknown') if isinstance(cfg,dict) else 'unknown'} "
      f"{cfg.get('name','unknown') if isinstance(cfg,dict) else 'unknown'}")
PY
}

resolve_config_name() {
  if [ "${CONFIG_NAME}" != "auto" ]; then echo "${CONFIG_NAME}"; return; fi
  local _s _sd cn; read -r _s _sd cn < <(read_ckpt_meta "${CKPT_DIR}"); echo "${cn}"
}

build_result_root() {
  local ckpt_dir="${1%/}"
  local layout
  layout="$(path_tail_two "$(dirname "${ckpt_dir}")")"
  local ts sd cn; read -r ts sd cn < <(read_ckpt_meta "${ckpt_dir}")
  echo "${RESULT_BASE}/${layout}/$(basename "${ckpt_dir}")_step${ts}_seed${sd}_ep${NUM_TRIALS_PER_TASK}/${RUN_TAG}"
}

# ─── 主流程 ────────────────────────────────────────────────
main() {
  local config_name
  config_name="$(resolve_config_name)"

  local result_root
  result_root="$(build_result_root "${CKPT_DIR}")"
  mkdir -p "${result_root}"

  local ts sd cn; read -r ts sd cn < <(read_ckpt_meta "${CKPT_DIR}")

  # 记录本次启动元信息
  {
    echo "CKPT_DIR=${CKPT_DIR}"
    echo "CONFIG_NAME=${config_name}"
    echo "LIBERO_PRO_ROOT=${LIBERO_PRO_ROOT}"
    echo "GPU_ID=${GPU_ID}"
    echo "PRO_SUFFIXES=${PRO_SUFFIXES[*]}"
    echo "RESULT_ROOT=${result_root}"
    echo "TRAIN: step=${ts} seed=${sd} config=${cn}"
    echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
    echo "SEED=${SEED}  POLICY_SEED=${POLICY_SEED}"
    echo "PORT_BASE=${PORT_BASE}"
    echo "START=$(date --iso-8601=seconds)"
  } > "${result_root}/launch_meta.txt"

  # 为每个 base suite 启动一个 tmux session
  local idx=0
  for base in "${BASE_SUITES[@]}"; do
    local port=$((PORT_BASE + idx))
    local short="${base#libero_}"
    local sess="${SESSION_PREFIX}_${RUN_TAG}_${short}"

    # 在 tmux 内依次跑每种扰动
    local cmd="cd '${OPENPI_ROOT}'"
    for suffix in "${PRO_SUFFIXES[@]}"; do
      local suite="${base}_${suffix}"
      local rdir="${result_root}/${suite}"
      cmd+=" && echo ''"
      cmd+=" && echo '====== [${suite}] started at '\$(date)' ======'"
      cmd+=" && mkdir -p '${rdir}/videos'"
      cmd+=" && EVAL_MAIN='examples/libero/main_pro.py'"
      cmd+=" LIBERO_PACKAGE_ROOT='${LIBERO_PRO_ROOT}'"
      cmd+=" CONFIG_NAME='${config_name}'"
      cmd+=" CKPT_DIR='${CKPT_DIR}'"
      cmd+=" SUITE='${suite}'"
      cmd+=" RESULT_DIR='${rdir}'"
      cmd+=" PORT='${port}'"
      cmd+=" GPU_ID='${GPU_ID}'"
      cmd+=" NUM_TRIALS_PER_TASK='${NUM_TRIALS_PER_TASK}'"
      cmd+=" RESIZE_SIZE='${RESIZE_SIZE}'"
      cmd+=" REPLAN_STEPS='${REPLAN_STEPS}'"
      cmd+=" NUM_STEPS_WAIT='${NUM_STEPS_WAIT}'"
      cmd+=" SEED='${SEED}'"
      cmd+=" POLICY_SEED='${POLICY_SEED}'"
      cmd+=" bash '${OPENPI_ROOT}/scripts/eval/run_eval_libero_single_suite.sh'"
      cmd+=" && echo '====== [${suite}] finished at '\$(date)' ======'"
    done
    cmd+=" && echo '' && echo 'All ${#PRO_SUFFIXES[@]} perturbations done for ${base}.'"
    cmd+=" ; echo 'Session ${sess} exited at '\$(date) ; sleep 86400"

    tmux new-session -d -s "${sess}" "${cmd}"

    # 错开 5 秒，避免 4 个 session 同时写 LIBERO config 竞争
    if [ "${idx}" -lt "$(( ${#BASE_SUITES[@]} - 1 ))" ]; then
      sleep 5
    fi
    idx=$((idx + 1))
  done

  # 输出汇总
  cat <<EOF | tee "${result_root}/attach_hint.txt"

Launched 4 tmux sessions on GPU ${GPU_ID}:
  Perturbation order: ${PRO_SUFFIXES[*]}
  Trials per task: ${NUM_TRIALS_PER_TASK}

  ${SESSION_PREFIX}_${RUN_TAG}_spatial  (port $((PORT_BASE)))
  ${SESSION_PREFIX}_${RUN_TAG}_object   (port $((PORT_BASE+1)))
  ${SESSION_PREFIX}_${RUN_TAG}_goal     (port $((PORT_BASE+2)))
  ${SESSION_PREFIX}_${RUN_TAG}_10       (port $((PORT_BASE+3)))

Result root:
  ${result_root}

监控命令：
  tmux ls | grep ${SESSION_PREFIX}_${RUN_TAG}
  for f in ${result_root}/*/client.log; do echo "\$(basename \$(dirname \$f))"; grep 'Total success' "\$f" 2>/dev/null; done
EOF
}

main "$@"
