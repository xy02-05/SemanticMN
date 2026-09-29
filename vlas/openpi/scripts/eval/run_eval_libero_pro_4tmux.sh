#!/usr/bin/env bash
set -euo pipefail

# 一键评测全部 5 种 LIBERO-PRO 扰动（lan/object/swap/task/env）。
#
# 架构：4 个 tmux session，每个绑定一个 base suite + 一个 GPU + 一个端口。
# session 内部依次跑 5 种扰动；某个 suite 跑完当前扰动后立刻启动下一个，
# 不需要等其他 suite —— 天然实现"哪个 GPU 空闲就在哪个 GPU 上接着跑"。

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
OPENPI_PYTHON="${OPENPI_PYTHON:-/root/miniconda3/envs/openpi/bin/python}"
LIBERO_PRO_ROOT="${LIBERO_PRO_ROOT:-/root/data/xuyuan1/Codes/mirror_neuron/vlas/LIBERO-PRO}"
read -r -a TMUX_CMD <<< "${TMUX_CMD:-tmux}"

# ========= 通常只需要改 checkpoint 路径 =========
CKPT_DIR="${CKPT_DIR:-/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/pi0_libero_full_qwen/new_DSN_new_egohod/30000}"
CONFIG_NAME="${CONFIG_NAME:-auto}"

RESULT_BASE="${RESULT_BASE:-${OPENPI_ROOT}/results/libero_pro}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SESSION_PREFIX="${SESSION_PREFIX:-pro5}"

# 评测参数
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-1000}"
POLICY_SEED="${POLICY_SEED:-${SEED}}"
SAVE_ACTION_FEATURES="${SAVE_ACTION_FEATURES:-}"

# 5 种扰动，可通过 PRO_SUFFIXES 环境变量自定义子集
# 例: PRO_SUFFIXES="lan object" 只跑两种
read -r -a PRO_SUFFIXES <<< "${PRO_SUFFIXES:-task swap env object lan}"
BASE_SUITES=(libero_spatial libero_object libero_goal libero_10)

# 端口自动分配
if [ -z "${PORT_BASE:-}" ]; then
  _tag_num="${RUN_TAG//[^0-9]/}"
  _tag_suffix="${_tag_num: -4}"
  PORT_BASE=$((24000 + 10#${_tag_suffix} * 4))
fi
PORTS=("${PORT_BASE}" "$((PORT_BASE+1))" "$((PORT_BASE+2))" "$((PORT_BASE+3))")

GPU_LAYOUT="${GPU_LAYOUT:-0 0 1 1}"
read -r -a GPU_IDS <<< "${GPU_LAYOUT}"
if [ "${#GPU_IDS[@]}" -ne 4 ]; then
  echo "GPU_LAYOUT must provide exactly 4 gpu ids, got: ${GPU_LAYOUT}" >&2; exit 1
fi

[ -d "${LIBERO_PRO_ROOT}" ] || { echo "LIBERO_PRO_ROOT not found: ${LIBERO_PRO_ROOT}" >&2; exit 1; }
[ -d "${CKPT_DIR}" ]        || { echo "CKPT_DIR not found: ${CKPT_DIR}" >&2; exit 1; }

# ─── 工具函数 ───────────────────────────────────────────

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
print(f"{meta.get('global_step','unknown')} {cfg.get('seed','unknown') if isinstance(cfg,dict) else 'unknown'} {cfg.get('name','unknown') if isinstance(cfg,dict) else 'unknown'}")
PY
}

resolve_config_name() {
  if [ "${CONFIG_NAME}" != "auto" ]; then echo "${CONFIG_NAME}"; return; fi
  local _s _sd cn; read -r _s _sd cn < <(read_ckpt_meta "${CKPT_DIR}"); echo "${cn}"
}

build_result_root() {
  local ckpt_dir="${1%/}"
  local layout; layout="$(path_tail_two "$(dirname "${ckpt_dir}")")"
  local ts sd cn; read -r ts sd cn < <(read_ckpt_meta "${ckpt_dir}")
  echo "${RESULT_BASE}/${layout}/$(basename "${ckpt_dir}")_step${ts}_seed${sd}_ep${NUM_TRIALS_PER_TASK}/${RUN_TAG}"
}

# ─── 主流程 ─────────────────────────────────────────────

main() {
  local config_name; config_name="$(resolve_config_name)"
  local result_root; result_root="$(build_result_root "${CKPT_DIR}")"
  mkdir -p "${result_root}"

  local ts sd cn; read -r ts sd cn < <(read_ckpt_meta "${CKPT_DIR}")
  {
    echo "CKPT_DIR=${CKPT_DIR}"
    echo "CONFIG_NAME=${config_name}"
    echo "LIBERO_PRO_ROOT=${LIBERO_PRO_ROOT}"
    echo "PRO_SUFFIXES=${PRO_SUFFIXES[*]}"
    echo "RESULT_ROOT=${result_root}"
    echo "TRAIN_STEP=${ts}  TRAIN_SEED=${sd}  TRAIN_CONFIG=${cn}"
    echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
    echo "SEED=${SEED}  POLICY_SEED=${POLICY_SEED}"
    echo "PORT_BASE=${PORT_BASE}  GPU_LAYOUT=${GPU_LAYOUT}"
    echo "START=$(date --iso-8601=seconds)"
  } > "${result_root}/launch_meta.txt"

  # 为每个 base suite 启动一个 tmux session，内部顺序跑所有扰动类型
  local idx
  for idx in "${!BASE_SUITES[@]}"; do
    local base="${BASE_SUITES[$idx]}"
    local port="${PORTS[$idx]}"
    local gpu="${GPU_IDS[$idx]}"
    local short="${base#libero_}"
    local sess="${SESSION_PREFIX}_${RUN_TAG}_${short}"

    # 构建 tmux 内执行的命令串：依次跑每种扰动
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
      cmd+=" GPU_ID='${gpu}'"
      cmd+=" NUM_TRIALS_PER_TASK='${NUM_TRIALS_PER_TASK}'"
      cmd+=" RESIZE_SIZE='${RESIZE_SIZE}'"
      cmd+=" REPLAN_STEPS='${REPLAN_STEPS}'"
      cmd+=" NUM_STEPS_WAIT='${NUM_STEPS_WAIT}'"
      cmd+=" SEED='${SEED}'"
      cmd+=" POLICY_SEED='${POLICY_SEED}'"
      cmd+=" SAVE_ACTION_FEATURES='${SAVE_ACTION_FEATURES}'"
      cmd+=" bash '${OPENPI_ROOT}/scripts/eval/run_eval_libero_single_suite.sh'"
      cmd+=" && echo '====== [${suite}] finished at '\$(date)' ======'"
    done
    # 最后输出汇总提示
    cmd+=" && echo '' && echo 'All ${#PRO_SUFFIXES[@]} perturbation types done for ${base}.'"
    cmd+=" ; echo 'Session ${sess} exited at '\$(date) ; sleep 86400"

    "${TMUX_CMD[@]}" new-session -d -s "${sess}" "${cmd}"

    # 错开 3 秒启动，避免多 session 同时写 LIBERO config 产生竞争
    if [ "${idx}" -lt "$(( ${#BASE_SUITES[@]} - 1 ))" ]; then
      sleep 3
    fi
  done

  {
    echo "Launched 4 tmux sessions, each running ${#PRO_SUFFIXES[@]} perturbation types sequentially:"
    echo "  Perturbation order: ${PRO_SUFFIXES[*]}"
    echo ""
    for idx in "${!BASE_SUITES[@]}"; do
      local short="${BASE_SUITES[$idx]#libero_}"
      echo "  ${SESSION_PREFIX}_${RUN_TAG}_${short}  (GPU ${GPU_IDS[$idx]}, port ${PORTS[$idx]})"
    done
    echo ""
    echo "Result root: ${result_root}"
    echo ""
    echo "监控进度："
    echo "  ${TMUX_CMD[*]} ls | grep ${SESSION_PREFIX}_${RUN_TAG}"
    echo "  for f in ${result_root}/*/client.log; do echo \"\$f\"; grep 'Total success' \"\$f\" 2>/dev/null; done"
  } | tee "${result_root}/attach_hint.txt"
}

main "$@"
