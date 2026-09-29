#!/usr/bin/env bash
set -euo pipefail

# 默认按 25000 / 20000 / 30000 的顺序评测，不对缺失 checkpoint 做回退。
OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OPENPI_ROOT}/checkpoints/pi0_libero_lora/raw}"
RESULT_BASE="${RESULT_BASE:-${OPENPI_ROOT}/results/libero}"
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_lora}"
CKPTS="${CKPTS:-25000 20000 30000}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
POLICY_SEED="${POLICY_SEED:-}"
PORT="${PORT:-8000}"
GPU_ID="${GPU_ID:-0}"
SKIP_SERVER="${SKIP_SERVER:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RESULT_BASE}"
cd "${OPENPI_ROOT}"
source ~/miniconda3/bin/activate openpi
export PYTHONPATH="${PYTHONPATH:-}:${OPENPI_ROOT}/third_party/libero"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl

resolve_ckpt_dir() {
  local ckpt="$1"
  local ckpt_dir="${CHECKPOINT_ROOT}/${ckpt}"
  if [ ! -d "${ckpt_dir}" ]; then
    echo "Missing checkpoint directory: ${ckpt_dir}" >&2
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
  echo "${RESULT_BASE}/${ckpt_layout}/${ckpt_tag}/${RUN_TAG}"
}

wait_for_server() {
  python - <<'PY2'
import asyncio
import os
import websockets

port = int(os.environ['WAIT_PORT'])
uri = f'ws://127.0.0.1:{port}'

async def main():
    for _ in range(600):
        try:
            async with websockets.connect(uri, open_timeout=1, close_timeout=1):
                return
        except Exception:
            await asyncio.sleep(1)
    raise SystemExit('server did not become ready in time')

asyncio.run(main())
PY2
}

run_one_ckpt() {
  local ckpt="$1"
  local ckpt_dir
  if ! ckpt_dir="$(resolve_ckpt_dir "${ckpt}")"; then
    printf '[CKPT_ERROR] %s %s missing_checkpoint\n' "$(date --iso-8601=seconds)" "${ckpt}" | tee -a "${RESULT_BASE}/queue_runner.log"
    return 0
  fi
  local result_root
  result_root="$(build_result_root "${ckpt_dir}")"
  mkdir -p "${result_root}/videos"

  {
    echo "# LIBERO Eval"
    echo "checkpoint_tag: ${ckpt}"
    echo "checkpoint_dir: ${ckpt_dir}"
    echo "config: ${CONFIG_NAME}"
    echo "suites: ${SUITES}"
    echo "num_trials_per_task: ${NUM_TRIALS_PER_TASK}"
    echo "policy_seed: ${POLICY_SEED}"
    echo
  } > "${result_root}/summary.md"

  local server_pid=""
  if [ "${SKIP_SERVER}" != "1" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    server_cmd=(python scripts/serve_policy.py --env LIBERO --port "${PORT}")
    if [ -n "${POLICY_SEED}" ]; then
      server_cmd+=(--policy-seed "${POLICY_SEED}")
    fi
    server_cmd+=(policy:checkpoint --policy.config="${CONFIG_NAME}" --policy.dir="${ckpt_dir}")
    "${server_cmd[@]}" > "${result_root}/server.log" 2>&1 &
    server_pid=$!
    export WAIT_PORT="${PORT}"
    if ! wait_for_server; then
      echo "- server_start: failed" >> "${result_root}/summary.md"
      printf '[CKPT_ERROR] %s %s server_start_failed\n' "$(date --iso-8601=seconds)" "${ckpt}" | tee -a "${RESULT_BASE}/queue_runner.log"
      if [ -n "${server_pid}" ]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
      fi
      return 0
    fi
  fi

  local suite
  for suite in ${SUITES}; do
    mkdir -p "${result_root}/videos/${suite}"
    printf '[SUITE_START] %s %s
' "$(date --iso-8601=seconds)" "${suite}" | tee -a "${result_root}/runner.log"
    if python examples/libero/main.py       --args.host 127.0.0.1       --args.port "${PORT}"       --args.resize-size "${RESIZE_SIZE}"       --args.replan-steps "${REPLAN_STEPS}"       --args.task-suite-name "${suite}"       --args.num-steps-wait "${NUM_STEPS_WAIT}"       --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"       --args.video-out-path "${result_root}/videos/${suite}"       --args.seed "${SEED}"       > "${result_root}/${suite}.log" 2>&1; then
      code=0
    else
      code=$?
    fi
    printf '[SUITE_END] %s %s code=%s
' "$(date --iso-8601=seconds)" "${suite}" "${code}" | tee -a "${result_root}/runner.log"
    rate=$(grep -F 'Total success rate:' "${result_root}/${suite}.log" | tail -n1 | sed 's/.*Total success rate: //')
    episodes=$(grep -F 'Total episodes:' "${result_root}/${suite}.log" | tail -n1 | sed 's/.*Total episodes: //')
    echo "- ${suite}: success_rate=${rate:-NA}, episodes=${episodes:-NA}, exit_code=${code}" >> "${result_root}/summary.md"
  done

  if [ -n "${server_pid}" ]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  echo "result_root: ${result_root}" >> "${result_root}/summary.md"
}

for ckpt in ${CKPTS}; do
  run_one_ckpt "${ckpt}"
done
