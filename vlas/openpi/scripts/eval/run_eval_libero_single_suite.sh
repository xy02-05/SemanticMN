#!/usr/bin/env bash
set -euo pipefail

# 单个 LIBERO suite 的独立评测脚本。
# 设计目标：
# 1. 每个 suite 独占一个 policy server，互不影响。
# 2. 只复用官方 serve_policy.py + 指定评测入口。
# 3. 失败时直接报错，方便当前调试。

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_lora}"
CKPT_DIR="${CKPT_DIR:?CKPT_DIR is required}"
SUITE="${SUITE:?SUITE is required}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
PORT="${PORT:?PORT is required}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
LIBERO_PACKAGE_ROOT="${LIBERO_PACKAGE_ROOT:-}"
EVAL_MAIN="${EVAL_MAIN:-examples/libero/main.py}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
POLICY_SEED="${POLICY_SEED:-}"
MAX_TASKS="${MAX_TASKS:-}"
# 是否在推理时提取并保存 action feature（设为 1 开启）
SAVE_ACTION_FEATURES="${SAVE_ACTION_FEATURES:-}"

mkdir -p "${RESULT_DIR}/videos"

cd "${OPENPI_ROOT}"
# conda 激活脚本会覆盖这两个变量；先保存调用者显式传入的评测环境。
REQUESTED_PYTHONPATH="${PYTHONPATH:-}"
REQUESTED_LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-}"
source ~/miniconda3/bin/activate openpi
export PYTHONPATH="${REQUESTED_PYTHONPATH:+${REQUESTED_PYTHONPATH}:}${OPENPI_ROOT}/third_party/libero"
if [ -n "${REQUESTED_LIBERO_CONFIG_PATH}" ]; then
  export LIBERO_CONFIG_PATH="${REQUESTED_LIBERO_CONFIG_PATH}"
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
# 4 路 server 并发时，PyTorch/BLAS 默认会把 CPU 线程开得过大，
# 导致 checkpoint 加载阶段长期卡在 CPU 线程竞争里，GPU 迟迟起不来。
# 这里统一收紧线程数，保证 4 个 suite 可以并发启动。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export LIBERO_PACKAGE_ROOT
if [ -n "${POLICY_SEED}" ]; then
  export PYTHONHASHSEED="${POLICY_SEED}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# policy_seed 和环境 seed 分开，前者只负责固定 diffusion policy 的随机采样。
SERVER_CMD=(
  python scripts/serve_policy.py
  --env LIBERO
  --port "${PORT}"
)
if [ -n "${POLICY_SEED}" ]; then
  SERVER_CMD+=(--policy-seed "${POLICY_SEED}")
fi
SERVER_CMD+=(
  policy:checkpoint
  --policy.config="${CONFIG_NAME}"
  --policy.dir="${CKPT_DIR}"
)
"${SERVER_CMD[@]}" > "${RESULT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

export WAIT_PORT="${PORT}"
python - <<'PY'
import os
import socket
import time

host = "127.0.0.1"
port = int(os.environ["WAIT_PORT"])

def main():
    for _ in range(600):
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise SystemExit("server did not become ready in time")

main()
PY

CLIENT_CMD=(
  python "${EVAL_MAIN}"
  --args.host 127.0.0.1
  --args.port "${PORT}"
  --args.resize-size "${RESIZE_SIZE}"
  --args.replan-steps "${REPLAN_STEPS}"
  --args.task-suite-name "${SUITE}"
  --args.num-steps-wait "${NUM_STEPS_WAIT}"
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"
  --args.video-out-path "${RESULT_DIR}/videos"
  --args.seed "${SEED}"
)

# demo 阶段只跑前几个 task，避免一开始就把完整评测全压上。
if [ -n "${MAX_TASKS}" ] && [ "$(basename "${EVAL_MAIN}")" = "main_pro.py" ]; then
  CLIENT_CMD+=(--args.max-tasks "${MAX_TASKS}")
fi

# 开启 action feature 存储（SAVE_ACTION_FEATURES=1）
if [ -n "${SAVE_ACTION_FEATURES}" ]; then
  CLIENT_CMD+=(--args.save-action-features --args.feature-save-dir "${RESULT_DIR}/action_features")
fi

"${CLIENT_CMD[@]}" > "${RESULT_DIR}/client.log" 2>&1
