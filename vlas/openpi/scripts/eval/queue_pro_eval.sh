#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LIBERO-PRO 多步数排队脚本
#
# 先等待 WAIT_TAG 对应的 30000 session 跑完，再依次 launch 剩余步数。
# 每个步数跑完后自动启动下一个，同一 GPU 上串行。
#
# 用法：
#   GPU_ID=0 CKPT_BASE=".../pi0_libero_full/raw" \
#     WAIT_TAG="20260417_031212" STEPS="20000 25000 15000 10000" \
#     nohup bash scripts/eval/queue_pro_eval.sh &
# ============================================================

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
CKPT_BASE="${CKPT_BASE:?需要设置 CKPT_BASE}"
GPU_ID="${GPU_ID:?需要设置 GPU_ID}"
STEPS="${STEPS:?需要设置 STEPS}"
WAIT_TAG="${WAIT_TAG:-}"
WAIT_PREFIX="${WAIT_PREFIX:-${SESSION_PREFIX}}"
PRO_SUFFIXES="${PRO_SUFFIXES:-lan object swap task env}"
SEED="${SEED:-1000}"
POLICY_SEED="${POLICY_SEED:-1000}"
SESSION_PREFIX="${SESSION_PREFIX:-pro${GPU_ID}}"

# 判断一组 PREFIX_TAG_* session 是否还有评测任务在跑
sessions_busy() {
    local tag="$1"
    local prefix="${2:-${SESSION_PREFIX}}"
    local sessions
    sessions=$(tmux ls 2>/dev/null | grep "^${prefix}_${tag}_" | cut -d: -f1 || true)
    [ -z "$sessions" ] && return 1

    for sess in $sessions; do
        local pid
        pid=$(tmux list-panes -t "$sess" -F '#{pane_pid}' 2>/dev/null || echo "")
        [ -z "$pid" ] && continue
        # 检查该 pane 进程是否有 python 子进程（server 或 client）
        if ps --ppid "$pid" -o comm= 2>/dev/null | grep -q "python"; then
            return 0
        fi
        # 也检查 pane 进程本身是否是 bash 在执行脚本（非 sleep）
        local child_cmds
        child_cmds=$(ps --ppid "$pid" -o comm= 2>/dev/null | tr '\n' ' ')
        # 如果有 bash 子进程（在执行 run_eval_libero_single_suite.sh），继续追踪
        if echo "$child_cmds" | grep -q "bash"; then
            for cpid in $(ps --ppid "$pid" -o pid= 2>/dev/null); do
                if ps --ppid "$cpid" -o comm= 2>/dev/null | grep -q "python"; then
                    return 0
                fi
            done
        fi
    done
    return 1
}

# 等待指定 tag 的 session 全部完成，然后清理
wait_and_cleanup() {
    local tag="$1"
    local prefix="${2:-${SESSION_PREFIX}}"
    echo "[$(date '+%H:%M:%S')] 等待 ${prefix}_${tag}_* 完成..."
    # 先等 30 秒让 server 启动，避免在进程还没 spawn 时误判为完成
    sleep 30
    while sessions_busy "$tag" "$prefix"; do
        sleep 60
    done
    echo "[$(date '+%H:%M:%S')] ${prefix}_${tag}_* 已完成，清理 session"
    for sess in $(tmux ls 2>/dev/null | grep "^${prefix}_${tag}_" | cut -d: -f1 || true); do
        tmux kill-session -t "$sess" 2>/dev/null || true
    done
    sleep 5
}

echo "========================================"
echo "LIBERO-PRO 多步数排队 (GPU ${GPU_ID})"
echo "  CKPT_BASE: ${CKPT_BASE}"
echo "  WAIT_TAG:  ${WAIT_TAG:-无}"
echo "  STEPS:     ${STEPS}"
echo "========================================"

# 先等当前 30000 跑完（可能使用不同的 prefix）
if [ -n "$WAIT_TAG" ]; then
    wait_and_cleanup "$WAIT_TAG" "$WAIT_PREFIX"
fi

for step in ${STEPS}; do
    ckpt_dir="${CKPT_BASE}/${step}"
    if [ ! -d "$ckpt_dir" ] || [ ! -f "$ckpt_dir/metadata.pt" ]; then
        echo "[SKIP] checkpoint 不存在: $ckpt_dir"
        continue
    fi

    echo ""
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] Launch step=${step}"
    echo "========================================"

    GPU_ID="${GPU_ID}" \
    PRO_SUFFIXES="${PRO_SUFFIXES}" \
    CKPT_DIR="${ckpt_dir}" \
    SEED="${SEED}" \
    POLICY_SEED="${POLICY_SEED}" \
    SESSION_PREFIX="${SESSION_PREFIX}" \
    bash "${OPENPI_ROOT}/scripts/eval/launch_pro_eval.sh"

    # 提取刚创建的 RUN_TAG（按 SESSION_PREFIX 过滤）
    new_tag=$(tmux ls 2>/dev/null | grep "^${SESSION_PREFIX}_" | sed "s/^${SESSION_PREFIX}_\([0-9_]*\)_.*/\1/" | sort -u | tail -1 || echo "")
    if [ -n "$new_tag" ]; then
        wait_and_cleanup "$new_tag"
    else
        echo "[ERROR] 未找到新创建的 session"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "[$(date '+%H:%M:%S')] GPU ${GPU_ID} 所有步数评测完成！"
echo "========================================"
