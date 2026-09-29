#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LIBERO-PRO 批量并行评测
#
# 在同一张 GPU 上同时 launch 多个 checkpoint 步数，
# 第1轮跑完后自动 launch 第2轮。
#
# 用法：
#   GPU_ID=0 CKPT_BASE=".../pi0_libero_full/raw" \
#     ROUND1="20000 25000" ROUND2="15000 10000" \
#     bash scripts/eval/launch_pro_batch.sh
# ============================================================

OPENPI_ROOT="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi"
CKPT_BASE="${CKPT_BASE:?需要 CKPT_BASE}"
GPU_ID="${GPU_ID:?需要 GPU_ID}"
ROUND1="${ROUND1:?需要 ROUND1，如 '20000 25000'}"
ROUND2="${ROUND2:-}"
PRO_SUFFIXES="${PRO_SUFFIXES:-lan object swap task env}"
SEED="${SEED:-1000}"
POLICY_SEED="${POLICY_SEED:-1000}"
PREFIX_BASE="${PREFIX_BASE:-g${GPU_ID}}"

# 检查 session 是否仍有 python 评测进程
sessions_busy() {
    local prefix="$1"
    local sessions
    sessions=$(tmux ls 2>/dev/null | grep "^${prefix}_" | cut -d: -f1 || true)
    [ -z "$sessions" ] && return 1
    for sess in $sessions; do
        local pid
        pid=$(tmux list-panes -t "$sess" -F '#{pane_pid}' 2>/dev/null || echo "")
        [ -z "$pid" ] && continue
        if ps --ppid "$pid" -o comm= 2>/dev/null | grep -q "python"; then
            return 0
        fi
        for cpid in $(ps --ppid "$pid" -o pid= 2>/dev/null); do
            if ps --ppid "$cpid" -o comm= 2>/dev/null | grep -q "python"; then
                return 0
            fi
        done
    done
    return 1
}

cleanup_sessions() {
    local prefix="$1"
    for sess in $(tmux ls 2>/dev/null | grep "^${prefix}_" | cut -d: -f1 || true); do
        tmux kill-session -t "$sess" 2>/dev/null || true
    done
}

launch_round() {
    local steps="$1"
    local prefixes=()

    for step in $steps; do
        local ckpt="${CKPT_BASE}/${step}"
        if [ ! -f "$ckpt/metadata.pt" ]; then
            echo "[SKIP] $ckpt 不存在"
            continue
        fi
        local pfx="${PREFIX_BASE}s${step}"
        prefixes+=("$pfx")

        echo "[$(date '+%H:%M:%S')] Launch step=${step} prefix=${pfx} on GPU ${GPU_ID}"
        GPU_ID="${GPU_ID}" \
        PRO_SUFFIXES="${PRO_SUFFIXES}" \
        CKPT_DIR="${ckpt}" \
        SEED="${SEED}" \
        POLICY_SEED="${POLICY_SEED}" \
        SESSION_PREFIX="${pfx}" \
        bash "${OPENPI_ROOT}/scripts/eval/launch_pro_eval.sh"
        # 错开几秒避免端口冲突
        sleep 3
    done

    # 等待本轮所有 session 完成
    echo "[$(date '+%H:%M:%S')] 等待本轮完成: ${prefixes[*]}"
    sleep 60
    while true; do
        local any_busy=0
        for pfx in "${prefixes[@]}"; do
            if sessions_busy "$pfx"; then
                any_busy=1
                break
            fi
        done
        if [ "$any_busy" -eq 0 ]; then
            break
        fi
        sleep 60
    done
    echo "[$(date '+%H:%M:%S')] 本轮完成，清理 sessions"
    for pfx in "${prefixes[@]}"; do
        cleanup_sessions "$pfx"
    done
    sleep 5
}

echo "========================================"
echo "LIBERO-PRO 批量并行 (GPU ${GPU_ID})"
echo "  CKPT_BASE: ${CKPT_BASE}"
echo "  ROUND1: ${ROUND1}"
echo "  ROUND2: ${ROUND2:-无}"
echo "========================================"

launch_round "${ROUND1}"

if [ -n "${ROUND2}" ]; then
    echo ""
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] 开始第2轮"
    echo "========================================"
    launch_round "${ROUND2}"
fi

echo ""
echo "========================================"
echo "[$(date '+%H:%M:%S')] GPU ${GPU_ID} 全部完成！"
echo "========================================"
