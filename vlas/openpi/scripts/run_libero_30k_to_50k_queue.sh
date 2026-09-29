#!/usr/bin/env bash

set -euo pipefail

WORK_ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
REPO_ROOT="$WORK_ROOT/mirror_neuron/vlas/openpi"
LAUNCHER="$REPO_ROOT/scripts/train_libero_50k_pair.sh"
LOCAL_BASE="${OPENPI_LOCAL_RUN_BASE:-/tmp/openpi_50k_runs}"
ARCHIVE_BASE="${OPENPI_ARCHIVE_BASE:-$WORK_ROOT/runs/openpi_50k_archive}"
GPU_PAIR="${GPU_PAIR:-0,1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_SUFFIX="${RUN_SUFFIX:-continue_30k_50k_gbs32_seed42}"
LOG_FILE="$WORK_ROOT/logs/openpi_30k_to_50k_queue.log"

CONFIGS=(
    pi0_libero_full_qwen_continue_30k_50k_gbs32
    pi0_libero_full_continue_30k_50k_gbs32
)

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

wait_for_run() {
    local config_name=$1
    local session_name="openpi-formal-${config_name}"
    local local_root="$LOCAL_BASE/$config_name/$RUN_SUFFIX"
    local archive_root="$ARCHIVE_BASE/$config_name/$RUN_SUFFIX"

    while tmux has-session -t "$session_name" 2>/dev/null; do
        step=$(
            grep -aoE 'Training: +[0-9]+%[^[:cntrl:]]*' "$local_root/launcher.log" 2>/dev/null |
                tail -n 1 || true
        )
        log "$config_name running ${step:-step unavailable}"
        sleep "$POLL_SECONDS"
    done

    if grep -qE 'Traceback|ChildFailedError|CUDA out of memory|OutOfMemoryError' \
        "$local_root/launcher.log" 2>/dev/null; then
        log "ERROR $config_name ended with a fatal marker"
        return 1
    fi
    if [ ! -e "$local_root/50000" ]; then
        log "ERROR $config_name ended without local checkpoint 50000"
        return 1
    fi

    while [ ! -d "$archive_root/50000" ]; do
        log "$config_name waiting for checkpoint 50000 archive"
        sleep "$POLL_SECONDS"
    done
    log "$config_name completed and archived checkpoint 50000"
}

mkdir -p "$(dirname "$LOG_FILE")"
for config_name in "${CONFIGS[@]}"; do
    log "launching $config_name on GPU pair $GPU_PAIR"
    CONFIG_NAME="$config_name" \
    EXP_NAME="$RUN_SUFFIX" \
    GPU_PAIR="$GPU_PAIR" \
    WANDB_ENABLED=false \
        "$LAUNCHER" formal | tee -a "$LOG_FILE"
    wait_for_run "$config_name"
done

log "ALL_CONTINUATION_RUNS_COMPLETED"
