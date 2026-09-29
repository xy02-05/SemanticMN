#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <local-run-dir> <archive-run-dir>" >&2
    exit 2
fi

LOCAL_RUN_DIR="$1"
ARCHIVE_RUN_DIR="$2"
LOG_DIR=/mnt/bn/2d-videos/xy/work/mirror_neuron/logs/checkpoint_offload
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="$(basename "$ARCHIVE_RUN_DIR")"
INTERVAL_SECONDS="${CHECKPOINT_WATCHDOG_INTERVAL_SECONDS:-30}"
STABLE_CHECKS="${CHECKPOINT_WATCHDOG_STABLE_CHECKS:-3}"
MIN_FREE_GIB="${CHECKPOINT_WATCHDOG_MIN_FREE_GIB:-300}"

mkdir -p "$LOCAL_RUN_DIR" "$ARCHIVE_RUN_DIR" "$LOG_DIR"

exec /mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_gr00t_n16_alignment/bin/python \
    "$SCRIPT_DIR/checkpoint_offload_watchdog.py" \
    --local-root "$LOCAL_RUN_DIR" \
    --archive-root "$ARCHIVE_RUN_DIR" \
    --interval-seconds "$INTERVAL_SECONDS" \
    --stable-checks "$STABLE_CHECKS" \
    --min-free-gib "$MIN_FREE_GIB" \
    >>"$LOG_DIR/${RUN_NAME}.log" 2>&1
