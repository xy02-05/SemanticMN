#!/usr/bin/env bash

set -euo pipefail

SESSION_REGEX="${1:?usage: $0 SESSION_REGEX GPU_CSV [POLL_SECONDS]}"
GPU_CSV="${2:?usage: $0 SESSION_REGEX GPU_CSV [POLL_SECONDS]}"
POLL_SECONDS="${3:-15}"
PYTHON_BIN="${FILL_PYTHON:-/mnt/bn/2d-videos/xy/tools/miniconda3/envs/joyai-interaction/bin/python}"
BURN_SCRIPT="${FILL_SCRIPT:-/mnt/bn/2d-videos/xy/tools/gpus/torch_tensor_burn.py}"

while tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Eq "$SESSION_REGEX"; do
    sleep "$POLL_SECONDS"
done

IFS=',' read -r -a gpu_ids <<<"$GPU_CSV"
for gpu in "${gpu_ids[@]}"; do
    session="mn-posttrain-fill-gpu${gpu}"
    memory_used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [ "$memory_used" -gt 1024 ]; then
        echo "GPU $gpu is already occupied (${memory_used} MiB); leaving it unchanged"
        continue
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
        continue
    fi
    tmux new-session -d -s "$session" \
        "env CUDA_VISIBLE_DEVICES=$gpu $PYTHON_BIN $BURN_SCRIPT --dtype tf32 --matrix-size 8192 --streams 2 --memory-fraction 0 --memory-chunk-mib 512 --log-seconds 10 --heartbeat-file /tmp/${session}.json --heartbeat-seconds 2 --stall-seconds 120"
done
