#!/usr/bin/env bash

set -euo pipefail

CODE_ROOT="${MIRROR_NEURON_CODE_ROOT:-/mnt/bn/2d-videos/xy/work/mirror_neuron_final}"
WORK_ROOT="${MIRROR_NEURON_WORK_ROOT:-/mnt/bn/2d-videos/xy/work/mirror_neuron}"
LOCAL_BASE="${OPENPI_LOCAL_RUN_BASE:-/tmp/mirror_neuron_repro/openpi_bridge_gbs128fix_8gpu_smoke}"
ARCHIVE_BASE="${OPENPI_ARCHIVE_BASE:-$WORK_ROOT/runs/reproduction/openpi_bridge_gbs128fix_8gpu_smoke}"
TARGET_SESSION_REGEX='^mn-svla-s[1-4]_'
FILL_PYTHON=/mnt/bn/2d-videos/xy/tools/miniconda3/envs/joyai-interaction/bin/python
BURN_SCRIPT=/mnt/bn/2d-videos/xy/tools/gpus/torch_tensor_burn.py

restore_fill() {
    for gpu in 0 1 2 3 4 5 6 7; do
        session="mn-posttrain-fill-gpu${gpu}"
        memory_used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        if [ "$memory_used" -le 1024 ] && ! tmux has-session -t "$session" 2>/dev/null; then
            tmux new-session -d -s "$session" \
                "env CUDA_VISIBLE_DEVICES=$gpu $FILL_PYTHON $BURN_SCRIPT --dtype tf32 --matrix-size 8192 --streams 2 --memory-fraction 0 --memory-chunk-mib 512 --log-seconds 10 --heartbeat-file /tmp/${session}.json --heartbeat-seconds 2 --stall-seconds 120"
        fi
    done
}
trap restore_fill EXIT

while tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Eq "$TARGET_SESSION_REGEX"; do
    sleep 15
done

# The post-training guard may have filled the cards first. Only stop sessions
# created by our own guard; unrelated processes are never touched.
for gpu in 0 1 2 3 4 5 6 7; do
    tmux kill-session -t "mn-posttrain-fill-gpu${gpu}" 2>/dev/null || true
done
sleep 10

for gpu in 0 1 2 3 4 5 6 7; do
    memory_used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [ "$memory_used" -gt 1024 ]; then
        echo "GPU $gpu is occupied after SpatialVLA completion (${memory_used} MiB); aborting smoke" >&2
        exit 1
    fi
done

run_smoke() {
    local config="$1"
    local experiment="$2"
    local port="$3"
    MODE=smoke \
    NUM_GPUS=8 \
    CUDA_DEVICES=0,1,2,3,4,5,6,7 \
    MIRROR_NEURON_WORK_ROOT="$WORK_ROOT" \
    OPENPI_LOCAL_RUN_BASE="$LOCAL_BASE" \
    OPENPI_ARCHIVE_BASE="$ARCHIVE_BASE" \
        bash "$CODE_ROOT/vlas/openpi/scripts/train_bridge_8gpu.sh" \
        "$config" "$experiment" "$port"
}

run_smoke \
    bridgev2_egohod_infonce_dsn_prepool_h512_8gpu \
    prepool_h512_gbs128fix_8gpu_smoke_seed42 \
    30451
run_smoke \
    bridgev2_egohod_infonce_dsn_postpool_h1024_8gpu \
    postpool_h1024_gbs128fix_8gpu_smoke_seed42 \
    30452

echo "BATCHFIX_8GPU_SMOKES_PASS"
