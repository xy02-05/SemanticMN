#!/usr/bin/env bash

set -euo pipefail

WORK_ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
REPO_ROOT="$WORK_ROOT/mirror_neuron/vlas/SpatialVLA"
CONFIG_ROOT="$WORK_ROOT/mirror_neuron/egovlpv2/egovlpv2/configs/ft"
TRAIN_SCRIPT="$REPO_ROOT/scripts/spatialvla_4b_finetune/train_bridge_embedding_experiment.sh"
LOCAL_BASE=/opt/tiger/rh2/rh2/init/spatialvla_bridge_runs
ARCHIVE_BASE="$WORK_ROOT/runs/spatialvla_bridge_archive"

STAMP="${RUN_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
NO_DIFF_NAME="merlin_egohod_dsn_no_diff_${STAMP}"
QWEN_NAME="merlin_qwen_dsn_full_${STAMP}"
HOST_TAG=$(hostname | tr -c '[:alnum:]_.-' '_')

mkdir -p "$LOCAL_BASE"

for session in spatialvla-merlin-no-diff spatialvla-merlin-qwen spatialvla-merlin-monitor; do
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 1
    fi
done

tmux new-session -d -s spatialvla-merlin-no-diff \
    "exec bash '$TRAIN_SCRIPT' \
        --run-name '$NO_DIFF_NAME' \
        --alignment-config '$CONFIG_ROOT/spatialvla_bridge_egohod_infonce_dsn_no_diff.json' \
        --cuda-devices '0,1' \
        --master-port 29641"

tmux new-session -d -s spatialvla-merlin-qwen \
    "exec bash '$TRAIN_SCRIPT' \
        --run-name '$QWEN_NAME' \
        --alignment-config '$CONFIG_ROOT/spatialvla_bridge_qwen3vl_infonce_dsn_full.json' \
        --cuda-devices '2,3' \
        --master-port 29642"

tmux new-session -d -s spatialvla-merlin-monitor \
    "exec /mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_spatialvla/bin/python \
        '$REPO_ROOT/scripts/monitor_bridge_training.py' \
        --run-root '$LOCAL_BASE/$NO_DIFF_NAME' \
        --archive-root '$ARCHIVE_BASE/$HOST_TAG/$NO_DIFF_NAME' \
        --run-root '$LOCAL_BASE/$QWEN_NAME' \
        --archive-root '$ARCHIVE_BASE/$HOST_TAG/$QWEN_NAME' \
        --target-step 1500 \
        --checkpoint-step 1000 \
        --poll-interval 30 \
        --status-file '$LOCAL_BASE/merlin_pair_${STAMP}_status.json' \
        2>&1 | tee '$LOCAL_BASE/merlin_pair_${STAMP}_monitor.log'"

cat <<EOF
no_diff_run=$NO_DIFF_NAME
qwen_run=$QWEN_NAME
status_file=$LOCAL_BASE/merlin_pair_${STAMP}_status.json
EOF
