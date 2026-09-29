#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CODE_ROOT="$(cd -- "$OPENPI_ROOT/../.." && pwd)"
WORK_ROOT="${MIRROR_NEURON_WORK_ROOT:-$(dirname "$CODE_ROOT")/mirror_neuron}"
ENV_ROOT="${OPENPI_ENV_ROOT:-/mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_openpi}"
LOCAL_BASE="${OPENPI_LOCAL_RUN_BASE:-/tmp/mirror_neuron_repro/openpi_bridge}"
ARCHIVE_BASE="${OPENPI_ARCHIVE_BASE:-$WORK_ROOT/runs/reproduction/openpi_bridge}"
CONFIG_NAME="${1:?usage: $0 CONFIG_NAME EXP_NAME [MASTER_PORT]}"
EXP_NAME="${2:?usage: $0 CONFIG_NAME EXP_NAME [MASTER_PORT]}"
MASTER_PORT="${3:-30310}"
NUM_GPUS=8
MODE="${MODE:-formal}"

case "$MODE" in
    smoke)
        NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
        ;;
    formal)
        NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
        ;;
    *)
        echo "MODE must be smoke or formal, got: $MODE" >&2
        exit 2
        ;;
esac

case "$CONFIG_NAME" in
    bridgev2_egohod_infonce_dsn_prepool_h512_8gpu|\
    bridgev2_egohod_infonce_dsn_postpool_h1024_8gpu) ;;
    *)
        echo "unsupported Bridge reproduction config: $CONFIG_NAME" >&2
        exit 2
        ;;
esac

for path in \
    "$ENV_ROOT/bin/torchrun" \
    "$WORK_ROOT/data/bridge-rlds/bridge_orig/1.0.0/dataset_info.json" \
    "$WORK_ROOT/data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl" \
    "$WORK_ROOT/weights/openpi/pi0_base_pytorch/model.safetensors"; do
    if [ ! -e "$path" ]; then
        echo "required path is missing: $path" >&2
        exit 1
    fi
done

local_root="$LOCAL_BASE/$CONFIG_NAME/$EXP_NAME"
archive_root="$ARCHIVE_BASE/$CONFIG_NAME/$EXP_NAME"
if [ -e "$local_root" ]; then
    echo "local run already exists: $local_root" >&2
    exit 1
fi
mkdir -p "$local_root" "$archive_root"+

cat >"$local_root/run_manifest.json" <<EOF
{
  "config_name": "$CONFIG_NAME",
  "exp_name": "$EXP_NAME",
  "num_gpus": $NUM_GPUS,
  "global_batch_size": 128,
  "per_device_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "mode": "$MODE",
  "num_train_steps": $NUM_TRAIN_STEPS,
  "save_interval": $SAVE_INTERVAL,
  "code_root": "$CODE_ROOT",
  "work_root": "$WORK_ROOT",
  "local_root": "$local_root",
  "archive_root": "$archive_root"
}
EOF
cp "$local_root/run_manifest.json" "$archive_root/run_manifest.json"
cp "$0" "$archive_root/launch_script.sh"

"$ENV_ROOT/bin/python" "$OPENPI_ROOT/scripts/checkpoint_offload_watcher.py" \
    --local-root "$local_root" \
    --archive-root "$archive_root" \
    --interval-seconds 10 \
    --stable-checks 3 \
    --min-free-gib 200 \
    >"$local_root/offload_watcher.log" 2>&1 &
echo "$!" >"$local_root/offload_watcher.pid"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$OPENPI_ROOT/src:$CODE_ROOT/egovlpv2:${PYTHONPATH:-}"
export MIRROR_NEURON_WORK_ROOT="$WORK_ROOT"
export MIRROR_NEURON_CODE_ROOT="$CODE_ROOT"
export HF_HOME="${HF_HOME:-/tmp/mirror_neuron_repro/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/tmp/mirror_neuron_repro/lerobot_cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$WORK_ROOT/cache/openpi}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$OPENPI_ROOT"
exec "$ENV_ROOT/bin/torchrun" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$NUM_GPUS" \
    --master-port="$MASTER_PORT" \
    scripts/train_pytorch_xy.py "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    --checkpoint-base-dir "$LOCAL_BASE" \
    --gradient-accumulation-steps 1 \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --no-wandb-enabled \
    2>&1 | tee "$local_root/training.log"
