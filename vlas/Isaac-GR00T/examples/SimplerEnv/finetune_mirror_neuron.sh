#!/usr/bin/env bash

set -x -euo pipefail

ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
REPO="$ROOT/mirror_neuron/vlas/Isaac-GR00T"
BASE_MODEL="$ROOT/weights/gr00t/GR00T-N1.6-3B"
DATASET="$ROOT/data/simplerenv_bridge/bridge_orig_lerobot"
ALIGNMENT_ENV=/mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_gr00t_n16_alignment

ALIGNMENT_BACKEND="${ALIGNMENT_BACKEND:-qwen}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-1.0}"
ALIGNMENT_CONFIG_PATH="${ALIGNMENT_CONFIG_PATH:-}"
MAX_STEPS="${MAX_STEPS:-1000}"
STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
WATCHDOG_INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-30}"
WATCHDOG_STABLE_CHECKS="${WATCHDOG_STABLE_CHECKS:-3}"
WATCHDOG_MIN_FREE_GIB="${WATCHDOG_MIN_FREE_GIB:-300}"
VALIDATE_ALIGNMENT_DATA="${VALIDATE_ALIGNMENT_DATA:-1}"
RUN_TAG="${RUN_TAG:-${ALIGNMENT_BACKEND}_w${ALIGNMENT_WEIGHT}_steps${MAX_STEPS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/mirror_neuron/gr00t_runs}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_TAG"
RUN_DIR="$OUTPUT_DIR/mirror_neuron_${RUN_TAG}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$ROOT/runs/gr00t_bridge_mirror_neuron_archive/$RUN_TAG}"
WATCHDOG_SESSION="gr00t-ckpt-${RUN_TAG//./_}"

case "$ALIGNMENT_BACKEND" in
    qwen)
        DEFAULT_ALIGNMENT_CONFIG="$REPO/examples/SimplerEnv/mirror_neuron_alignment.json"
        ;;
    egohod)
        DEFAULT_ALIGNMENT_CONFIG="$REPO/examples/SimplerEnv/mirror_neuron_alignment_egohod.json"
        ;;
    *)
        echo "Unsupported ALIGNMENT_BACKEND: $ALIGNMENT_BACKEND" >&2
        exit 1
        ;;
esac
ALIGNMENT_CONFIG="${ALIGNMENT_CONFIG_PATH:-$DEFAULT_ALIGNMENT_CONFIG}"
if [ ! -f "$ALIGNMENT_CONFIG" ]; then
    echo "Alignment config does not exist: $ALIGNMENT_CONFIG" >&2
    exit 1
fi

if [ -z "$RESUME_CHECKPOINT" ] && [ -e "$OUTPUT_DIR" ]; then
    echo "Output directory already exists: $OUTPUT_DIR" >&2
    echo "Use a new RUN_TAG so each weight starts from the same base checkpoint." >&2
    exit 1
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ ! -d "$RESUME_CHECKPOINT" ]; then
        echo "Resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
        exit 1
    fi
    RESUME_ARGS=(--no-auto-resume --resume-from-checkpoint "$RESUME_CHECKPOINT")
else
    RESUME_ARGS=(--no-auto-resume)
fi
if [ -n "$STOP_AFTER_STEPS" ]; then
    STOP_ARGS=(--stop-after-steps "$STOP_AFTER_STEPS")
else
    STOP_ARGS=()
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export USE_WANDB="${USE_WANDB:-0}"
export MASTER_PORT="${MASTER_PORT:-29761}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PATH="$ALIGNMENT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ALIGNMENT_ENV/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_ROOT" "$ARCHIVE_ROOT"
rm -f "$RUN_DIR/.training_complete"
if [ -n "$RESUME_CHECKPOINT" ]; then
    for pattern in \
        "model*.safetensors" \
        "pytorch_model*.bin" \
        "model*.index.json" \
        "pytorch_model*.index.json" \
        "config.json" \
        "training_args.bin"; do
        for model_link in "$RUN_DIR"/$pattern; do
            if [ -L "$model_link" ]; then
                rm "$model_link"
            fi
        done
    done
fi
if [ "$VALIDATE_ALIGNMENT_DATA" = "1" ]; then
    /mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_gr00t_n16_alignment/bin/python \
        "$REPO/scripts/validate_bridge_alignment_data.py" \
        --dataset-root "$DATASET" \
        --task-meta "$ROOT/data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl" \
        --qwen-embedding "$ROOT/data/embedding/bridge_qwen3vl_text_features.npz" \
        --qwen-index "$ROOT/data/embedding/bridge_qwen3vl_text_index.json" \
        --egohod-embedding "$ROOT/data/embedding/bridge_egohod_proj_text_features.npz"
fi

if ! tmux has-session -t "$WATCHDOG_SESSION" 2>/dev/null; then
    tmux new-session -d -s "$WATCHDOG_SESSION" \
        env \
        CHECKPOINT_WATCHDOG_INTERVAL_SECONDS="$WATCHDOG_INTERVAL_SECONDS" \
        CHECKPOINT_WATCHDOG_STABLE_CHECKS="$WATCHDOG_STABLE_CHECKS" \
        CHECKPOINT_WATCHDOG_MIN_FREE_GIB="$WATCHDOG_MIN_FREE_GIB" \
        bash "$REPO/scripts/run_checkpoint_offload_watchdog.sh" "$RUN_DIR" "$ARCHIVE_ROOT"
fi

cd "$REPO"

if NUM_GPUS="$NUM_GPUS" \
    MAX_STEPS="$MAX_STEPS" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    SAVE_STEPS="$SAVE_STEPS" \
    SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
    LOGGING_STEPS="$LOGGING_STEPS" \
    DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
    bash examples/finetune.sh \
        --base-model-path "$BASE_MODEL" \
        --dataset-path "$DATASET" \
        --embodiment-tag OXE_WIDOWX \
        --output-dir "$OUTPUT_DIR" \
        --experiment-name "mirror_neuron_${RUN_TAG}" \
        --state-dropout-prob 0.8 \
        -- \
        --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
        "${RESUME_ARGS[@]}" \
        "${STOP_ARGS[@]}" \
        --use-alignment \
        --alignment-config-path "$ALIGNMENT_CONFIG" \
        --alignment-loss-weight "$ALIGNMENT_WEIGHT"; then
    COMPLETED_STEP="$(python - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
states = []
for path in run_dir.glob("checkpoint-*/trainer_state.json"):
    with path.open() as handle:
        states.append(int(json.load(handle)["global_step"]))
if not states:
    raise RuntimeError(f"no trainer_state.json found under {run_dir}")
print(max(states))
PY
)"
    printf '%s\n' "$COMPLETED_STEP" >"$RUN_DIR/.training_complete"
else
    exit "$?"
fi
