#!/usr/bin/env bash

set -euo pipefail

WORK_ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
REPO_ROOT="$WORK_ROOT/mirror_neuron/vlas/SpatialVLA"
EGO_ROOT="$WORK_ROOT/mirror_neuron/egovlpv2"
ENV_ROOT="${SPATIALVLA_ENV_ROOT:-/mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_spatialvla_h20}"
PYTHON="$ENV_ROOT/bin/python"
TORCHRUN="$ENV_ROOT/bin/torchrun"
MODEL_PATH="$WORK_ROOT/weights/spatialvla/spatialvla-4b-224-pt"
DATA_ROOT="$WORK_ROOT/data/spatialvla_bridge"
ARCHIVE_BASE="$WORK_ROOT/runs/spatialvla_bridge_archive"
LOCAL_BASE="${SPATIALVLA_LOCAL_BASE:-/opt/tiger/rh2/rh2/init/spatialvla_bridge_runs}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-12}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
FIX_RAW_LENGTH="${FIX_RAW_LENGTH:-}"
ALIGNMENT_LOSS_WEIGHT="${ALIGNMENT_LOSS_WEIGHT:-1.0}"

RUN_NAME=
ALIGNMENT_CONFIG=
CUDA_DEVICES=
MASTER_PORT=

usage() {
    cat <<'EOF'
Usage:
  train_bridge_embedding_experiment.sh \
    --run-name <name> \
    --alignment-config <json> \
    --cuda-devices <0,1> \
    --master-port <port>
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --alignment-config)
            ALIGNMENT_CONFIG="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for variable in RUN_NAME ALIGNMENT_CONFIG CUDA_DEVICES MASTER_PORT; do
    if [ -z "${!variable}" ]; then
        echo "Missing required argument: $variable" >&2
        exit 2
    fi
done

# 按可见卡列表启动等量训练进程，兼容两卡和四卡实验。
NUM_GPUS="$(awk -F, '{print NF}' <<<"$CUDA_DEVICES")"

if [ ! -f "$ALIGNMENT_CONFIG" ]; then
    echo "Alignment config does not exist: $ALIGNMENT_CONFIG" >&2
    exit 1
fi
ALIGNMENT_CONFIG="$(realpath "$ALIGNMENT_CONFIG")"
if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo "SpatialVLA pretrained weight is incomplete: $MODEL_PATH" >&2
    exit 1
fi
if [ ! -f "$DATA_ROOT/bridge_orig/1.0.0/dataset_info.json" ]; then
    echo "Bridge RLDS dataset is incomplete: $DATA_ROOT/bridge_orig/1.0.0" >&2
    exit 1
fi
if [ ! -f "$DATA_ROOT/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl" ]; then
    echo "Bridge task mapping is missing" >&2
    exit 1
fi

HOST_TAG=$(hostname | tr -c '[:alnum:]_.-' '_')
LOCAL_RUN_DIR="$LOCAL_BASE/$RUN_NAME"
ARCHIVE_RUN_DIR="$ARCHIVE_BASE/$HOST_TAG/$RUN_NAME"
STOP_FILE="$LOCAL_RUN_DIR/.stop_archive"
ARCHIVE_LOG="$LOCAL_RUN_DIR/archive_watcher.log"
TRAIN_LOG="$LOCAL_RUN_DIR/training.log"

if [ -e "$LOCAL_RUN_DIR" ]; then
    echo "Local output already exists: $LOCAL_RUN_DIR" >&2
    exit 1
fi

mkdir -p "$LOCAL_RUN_DIR" "$ARCHIVE_RUN_DIR"
cp "$ALIGNMENT_CONFIG" "$LOCAL_RUN_DIR/alignment_config.json"
cp "$0" "$LOCAL_RUN_DIR/launch_script.sh"

cat >"$LOCAL_RUN_DIR/run_manifest.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "host": "$(hostname)",
  "cuda_devices": "$CUDA_DEVICES",
  "num_gpus": $NUM_GPUS,
  "master_port": $MASTER_PORT,
  "model_path": "$MODEL_PATH",
  "data_root": "$DATA_ROOT",
  "alignment_config": "$ALIGNMENT_CONFIG",
  "local_run_dir": "$LOCAL_RUN_DIR",
  "archive_run_dir": "$ARCHIVE_RUN_DIR",
  "alignment_loss_weight": $ALIGNMENT_LOSS_WEIGHT,
  "per_device_batch_size": $PER_DEVICE_BATCH_SIZE,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "num_train_epochs": $NUM_TRAIN_EPOCHS,
  "save_steps": $SAVE_STEPS,
  "seed": 42
}
EOF
cp "$LOCAL_RUN_DIR/run_manifest.json" "$ARCHIVE_RUN_DIR/run_manifest.json"
cp "$LOCAL_RUN_DIR/alignment_config.json" "$ARCHIVE_RUN_DIR/alignment_config.json"
echo "$$" >"$LOCAL_RUN_DIR/train.pid"

nohup "$PYTHON" "$REPO_ROOT/scripts/checkpoint_archive_watcher.py" \
    --local-root "$LOCAL_RUN_DIR" \
    --archive-root "$ARCHIVE_RUN_DIR" \
    --poll-interval 10 \
    --stable-polls 2 \
    --stable-interval 10 \
    --stop-file "$STOP_FILE" \
    >"$ARCHIVE_LOG" 2>&1 &
echo "$!" >"$LOCAL_RUN_DIR/archive_watcher.pid"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export MASTER_PORT
export PYTHONWARNINGS="ignore::DeprecationWarning"
GL_ROOT=/mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_gr00t_n16/lib
export LD_PRELOAD="$GL_ROOT/libGLdispatch.so.0:$GL_ROOT/libXau.so.6:$GL_ROOT/libXdmcp.so.6:$GL_ROOT/libxcb.so.1:$GL_ROOT/libX11.so.6:$GL_ROOT/libGLX.so.0:$GL_ROOT/libGL.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12/bin:$ENV_ROOT/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12
export PYTHONPATH="$REPO_ROOT:$EGO_ROOT:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TOKENIZERS_PARALLELISM=true
export LAUNCHER=pytorch
export OMP_NUM_THREADS=1
# 当前 H20 驱动上的 cuBLASLt addmm heuristic 会触发 SIGFPE；
# 仅让带 bias 的 Linear 回退到传统 cuBLAS，不改变模型与数值定义。
export DISABLE_ADDMM_CUDA_LT=1

cd "$REPO_ROOT"

EXTRA_DATA_ARGS=()
if [ -n "$FIX_RAW_LENGTH" ]; then
    EXTRA_DATA_ARGS+=(--fix_raw_length "$FIX_RAW_LENGTH")
fi

{
    echo "run_name=$RUN_NAME"
    echo "host=$(hostname)"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "local_run_dir=$LOCAL_RUN_DIR"
    echo "archive_run_dir=$ARCHIVE_RUN_DIR"
    echo "alignment_config=$ALIGNMENT_CONFIG"
    echo "started_at=$(date -Is)"
} | tee "$TRAIN_LOG"

exec "$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$NUM_GPUS" \
    --master-port "$MASTER_PORT" \
    --module train.spatialvla_finetune_align_v2 \
    --model_name_or_path "$MODEL_PATH" \
    --lora 32 \
    --lora_alpha 32 \
    --lora_target linear \
    --ignore_data_skip True \
    --data_root_dir "$DATA_ROOT" \
    --data_mix bridge_orig \
    --task_filename tasks_with_id.jsonl \
    "${EXTRA_DATA_ARGS[@]}" \
    --shuffle_buffer_size 8192 \
    --tsfm_thread_muti 12 \
    --read_thread_muti 12 \
    --obs_backward_steps 0 \
    --obs_backward_delta 1 \
    --action_forward_steps 3 \
    --freeze_egovlpv2_model true \
    --flash_attn False \
    --output_dir "$LOCAL_RUN_DIR" \
    --overwrite_output_dir False \
    --freeze_vision_tower False \
    --dataloader_num_workers 2 \
    --bf16 True \
    --tf32 True \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 3 \
    --learning_rate 0.0001 \
    --weight_decay 0.0 \
    --warmup_ratio 0.005 \
    --lr_scheduler_type linear \
    --logging_steps "$LOGGING_STEPS" \
    --max_grad_norm 1.0 \
    --seed 42 \
    --do_train True \
    --deepspeed scripts/zero1.json \
    --grad_checkpoint True \
    --report_to tensorboard \
    --log_level warning \
    --egovlpv2_config_path "$ALIGNMENT_CONFIG" \
    --vlm_loss_weight 0 \
    --alignment_loss_weight "$ALIGNMENT_LOSS_WEIGHT" \
    --use_egovlpv2 false \
    --use_alignment true \
    --vlm_mode embedding \
    > >(tee -a "$TRAIN_LOG") 2>&1
