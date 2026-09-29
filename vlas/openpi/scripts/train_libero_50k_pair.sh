#!/usr/bin/env bash

set -euo pipefail

WORK_ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
REPO_ROOT="$WORK_ROOT/mirror_neuron/vlas/openpi"
ENV_ROOT="${OPENPI_ENV_ROOT:-/mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_openpi}"
LOCAL_BASE="${OPENPI_LOCAL_RUN_BASE:-/tmp/openpi_50k_runs}"
ARCHIVE_BASE="${OPENPI_ARCHIVE_BASE:-$WORK_ROOT/runs/openpi_50k_archive}"
TRAINING_LOCK="${OPENPI_50K_TRAINING_LOCK:-/tmp/openpi_50k_training.lock}"
MODE="${1:-formal}"
GPU_PAIR="${GPU_PAIR:-0,1}"
MASTER_PORT="${MASTER_PORT:-29671}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
case "$WANDB_ENABLED" in
    true|1)
        WANDB_CLI_FLAG=--wandb-enabled
        ;;
    false|0)
        WANDB_CLI_FLAG=--no-wandb-enabled
        ;;
    *)
        echo "WANDB_ENABLED must be true/false or 1/0" >&2
        exit 2
        ;;
esac

case "$MODE" in
    smoke)
        MAX_STEPS="${MAX_STEPS:-100}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
        RUN_SUFFIX="${RUN_SUFFIX:-smoke_$(date '+%Y%m%d_%H%M%S')}"
        ;;
    formal)
        MAX_STEPS="${MAX_STEPS:-50000}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
        RUN_SUFFIX="${RUN_SUFFIX:-50k_gbs32_seed42}"
        ;;
    *)
        echo "usage: $0 [smoke|formal]" >&2
        exit 2
        ;;
esac

mkdir -p "$LOCAL_BASE" "$ARCHIVE_BASE"
export CUDA_VISIBLE_DEVICES="$GPU_PAIR"
export PYTHONPATH="$REPO_ROOT/src:$WORK_ROOT/mirror_neuron/egovlpv2:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/tmp/openpi_hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/tmp/openpi_lerobot_cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$WORK_ROOT/cache/openpi}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

launch_run() {
    local config_name=$1
    local exp_name=$2
    local local_root="$LOCAL_BASE/$config_name/$exp_name"
    local archive_root="$ARCHIVE_BASE/$config_name/$exp_name"
    local train_session="openpi-${MODE}-${config_name}"
    local watcher_session="openpi-watch-${MODE}-${config_name}"
    local train_log="$local_root/launcher.log"
    local watcher_log="$local_root/offload_watcher.log"

    if tmux has-session -t "$train_session" 2>/dev/null; then
        echo "tmux session exists: $train_session" >&2
        exit 1
    fi
    if tmux has-session -t "$watcher_session" 2>/dev/null; then
        echo "tmux session exists: $watcher_session" >&2
        exit 1
    fi
    if [ -e "$local_root" ]; then
        echo "local run exists: $local_root" >&2
        exit 1
    fi

    # 正式训练期间保留共享锁，避免特征提取任务并发抢占同一组 GPU。
    printf '%s config=%s gpu_pair=%s\n' \
        "$(date --iso-8601=seconds)" "$config_name" "$GPU_PAIR" >>"$TRAINING_LOCK"
    mkdir -p "$local_root" "$archive_root"
    if [ -n "${SOURCE_CHECKPOINT:-}" ]; then
        if [ ! -d "$SOURCE_CHECKPOINT" ]; then
            echo "source checkpoint is missing: $SOURCE_CHECKPOINT" >&2
            exit 1
        fi
        ln -s "$SOURCE_CHECKPOINT" "$local_root/30000"
    fi
    cp "$0" "$local_root/launch_script.sh"
    cat >"$local_root/run_manifest.json" <<EOF
{
  "mode": "$MODE",
  "config_name": "$config_name",
  "exp_name": "$exp_name",
  "gpu_pair": "$GPU_PAIR",
  "master_port": $MASTER_PORT,
  "global_batch_size": 32,
  "per_device_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "max_steps": $MAX_STEPS,
  "save_interval": $SAVE_INTERVAL,
  "training_lock": "$TRAINING_LOCK",
  "local_root": "$local_root",
  "archive_root": "$archive_root"
}
EOF
    cp "$local_root/run_manifest.json" "$archive_root/run_manifest.json"

    tmux new-session -d -s "$watcher_session" \
        "exec '$ENV_ROOT/bin/python' '$REPO_ROOT/scripts/checkpoint_offload_watcher.py' \
            --local-root '$local_root' \
            --archive-root '$archive_root' \
            --interval-seconds 10 \
            --stable-checks 3 \
            --min-free-gib 200 \
            2>&1 | tee -a '$watcher_log'"

    tmux new-session -d -s "$train_session" \
        "cd '$REPO_ROOT' && \
         export CUDA_VISIBLE_DEVICES='$GPU_PAIR' PYTHONPATH='$PYTHONPATH' \
                HF_HOME='$HF_HOME' HF_LEROBOT_HOME='$HF_LEROBOT_HOME' \
                OPENPI_DATA_HOME='$OPENPI_DATA_HOME' \
                PYTHONNOUSERSITE=1 \
                XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
         exec '$ENV_ROOT/bin/torchrun' \
            --standalone --nnodes=1 --nproc-per-node=2 --master-port='$MASTER_PORT' \
            scripts/train_pytorch_xy.py '$config_name' \
            --exp-name '$exp_name' \
            --checkpoint-base-dir '$LOCAL_BASE' \
            --num-train-steps '$MAX_STEPS' \
            --save-interval '$SAVE_INTERVAL' \
            '$WANDB_CLI_FLAG' \
            2>&1 | tee -a '$train_log'"

    echo "train_session=$train_session"
    echo "watcher_session=$watcher_session"
    echo "local_root=$local_root"
    echo "archive_root=$archive_root"
}

CONFIG_NAME="${CONFIG_NAME:-}"
if [ -z "$CONFIG_NAME" ]; then
    echo "CONFIG_NAME must select one supported LIBERO training config" >&2
    exit 2
fi
case "$CONFIG_NAME" in
    pi0_libero_full_50k_gbs32|pi0_libero_full_qwen_50k_gbs32|\
    pi0_libero_full_qwen_recon001_prepool_50k_gbs32|\
    pi0_libero_full_qwen_recon001_postpool_50k_gbs32|\
    pi0_libero_full_qwen_dsn_full_30k_gbs32|\
    pi0_libero_full_qwen_dsn_no_recon_30k_gbs32|\
    pi0_libero_full_qwen_dsn_no_diff_30k_gbs32|\
    pi0_libero_full_continue_30k_50k_gbs32|\
    pi0_libero_full_qwen_continue_30k_50k_gbs32) ;;
    *)
        echo "unsupported CONFIG_NAME: $CONFIG_NAME" >&2
        exit 2
        ;;
esac

case "$CONFIG_NAME" in
    pi0_libero_full_continue_30k_50k_gbs32)
        SOURCE_CHECKPOINT="$WORK_ROOT/mirror_neuron/vlas/openpi/checkpoints/pi0_libero_full/raw/30000"
        ;;
    pi0_libero_full_qwen_continue_30k_50k_gbs32)
        SOURCE_CHECKPOINT="$WORK_ROOT/mirror_neuron/vlas/openpi/checkpoints/pi0_libero_full_qwen/new_DSN_new/30000"
        ;;
    *)
        SOURCE_CHECKPOINT=
        ;;
esac

if [ "$MODE" = smoke ] && [ -n "$SOURCE_CHECKPOINT" ] && [ "$MAX_STEPS" -eq 100 ]; then
    MAX_STEPS=30100
fi

launch_run "$CONFIG_NAME" "${EXP_NAME:-$RUN_SUFFIX}"
