set -x

DEBUG=true
if [ "$DEBUG" = true ]; then
  GPUS=1
  GPUS_PER_NODE=1
  PER_DEVICE_BATCH_SIZE=2
  shuffle_buffer_size=2
  mixture=bridge_orig
  NUM_WORKERS=0
  TORCH_RUN_ARGS="--standalone --nnodes=1"
  save_steps=50
fi

GPUS=${GPUS:-48}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-$((GPUS * PER_DEVICE_BATCH_SIZE))}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

mixture=${mixture:-oxe_spatial_vla_plus}
NUM_WORKERS=${NUM_WORKERS:-1}
shuffle_buffer_size=${shuffle_buffer_size:-8192} # large buffer for better shuffling, we use 65536 in pretrain

lr=2e-5
save_steps=${save_steps:-20000}

note=paligemma3b_zoe_lr${lr}_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACC}_node$((GPUS / GPUS_PER_NODE))_gpu${GPUS}
cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")

# resume training from ckpt
resume_path=
fix_raw_length=${fix_raw_length:-0}
OUTPUT_DIR=${resume_path:-outputs/spatialvla_paligemma2_3b_pretrain/$date_dir/${cur_time}_${mixture}_${note}}
mkdir -p $OUTPUT_DIR

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
# export LD_PRELOAD=../libtcmalloc.so.4.5.3 # optional, for better memory management
# export TRITON_CACHE_DIR=~/.triton

cp $(realpath "$0") ${OUTPUT_DIR}
export LAUNCHER="pytorch"
TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--nnodes $NODES --nproc-per-node $GPUS_PER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"}

torchrun $TORCH_RUN_ARGS \
  train/spatialvla_pretrain.py \
  --fix_raw_length ${fix_raw_length} \
  --ignore_data_skip True \
  --data_root_dir /oss/vla_ptm_hwfile/DATA/open_x_embodiment_converted \
  --data_mix ${mixture} \
  --shuffle_buffer_size ${shuffle_buffer_size} \
  --obs_backward_steps 0 \
  --obs_backward_delta 1 \
  --action_forward_steps 3 \
  --vision_zoe_path ../pretrained/zoedepth-nyu-kitti \
  --vlm_path ../pretrained/paligemma2-3b-pt-224 \
  --use_vision_zoe True \
  --flash_attn True \
  --output_dir ${OUTPUT_DIR} \
  --overwrite_output_dir False \
  --freeze_llm_embed True \
  --freeze_vision_tower False \
  --dataloader_num_workers ${NUM_WORKERS} \
  --bf16 True \
  --tf32 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy steps \
  --save_steps ${save_steps} \
  --save_total_limit 3 \
  --learning_rate ${lr} \
  --weight_decay 0.0 \
  --warmup_ratio 0.005 \
  --lr_scheduler_type linear \
  --logging_steps 500 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --deepspeed scripts/zero1.json \
  --action_config scripts/action_config.json \
  --intrinsic_config_path scripts/intrinsics.json \
  --normalized_statistic_path scripts/gs_spatialvla_plus.json \
  --min_sigma 0.5 \
  --report_to tensorboard \
  --log_level warning