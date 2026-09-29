#!/bin/bash
# =================================================================
# SpatialVLA 4B MIMIC-VLA 对齐训练脚本 (v2版本)
# 
# 功能：使用 spatialvla_finetune_align_v2.py 进行训练
# 参数：参考 finetune_full_xy_mimic_align_new.sh 配置
# 适配：使用accelerate而非HF Trainer
# =================================================================

set -x

cd /root/data/xuyuan1/Codes/mirror_neuron/vlas/SpatialVLA
export PYTHONWARNINGS="ignore::DeprecationWarning"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/lib64
export PATH=$PATH:/usr/local/cuda-12/bin
export CUDA_HOME="/usr/local/cuda-12"
. ~/miniconda3/bin/activate spatialvla
export CUDA_VISIBLE_DEVICES=0,1
DEBUG=true
if [ "$DEBUG" = true ]; then
  GPUS=2
  GPUS_PER_NODE=2
  PER_DEVICE_BATCH_SIZE=12
  shuffle_buffer_size=8192
  tsfm_thread_muti=12
  read_thread_muti=12
  mixture=bridge_orig
  NUM_WORKERS=2
  TORCH_RUN_ARGS="--standalone --nnodes=1 --nproc-per-node $GPUS_PER_NODE"
  BATCH_SIZE=192
  save_steps=1000
fi

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-$((GPUS * PER_DEVICE_BATCH_SIZE))}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

mixture=bridge_orig
mixture=${mixture:-oxe_magic_soup_plus}
NUM_WORKERS=${NUM_WORKERS:-1}
shuffle_buffer_size=${shuffle_buffer_size:-8192}
tsfm_thread_muti=${tsfm_thread_muti:-1}
read_thread_muti=${read_thread_muti:-1}

lr=1e-4
lora=32
lora_alpha=32
lora_target="linear"
epoch=2
save_steps=${save_steps:-10}

# MIMIC-VLA specific parameters
egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/task_id_egohod_mix.json"
vlm_loss_weight=0
alignment_loss_weight=0.5
use_egovlpv2=false
use_alignment=true
vlm_mode="embedding"
task_filename="tasks_with_id.jsonl"  # task_id版本使用带task_id的任务映射

cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")

# resume training from ckpt
model_name_or_path="/root/data/xuyuan1/dataset/spatialvla-4b-224-pt/spatialvla-4b-224-pt"
note=$(basename $model_name_or_path)_mimic_align_v2_lr${lr}_bs${PER_DEVICE_BATCH_SIZE}_node$((GPUS / GPUS_PER_NODE))_gpu${GPUS}
OUTPUT_DIR=${resume_path:-outputs/spatialvla_4b_mimic_align_v2/$date_dir/align1_egohod_mix_w0.5_${cur_time}_${mixture}_${note}}
mkdir -p $OUTPUT_DIR

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3

cp $(realpath "$0") ${OUTPUT_DIR}

export LAUNCHER="pytorch"
TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--nnodes $NODES --nproc-per-node $GPUS_PER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"}

torchrun $TORCH_RUN_ARGS \
  train/spatialvla_finetune_align_v2.py \
  --model_name_or_path ${model_name_or_path} \
  ${ADAPT_ARGS} \
  --lora ${lora} \
  --lora_alpha ${lora_alpha} \
  --lora_target ${lora_target} \
  --ignore_data_skip True \
  --data_root_dir /root/data/xuyuan1/Codes/mirror_neuron/data/ \
  --data_mix ${mixture} \
  --task_filename ${task_filename} \
  --shuffle_buffer_size ${shuffle_buffer_size} \
  --tsfm_thread_muti ${tsfm_thread_muti} \
  --read_thread_muti ${read_thread_muti} \
  --obs_backward_steps 0 \
  --obs_backward_delta 1 \
  --action_forward_steps 3 \
  --freeze_egovlpv2_model true \
  --flash_attn False \
  --output_dir ${OUTPUT_DIR} \
  --overwrite_output_dir False \
  --freeze_vision_tower False \
  --dataloader_num_workers ${NUM_WORKERS} \
  --bf16 True \
  --tf32 True \
  --num_train_epochs ${epoch} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy steps \
  --save_steps ${save_steps} \
  --save_total_limit 3 \
  --learning_rate ${lr} \
  --weight_decay 0.0 \
  --warmup_ratio 0.005 \
  --lr_scheduler_type linear \
  --logging_steps 20 \
  --max_grad_norm 1.0 \
  --do_train True \
  --deepspeed scripts/zero1.json \
  --grad_checkpoint True \
  --report_to tensorboard \
  --log_level warning \
  --egovlpv2_config_path ${egovlpv2_config_path} \
  --vlm_loss_weight ${vlm_loss_weight} \
  --alignment_loss_weight ${alignment_loss_weight} \
  --use_egovlpv2 ${use_egovlpv2} \
  --use_alignment ${use_alignment} \
  --vlm_mode ${vlm_mode} \
  2>&1 | tee ${OUTPUT_DIR}/training.log