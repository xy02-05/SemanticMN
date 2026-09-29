#!/bin/bash
# =================================================================
# SpatialVLA × LIBERO 评测 Launcher  (joint_align ckpt-36000 专用)
#
# 与 scripts/eval_libero.sh 的关系：
#   - 完全复用 eval/eval_libero.py 的全部超参（rotate180 / center_crop 0.9 /
#     ensemble alpha 0.1 / invert_gripper / replan 4）——保持单点真实
#   - 仅改了「默认 CKPT 路径」与「输出目录命名空间」
#   - ckpt 是 PEFT 格式（adapter_config.json + modules_to_save: spatial_embed_tokens），
#     eval_libero.py 的 _load_model_and_processor 会自动走 PEFT 分支：
#       base_model = AutoModel.from_pretrained(adapter_cfg.base_model_name_or_path)
#       PeftModel.from_pretrained(base_model, ckpt) → merge_and_unload
#   - 内参 / bin_policy：训练时 scripts/intrinsics.json + scripts/gs_libero_mix.json
#     已注入 _processor 并随 ckpt 落到 processor_config.json，AutoProcessor.from_pretrained
#     会原样恢复；eval_libero.py 加了 fx≈270.39 + r_bins[1]≈0.3644 的 sanity assert。
#
# 用法：
#   bash scripts/eval_libero_align_ckpt36k.sh [GPU] [CKPT] [SUITES] [NUM_TRIALS] [MAX_TASKS]
# 例：
#   bash scripts/eval_libero_align_ckpt36k.sh 0                                    # 全量 4 suite × 10 trial
#   bash scripts/eval_libero_align_ckpt36k.sh 0 "" libero_spatial 1 1             # smoke test
# =================================================================

set -x

cd /root/data/xuyuan1/Codes/mirror_neuron/vlas/SpatialVLA
export PYTHONWARNINGS="ignore::DeprecationWarning"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/lib64
export PATH=$PATH:/usr/local/cuda-12/bin
export CUDA_HOME="/usr/local/cuda-12"
. ~/miniconda3/bin/activate spatialvla

# ============ 参数解析（带空字符串保护） ============
GPU_ID=${1:-0}
CKPT_PATH=${2:-/root/data/xuyuan1/Codes/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_libero_joint/2026-05-06/20-36-54_libero_mix_joint_align_w1_lr2.5e-4_bs16_lora32_ep100/checkpoint-36000}
# 允许 "" 占位以使用默认值
[ -z "$CKPT_PATH" ] && CKPT_PATH=/root/data/xuyuan1/Codes/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_libero_joint/2026-05-06/20-36-54_libero_mix_joint_align_w1_lr2.5e-4_bs16_lora32_ep100/checkpoint-36000
SUITES=${3:-libero_spatial,libero_object,libero_goal,libero_10}
NUM_TRIALS=${4:-10}
MAX_TASKS=${5:--1}

export CUDA_VISIBLE_DEVICES=$GPU_ID
# MuJoCo / robosuite 离屏渲染
export MUJOCO_GL=egl
# 用 SpatialVLA 独立的 LIBERO 配置目录，避免污染 ~/.libero
export LIBERO_CONFIG_PATH="$(pwd)/.libero_config"

# ============ 输出目录（独立 align 命名空间，方便和 raw baseline 对比） ============
ckpt_tag=$(basename $(dirname $CKPT_PATH))_$(basename $CKPT_PATH)
date_tag=$(date +%Y-%m-%d)
time_tag=$(date +%H-%M-%S)
OUTPUT_DIR=outputs/eval_libero_align/${date_tag}/${time_tag}_${ckpt_tag}_trials${NUM_TRIALS}
mkdir -p $OUTPUT_DIR

cp $(realpath "$0") ${OUTPUT_DIR}/

# 让 PYTHONPATH 找到 eval/action_ensembler.py 与 LIBERO 本地副本
export PYTHONPATH="${PYTHONPATH}:$(pwd)/eval:$(pwd)/LIBERO"

python eval/eval_libero.py \
  --ckpt-path "$CKPT_PATH" \
  --unnorm-key libero_mix_no_noops/1.0.0 \
  --dtype bf16 \
  --attn-implementation eager \
  --suites "$SUITES" \
  --num-trials-per-task $NUM_TRIALS \
  --max-tasks-per-suite $MAX_TASKS \
  --num-steps-wait 10 \
  --use-center-crop \
  --crop-pct 0.9 \
  --rotate-image-180 \
  --use-action-ensemble \
  --action-ensemble-alpha 0.1 \
  --replan-steps 4 \
  --action-clip \
  --invert-gripper \
  --gripper-threshold 0.5 \
  --video-out-dir $OUTPUT_DIR \
  --save-videos \
  --video-fps 10 \
  --seed 7 \
  --deterministic \
  2>&1 | tee ${OUTPUT_DIR}/eval.log
