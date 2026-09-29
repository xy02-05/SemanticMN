#!/bin/bash
# =================================================================
# SpatialVLA × LIBERO 评测 Launcher
#
# 与 vlas/openpi/scripts/eval/run_eval_libero_*.sh **完全独立**：
#   - libero 包从 vlas/SpatialVLA/LIBERO/ 注入 sys.path（隔离 site-packages 版）
#   - video / log 输出到 vlas/SpatialVLA/outputs/eval_libero/（不与 openpi 混）
#   - conda env 用 spatialvla（与 openpi env 不同）
#
# 用法：
#   bash scripts/eval_libero.sh [GPU_ID] [CKPT_PATH] [SUITES] [NUM_TRIALS]
# 例：
#   bash scripts/eval_libero.sh 0 outputs/.../checkpoint-2000 libero_spatial 3
#   bash scripts/eval_libero.sh 1 outputs/.../checkpoint-2000 libero_spatial,libero_object,libero_goal,libero_10 50
# =================================================================

set -x

cd /root/data/xuyuan1/Codes/mirror_neuron/vlas/SpatialVLA
export PYTHONWARNINGS="ignore::DeprecationWarning"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/lib64
export PATH=$PATH:/usr/local/cuda-12/bin
export CUDA_HOME="/usr/local/cuda-12"
. ~/miniconda3/bin/activate spatialvla

GPU_ID=${1:-0}
CKPT_PATH=${2:-/root/data/xuyuan1/Codes/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_libero_joint/2026-05-06/10-48-08_libero_mix_joint_align_w1_lr2.5e-4_bs8_lora32_ep100/checkpoint-2000}
SUITES=${3:-libero_spatial,libero_object,libero_goal,libero_10}
NUM_TRIALS=${4:-10}   # 每个 task 跑 10 个 trial（用户约定）；可命令行覆盖
MAX_TASKS=${5:--1}   # smoke test 时设 1，全量评测留 -1

export CUDA_VISIBLE_DEVICES=$GPU_ID
# MuJoCo / robosuite 离屏渲染
export MUJOCO_GL=egl
# 用 SpatialVLA 独立的 LIBERO 配置目录，隔离 ~/.libero（避免污染或冲突 openpi 评测）
export LIBERO_CONFIG_PATH="$(pwd)/.libero_config"

# Output dir 独立到 SpatialVLA 命名空间，绝不与 openpi 共享
ckpt_tag=$(basename $(dirname $CKPT_PATH))_$(basename $CKPT_PATH)
date_tag=$(date +%Y-%m-%d)
time_tag=$(date +%H-%M-%S)
OUTPUT_DIR=outputs/eval_libero/${date_tag}/${time_tag}_${ckpt_tag}_trials${NUM_TRIALS}
mkdir -p $OUTPUT_DIR

# 复制脚本本身做为运行 snapshot
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
