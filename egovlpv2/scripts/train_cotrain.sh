#!/bin/bash
# ===========================================================
# EgoHOD FHO Cotrain 训练脚本
# 
# 支持多卡训练和EK100-CLS评测集成
#
# 使用方式:
#   ./train_cotrain.sh                              # 默认单卡训练
#   ./train_cotrain.sh --gpus 2                     # 2卡训练
#   ./train_cotrain.sh --config /path/to/config.json # 指定配置
#   ./train_cotrain.sh --eval                       # 训练后评测
#   ./train_cotrain.sh --eval_only                  # 仅评测（不训练）
#   ./train_cotrain.sh --eval_subset 0.05           # 评测5%的数据(均匀采样)
# ===========================================================

set -e

# 脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EGOVLPV2_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
TESTS_ROOT="${EGOVLPV2_ROOT}/../tests/egohod_eval"

# 默认参数
CONFIG_FILE="${EGOVLPV2_ROOT}/egovlpv2/configs/ft/cotrain.json"
NUM_GPUS=1
PORT=52142
DO_EVAL=false
EVAL_ONLY=false
EVAL_SUBSET=1.0

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)      CONFIG_FILE="$2"; shift 2 ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --port)        PORT="$2"; shift 2 ;;
        --eval)        DO_EVAL=true; shift ;;
        --eval_only)   EVAL_ONLY=true; shift ;;
        --eval_subset) EVAL_SUBSET="$2"; shift 2 ;;
        --save_dir)    SAVE_DIR="$2"; shift 2 ;;
        --lora_ckpt)   LORA_CKPT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 设置保存目录
SAVE_DIR=${SAVE_DIR:-"${EGOVLPV2_ROOT}/outputs/cotrain_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$SAVE_DIR"

# 激活环境
source ~/miniconda3/bin/activate spatialvla

# 设置PYTHONPATH
export PYTHONPATH="${EGOVLPV2_ROOT}:${EGOVLPV2_ROOT}/../vlms/EgoHOD:${PYTHONPATH}"

echo "=========================================================="
echo "EgoHOD FHO Cotrain"
echo "=========================================================="
echo "配置文件:   $CONFIG_FILE"
echo "GPU数量:    $NUM_GPUS"
echo "保存目录:   $SAVE_DIR"
echo "评测子集:   ${EVAL_SUBSET} (1.0=全部)"
echo "=========================================================="

# ========================
# 训练
# ========================
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo ">>> 开始训练..."
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
        echo "使用GPU: $GPU_IDS"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python "${EGOVLPV2_ROOT}/multinode_train_charades.py" \
            --config "$CONFIG_FILE" \
            --save_dir "$SAVE_DIR" \
            --port "$PORT" \
            --print_freq 100 \
            --no_val
    else
        python "${EGOVLPV2_ROOT}/multinode_train_charades.py" \
            --config "$CONFIG_FILE" \
            --save_dir "$SAVE_DIR" \
            --port "$PORT" \
            --print_freq 100 \
            --no_val
    fi
    
    echo ">>> 训练完成！模型保存在: $SAVE_DIR"
fi

# ========================
# 评测
# ========================
if [ "$DO_EVAL" = true ] || [ "$EVAL_ONLY" = true ]; then
    echo ""
    echo ">>> 开始 EK100-CLS 评测..."
    
    # 确定 checkpoint 路径
    if [ -n "$LORA_CKPT" ]; then
        CKPT_PATH="$LORA_CKPT"
    else
        # 查找最新的checkpoint (排除 _train_state.pth 文件)
        CKPT_PATH=$(find "$SAVE_DIR" -name "checkpoint-epoch*.pth" -type f 2>/dev/null | grep -v "_train_state" | sort -V | tail -1)
    fi
    
    if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
        echo "警告: 未找到checkpoint，仅评测原始模型"
        python "${TESTS_ROOT}/eval_ek100_cls.py" \
            --mode original \
            --eval_subset "$EVAL_SUBSET"
    else
        echo "评测checkpoint: $CKPT_PATH"
        python "${TESTS_ROOT}/eval_ek100_cls.py" \
            --mode compare \
            --lora_checkpoint "$CKPT_PATH" \
            --eval_subset "$EVAL_SUBSET"
    fi
fi

echo ""
echo "=========================================================="
echo "完成！"
echo "=========================================================="
