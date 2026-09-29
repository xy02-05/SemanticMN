#!/bin/bash
# pi0.5 LIBERO 完整 pipeline：train extract → eval_similarity → train_probe
# 假定 test split 已经提取（pi05_libero_official_test.npz 存在），由本脚本之外串行先跑
# 使用 GPU 1（避开 GPU 0 上的训练）

set -e
cd /root/data/xuyuan1/Codes/mirror_neuron/data_process/libero
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi

DEVICE=${1:-cuda:1}
BS=${2:-16}
LOG_DIR=/tmp
TS=$(date +%Y%m%d-%H%M%S)

echo "================================================"
echo "[1/3] Extract train split features (clean mode)"
echo "================================================"
python extract_features.py \
    --checkpoint pi05_libero_official --group pi05 \
    --split train --batch_size $BS \
    --device $DEVICE --feature_mode clean 2>&1 | tee $LOG_DIR/pi05_extract_train_$TS.log

echo "================================================"
echo "[2/3] Run eval_similarity (CKA / SVCCA / KNN)"
echo "================================================"
python eval_similarity.py \
    --checkpoint pi05_libero_official \
    --split test --text_type qwen3 2>&1 | tee $LOG_DIR/pi05_eval_similarity_$TS.log

echo "================================================"
echo "[3/3] Train linear probe (action → text)"
echo "================================================"
python train_probe.py \
    --checkpoint pi05_libero_official \
    --text_type qwen3 --device $DEVICE 2>&1 | tee $LOG_DIR/pi05_train_probe_$TS.log

echo ""
echo "================================================"
echo "DONE."
echo "  features: outputs/features/pi05_libero_official_{train,test}.npz"
echo "  similarity JSON: outputs/results/similarity_pi05_libero_official_test_qwen3.json"
echo "  probe JSON: outputs/probes/pi05_libero_official_qwen3/probe_results.json"
echo "================================================"
