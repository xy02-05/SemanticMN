#!/bin/bash
# 全量 probe 重跑：200 epoch，记录每 10 epoch 结果，best_after_50
# 13 checkpoints × 2 text types × 2 probe types = 52 runs
# 串行执行（GPU 不够并行）

set -e
cd /root/data/xuyuan1/Codes/mirror_neuron/data_process/libero

CKPTS="pretrained step_5k step_10k step_15k step_20k step_25k step_30k new_dsn_new_5k new_dsn_new_10k new_dsn_new_15k new_dsn_new_20k new_dsn_new_25k new_dsn_new_30k"
TEXT_TYPES="qwen3 egohod"
EPOCHS=200
MIN_BEST=50

echo "========== Task-level rollout probes (32train/8test) =========="
for text in $TEXT_TYPES; do
  for ckpt in $CKPTS; do
    echo ""
    echo ">>> task-level: $ckpt / $text"
    python3 train_probe.py \
      --checkpoint "$ckpt" \
      --text_type "$text" \
      --probe_train_mode rollout \
      --probe_test_mode rollout \
      --epochs $EPOCHS \
      --min_best_epoch $MIN_BEST \
      --layers 10
  done
done

echo ""
echo "========== Suite-level probes (30train/Goal10test) =========="
for text in $TEXT_TYPES; do
  for ckpt in $CKPTS; do
    echo ""
    echo ">>> suite-level: $ckpt / $text"
    python3 train_probe.py \
      --checkpoint "$ckpt" \
      --text_type "$text" \
      --probe_train_mode rollout \
      --probe_test_mode rollout \
      --epochs $EPOCHS \
      --min_best_epoch $MIN_BEST \
      --layers 10 \
      --suite_split
  done
done

echo ""
echo "========== ALL PROBES DONE =========="
