#!/bin/bash
# 批量运行所有缺失的 probe
# 用法: bash run_all_probes.sh

cd /root/data/xuyuan1/Codes/mirror_neuron/data_process/libero
FEAT_DIR="outputs/features"

echo "====== Task-level rollout probes (32/8 split) ======"
for ckpt in step_5k step_10k new_dsn_new_20k new_dsn_new_25k new_dsn_new_30k; do
    if [ ! -f "outputs/probes/${ckpt}_qwen3_rollout/probe_results.json" ]; then
        if [ -f "${FEAT_DIR}/${ckpt}_train_rollout.npz" ] && [ -f "${FEAT_DIR}/${ckpt}_test_rollout.npz" ]; then
            echo ">>> Running task-level probe: $ckpt"
            python train_probe.py --checkpoint $ckpt --probe_train_mode rollout --probe_test_mode rollout --epochs 100
        else
            echo ">>> Skipping $ckpt: features not ready"
        fi
    else
        echo ">>> Already done: $ckpt"
    fi
done

echo ""
echo "====== Suite-level probes (30/Goal10) ======"
for ckpt in pretrained step_5k step_10k step_30k new_dsn_new_5k new_dsn_new_10k new_dsn_new_15k new_dsn_new_20k new_dsn_new_25k new_dsn_new_30k; do
    if [ ! -f "outputs/probes/${ckpt}_qwen3_suite_goal_rollout/probe_results.json" ]; then
        if [ -f "${FEAT_DIR}/${ckpt}_train_rollout.npz" ] && [ -f "${FEAT_DIR}/${ckpt}_test_rollout.npz" ]; then
            echo ">>> Running suite probe: $ckpt"
            python train_probe_suite.py --checkpoint $ckpt --feature_mode rollout --eval_suite goal --epochs 200
        else
            echo ">>> Skipping $ckpt: features not ready"
        fi
    else
        echo ">>> Already done: $ckpt"
    fi
done

echo ""
echo "====== Chunk-level probes (32/8 split) ======"
for ckpt in pretrained step_5k step_10k step_30k new_dsn_new_5k new_dsn_new_10k new_dsn_new_15k new_dsn_new_20k new_dsn_new_25k new_dsn_new_30k; do
    if [ ! -f "outputs/probes/${ckpt}_qwen3_rollout_chunk/probe_results.json" ]; then
        if [ -f "${FEAT_DIR}/${ckpt}_train_rollout.npz" ] && [ -f "${FEAT_DIR}/${ckpt}_test_rollout.npz" ]; then
            echo ">>> Running chunk probe: $ckpt"
            python train_probe.py --checkpoint $ckpt --probe_train_mode rollout --probe_test_mode rollout --granularity chunk --epochs 100
        else
            echo ">>> Skipping $ckpt: features not ready"
        fi
    else
        echo ">>> Already done: $ckpt"
    fi
done

echo ""
echo "====== Done ======"
