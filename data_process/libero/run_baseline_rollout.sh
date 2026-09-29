#!/bin/bash
# Baseline (pi0_libero_full/raw) rollout feature extraction: 5k-25k
# Run 3 in parallel (each ~8GB GPU), sequentially for train after test
source ~/miniconda3/bin/activate openpi
cd /root/data/xuyuan1/Codes/mirror_neuron/data_process/libero

for step in 5k 10k 15k 20k 25k; do
    echo "========== step_${step} test rollout =========="
    python extract_features.py --checkpoint step_${step} --split test --feature_mode rollout
    echo "========== step_${step} train rollout =========="
    python extract_features.py --checkpoint step_${step} --split train --feature_mode rollout
done
echo "ALL BASELINE ROLLOUT DONE"
