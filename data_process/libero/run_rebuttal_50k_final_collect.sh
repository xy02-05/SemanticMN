#!/usr/bin/env bash

set -euo pipefail

# 等待 35k-50k 的 Raw/Ours probe 全部完成，再生成最终审计表和曲线。
SCRIPT_DIR=/mnt/bn/2d-videos/xy/work/mirror_neuron/data_process/libero
PYTHON=/mnt/bn/2d-videos/xy/tools/miniconda3/envs/mirror_openpi/bin/python
DONE_MARKER="$SCRIPT_DIR/outputs/results/rebuttal_50k_semantic_curves.done"
STEPS=(35 40 45 50)

export MIRROR_PROJECT_ROOT=/mnt/bn/2d-videos/xy
export MIRROR_WORK_ROOT=/mnt/bn/2d-videos/xy/work/mirror_neuron
export MIRROR_DATA_ROOT="$MIRROR_WORK_ROOT/data"
export MIRROR_ANALYSIS_ROOT="$MIRROR_WORK_ROOT/analysis"
export OPENPI_ROOT="$MIRROR_WORK_ROOT/vlas/openpi"
export PYTHONPATH="$MIRROR_ANALYSIS_ROOT:$OPENPI_ROOT/src"
export PYTHONNOUSERSITE=1

cd "$SCRIPT_DIR"
rm -f "$DONE_MARKER"

for step in "${STEPS[@]}"; do
    for checkpoint in "step_${step}k" "new_dsn_new_${step}k"; do
        result="outputs/probes/${checkpoint}_qwen3_rollout/probe_results.json"
        while [ ! -f "$result" ]; do
            echo "[WAIT] $(date --iso-8601=seconds) $result"
            sleep 60
        done
    done
done

while pgrep -f '[t]rain_probe.py' >/dev/null; do
    echo "[WAIT] $(date --iso-8601=seconds) train_probe.py"
    sleep 60
done

"$PYTHON" collect_rebuttal_50k_semantic_metrics.py
touch "$DONE_MARKER"
echo "[FINAL_COLLECT_DONE] $(date --iso-8601=seconds)"
