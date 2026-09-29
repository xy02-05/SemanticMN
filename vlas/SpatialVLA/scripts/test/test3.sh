model_name=spatialvla
tasks=(
    bridge.sh
)
ckpts=(
   /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_mimic_align_v2/2025-12-27/align1_egohod_freezevlm_w1_l10_fg_02-01-58_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_egohod_lr1e-4_bs12_node1_gpu2/checkpoint-8000
   /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_mimic_align_v2/2025-12-27/align1_egohod_freezevlm_w1_l10_fg_02-01-58_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_egohod_lr1e-4_bs12_node1_gpu2/checkpoint-4000
   /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_mimic_align_v2/2025-12-27/align1_egohod_freezevlm_w1_l10_fg_02-01-58_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_egohod_lr1e-4_bs12_node1_gpu2/checkpoint-6000
   /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_mimic_align_v2/2025-12-27/align1_egohod_freezevlm_w1_l10_fg_02-01-58_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_egohod_lr1e-4_bs12_node1_gpu2/checkpoint-10000
)

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd /mnt/nvmepool/xuyuan/Codes/SimplerEnv-OpenVLA
. ~/miniconda3/bin/activate simpler_env

for ckpt_path in ${ckpts[@]}; do
    # 🤗 NOTE: set hf cache to avoid confilcts
    # base_dir=$(dirname $ckpt_path)
    # export HF_MODULES_CACHE=$base_dir/hf_cache/modules
    # mkdir -p $HF_MODULES_CACHE
    # logging_dir=$base_dir/simpler_env/$(basename $ckpt_path)${action_ensemble_temp}
  
    logging_dir=results/results_align_mean_nofg/$(basename $ckpt_path)

    mkdir -p $logging_dir
    for i in ${!tasks[@]}; do
        task=${tasks[$i]}
        echo "🚀 running $task ..."
        device=0
        bash scripts/$task $ckpt_path $model_name $logging_dir $device
    done

    # statistics evalution results
    echo "🚀 all tasks DONE! Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root $logging_dir \
        >>$logging_dir/total.metrics
done
