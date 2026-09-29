model_name=spatialvla
tasks=(
    bridge.sh
    # drawer_variant_agg.sh
    # drawer_visual_matching.sh
    # move_near_variant_agg.sh
    # move_near_visual_matching.sh
    # pick_coke_can_variant_agg.sh
    # pick_coke_can_visual_matching.sh

    # put_in_drawer_variant_agg.sh
    # put_in_drawer_visual_matching.sh
)

ckpts=(
    /data/xuyuan/UniVLA_env/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_mimic_align_v2/2025-10-12/cotrain_07-58-47_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_lr1e-4_bs12_node1_gpu2/checkpoint-18000
    /data/xuyuan/UniVLA_env/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_mimic_align_v2/2025-10-12/cotrain_07-58-47_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_lr1e-4_bs12_node1_gpu2/checkpoint-20000
    /data/xuyuan/UniVLA_env/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_mimic_align_v2/2025-10-12/cotrain_07-58-47_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_lr1e-4_bs12_node1_gpu2/checkpoint-22000
    /data/xuyuan/UniVLA_env/mirror_neuron/vlas/SpatialVLA/outputs/spatialvla_4b_mimic_align_v2/2025-10-12/cotrain_07-58-47_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_lr1e-4_bs12_node1_gpu2/checkpoint-24000
)

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd /data/xuyuan/UniVLA_env/project/SimplerEnv-OpenVLA
. ~/miniconda3/bin/activate simpler_env

action_ensemble_temp=-0.8
for ckpt_path in ${ckpts[@]}; do
    # 🤗 NOTE: set hf cache to avoid confilcts
    # base_dir=$(dirname $ckpt_path)
    # export HF_MODULES_CACHE=$base_dir/hf_cache/modules
    # mkdir -p $HF_MODULES_CACHE
    # logging_dir=$base_dir/simpler_env/$(basename $ckpt_path)${action_ensemble_temp}
  
    logging_dir=results_cotrain/$(basename $ckpt_path)${action_ensemble_temp}

    mkdir -p $logging_dir
    cp $0 $logging_dir
    for i in ${!tasks[@]}; do
        task=${tasks[$i]}
        echo "🚀 running $task ..."
        device=0
        bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device
    done

    # statistics evalution results
    echo "🚀 all tasks DONE! Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root $logging_dir \
        >>$logging_dir/total.metrics
done
