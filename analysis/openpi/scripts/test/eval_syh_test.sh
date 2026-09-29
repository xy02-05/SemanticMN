#!/bin/bash

task_name=pick_single_bottle
task_config=demo_randomized
train_config_name=robotwin_rlds_hard_48x50_align_fg_ego_40x50_128_egohod
model_name=align_w0.1_10x100_egohod_nofg_meanpool
checkpoint_id=40000
seed=999
gpu_id=7

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/RoboTwin_new/policy/pi0
source ~/miniconda3/bin/activate openpi

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_short_short.py --config policy/pi0/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name}_${checkpoint_id} \
    --checkpoint_id ${checkpoint_id} \
    --seed ${seed} \
    --policy_name pi0 



# bash /data/xuyuan/root/Codes/RoboTwin/policy/pi0/eval_syh_short.sh open_laptop demo_clean open_laptop-easy  open_laptop-easy 10000 0 0
