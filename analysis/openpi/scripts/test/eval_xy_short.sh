#!/bin/bash

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
checkpoint_id=${5}
seed=${6}
gpu_id=${7}

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd /data/xuyuan/root/Codes/RoboTwin/policy/pi0
source ~/miniconda3/bin/activate openpi
# source /data/xuyuan/UniVLA_env/project/RoboTwin/policy/pi0/.venv/bin/activate 
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_short.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name}_${checkpoint_id} \
    --checkpoint_id ${checkpoint_id} \
    --seed ${seed} \
    --policy_name ${policy_name} 



# bash /data/xuyuan/root/Codes/RoboTwin/policy/pi0/eval_syh_short.sh open_laptop demo_clean open_laptop-easy  open_laptop-easy 10000 0 0
