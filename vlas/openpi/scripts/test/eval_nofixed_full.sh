#!/bin/bash

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
checkpoint_id=${5}
seed=${6}
gpu_id=${7}
test_num=${8:-40}  # 测试样本数量，默认40
step_lim=${9:-_eval_step_limit.yml}  # 执行步数上限，可选参数

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mvideo quality: ${video_quality}\033[0m"
echo -e "\033[33mtest num: ${test_num}\033[0m"
if [ -n "${step_lim}" ]; then
    echo -e "\033[33mstep limit: ${step_lim}\033[0m"
fi

cd /data/xuyuan/UniVLA_env/mirror_neuron/vlas/RoboTwin_new/policy/pi0
source ~/miniconda3/bin/activate openpi
# source /data/xuyuan/UniVLA_env/project/RoboTwin/policy/pi0/.venv/bin/activate 
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_short_short.py --config policy/$policy_name/deploy_policy.yml --use_zero_noise \
    --test_num ${test_num} \
    --step_lim ${step_lim} \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name}_${checkpoint_id} \
    --checkpoint_id ${checkpoint_id} \
    --seed ${seed} \
    --policy_name ${policy_name} 
