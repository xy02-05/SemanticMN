#!/bin/bash

# 定义任务名称数组
task_names=("click_bell" "move_playingcard_away" "pick_diverse_bottles" "place_a2b_right" "place_object_basket" "stack_bowls_three")

# 循环遍历所有任务并执行
for task_name in "${task_names[@]}"; do
    echo "开始执行任务: $task_name"
    bash /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/openpi/scripts/test/eval_50_0.75.sh \
        "$task_name" \
        demo_randomized \
        robotwin_rlds_hard_48x50_raw_ego_40x50_128 \
        raw_ego_40x50 \
        30000 \
        0 \
        1 \
        50 \
        _eval_step_limit_new_3.yml
    
    # 检查上一个命令是否执行成功
    if [ $? -ne 0 ]; then
        echo "任务 $task_name 执行失败，退出循环"
        exit 1
    fi
    
    echo "任务 $task_name 执行完成"
    echo "----------------------------------------"
done

echo "所有任务执行完毕"

# 定义任务名称数组
task_names=("click_bell" "move_playingcard_away" "pick_diverse_bottles" "place_a2b_right" "place_object_basket" "stack_bowls_three")

# 循环遍历所有任务并执行
for task_name in "${task_names[@]}"; do
    echo "开始执行任务: $task_name"
    bash /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/openpi/scripts/test/eval_50_0.75.sh \
        "$task_name" \
        demo_randomized \
        robotwin_rlds_hard_48x50_raw_ego_40x50_128 \
        raw_ego_40x50 \
        40000 \
        0 \
        1 \
        50 \
        _eval_step_limit_new_3.yml
    
    # 检查上一个命令是否执行成功
    if [ $? -ne 0 ]; then
        echo "任务 $task_name 执行失败，退出循环"
        exit 1
    fi
    
    echo "任务 $task_name 执行完成"
    echo "----------------------------------------"
done

echo "所有任务执行完毕"