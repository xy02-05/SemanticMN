#!/bin/bash

# 定义任务名称数组
task_names=(
    "click_alarmclock"
    "move_stapler_pad"
    "pick_dual_bottles"
    "place_a2b_left"
    "place_can_basket"
    "place_container_plate"
    "place_object_stand"
    "put_object_cabinet"
    "stack_bowls_two"
    "stack_blocks_three"
)
# 循环遍历所有任务并执行
for task_name in "${task_names[@]}"; do
    echo "开始执行任务: $task_name"
    bash /root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/scripts/test/eval_50_0.75.sh \
        "$task_name" \
        demo_randomized \
        robotwin_rlds_hard_48x50_raw_ego_40x50_128 \
        raw_ego_10x100 \
        50000 \
        0 \
        0 \
        50 \
        _eval_step_limit.yml
    
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
task_names=(
    "click_alarmclock"
    "move_stapler_pad"
    "pick_dual_bottles"
    "place_a2b_left"
    "place_can_basket"
    "place_container_plate"
    "place_object_stand"
    "put_object_cabinet"
    "stack_bowls_two"
    "stack_blocks_three"
)
# 循环遍历所有任务并执行
for task_name in "${task_names[@]}"; do
    echo "开始执行任务: $task_name"
    bash /root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/scripts/test/eval_50_0.75.sh \
        "$task_name" \
        demo_randomized \
        robotwin_rlds_hard_48x50_align_chunk50 \
        align_w0.1_egovideo_chunk50_composite_pos2_non_verb_neg_freeze \
        50000 \
        0 \
        0 \
        50 \
        _eval_step_limit.yml
    
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
task_names=(
    "click_alarmclock"
    "move_stapler_pad"
    "pick_dual_bottles"
    "place_a2b_left"
    "place_can_basket"
    "place_container_plate"
    "place_object_stand"
    "put_object_cabinet"
    "stack_bowls_two"
    "stack_blocks_three"
)
# 循环遍历所有任务并执行
for task_name in "${task_names[@]}"; do
    echo "开始执行任务: $task_name"
    bash /root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/scripts/test/eval_50_0.75.sh \
        "$task_name" \
        demo_randomized \
        robotwin_rlds_hard_48x50_align_fg_ego_40x50_128_egohod \
        align_w0.1_10x100_egohod_nofg_meanpool \
        50000 \
        0 \
        0 \
        50 \
        _eval_step_limit.yml
    
    # 检查上一个命令是否执行成功
    if [ $? -ne 0 ]; then
        echo "任务 $task_name 执行失败，退出循环"
        exit 1
    fi
    
    echo "任务 $task_name 执行完成"
    echo "----------------------------------------"
done

echo "所有任务执行完毕"