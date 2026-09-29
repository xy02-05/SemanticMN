#!/bin/bash

unset CUDA_VISIBLE_DEVICES

# 定义任务名称数组
task_names=("click_alarmclock" "click_bell" "dump_bin_bigbin" "grab_roller" "handover_mic" 
            "move_playingcard_away" "open_laptop" "open_microwave" "press_stapler" "stack_bowls_three")

# 循环遍历所有任务并执行
for task_name in "${task_names[@]}"; do
    echo "开始执行任务: $task_name"
    bash /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/openpi/scripts/test/eval_50_0.75.sh \
        "$task_name" \
        demo_randomized \
        robotwin_rlds_hard_48x50_align_fg_ego \
        align_w0.1_fb_sigmoid \
        30000 \
        0 \
        2
    
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
task_names=("click_alarmclock" "click_bell" "dump_bin_bigbin" "grab_roller" "handover_mic" 
            "move_playingcard_away" "open_laptop" "open_microwave" "press_stapler" "stack_bowls_three")

# 循环遍历所有任务并执行
for task_name in "${task_names[@]}"; do
    echo "开始执行任务: $task_name"
    bash /mnt/nvmepool/xuyuan/Codes/mirror_neuron/vlas/openpi/scripts/test/eval_50_0.75.sh \
        "$task_name" \
        demo_randomized \
        robotwin_rlds_hard_48x50_align_fg_ego \
        align_w0.1_fb_sigmoid \
        40000 \
        0 \
        2
    
    # 检查上一个命令是否执行成功
    if [ $? -ne 0 ]; then
        echo "任务 $task_name 执行失败，退出循环"
        exit 1
    fi
    
    echo "任务 $task_name 执行完成"
    echo "----------------------------------------"
done

echo "所有任务执行完毕"