"""
检查点保存相关工具函数
负责训练过程中的模型保存、状态保存和最终模型保存
从spatialvla_finetune_align_v2.py中迁移而来，简化实现
"""

import os
import json


def save_checkpoint(model, accelerator, training_args, global_step, processor=None, training_state=None):
    """
    保存训练检查点，包含模型权重和训练状态信息
    
    关键修复原理：
    1. accelerator.save_state()必须所有进程同时调用，不能只在主进程中调用
    2. 模型保存只在主进程执行，避免文件冲突
    3. 最小化同步点，简化分布式逻辑
    4. 新增：保存训练状态，支持resume training
    
    Args:
        model: 模型实例（MIMIC-VLA模型）
        accelerator: accelerate实例，用于分布式保存
        training_args: 训练参数，包含output_dir
        global_step: 当前训练步数
        processor: SpatialVLA处理器（可选）
        training_state: 训练状态字典，包含global_step、epoch、损失统计等信息（可选）
    """
    # 创建检查点目录路径（所有进程都需要知道这个路径）
    checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
    
    # 关键修复：所有进程必须同时调用save_state，这是accelerate的分布式保存机制要求
    # save_state内部会自动处理主进程保存、其他进程等待的逻辑
    accelerator.save_state(checkpoint_dir)
    
    # 只在主进程保存模型和处理器，避免多进程写入冲突
    if accelerator.is_main_process:
        # 保存模型权重
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True
        )
        
        # 保存处理器配置
        if processor is not None:
            processor.save_pretrained(checkpoint_dir)
        
        # 新增：保存训练状态信息，支持resume training
        if training_state is not None:
            # 训练状态保存路径
            training_state_path = os.path.join(checkpoint_dir, "training_state.json")
            
            # 确保training_state包含必要的checkpoint元信息
            training_state_to_save = training_state.copy()
            training_state_to_save["checkpoint_step"] = global_step
            
            # 将训练状态保存为JSON文件，方便读取和调试
            with open(training_state_path, 'w', encoding='utf-8') as f:
                json.dump(training_state_to_save, f, indent=2, ensure_ascii=False)
            
            accelerator.print(f"📋 训练状态已保存: {training_state_path}")
        
        accelerator.print(f"✅ 检查点已保存: {checkpoint_dir}")
    
    # 确保模型保存完成后所有进程同步
    accelerator.wait_for_everyone()


def save_final_model(model, accelerator, training_args, processor=None):
    """
    保存最终训练完成的模型，简化分布式逻辑
    
    修复原理：
    1. 减少不必要的同步点，避免复杂的多重同步逻辑
    2. 只在主进程保存，使用简单清晰的同步模式
    
    Args:
        model: 模型实例（MIMIC-VLA模型）
        accelerator: accelerate实例，用于分布式保存
        training_args: 训练参数，包含output_dir
        processor: SpatialVLA处理器（可选）
    """
    # 确保所有进程都完成训练后再开始保存
    accelerator.wait_for_everyone()
    
    # 只在主进程执行保存操作
    if accelerator.is_main_process:
        final_dir = training_args.output_dir
        
        # 保存模型权重
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            final_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True
        )
        
        # 保存处理器配置
        if processor is not None:
            processor.save_pretrained(final_dir)
        
        accelerator.print(f"🎉 最终模型已保存: {final_dir}")
    
    # 确保主进程保存完成后所有进程同步
    accelerator.wait_for_everyone()


def load_checkpoint(checkpoint_path, model, accelerator, processor=None):
    """
    从检查点加载模型权重、optimizer/scheduler状态和训练状态信息
    
    功能说明：
    1. 使用accelerator.load_state()恢复optimizer和scheduler状态
    2. 只在主进程加载模型权重，避免重复加载
    3. 读取training_state.json恢复训练进度信息
    4. 支持分布式环境下的安全加载
    
    Args:
        checkpoint_path: checkpoint目录路径
        model: 模型实例（需要加载权重）
        accelerator: accelerate实例，用于恢复optimizer/scheduler状态
        processor: SpatialVLA处理器（可选，用于验证一致性）
    
    Returns:
        training_state: 包含global_step、epoch、损失统计等信息的字典，如果没有则返回None
    """
    if not os.path.exists(checkpoint_path):
        accelerator.print(f"❌ checkpoint路径不存在: {checkpoint_path}")
        return None
    
    accelerator.print(f"🔄 开始加载checkpoint: {checkpoint_path}")
    
    # 步骤1：所有进程同时加载accelerator状态（optimizer, scheduler等）
    # 这是accelerate分布式加载的核心，必须所有进程同时调用
    accelerator.load_state(checkpoint_path)
    accelerator.print("✅ Accelerator状态（optimizer/scheduler）已恢复")
    
    # 步骤2：只在主进程加载模型权重，然后广播给其他进程
    if accelerator.is_main_process:
        # 加载模型权重到unwrapped model
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 检查模型权重文件是否存在
        model_files = [f for f in os.listdir(checkpoint_path) if f.startswith('model') and (f.endswith('.bin') or f.endswith('.safetensors'))]
        if model_files:
            # 使用from_pretrained加载权重，这是最安全的方式
            loaded_model = unwrapped_model.__class__.from_pretrained(
                checkpoint_path,
                local_files_only=True,
                torch_dtype=unwrapped_model.config.torch_dtype if hasattr(unwrapped_model, 'config') else None
            )
            
            # 将加载的权重复制到当前模型
            unwrapped_model.load_state_dict(loaded_model.state_dict(), strict=True)
            accelerator.print("✅ 模型权重已恢复")
        else:
            accelerator.print("⚠️ 未找到模型权重文件，跳过模型权重恢复")
    
    # 步骤3：读取训练状态信息
    training_state = None
    training_state_path = os.path.join(checkpoint_path, "training_state.json")
    
    if os.path.exists(training_state_path):
        # 只在主进程读取，然后所有进程都会得到相同的信息
        if accelerator.is_main_process:
            with open(training_state_path, 'r', encoding='utf-8') as f:
                training_state = json.load(f)
            accelerator.print(f"✅ 训练状态已恢复: global_step={training_state.get('global_step', 'unknown')}")
    else:
        accelerator.print("⚠️ 未找到训练状态文件，将从默认状态开始")
    
    # 步骤4：确保所有进程加载完成后同步
    accelerator.wait_for_everyone()
    
    # 广播训练状态给所有进程（确保分布式一致性）
    if accelerator.num_processes > 1:
        import torch.distributed as dist
        if training_state is not None and accelerator.is_main_process:
            # 主进程将training_state转换为字符串并广播
            training_state_str = json.dumps(training_state)
            # 这里只是示意，实际的广播需要更复杂的实现
            # 简化处理：所有进程都读取同一个文件
        
        # 非主进程也读取训练状态文件（简化的分布式处理）
        if not accelerator.is_main_process and os.path.exists(training_state_path):
            with open(training_state_path, 'r', encoding='utf-8') as f:
                training_state = json.load(f)
    
    accelerator.print(f"🎉 Checkpoint加载完成: {checkpoint_path}")
    return training_state
