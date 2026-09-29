"""
Accelerator和DeepSpeed配置工具函数
从spatialvla_finetune_align_v2.py中迁移而来，简化主文件结构
"""

import os
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration


def create_accelerator(training_args, model_args=None):
    """
    创建accelerator实例，配置DeepSpeed、混合精度和分布式训练
    
    功能：
    1. 根据training_args自动配置DeepSpeed参数
    2. 设置混合精度训练（bf16/fp16）
    3. 配置分布式训练和设备映射
    4. 支持gradient checkpointing和accumulation
    5. 自动配置wandb和tensorboard logging
    
    Args:
        training_args: HuggingFace TrainingArguments实例
        model_args: 模型相关参数（可选）
        
    Returns:
        Accelerator: 配置好的accelerator实例
    """
    logs_dir = os.path.join(training_args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    project_config = ProjectConfiguration(
        project_dir=training_args.output_dir,
        logging_dir=logs_dir
    )
    
    # 配置DeepSpeed插件
    # 注意：新版本的DeepSpeedPlugin使用hf_ds_config参数而不是config_file
    # hf_ds_config参数接受DeepSpeed配置文件路径或配置字典
    deepspeed_plugin = None
    if training_args.deepspeed:
        # 读取DeepSpeed配置文件并添加批次大小信息
        import json
        with open(training_args.deepspeed, 'r') as f:
            ds_config = json.load(f)
        
        # 如果配置中的train_micro_batch_size_per_gpu是"auto"，则用training_args中的值替换
        if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
            ds_config["train_micro_batch_size_per_gpu"] = training_args.per_device_train_batch_size
            print(f"🔧 DeepSpeed配置更新: train_micro_batch_size_per_gpu = {training_args.per_device_train_batch_size}")
        
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
    
    # 确定混合精度类型
    mixed_precision = "no"
    if training_args.bf16:
        mixed_precision = "bf16"
    elif training_args.fp16:
        mixed_precision = "fp16"
    
    # 根据training_args.report_to参数配置logging后端
    # 支持tensorboard、wandb或两者同时使用
    log_with = []
    if hasattr(training_args, 'report_to') and training_args.report_to:
        # report_to可能是字符串、列表，或逗号分隔的字符串
        if isinstance(training_args.report_to, list):
            report_to_list = training_args.report_to
        elif isinstance(training_args.report_to, str):
            # 处理逗号分隔的字符串，如："tensorboard,wandb"
            report_to_list = [backend.strip() for backend in training_args.report_to.split(',')]
        else:
            report_to_list = [str(training_args.report_to)]
        
        # 过滤有效的backend
        for backend in report_to_list:
            if backend == "tensorboard":
                log_with.append("tensorboard")
            elif backend == "wandb":
                log_with.append("wandb")
        
        print(f"🔍 解析report_to参数: 原始='{training_args.report_to}' -> 后端列表={log_with}")
        
        # 如果没有指定有效的backend，默认使用tensorboard
        if not log_with:
            log_with = ["tensorboard"]
            print("⚠️ 未找到有效的logging后端，默认使用tensorboard")
    else:
        # 默认使用tensorboard
        log_with = ["tensorboard"]
        print("📊 使用默认logging后端: tensorboard")
    
    # 创建accelerator
    # 注意：这里不传入log_with参数，避免在accelerator创建时就初始化tracker
    # tracker的初始化将在主训练脚本中通过init_trackers()方法统一管理
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,  # 移除此参数，避免提前创建tracker导致多个tensorboard文件
        project_config=project_config,
        deepspeed_plugin=deepspeed_plugin,
        step_scheduler_with_optimizer=False,  # 手动控制scheduler步进
    )
    
    print("✅ Accelerator配置完成:")
    print(f"   - 混合精度: {mixed_precision}")
    print(f"   - 梯度累积步数: {training_args.gradient_accumulation_steps}")
    print(f"   - DeepSpeed: {'启用' if deepspeed_plugin else '禁用'}")
    print(f"   - Logging后端: {log_with}")
    print(f"   - Logging目录: {logs_dir}")
    print(f"   - 设备: {accelerator.device}")
    print(f"   - 进程数: {accelerator.num_processes}")
    print(f"   - 主进程: {'是' if accelerator.is_main_process else '否'}")
    
    # 注意：不在accelerator创建时初始化trackers，避免重复初始化导致多个tensorboard文件
    # 将tracker初始化交给主训练脚本在合适的时机进行，确保只有一个logging实例
    
    # 预准备tracker配置，供主训练脚本使用
    tracker_config = {
        # 优化器相关超参数
        "learning_rate": training_args.learning_rate,          # 学习率
        "batch_size": training_args.per_device_train_batch_size,  # 每设备批次大小
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,  # 梯度累积步数
        "num_train_epochs": training_args.num_train_epochs,    # 训练轮数
        "warmup_ratio": training_args.warmup_ratio,            # 预热比例
        "weight_decay": training_args.weight_decay,            # 权重衰减
        "lr_scheduler_type": training_args.lr_scheduler_type,  # 学习率调度器类型
        
        # 混合精度训练配置
        "mixed_precision": "bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "no"),
        
        # DeepSpeed相关配置
        "deepspeed_enabled": bool(training_args.deepspeed),    # 是否启用DeepSpeed
        
        # 模型架构相关参数（从model_args获取）
        # 兼容不同训练脚本：没有字段时使用默认值
        "lora_rank": getattr(model_args, "lora", 0) or 0,  # LoRA秩，0表示不使用LoRA
        "use_egovlpv2": getattr(model_args, "use_egovlpv2", False),  # 是否使用EgoVLPv2
        "use_alignment": getattr(model_args, "use_alignment", False),  # 是否使用对齐损失
        "vlm_loss_weight": getattr(model_args, "vlm_loss_weight", 0.0),  # VLM损失权重
        "alignment_loss_weight": getattr(model_args, "alignment_loss_weight", 0.0),  # 对齐损失权重
    }
    
    # 生成实验名称，基于output_dir的basename
    experiment_name = os.path.basename(training_args.output_dir)
    
    # 将配置信息存储到accelerator对象中，供主训练脚本使用
    accelerator._spatialvla_tracker_config = tracker_config
    accelerator._spatialvla_experiment_name = experiment_name
    accelerator._spatialvla_log_with = log_with  # 保存logging后端配置，供init_trackers使用
    
    print(f"🚀 Accelerator配置完成，tracker配置已准备:")
    print(f"   - 实验名称: {experiment_name}")
    print(f"   - 项目名称: spatialvla")  
    print(f"   - 配置参数数量: {len(tracker_config)}")
    print(f"   - 准备启用的trackers: {log_with}")
    print(f"   - 注意：trackers将在主训练脚本中统一初始化")

    
    return accelerator


def update_deepspeed_scheduler_config(accelerator, training_args, num_training_steps, warmup_min_lr_ratio = 0):
    """
    更新DeepSpeed配置中scheduler参数的"auto"值为实际计算值
    
    功能：
    1. 计算warmup步数（基于总训练步数和预热比例）
    2. 计算预热阶段的最小和最大学习率
    3. 将DeepSpeed配置中的"auto"参数替换为实际值
    4. 支持warmup_num_steps、total_num_steps、warmup_min_lr、warmup_max_lr的自动配置
    
    Args:
        accelerator: Accelerator实例，用于访问DeepSpeed插件和配置
        training_args: HuggingFace TrainingArguments实例，包含学习率和预热设置
        num_training_steps: 总训练步数，用于计算预热步数和调度器配置
        
    Returns:
        None: 直接修改accelerator中的DeepSpeed配置
    """
    # 检查是否使用了DeepSpeed
    if not hasattr(accelerator, 'deepspeed_plugin') or accelerator.deepspeed_plugin is None:
        print("⚠️ 未使用DeepSpeed，跳过scheduler配置更新")
        return
    
    # 获取DeepSpeed配置
    # HfDeepSpeedConfig对象通过.config属性访问原始配置字典
    ds_config = accelerator.deepspeed_plugin.hf_ds_config.config
    
    # 检查scheduler配置是否存在
    if "scheduler" not in ds_config or "params" not in ds_config["scheduler"]:
        print("⚠️ DeepSpeed配置中未找到scheduler参数，跳过更新")
        return
    
    scheduler_params = ds_config["scheduler"]["params"]
    
    # 计算预热步数：优先使用warmup_ratio，否则使用固定的warmup_steps
    warmup_steps = int(num_training_steps * training_args.warmup_ratio) if training_args.warmup_ratio > 0 else training_args.warmup_steps
    
    # 计算预热阶段的学习率范围
    # base_lr使用training_args中的学习率作为目标学习率
    base_lr = training_args.learning_rate
    warmup_min_lr = base_lr * warmup_min_lr_ratio
    warmup_max_lr = base_lr         # 预热结束学习率为目标学习率
    
    # 替换scheduler配置中的"auto"参数为实际计算值
    # 只有当参数值为"auto"时才进行替换，保持用户自定义值不变
    if scheduler_params.get("warmup_min_lr") == "auto":
        scheduler_params["warmup_min_lr"] = warmup_min_lr
        
    if scheduler_params.get("warmup_max_lr") == "auto":
        scheduler_params["warmup_max_lr"] = warmup_max_lr
        
    if scheduler_params.get("warmup_num_steps") == "auto":
        scheduler_params["warmup_num_steps"] = warmup_steps
        
    if scheduler_params.get("total_num_steps") == "auto":
        scheduler_params["total_num_steps"] = num_training_steps
    
    # 输出更新后的配置信息，便于调试和验证
    print(f"🔧 DeepSpeed scheduler配置更新完成:")
    print(f"   - 基础学习率: {base_lr}")
    print(f"   - 预热步数: {warmup_steps} (总步数: {num_training_steps}, 预热比例: {training_args.warmup_ratio})")
    print(f"   - warmup_min_lr: {scheduler_params.get('warmup_min_lr', 'N/A')}")
    print(f"   - warmup_max_lr: {scheduler_params.get('warmup_max_lr', 'N/A')}")
    print(f"   - warmup_num_steps: {scheduler_params.get('warmup_num_steps', 'N/A')}")
    print(f"   - total_num_steps: {scheduler_params.get('total_num_steps', 'N/A')}")
    
    print("✅ DeepSpeed scheduler配置更新成功")
