"""
优化器和学习率调度器工具函数
为MIMIC-VLA多模型训练创建统一的优化器和调度器
与mimic_vla_trainer.py保持完全一致的实现逻辑
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
# 导入transformers和accelerate的scheduler函数，支持标准库实现
from transformers.optimization import get_scheduler
from accelerate.utils import DummyScheduler

# 导入EgoVLPv2优化器相关函数
from egovlpv2.set_optim_schedule import get_egovlpv2_optimizer_grouped_parameters

# 导入调试相关模块
import logging

logger = logging.getLogger(__name__)


def get_decay_parameter_names(model):
    """
    获取所有需要应用weight decay的参数名称，与HF Trainer的逻辑一致
    
    Args:
        model: 模型实例
        
    Returns:
        List[str]: 需要weight decay的参数名称列表
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def get_spatial_vla_parameters(model, training_args):
    """
    获取SpatialVLA模型的参数组，使用HF标准逻辑支持LoRA
    
    Args:
        model: MIMIC-VLA模型实例
        training_args: 训练参数配置
        
    Returns:
        List[dict]: SpatialVLA参数组列表
    """
    spatial_vla = model.spatial_vla
    
    # 使用HF标准的decay参数检测逻辑
    decay_parameters = get_decay_parameter_names(spatial_vla)
    
    spatial_params = [
        {
            "params": [
                p for n, p in spatial_vla.named_parameters() 
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in spatial_vla.named_parameters() 
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    return spatial_params


def get_egovlpv2_parameters(model, egovlpv2_config):
    """
    获取EgoVLPv2模型的参数组，使用多组学习率策略
    
    Args:
        model: MIMIC-VLA模型实例
        egovlpv2_config: EgoVLPv2配置字典
        
    Returns:
        List[dict]: EgoVLPv2参数组列表
    """
    egovlpv2_model = model.egovlpv2_model
    
    # 直接调用EgoVLPv2的参数分组函数
    return get_egovlpv2_optimizer_grouped_parameters(
        model=egovlpv2_model, 
        config=egovlpv2_config, 
        check_requires_grad=False
    )


def get_egohod_parameters(model, egovlpv2_config):
    """
    获取EgoHOD模型的参数组（参考EgoHOD官方main_pretrain.py）
    
    EgoHOD使用timm的add_weight_decay分组：
    - bias和norm层不使用weight decay
    - 其他层使用配置的weight decay
    
    Args:
        model: MIMIC-VLA模型实例
        egovlpv2_config: EgoHOD配置（包含optimizer.args.weight_decay）
        
    Returns:
        List[dict]: EgoHOD参数组列表
    """
    import timm.optim.optim_factory as optim_factory
    
    egohod_model = model.egovlpv2_model
    
    # 从配置获取weight_decay（参考EgoHOD官方代码）
    optimizer_config = egovlpv2_config.config.get('optimizer', {})
    weight_decay = optimizer_config.get('args', {}).get('weight_decay', 0.01)
    lr = optimizer_config.get('args', {}).get('lr', 5e-5)
    
    # 使用timm的add_weight_decay创建参数组（EgoHOD官方使用这个）
    param_groups = optim_factory.param_groups_weight_decay(egohod_model, weight_decay)
    
    # 设置学习率
    for group in param_groups:
        group['lr'] = lr
    
    return param_groups


def get_alignment_parameters(model, alignment_config):
    """
    获取Alignment模型的参数组，使用HF Trainer兼容的参数分组策略
    
    Args:
        model: MIMIC-VLA模型实例
        alignment_config: Alignment配置字典
        
    Returns:
        List[dict]: Alignment参数组列表
    """
    alignment_model = model.alignment_model
    
    # 从配置中读取学习率和权重衰减
    alignment_optimizer_config = alignment_config['alignment']['optimizer']
    lr = alignment_optimizer_config['args']['lr']
    wd = alignment_optimizer_config['args']['weight_decay']
    print(f"🔧 Alignment学习率: {lr}, 权重衰减: {wd}")

    decay_parameters = get_decay_parameter_names(alignment_model)
    
    alignment_params = [
        {
            "params": [
                p for n, p in alignment_model.named_parameters() 
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p for n, p in alignment_model.named_parameters() 
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    
    return alignment_params


def create_optimizer_and_scheduler(
    model, 
    training_args, 
    num_training_steps, 
    egovlpv2_config=None, 
    alignment_config=None
):
    """
    创建优化器和学习率调度器，与mimic_vla_trainer.py完全一致的实现
    
    功能：
    1. 为MIMIC-VLA三个模型创建统一优化器，使用相同的参数分组策略
    2. SpatialVLA：使用HF标准逻辑，支持LoRA
    3. EgoVLPv2：使用多组学习率策略，从配置读取参数
    4. Alignment：从配置文件读取学习率和权重衰减设置
    5. 创建线性预热+Cosine退火学习率调度器
    6. 添加参数检查和统计输出功能
    
    Args:
        model: MIMIC-VLA模型实例
        training_args: HF训练参数配置
        num_training_steps: 总训练步数
        egovlpv2_config: EgoVLPv2配置字典（可选）
        alignment_config: Alignment配置字典（可选）
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # === 构建多模型参数组，与mimic_vla_trainer.py完全一致 ===
    optimizer_grouped_parameters = []
    
    print("🔧 开始创建MIMIC-VLA多模型优化器...")
    
    # 1. SpatialVLA参数组（支持LoRA）
    spatial_params = get_spatial_vla_parameters(model, training_args)
    optimizer_grouped_parameters.extend(spatial_params)
    print(f"✅ SpatialVLA参数组: {len(spatial_params)}组")
    
    # 2. VLM参数组（EgoVLPv2/EgoHOD co-training模式）
    # freeze模式（egohod/egovideo/embedding/clip）不需要创建优化器
    if training_args.freeze_egovlpv2_model:
        print(f"⏭️ VLM参数已冻结，跳过优化器创建")
    elif hasattr(model, 'egovlpv2_model') and model.egovlpv2_model is not None:
        trainable_params = sum(p.numel() for p in model.egovlpv2_model.parameters() if p.requires_grad)
        
        if trainable_params == 0:
            print(f"ℹ️ VLM模型参数已全部冻结")
        else:
            if egovlpv2_config is None:
                raise ValueError("egovlpv2_config参数必须提供")
            
            # 根据vlm_mode选择参数分组方式
            vlm_mode = getattr(model, 'vlm_mode', 'egovlpv2')
            if vlm_mode == 'egohod':
                # EgoHOD co-training：使用timm风格参数分组
                vlm_params = get_egohod_parameters(model, egovlpv2_config)
                print(f"✅ EgoHOD参数组: {len(vlm_params)}组 ({trainable_params:,} 可训练参数)")
            else:
                # EgoVLPv2：使用多组学习率策略
                vlm_params = get_egovlpv2_parameters(model, egovlpv2_config)
                print(f"✅ EgoVLPv2参数组: {len(vlm_params)}组 ({trainable_params:,} 可训练参数)")
            
            optimizer_grouped_parameters.extend(vlm_params)
    
    # 3. Alignment参数组（如果启用）
    if hasattr(model, 'alignment_model') and model.alignment_model is not None:
        if alignment_config is None:
            raise ValueError("alignment_config参数必须提供用于Alignment优化器配置")
        alignment_params = get_alignment_parameters(model, alignment_config)
        optimizer_grouped_parameters.extend(alignment_params)
        print(f"✅ Alignment参数组: {len(alignment_params)}组")
    
    # ✅ 过滤空参数组（DeepSpeed ZeRO不支持空参数组）
    def has_trainable_params(group):
        return any(p.requires_grad for p in group["params"])
    
    non_empty_params = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0 and has_trainable_params(g)]
    if len(non_empty_params) < len(optimizer_grouped_parameters):
        print(f"⚠️ 过滤掉 {len(optimizer_grouped_parameters) - len(non_empty_params)} 个空/冻结参数组")
    optimizer_grouped_parameters = non_empty_params
    
    # 创建AdamW优化器
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )
    
    # 创建学习率调度器
    warmup_steps = int(num_training_steps * training_args.warmup_ratio) if training_args.warmup_ratio > 0 else training_args.warmup_steps
    
    if warmup_steps > 0:
        # 线性预热 + Cosine退火
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-8, total_iters=warmup_steps),
                LinearLR(optimizer, start_factor=1, end_factor=1e-8, total_iters=num_training_steps - warmup_steps),
            ],
            milestones=[warmup_steps]
        )
    else:
        # 仅Cosine退火
        scheduler = LinearLR(optimizer, start_factor=1, total_iters=num_training_steps)
    
    # === 参数统计和调试输出 ===
    total_params = sum(len(group["params"]) for group in optimizer_grouped_parameters)
    total_trainable = sum(p.numel() for group in optimizer_grouped_parameters for p in group["params"])
    
    # 参数检查：验证各模型的参数是否正确包含在优化器中
    _inspect_optimizer_coverage(model, optimizer)
    
    print("✅ 优化器和调度器创建完成:")
    print(f"   - 参数组数量: {len(optimizer_grouped_parameters)}")
    print(f"   - 总参数张量数: {total_params}")
    print(f"   - 可训练参数数: {total_trainable:,}")
    print(f"   - 主学习率: {training_args.learning_rate}")
    print(f"   - 总训练步数: {num_training_steps} with warm up {warmup_steps} and decay steps {num_training_steps - warmup_steps}")
    
    return optimizer, scheduler


def _inspect_optimizer_coverage(model, optimizer):
    """
    验证优化器是否正确包含各模型组件的可训练参数
    
    该函数检查SpatialVLA、EgoVLPv2和Alignment模型的所有requires_grad=True的参数
    是否都已正确包含在优化器的参数组中，确保训练时所有参数都能被优化。
    
    Args:
        model: MIMIC-VLA模型实例
        optimizer: 已创建的优化器实例
    """
    print("🔍 正在验证优化器参数覆盖率...")
    
    # 获取优化器中所有参数的ID集合
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            optimizer_param_ids.add(id(param))
    
    # 检查SpatialVLA模型
    _check_model_coverage(model.spatial_vla, optimizer_param_ids, "SpatialVLA")
    
    # 检查EgoVLPv2模型（如果存在）
    if hasattr(model, 'egovlpv2_model') and model.egovlpv2_model is not None:
        _check_model_coverage(model.egovlpv2_model, optimizer_param_ids, "EgoVLPv2")
    
    # 检查Alignment模型（如果存在）
    if hasattr(model, 'alignment_model') and model.alignment_model is not None:
        _check_model_coverage(model.alignment_model, optimizer_param_ids, "Alignment")
    
    print("✅ 优化器参数覆盖率验证完成")


def _check_model_coverage(model_component, optimizer_param_ids, model_name):
    """
    检查单个模型组件的参数覆盖率
    
    Args:
        model_component: 要检查的模型部分
        optimizer_param_ids: 优化器中所有参数的ID集合
        model_name: 模型名称（用于日志输出）
    """
    if model_component is None:
        print(f"   - {model_name}: 组件不存在，跳过检查")
        return
    
    # 获取模型组件中所有可训练参数的ID
    model_param_ids = set(id(p) for p in model_component.parameters() if p.requires_grad)
    
    if not model_param_ids:
        print(f"   - {model_name}: ⚠️  没有可训练参数")
        return
    
    # 计算覆盖率
    found_count = len(model_param_ids.intersection(optimizer_param_ids))
    total_count = len(model_param_ids)
    
    if found_count == total_count:
        print(f"   - {model_name}: ✅ 全部 {total_count} 个可训练参数已包含")
    else:
        print(f"   - {model_name}: ❌ 只有 {found_count}/{total_count} 个参数被包含")


def inspect_optimizer_param_groups(optimizer, stage_name, accelerator):
    """
    检查并输出optimizer参数组的详细信息，用于调试DeepSpeed对参数组的影响
    
    功能说明：
    1. 统计每个参数组的参数数量、参数总数
    2. 显示每个参数组的学习率、权重衰减等超参数
    3. 检查参数是否在正确的设备上
    4. 分析DeepSpeed ZeRO是否改变了参数组织结构
    5. 输出参数组的内存占用情况
    6. 检测和说明DeepSpeed参数组合并和ZeRO分片的影响
    
    参数说明：
        optimizer: 要检查的优化器实例（可能是原始optimizer或DeepSpeed包装后的optimizer）
        stage_name: 检查阶段名称，用于区分不同时机的检查结果
        accelerator: accelerator实例，用于设备信息和主进程控制
    """
    # 只在主进程输出，避免重复日志
    if not accelerator.is_main_process:
        return
    
    accelerator.print(f"\n📊 ========== Optimizer参数组检查 ({stage_name}) ==========")
    
    # 检查optimizer的类型，判断是否被DeepSpeed包装
    optimizer_type = type(optimizer).__name__
    accelerator.print(f"🔧 Optimizer类型: {optimizer_type}")
    
    # 标记是否为DeepSpeed优化器
    is_deepspeed = 'deepspeed' in optimizer_type.lower() or 'DeepSpeed' in optimizer_type
    
    # 获取参数组信息
    if hasattr(optimizer, 'param_groups'):
        param_groups = optimizer.param_groups
    elif hasattr(optimizer, 'optimizer') and hasattr(optimizer.optimizer, 'param_groups'):
        # DeepSpeed包装的optimizer
        param_groups = optimizer.optimizer.param_groups
        accelerator.print(f"🔍 检测到DeepSpeed包装，实际optimizer类型: {type(optimizer.optimizer).__name__}")
        is_deepspeed = True
    else:
        accelerator.print("❌ 无法找到param_groups属性")
        return
    
    accelerator.print(f"📈 参数组总数: {len(param_groups)}")
    
    # 统计总参数数量和内存使用
    total_params = 0
    total_elements = 0
    
    # 按学习率分组统计，分析参数组合并情况
    lr_groups = {}
    
    # 遍历每个参数组，输出详细信息
    for i, group in enumerate(param_groups):
        group_params = group['params']
        group_param_count = len(group_params)
        
        # 计算该组参数的总元素数
        group_elements = sum(p.numel() for p in group_params if p is not None)
        total_params += group_param_count
        total_elements += group_elements
        
        # 获取该组的超参数设置
        lr = group.get('lr', 'N/A')
        weight_decay = group.get('weight_decay', 'N/A')
        
        # 按学习率分组统计
        lr_key = f"lr:{lr}_wd:{weight_decay}"
        if lr_key not in lr_groups:
            lr_groups[lr_key] = {'count': 0, 'elements': 0, 'groups': []}
        lr_groups[lr_key]['count'] += group_param_count
        lr_groups[lr_key]['elements'] += group_elements
        lr_groups[lr_key]['groups'].append(i+1)
        
        accelerator.print(f"  📌 参数组 {i+1}:")
        accelerator.print(f"     - 参数张量数: {group_param_count}")
        accelerator.print(f"     - 参数元素总数: {group_elements:,}")
        accelerator.print(f"     - 学习率: {lr}")
        accelerator.print(f"     - 权重衰减: {weight_decay}")
        
        # 检查前几个参数的设备信息（避免输出过多信息）
        if group_params:
            sample_param = group_params[0]
            if hasattr(sample_param, 'device'):
                accelerator.print(f"     - 参数设备: {sample_param.device}")
            if hasattr(sample_param, 'dtype'):
                accelerator.print(f"     - 参数数据类型: {sample_param.dtype}")
            
            # 检查参数是否需要梯度
            requires_grad_count = sum(1 for p in group_params if p.requires_grad)
            accelerator.print(f"     - 需要梯度的参数: {requires_grad_count}/{group_param_count}")
    
    # 输出按超参数分组的统计
    accelerator.print(f"📋 按超参数分组统计:")
    for lr_key, stats in lr_groups.items():
        accelerator.print(f"   - {lr_key}: {stats['count']}张量, {stats['elements']:,}元素, 组{stats['groups']}")
    
    # 输出汇总信息
    accelerator.print(f"📊 汇总统计:")
    accelerator.print(f"   - 总参数张量数: {total_params}")
    accelerator.print(f"   - 总参数元素数: {total_elements:,}")
    accelerator.print(f"   - 估计内存占用: {total_elements * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 如果是DeepSpeed，输出ZeRO分片信息
    if is_deepspeed:
        accelerator.print(f"🚀 DeepSpeed ZeRO分片信息:")
        accelerator.print(f"   - 当前GPU参数数: {total_elements:,}")
        accelerator.print(f"   - 参数组数量变化: 原始10组 → 当前{len(param_groups)}组 (合并了相同超参数的组)")
    
    # 如果是DeepSpeed optimizer，尝试获取额外的DeepSpeed特定信息
    if hasattr(optimizer, 'mpu'):
        accelerator.print(f"   - MPU (模型并行): {optimizer.mpu is not None}")
    
    if hasattr(optimizer, 'overflow'):
        accelerator.print(f"   - 是否检测到溢出: {optimizer.overflow}")
    
    # 检查ZeRO相关信息
    if hasattr(optimizer, 'zero_optimization'):
        accelerator.print(f"   - ZeRO优化: 启用")
    
    accelerator.print(f"========================================\n")