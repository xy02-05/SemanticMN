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
    # 标量温度参数不应施加weight decay，否则会持续把log_temperature往0推，
    # 等价于把temperature往1.0拉，干扰对比学习的自适应缩放。
    decay_parameters = [
        name for name in decay_parameters
        if "bias" not in name and "temperature" not in name
    ]
    return decay_parameters

def get_pi0_parameters(model, optimizer_config, base_lr):
    """
    获取PI0模型的参数组，使用OpenPI原生的简单分组策略
    
    Args:
        model: PI0PytorchAlign模型实例（包装后的模型）
        optimizer_config: OpenPI的优化器配置（包含weight_decay等参数）
        base_lr: PI0的基础学习率（通常使用peak_lr）
        
    Returns:
        List[dict]: PI0参数组列表（包含base_lr字段）
    """
    # 获取PI0Pytorch内部模型（可能被LoRA包装）
    pi0_model = model.pi0_pytorch
    
    # 分组：decay参数 vs no-decay参数
    # 保存base_lr，用于后续的scheduler缩放
    pi0_params = [
        {
            "params": [p for n, p in pi0_model.named_parameters() if p.requires_grad],
            "weight_decay": optimizer_config.weight_decay,
            "base_lr": base_lr,  # 保存基础学习率
        }
    ]
    
    return pi0_params 


def get_egovlpv2_parameters(model, egovlpv2_config, default_base_lr):
    """
    获取EgoVLPv2模型的参数组，使用多组学习率策略
    
    Args:
        model: MIMIC-VLA模型实例
        egovlpv2_config: EgoVLPv2配置字典
        default_base_lr: 默认基础学习率（如果参数组未设置lr时使用）
        
    Returns:
        List[dict]: EgoVLPv2参数组列表（每个组都包含base_lr字段）
    """
    egovlpv2_model = model.egovlpv2_model
    
    # 直接调用EgoVLPv2的参数分组函数
    # check_requires_grad=True: 仅包含requires_grad=True的参数
    # LoRA模式下只有adapter参数可训练，冻结参数不需要加入optimizer（节省显存）
    egovlpv2_params = get_egovlpv2_optimizer_grouped_parameters(
        model=egovlpv2_model, 
        config=egovlpv2_config, 
        check_requires_grad=True
    )
    
    # 为每个参数组设置base_lr：如果已经设置了lr，使用该lr；否则使用default_base_lr
    for group in egovlpv2_params:
        group['base_lr'] = group['lr']  # 使用已设置的lr作为base_lr

    
    return egovlpv2_params


def get_alignment_parameters(model, alignment_config):
    """
    获取Alignment模型的参数组，使用HF Trainer兼容的参数分组策略
    
    Args:
        model: MIMIC-VLA模型实例
        alignment_config: Alignment配置字典
        
    Returns:
        List[dict]: Alignment参数组列表（每个组都包含base_lr字段）
    """
    alignment_model = model.alignment_model
    
    # 从配置中读取学习率和权重衰减
    alignment_optimizer_config = alignment_config['alignment']['optimizer']
    lr = alignment_optimizer_config['args']['lr']
    wd = alignment_optimizer_config['args']['weight_decay']
    print(f"🔧 Alignment学习率: {lr}, 权重衰减: {wd}")

    decay_parameters = get_decay_parameter_names(alignment_model)
    
    # 保存base_lr，用于后续的scheduler缩放
    alignment_params = [
        {
            "params": [
                p for n, p in alignment_model.named_parameters() 
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": wd,
            "lr": lr,
            "base_lr": lr,  # 保存基础学习率
        },
        {
            "params": [
                p for n, p in alignment_model.named_parameters() 
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": lr,
            "base_lr": lr,  # 保存基础学习率
        },
    ]
    
    return alignment_params


def create_optimizer_and_scheduler(
    model, 
    optimizer_config,
    lr_schedule_config,
    num_training_steps, 
    egovlpv2_config=None, 
    alignment_config=None,
    freeze_egovlpv2=False
):
    """
    创建优化器和学习率调度器，适配OpenPI的配置结构
    
    功能：
    1. 为PI0PytorchAlign三个模型创建统一优化器
    2. PI0：使用OpenPI原生参数分组（简单的decay/no-decay）
    3. EgoVLPv2：使用多组学习率策略，从配置读取参数
    4. Alignment：从配置文件读取学习率和权重衰减设置
    5. 不创建调度器（OpenPI在训练循环中手动管理学习率）
    
    Args:
        model: PI0PytorchAlign模型实例
        optimizer_config: OpenPI的优化器配置（AdamW dataclass）
        lr_schedule_config: OpenPI的学习率配置（仅用于显示，不创建scheduler）
        num_training_steps: 总训练步数（用于统计）
        egovlpv2_config: EgoVLPv2配置字典（可选）
        alignment_config: Alignment配置字典（可选）
        freeze_egovlpv2: 是否冻结EgoVLPv2模型
        
    Returns:
        optimizer: PyTorch优化器（不返回scheduler，由训练循环管理）
    """
    # === 构建多模型参数组 ===
    optimizer_grouped_parameters = []
    peak_lr = lr_schedule_config.peak_lr
    
    print("🔧 开始创建OpenPI多模型优化器...")
    
    # 1. PI0参数组（使用OpenPI原生分组，base_lr使用peak_lr）
    pi0_params = get_pi0_parameters(model, optimizer_config, base_lr=peak_lr)
    optimizer_grouped_parameters.extend(pi0_params)
    print(f"✅ PI0参数组: {len(pi0_params)}组, base_lr={peak_lr}")
    
    # 2. EgoVLPv2参数组（如果启用）
    if freeze_egovlpv2:
        print(f"🔒 Freeze EgoVLPv2模型，不加入优化器")
    elif hasattr(model, 'egovlpv2_model') and model.egovlpv2_model is not None:
        if egovlpv2_config is None:
            raise ValueError("egovlpv2_config参数必须提供用于EgoVLPv2优化器配置")
        egovlpv2_params = get_egovlpv2_parameters(model, egovlpv2_config, default_base_lr=peak_lr)
        optimizer_grouped_parameters.extend(egovlpv2_params)
        # 显示EgoVLPv2的base_lr信息
        egovlpv2_base_lrs = [g.get('base_lr', 'N/A') for g in egovlpv2_params]
        print(f"✅ EgoVLPv2参数组: {len(egovlpv2_params)}组, base_lrs={egovlpv2_base_lrs}")
    
    # 3. Alignment参数组（如果启用）
    if hasattr(model, 'alignment_model') and model.alignment_model is not None:
        if alignment_config is None:
            raise ValueError("alignment_config参数必须提供用于Alignment优化器配置")
        alignment_params = get_alignment_parameters(model, alignment_config)
        optimizer_grouped_parameters.extend(alignment_params)
        alignment_base_lr = alignment_params[0].get('base_lr', 'N/A') if alignment_params else 'N/A'
        print(f"✅ Alignment参数组: {len(alignment_params)}组, base_lr={alignment_base_lr}")
    
    # 创建AdamW优化器（使用OpenPI的配置参数）
    # 注意：每个参数组可能已经设置了lr，如果没有设置，会使用这里的默认lr
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=peak_lr,  # 默认学习率（仅用于未设置lr的参数组）
        betas=(optimizer_config.b1, optimizer_config.b2),
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )
    
    # 确保所有参数组都有base_lr字段（如果没有，使用当前lr作为base_lr）
    for group in optimizer.param_groups:
        if 'base_lr' not in group:
            # 如果参数组没有base_lr，使用当前lr作为base_lr
            group['base_lr'] = group.get('lr', peak_lr)
    
    # === 参数统计和调试输出 ===
    total_params = sum(len(group["params"]) for group in optimizer_grouped_parameters)
    total_trainable = sum(p.numel() for group in optimizer_grouped_parameters for p in group["params"])
    
    # 参数检查：验证各模型的参数是否正确包含在优化器中
    _inspect_optimizer_coverage(model, optimizer)
    
    print("✅ 优化器创建完成:")
    print(f"   - 参数组数量: {len(optimizer_grouped_parameters)}")
    print(f"   - 总参数张量数: {total_params}")
    print(f"   - 可训练参数数: {total_trainable:,}")
    print(f"   - 初始学习率: {lr_schedule_config.peak_lr}")
    print(f"   - AdamW配置: b1={optimizer_config.b1}, b2={optimizer_config.b2}, eps={optimizer_config.eps}")
    print(f"   - Weight decay: {optimizer_config.weight_decay}")
    
    return optimizer


def _inspect_optimizer_coverage(model, optimizer):
    """
    验证优化器是否正确包含各模型组件的可训练参数
    
    该函数检查PI0、EgoVLPv2和Alignment模型的所有requires_grad=True的参数
    是否都已正确包含在优化器的参数组中，确保训练时所有参数都能被优化。
    
    Args:
        model: PI0PytorchAlign模型实例
        optimizer: 已创建的优化器实例
    """
    print("🔍 正在验证优化器参数覆盖率...")
    
    # 获取优化器中所有参数的ID集合
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            optimizer_param_ids.add(id(param))
    
    # 检查PI0模型（内部模型）
    _check_model_coverage(model.pi0_pytorch, optimizer_param_ids, "PI0Pytorch")
    
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
    检查并输出optimizer参数组的详细信息，便于在OpenPI中比对PI0/Ego/Alignment各自的学习率设置
    
    功能说明：
    1. 统计每个参数组的参数数量、参数总数
    2. 显示每个参数组的学习率、权重衰减等超参数（自动补全默认值）
    3. 检查参数所在设备与dtype，辅助定位DeepSpeed/Accelerate改写
    4. 汇总各学习率配置的参数数量，方便排查配置是否丢失
    
    参数说明：
        optimizer: 要检查的优化器实例（可能是原始optimizer或DeepSpeed包装后的optimizer）
        stage_name: 检查阶段名称，用于区分不同时机的检查结果
        accelerator: accelerator实例，用于主进程控制
    """
    # 只在主进程输出，避免重复日志
    if not accelerator.is_main_process:
        return

    logger.info("")
    logger.info(f"📊 ========== Optimizer参数组检查 ({stage_name}) ==========")

    # 检查optimizer的类型，判断是否被DeepSpeed包装
    optimizer_type = type(optimizer).__name__
    logger.info(f"🔧 Optimizer类型: {optimizer_type}")

    # 标记是否为DeepSpeed优化器
    is_deepspeed = 'deepspeed' in optimizer_type.lower() or 'DeepSpeed' in optimizer_type

    # 获取真实optimizer（DeepSpeed会在optimizer.optimizer中封装真正的AdamW）
    base_optimizer = optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer

    # 获取参数组信息
    if hasattr(optimizer, 'param_groups'):
        param_groups = optimizer.param_groups
    elif hasattr(base_optimizer, 'param_groups'):
        param_groups = base_optimizer.param_groups
        logger.info(f"🔍 检测到DeepSpeed包装，实际optimizer类型: {type(base_optimizer).__name__}")
        is_deepspeed = True
    else:
        logger.info("❌ 无法找到param_groups属性")
        return

    defaults = getattr(base_optimizer, "defaults", {})
    logger.info(f"📈 参数组总数: {len(param_groups)}")

    # 统计总参数数量和内存使用
    total_params = 0
    total_elements = 0

    # 按学习率分组统计，分析参数组合并情况
    lr_groups = {}

    # 遍历每个参数组，输出详细信息
    for i, group in enumerate(param_groups):
        raw_params = group.get('params', [])
        group_params = [p for p in raw_params if p is not None]
        group_param_count = len(group_params)

        # 计算该组参数的总元素数
        group_elements = sum(p.numel() for p in group_params)
        total_params += group_param_count
        total_elements += group_elements

        # 获取该组的超参数设置，若未显式指定则回退到默认值
        lr = group.get('lr', defaults.get('lr', 'N/A'))
        weight_decay = group.get('weight_decay', defaults.get('weight_decay', 'N/A'))

        # 按学习率分组统计
        lr_key = f"lr:{lr}_wd:{weight_decay}"
        if lr_key not in lr_groups:
            lr_groups[lr_key] = {'count': 0, 'elements': 0, 'groups': []}
        lr_groups[lr_key]['count'] += group_param_count
        lr_groups[lr_key]['elements'] += group_elements
        lr_groups[lr_key]['groups'].append(i + 1)

        logger.info(f"  📌 参数组 {i + 1}:")
        logger.info(f"     - 参数张量数: {group_param_count}")
        logger.info(f"     - 参数元素总数: {group_elements:,}")
        logger.info(f"     - 学习率: {lr}")
        logger.info(f"     - 权重衰减: {weight_decay}")

        # 检查代表性参数的设备与dtype，帮助定位是否被移动到CPU/NVMe
        sample_param = next((p for p in group_params if hasattr(p, 'device')), None)
        if sample_param is not None:
            logger.info(f"     - 参数设备: {sample_param.device}")
        sample_param = next((p for p in group_params if hasattr(p, 'dtype')), sample_param)
        if sample_param is not None and hasattr(sample_param, 'dtype'):
            logger.info(f"     - 参数数据类型: {sample_param.dtype}")

        # 检查参数是否需要梯度
        requires_grad_count = sum(1 for p in group_params if getattr(p, 'requires_grad', False))
        logger.info(f"     - 需要梯度的参数: {requires_grad_count}/{group_param_count}")

    # 输出按超参数分组的统计
    logger.info("📋 按超参数分组统计:")
    for lr_key, stats in lr_groups.items():
        logger.info(f"   - {lr_key}: {stats['count']}张量, {stats['elements']:,}元素, 组{stats['groups']}")

    # 输出汇总信息
    logger.info("📊 汇总统计:")
    logger.info(f"   - 总参数张量数: {total_params}")
    logger.info(f"   - 总参数元素数: {total_elements:,}")
    logger.info(f"   - 估计内存占用: {total_elements * 4 / 1024 / 1024:.2f} MB (float32)")

    # 如果是DeepSpeed，输出ZeRO分片信息
    if is_deepspeed:
        logger.info("🚀 DeepSpeed ZeRO分片信息:")
        logger.info(f"   - 当前GPU参数数: {total_elements:,}")
        logger.info(f"   - 参数组数量变化: 原始10组 → 当前{len(param_groups)}组 (合并了相同超参数的组)")

    # 如果是DeepSpeed optimizer，尝试获取额外的DeepSpeed特定信息
    if hasattr(optimizer, 'mpu'):
        logger.info(f"   - MPU (模型并行): {optimizer.mpu is not None}")

    if hasattr(optimizer, 'overflow'):
        logger.info(f"   - 是否检测到溢出: {optimizer.overflow}")

    # 检查ZeRO相关信息
    if hasattr(optimizer, 'zero_optimization'):
        logger.info("   - ZeRO优化: 启用")

    logger.info("========================================")
