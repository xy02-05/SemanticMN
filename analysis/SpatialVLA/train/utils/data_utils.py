"""
数据加载相关工具函数
包含训练采样器、数据加载器创建、评估和训练循环等功能
从spatialvla_finetune_align_v2.py和monkey_patch.py中迁移而来，删除重复代码
"""

import os
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers.utils.logging import get_logger
from transformers.trainer import LengthGroupedSampler, RandomSampler, has_length, is_datasets_available, seed_worker, _is_peft_model

# 导入SpatialVLA相关组件
from train.monkey_patch import concat_pad_data_collator, LengthGroupedSampler

logger = get_logger(__name__)


def create_train_sampler(
    train_dataset, 
    group_by_length: bool = False,
    train_batch_size: int = 1,
    world_size: int = 1,
    gradient_accumulation_steps: int = 1,
    tokenizer = None
) -> Optional[torch.utils.data.Sampler]:
    """
    创建训练采样器的无self版本，适配accelerate+deepspeed训练
    
    功能：
    1. 支持SpatialVLA多数据集的长度分组采样
    2. 支持分布式训练的world_size调整
    3. 兼容原有的group_by_length配置
    4. 独立于Trainer类，可直接在accelerate环境中使用
    
    Args:
        train_dataset: 训练数据集，支持多数据集（datasets属性）
        group_by_length: 是否按长度分组采样
        train_batch_size: 训练批次大小
        world_size: 分布式训练的进程数
        gradient_accumulation_steps: 梯度累积步数
        tokenizer: 分词器，用于获取model_input_names
        
    Returns:
        Optional[torch.utils.data.Sampler]: 采样器实例或None
    """
    if train_dataset is None:
        return None
    
    # 构建采样器
    if group_by_length:
        lengths = []
        # 支持多数据集场景：从每个子数据集提取长度信息
        if hasattr(train_dataset, 'datasets'):
            for dataset in train_dataset.datasets:
                if hasattr(dataset, 'length'):
                    lengths = lengths + dataset.length
        else:
            # 单数据集场景：尝试从数据集直接获取长度
            if hasattr(train_dataset, 'length'):
                lengths = train_dataset.length
            else:
                # 回退：从tokenizer推断输入字段名
                model_input_name = tokenizer.model_input_names[0] if tokenizer is not None else 'input_ids'
                lengths = [len(feature[model_input_name]) for feature in train_dataset]
        
        model_input_name = tokenizer.model_input_names[0] if tokenizer is not None else None
        return LengthGroupedSampler(
            batch_size=train_batch_size,
            world_size=world_size * gradient_accumulation_steps,
            dataset=train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        return RandomSampler(train_dataset)

def create_train_dataloader(
    train_dataset,
    batch_size: int,
    data_collator,
    sampler = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
    use_raw_dataloader: bool = False,
    accelerator = None
) -> DataLoader:
    """
    创建训练数据加载器的无self版本，适配accelerate+deepspeed训练
    
    功能：
    1. 支持自定义数据整理器（如concat_pad_data_collator）
    2. 支持自定义采样器（如LengthGroupedSampler）
    3. 支持accelerate的分布式和混合精度包装
    4. 支持SpatialVLA的use_raw_dataloader模式
    5. 独立于Trainer类，直接在accelerate环境中使用
    
    Args:
        train_dataset: 训练数据集
        batch_size: 批次大小
        data_collator: 数据整理器函数
        sampler: 采样器实例（可选）
        num_workers: 数据加载工作线程数
        pin_memory: 是否将数据固定在内存中
        persistent_workers: 是否保持工作线程持久化
        drop_last: 是否丢弃最后不完整的批次
        use_raw_dataloader: 是否使用原始DataLoader（不经过accelerate包装）
        accelerator: accelerate实例，用于prepare DataLoader
        
    Returns:
        DataLoader: 配置好的训练数据加载器
    """
    if train_dataset is None:
        raise ValueError("训练需要提供train_dataset")

    # 构建DataLoader参数
    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    # 对于非IterableDataset，设置采样器和其他参数
    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        if sampler is not None:
            dataloader_params["sampler"] = sampler
        dataloader_params["drop_last"] = drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    # 创建DataLoader
    dataloader = DataLoader(train_dataset, **dataloader_params)
    
    # 根据use_raw_dataloader决定是否通过accelerate包装
    if use_raw_dataloader or getattr(train_dataset, 'use_raw_dataloader', True):
        return dataloader
    elif accelerator is not None:
        return accelerator.prepare(dataloader)


def create_bridge_dataloader_and_sampler(
    data_args,
    training_args,
    output_dir: str,
    vla_processor = None,
    tokenizer = None,
    accelerator = None
):
    """
    为Bridge v2数据集创建数据加载器和采样器，集成无self版本的函数
    
    该函数是create_train_sampler和create_train_dataloader的高级封装，
    专门针对SpatialVLA的Bridge数据集训练场景进行优化。
    
    功能：
    1. 构建Bridge v2训练和验证数据集
    2. 创建适配的长度分组采样器
    3. 配置支持SpatialVLA特殊需求的数据加载器
    4. 与accelerate+deepspeed训练环境完美集成
    
    Args:
        data_args: 数据相关参数（DataTrainingArguments实例）
        training_args: 训练相关参数（TrainingArguments实例）  
        output_dir: 输出目录路径
        vla_processor: SpatialVLA处理器（可选）
        tokenizer: 分词器
        accelerator: accelerate实例
        
    Returns:
        tuple: (train_dataloader, eval_dataloader, train_sampler)
    """
    # 导入数据集构建函数
    from data.dataset import build_datasets
    
    # 构建数据集（与原有逻辑保持一致）
    train_dataset, eval_dataset = build_datasets(
        data_args,
        output_dir,
        vla_processor=vla_processor,
    )
    
    # 创建训练采样器
    train_sampler = create_train_sampler(
        train_dataset=train_dataset,
        group_by_length=getattr(training_args, 'group_by_length', False),
        train_batch_size=training_args.per_device_train_batch_size,
        world_size=training_args.world_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        tokenizer=tokenizer
    )
    
    # 创建训练数据加载器
    train_dataloader = create_train_dataloader(
        train_dataset=train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        data_collator=concat_pad_data_collator,  # 使用SpatialVLA专用的数据整理器
        sampler=train_sampler,
        num_workers=getattr(training_args, 'dataloader_num_workers', 0),
        pin_memory=getattr(training_args, 'dataloader_pin_memory', True),
        persistent_workers=getattr(training_args, 'dataloader_persistent_workers', False),
        drop_last=getattr(training_args, 'dataloader_drop_last', False),
        use_raw_dataloader=getattr(train_dataset, 'use_raw_dataloader', False),
        accelerator=accelerator
    )
    
    # 创建验证数据加载器（如果有验证集）
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = create_train_dataloader(
            train_dataset=eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            data_collator=concat_pad_data_collator,
            sampler=None,  # 验证时通常不需要特殊采样器
            num_workers=getattr(training_args, 'dataloader_num_workers', 0),
            pin_memory=getattr(training_args, 'dataloader_pin_memory', True),
            persistent_workers=False,  # 验证时不需要持久化工作线程
            drop_last=False,  # 验证时不丢弃数据
            use_raw_dataloader=getattr(eval_dataset, 'use_raw_dataloader', False),
            accelerator=accelerator
        )
    
    print("✅ Bridge v2数据加载器创建完成:")
    print(f"   - 训练数据集大小: {len(train_dataset) if train_dataset else 'N/A'} {len(train_dataloader)} with world_size: {training_args.world_size} and gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"   - 验证数据集大小: {len(eval_dataset) if eval_dataset else 'N/A'}")
    print(f"   - 采样器类型: {type(train_sampler).__name__ if train_sampler else 'None'}")
    print(f"   - 批次大小: {training_args.per_device_train_batch_size}")
    
    return train_dataloader, eval_dataloader, train_sampler