# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from transformers import (
    get_constant_schedule,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

# ✅ 兼容新版transformers：AdamW已移至torch.optim
try:
    from transformers.optimization import AdamW
except ImportError:
    from torch.optim import AdamW


def get_egovlpv2_optimizer_grouped_parameters(model, config, check_requires_grad=False):
    """
    获取EgoVLPv2模型的参数组，使用多组学习率策略
    
    Args:
        model: EgoVLPv2模型
        config: EgoVLPv2配置
        check_requires_grad: 是否检查requires_grad (默认False，用于兼容原始set_schedule)
        
    Returns:
        list: 参数组列表
    """
    lr = config["optimizer"]["args"]["lr"]
    wd = config["optimizer"]["args"]["weight_decay"]
    lr_mult_head = config["optimizer"]["args"]["lr_mult_head"]
    lr_mult_cross_modal = config["optimizer"]["args"]["lr_mult_cross_modal"]
    
    # 参数分类规则（基于EgoVLPv2的set_schedule函数）
    no_decay = [
        "bias", "LayerNorm.bias", "LayerNorm.weight", 
        "norm.bias", "norm.weight", "norm1.bias", "norm1.weight",
        "norm2.bias", "norm2.weight",
    ]
    head_names = ["mlm_score", "itm_score", "txt_proj", "vid_proj"]
    cross_modal_names = ["cross_modal", "i2t", "t2i"]
    
    optimizer_grouped_parameters = [
        # 普通参数 + weight decay
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
                and (not check_requires_grad or p.requires_grad)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        # 普通参数 + 无weight decay
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
                and (not check_requires_grad or p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        # head参数 + weight decay
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
                and (not check_requires_grad or p.requires_grad)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_head,
        },
        # head参数 + 无weight decay
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
                and (not check_requires_grad or p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_head,
        },
        # cross_modal参数 + weight decay
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
                and (not check_requires_grad or p.requires_grad)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_cross_modal,
        },
        # cross_modal参数 + 无weight decay
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
                and (not check_requires_grad or p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_cross_modal,
        },
    ]
    
    return optimizer_grouped_parameters


def create_egovlpv2_optimizer(optimizer_grouped_parameters, config, lr=None):
    """
    创建EgoVLPv2优化器
    
    Args:
        optimizer_grouped_parameters: 分组的参数列表
        config: EgoVLPv2配置
        lr: 学习率（可选，如果不提供则从config中获取）
        
    Returns:
        torch.optim.Optimizer: 优化器实例
    """
    if lr is None:
        lr = config["optimizer"]["args"]["lr"]
    
    optim_type = config["optimizer"]["type"]
    
    if optim_type == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")
    
    return optimizer


def create_egovlpv2_scheduler(optimizer, config_yaml, max_steps, warmup_steps):
    """
    创建EgoVLPv2学习率调度器
    
    Args:
        optimizer: 优化器实例
        config_yaml: YAML配置（包含end_lr和decay_power）
        max_steps: 最大训练步数
        warmup_steps: 预热步数
        
    Returns:
        torch.optim.lr_scheduler: 学习率调度器
    """
    end_lr = config_yaml["end_lr"]
    decay_power = config_yaml["decay_power"]
    
    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )
    
    return scheduler


def set_schedule(model, config, config_yaml, max_steps, warmup_steps):
    """
    原始的set_schedule函数，现在使用拆分的函数重构
    """
    # 获取参数组
    optimizer_grouped_parameters = get_egovlpv2_optimizer_grouped_parameters(model, config)
    
    # 创建优化器
    optimizer = create_egovlpv2_optimizer(optimizer_grouped_parameters, config)
    
    # 创建调度器
    scheduler = create_egovlpv2_scheduler(optimizer, config_yaml, max_steps, warmup_steps)

    return optimizer, scheduler


# === 原始未重构的函数，保持兼容性 ===

def set_schedule_constant(model, config, config_yaml, max_steps, warmup_steps):
    
    lr = config["optimizer"]["args"]["lr"]
    wd = config["optimizer"]["args"]["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["mlm_score", "itm_score", "txt_proj", "vid_proj"]
    cross_modal_names = ["cross_modal", "i2t", "t2i"]
    lr_mult_head = config["optimizer"]["args"]["lr_mult_head"]
    lr_mult_cross_modal = config["optimizer"]["args"]["lr_mult_cross_modal"]
    end_lr = config_yaml["end_lr"]
    decay_power = config_yaml["decay_power"]
    optim_type = config["optimizer"]["type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_cross_modal,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_cross_modal,
        },
    ]

    if optim_type == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr)
        #optimizer = AdamW(list(model.txt_proj.parameters()) + list(model.vid_proj.parameters()), lr=lr, eps=1e-8, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    scheduler = get_constant_schedule(optimizer)

    return optimizer, scheduler
