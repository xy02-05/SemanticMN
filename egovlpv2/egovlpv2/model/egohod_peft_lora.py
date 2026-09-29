# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
EgoHOD Text Model LoRA 集成模块

本模块提供了使用 Hugging Face PEFT 库为 EgoHOD 的 TextTransformer 添加 LoRA 的功能。

主要功能：
1. LoRA 配置验证和转换
2. 将 PEFT LoRA 应用到 TextTransformer
3. 可训练参数统计和调试工具
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict


def create_lora_config(
    r: int = 16,
    lora_alpha: float = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    layers_to_transform: Optional[List[int]] = None,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
) -> LoraConfig:
    """
    创建 PEFT LoraConfig 配置对象，支持细粒度层选择
    
    参数说明：
        r: LoRA 秩，控制低秩矩阵的维度（典型值：4, 8, 16, 32, 64）
        lora_alpha: 缩放因子，实际缩放 = lora_alpha/r（通常设置为 2*r）
        lora_dropout: LoRA 层的 dropout 率
        target_modules: 要应用 LoRA 的模块名称列表
            - None: 默认为 ["c_fc", "c_proj"]（推荐，只对MLP应用LoRA）
            - 可选值：
                - ["c_fc", "c_proj"]: MLP层（推荐）
                - ["c_fc", "c_proj", "text_projection"]: MLP + 投影层
            - 注意：
                - 对"text_projection"应用LoRA需要先转换为Linear层
                - 不建议对"out_proj"应用LoRA（MultiheadAttention内部，bias兼容性问题）
        layers_to_transform: 要应用 LoRA 的层索引列表
            - None: 对所有层应用
            - 示例：[8, 9, 10, 11] 只对最后4层应用
            - 示例：[0, 1, 2, 3] 只对前4层应用
        bias: 是否训练 bias 参数
            - "none": 不训练任何 bias（推荐）
            - "all": 训练所有 bias
            - "lora_only": 只训练 LoRA 相关的 bias
        task_type: 任务类型（PEFT 内部使用）
    
    返回：
        LoraConfig: PEFT LoRA 配置对象
    
    注意事项：
        1. target_modules 需要匹配 TextTransformer 的模块命名
        2. EgoHOD TextTransformer 的模块结构：
           - transformer.resblocks.{i}.attn.out_proj: 注意力输出投影（推荐）
           - transformer.resblocks.{i}.mlp.c_fc: MLP 第一层（推荐）
           - transformer.resblocks.{i}.mlp.c_proj: MLP 第二层（推荐）
        3. 使用 layers_pattern 参数通过正则表达式匹配特定层
    """
    # 默认目标模块：只对 MLP 应用 LoRA
    # 注意：不对out_proj应用LoRA，因为它在MultiheadAttention内部，
    # 当bias="none"时会与PyTorch的MultiheadAttention实现冲突
    if target_modules is None:
        target_modules = ["c_fc", "c_proj"]
    
    # 处理层选择逻辑
    # PEFT的layers_to_transform直接支持层索引列表
    if layers_to_transform is not None:
        print(f"✓ 应用 LoRA 到指定层: {layers_to_transform}")
    
    # 创建 LoraConfig
    # 注意：直接使用layers_to_transform参数，PEFT会自动处理
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        bias=bias,
        task_type=task_type,
    )
    
    return config


def apply_lora_to_text_model(
    text_model: nn.Module,
    lora_config: LoraConfig,
) -> nn.Module:
    """
    将 LoRA 应用到 TextTransformer
    
    参数：
        text_model: TextTransformer 实例
        lora_config: PEFT LoraConfig 配置
    
    返回：
        应用了 LoRA 的模型（PeftModel 包装）
    
    工作流程：
        1. 使用 get_peft_model 将 LoRA 应用到模型
        2. PEFT 会自动：
           - 找到所有匹配 target_modules 的层
           - 在这些层旁边添加 lora_A 和 lora_B 参数
           - 将原始参数设置为 requires_grad=False
           - 将 LoRA 参数设置为 requires_grad=True
        3. 返回包装后的模型
    
    注意事项：
        1. 必须在加载预训练权重之后调用
        2. 返回的是 PeftModel 类型，保持原有接口不变
        3. 可以通过 model.print_trainable_parameters() 查看参数统计
    """
    # 应用 PEFT LoRA
    peft_model = get_peft_model(text_model, lora_config)
    
    return peft_model


def print_lora_info(model: nn.Module) -> Dict[str, Any]:
    """
    打印和返回 LoRA 应用信息（支持细粒度层分析）
    
    参数：
        model: 应用了 LoRA 的模型
    
    返回：
        包含 LoRA 信息的字典
    
    信息包括：
        - 应用了 LoRA 的层列表（按层索引分组）
        - 可训练参数统计
        - LoRA 参数统计
    """
    info = {
        "lora_layers": [],
        "lora_layer_indices": set(),  # 应用了 LoRA 的层索引
        "trainable_params": 0,
        "all_params": 0,
        "lora_params": 0,
    }
    
    # 收集应用了 LoRA 的层
    print("\n" + "="*60)
    print("LoRA 应用信息")
    print("="*60)
    
    lora_layers = []
    lora_layer_indices = set()
    
    for name, module in model.named_modules():
        # 检查是否包含 lora_A 参数（PEFT 的 LoRA 标志）
        if any(n.startswith('lora_') for n, _ in module.named_parameters(recurse=False)):
            lora_layers.append(name)
            # 提取层索引（例如从 "transformer.resblocks.8.mlp.c_fc" 提取 8）
            if 'resblocks.' in name:
                try:
                    layer_idx = int(name.split('resblocks.')[1].split('.')[0])
                    lora_layer_indices.add(layer_idx)
                except:
                    pass
            print(f"✓ LoRA 应用到: {name}")
    
    info["lora_layers"] = lora_layers
    info["lora_layer_indices"] = sorted(lora_layer_indices)
    
    # 打印层索引摘要
    if lora_layer_indices:
        print(f"\n应用 LoRA 的层索引: {sorted(lora_layer_indices)}")
    
    # 统计参数
    print("\n" + "-"*60)
    print("参数统计")
    print("-"*60)
    
    trainable_params = 0
    all_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()
    
    info["trainable_params"] = trainable_params
    info["all_params"] = all_params
    info["lora_params"] = lora_params
    
    print(f"总参数量:        {all_params:,}")
    print(f"可训练参数量:    {trainable_params:,}")
    print(f"LoRA 参数量:     {lora_params:,}")
    print(f"可训练参数比例:  {100 * trainable_params / all_params:.4f}%")
    print("="*60 + "\n")
    
    return info


def validate_lora_config(lora_config_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    验证 LoRA 配置参数的合法性
    
    参数：
        lora_config_dict: 从配置文件读取的 LoRA 配置字典
    
    返回：
        验证并补充默认值后的配置字典，如果 enabled=False 则返回 None
    
    配置格式示例：
        {
            "enabled": true,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["out_proj", "c_fc", "c_proj"],
            "layers_to_transform": [8, 9, 10, 11],
            "bias": "none"
        }
    """
    if lora_config_dict is None:
        return None
    
    # 检查是否启用
    if not lora_config_dict.get("enabled", False):
        return None
    
    # 验证和设置默认值
    config = {
        "enabled": True,
        "r": lora_config_dict.get("r", 16),
        "lora_alpha": lora_config_dict.get("lora_alpha", 32),
        "lora_dropout": lora_config_dict.get("lora_dropout", 0.05),
        "target_modules": lora_config_dict.get("target_modules", None),
        "layers_to_transform": lora_config_dict.get("layers_to_transform", None),
        "bias": lora_config_dict.get("bias", "none"),
    }
    
    # 验证参数范围
    assert config["r"] > 0, f"LoRA rank must be positive, got {config['r']}"
    assert config["lora_alpha"] > 0, f"LoRA alpha must be positive, got {config['lora_alpha']}"
    assert 0 <= config["lora_dropout"] < 1, f"LoRA dropout must be in [0, 1), got {config['lora_dropout']}"
    assert config["bias"] in ["none", "all", "lora_only"], f"Invalid bias option: {config['bias']}"
    
    # 验证 layers_to_transform
    if config["layers_to_transform"] is not None:
        assert all(isinstance(i, int) and i >= 0 for i in config["layers_to_transform"]), \
            "layers_to_transform must be a list of non-negative integers"
    
    # 验证 target_modules
    if config["target_modules"] is not None:
        valid_targets = ["in_proj_weight", "out_proj", "c_fc", "c_proj", "text_projection"]
        for module in config["target_modules"]:
            assert module in valid_targets, \
                f"Invalid target_module: {module}. Valid options: {valid_targets}"
    
    return config


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    提取模型中的 LoRA 参数（包括 bias）
    
    参数：
        model: 应用了 LoRA 的模型（PeftModel）
    
    返回：
        只包含 LoRA 参数的 state_dict（包括 lora_A, lora_B 和相关 bias）
    
    用途：
        可以单独保存 LoRA 权重，文件大小通常只有几MB
    
    注意：
        使用 PEFT 自带的 get_peft_model_state_dict 方法，正确处理 bias 参数
    """
    # 使用 PEFT 自带方法，自动处理 LoRA 参数和 bias
    lora_state_dict = get_peft_model_state_dict(model)
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, lora_state_dict: Dict[str, torch.Tensor]):
    """
    加载 LoRA 参数到模型（包括 bias）
    
    参数：
        model: 应用了 LoRA 的模型（PeftModel）
        lora_state_dict: LoRA 参数字典
    
    用途：
        从单独保存的 LoRA 权重文件恢复
    
    注意：
        使用 PEFT 自带的 set_peft_model_state_dict 方法，正确处理 bias 参数
    """
    # 使用 PEFT 自带方法，自动处理 LoRA 参数和 bias
    set_peft_model_state_dict(model, lora_state_dict)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    统计模型参数数量
    
    参数：
        model: 模型
        trainable_only: 是否只统计可训练参数
    
    返回：
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def convert_text_projection_to_linear(text_model: nn.Module) -> nn.Module:
    """
    将TextTransformer的text_projection从Parameter转换为Linear层
    
    这样可以对投影层应用LoRA。
    
    参数：
        text_model: TextTransformer实例
    
    返回：
        修改后的text_model
    
    注意：
        - 只有当text_projection存在且为Parameter时才转换
        - 转换后保持相同的权重值
        - Linear层不使用bias（与原Parameter行为一致）
        - 需要在应用LoRA之前调用此函数
    """
    if hasattr(text_model, 'text_projection') and text_model.text_projection is not None:
        # 检查是否为Parameter
        if isinstance(text_model.text_projection, nn.Parameter):
            # 获取原始权重的形状 [width, output_dim]
            weight = text_model.text_projection.data
            width, output_dim = weight.shape
            
            # 创建Linear层（不使用bias）
            # 注意：nn.Linear的权重形状是 [output_dim, input_dim]，需要转置
            linear_proj = nn.Linear(width, output_dim, bias=False)
            linear_proj.weight.data = weight.t()  # 转置以匹配Linear层的权重格式
            
            # 删除原始Parameter并添加Linear层作为module
            # 必须先删除Parameter，否则会报错
            delattr(text_model, 'text_projection')
            text_model.text_projection = linear_proj
            
            print(f"✓ text_projection转换为Linear层: [{width}] -> [{output_dim}]")
    
    return text_model


# ============================================================================
# Video LoRA 相关函数（复用text LoRA的核心逻辑）
# ============================================================================

def validate_video_lora_config(lora_config_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    验证 Video LoRA 配置参数的合法性
    
    VisionTransformer使用FlashMLP，模块命名与TextTransformer不同：
        - transformer.resblocks.{i}.mlp.fc1: MLP第一层（对应TextTransformer的c_fc）
        - transformer.resblocks.{i}.mlp.fc2: MLP第二层（对应TextTransformer的c_proj）
        - transformer.resblocks.{i}.attn.out_proj: 注意力输出投影
        - image_projection: 输出投影（需转换为Linear才能应用LoRA）
    
    配置格式示例：
        {
            "enabled": true,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["fc1", "fc2"],  # 注意：VisionTransformer使用fc1/fc2
            "layers_to_transform": [8, 9, 10, 11],
            "bias": "none"
        }
    """
    if lora_config_dict is None:
        return None
    
    if not lora_config_dict.get("enabled", False):
        return None
    
    # 验证和设置默认值
    config = {
        "enabled": True,
        "r": lora_config_dict.get("r", 16),
        "lora_alpha": lora_config_dict.get("lora_alpha", 32),
        "lora_dropout": lora_config_dict.get("lora_dropout", 0.05),
        "target_modules": lora_config_dict.get("target_modules", None),
        "layers_to_transform": lora_config_dict.get("layers_to_transform", None),
        "bias": lora_config_dict.get("bias", "none"),
    }
    
    # 默认target_modules为VisionTransformer的MLP层
    if config["target_modules"] is None:
        config["target_modules"] = ["fc1", "fc2"]
    
    # 验证参数范围
    assert config["r"] > 0, f"LoRA rank must be positive, got {config['r']}"
    assert config["lora_alpha"] > 0, f"LoRA alpha must be positive"
    assert 0 <= config["lora_dropout"] < 1, f"LoRA dropout must be in [0, 1)"
    assert config["bias"] in ["none", "all", "lora_only"], f"Invalid bias option"
    
    # 验证 layers_to_transform
    if config["layers_to_transform"] is not None:
        assert all(isinstance(i, int) and i >= 0 for i in config["layers_to_transform"]), \
            "layers_to_transform must be a list of non-negative integers"
    
    # 验证 target_modules（VisionTransformer使用fc1/fc2而非c_fc/c_proj）
    # 注意：当 use_flash_attn=False 时，VisionTransformer 使用 c_fc/c_proj
    valid_targets = ["fc1", "fc2", "out_proj", "image_projection", "c_fc", "c_proj"]
    for module in config["target_modules"]:
        assert module in valid_targets, \
            f"Invalid video target_module: {module}. Valid: {valid_targets}"
    
    return config


def convert_image_projection_to_linear(vision_model: nn.Module) -> nn.Module:
    """
    将VisionTransformer的image_projection从Parameter转换为Linear层
    
    类似于convert_text_projection_to_linear，便于对投影层应用LoRA。
    
    参数：
        vision_model: VisionTransformer实例
    
    返回：
        修改后的vision_model
    
    注意：
        - VisionTransformer的image_projection形状为 [width, output_dim]
        - Linear层权重形状是 [output_dim, input_dim]，需要转置
    """
    if hasattr(vision_model, 'image_projection') and vision_model.image_projection is not None:
        if isinstance(vision_model.image_projection, nn.Parameter):
            weight = vision_model.image_projection.data
            width, output_dim = weight.shape
            
            # 创建Linear层（不使用bias）
            linear_proj = nn.Linear(width, output_dim, bias=False)
            linear_proj.weight.data = weight.t()  # 转置匹配Linear权重格式
            
            delattr(vision_model, 'image_projection')
            vision_model.image_projection = linear_proj
            
            print(f"✓ image_projection转换为Linear层: [{width}] -> [{output_dim}]")
    
    return vision_model


def apply_lora_to_vision_model(
    vision_model: nn.Module,
    lora_config: LoraConfig,
) -> nn.Module:
    """
    将 LoRA 应用到 VisionTransformer
    
    与apply_lora_to_text_model逻辑完全一致，复用PEFT的get_peft_model。
    
    参数：
        vision_model: VisionTransformer 实例
        lora_config: PEFT LoraConfig 配置
    
    返回：
        应用了 LoRA 的模型（PeftModel 包装）
    """
    peft_model = get_peft_model(vision_model, lora_config)
    return peft_model

