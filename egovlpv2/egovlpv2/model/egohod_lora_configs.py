# EgoHOD模型LoRA配置示例

"""
本文件展示如何在训练配置中使用Text LoRA和Video LoRA

支持对Text和Video模型分别配置LoRA，可独立或联合使用。
"""

# ============================================================================
# Text LoRA 配置示例（TextTransformer，12层）
# ============================================================================

# Text配置1: 只对最后4层的MLP应用LoRA（推荐）
text_lora_last_4_layers = {
    "enabled": True,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["c_fc", "c_proj"],  # MLP层
    "layers_to_transform": [8, 9, 10, 11],
    "bias": "none"
}

# Text配置2: 所有层的MLP
text_lora_all_layers = {
    "enabled": True,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["c_fc", "c_proj"],
    "layers_to_transform": None,  # 所有层
    "bias": "none"
}

# Text配置3: 含text_projection的LoRA
text_lora_with_projection = {
    "enabled": True,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["c_fc", "c_proj", "text_projection"],
    "layers_to_transform": [8, 9, 10, 11],
    "bias": "none"
}


# ============================================================================
# Video LoRA 配置示例（VisionTransformer，ViT-B/16有12层）
# 注意：VisionTransformer使用FlashMLP，模块名为fc1/fc2，不是c_fc/c_proj
# ============================================================================

# Video配置1: 只对最后4层的MLP应用LoRA（推荐）
video_lora_last_4_layers = {
    "enabled": True,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["fc1", "fc2"],  # VisionTransformer的MLP层命名
    "layers_to_transform": [8, 9, 10, 11],
    "bias": "none"
}

# Video配置2: 所有层的MLP
video_lora_all_layers = {
    "enabled": True,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["fc1", "fc2"],
    "layers_to_transform": None,
    "bias": "none"
}

# Video配置3: 含image_projection的LoRA
video_lora_with_projection = {
    "enabled": True,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["fc1", "fc2", "image_projection"],
    "layers_to_transform": [8, 9, 10, 11],
    "bias": "none"
}

# 配置禁用
lora_disabled = None


# ============================================================================
# 在训练配置JSON中的使用示例
# ============================================================================
"""
{
    "arch": {
        "type": "EgoHODModel",
        "args": {
            "video_params": {"num_frames": 4},
            "text_params": {},
            "projection_dim": 512,
            "load_checkpoint": "weights/ViT-B-16.pt",
            "egohod_checkpoint_path": "weights/base_best.pt",
            "lora_config": {
                "enabled": true,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["c_fc", "c_proj"],
                "layers_to_transform": [8, 9, 10, 11],
                "bias": "none"
            },
            "video_lora_config": {
                "enabled": true,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["c_fc", "c_proj"],
                "layers_to_transform": [8, 9, 10, 11],
                "bias": "none"
            }
        }
    }
}
"""


# ============================================================================
# 参数量对比（ViT-B/16）
# ============================================================================
"""
TextTransformer (12层, width=512):
- Token embedding: 512 * 49408 ≈ 25.3M
- 12层transformer: 约 38M
- 总计: 约 63.6M

VisionTransformer (12层, width=768):
- Patch embedding + pos: 约 0.6M
- 12层transformer: 约 85M
- 总计: 约 86M

LoRA参数量估算 (rank=16, 最后4层MLP):
- Text: 每层(512→2048→512) ≈ 82K, 4层 ≈ 0.33M
- Video: 每层(768→3072→768) ≈ 123K, 4层 ≈ 0.49M
"""


# ============================================================================
# 训练建议
# ============================================================================
"""
1. 推荐配置（平衡性能和效率）:
   - rank: 16, alpha: 32
   - layers_to_transform: [8, 9, 10, 11] (最后4层)
   - target_modules: ["c_fc", "c_proj"] (只MLP)

2. 小数据集/快速实验:
   - rank: 8, layers_to_transform: [10, 11]

3. 大数据集/追求性能:
   - rank: 32-64, layers_to_transform: [6,7,8,9,10,11]

4. 学习率: LoRA参数使用1e-4到5e-4

5. Text vs Video LoRA选择:
   - 语言理解任务: 只用Text LoRA
   - 视觉理解任务: 只用Video LoRA
   - 多模态对齐: 两者都用
"""





