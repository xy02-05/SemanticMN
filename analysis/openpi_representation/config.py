"""
OpenPI 表征分析配置文件
定义模型路径、数据路径、提取参数、输出目录

对标 bridge_representation/config.py（SpatialVLA 版本）
"""
import os
import json
from collections import defaultdict

# ===================== 代码路径 =====================
OPENPI_DIR = "/root/data/xuyuan1/Codes/analysis/openpi"

# ===================== 数据路径 =====================
# 复用 SpatialVLA 的数据子集（保证所有模型输入 100% 一致）
SPATIALVLA_OUTPUT_DIR = "/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs"
# v2: 使用完整 67-task 子集 (3350 轨迹), 原来只用 35 task (1750 轨迹)
SUBSET_DIR = os.path.join(SPATIALVLA_OUTPUT_DIR, "data_subset")
# 旧 35-task 子集路径 (备份引用)
SUBSET_DIR_35TASKS = os.path.join(SPATIALVLA_OUTPUT_DIR, "data_subset_35tasks")

# norm_stats 路径（OpenPI bridgev2 训练用的归一化统计量）
NORM_STATS_PATH = (
    "/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/"
    "bridgev2_rlds_train/raw/50000/assets/bridgev2/norm_stats.json"
)

# task 元数据（与 SpatialVLA 共用）
META_DIR = "/root/data/xuyuan1/dataset/bridge_orig/bridge_orig_lerobot/meta"
TASKS_WITH_ID_PATH = os.path.join(META_DIR, "tasks_with_id.jsonl")

# ===================== 模型路径 =====================
# 预训练模型（PI0 base）
PRETRAINED_PATH = "/root/data/xuyuan1/dataset/pi0_base_pytorch"

# 全参数微调模型（bridgev2_rlds_train/raw，取最新 step）
RAW_FT_PATH = (
    "/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/"
    "bridgev2_rlds_train/raw/50000"
)

# 对齐模型（LoRA 已合并到 model.safetensors）
ALIGNED_PATH = (
    "/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/checkpoints/"
    "bridgev2_rlds_train_align_lora/align_lora_task_id_0.1/50000"
)

MODEL_CONFIGS = {
    "pretrained": {
        "weight_path": PRETRAINED_PATH,
        "label": "Pretrained (PI0 Base)",
    },
    "raw_ft": {
        "weight_path": RAW_FT_PATH,
        "label": "Raw FT (bridgev2)",
    },
    "aligned": {
        "weight_path": ALIGNED_PATH,
        "label": "Aligned (LoRA+align)",
    },
}

# ===================== 模型配置 =====================
# PI0 模型参数（与训练时一致）
PI0_MODEL_CONFIG = {
    "action_dim": 32,
    "action_horizon": 5,
    "max_token_len": 48,       # Pi0 默认
    "use_learnable_token": False,
    "dtype": "bfloat16",
}

# ===================== 提取参数 =====================
# gemma_expert (gemma_300m) 共 18 层，选 10 个代表性层
LAYER_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]

# 每个 task 采样的轨迹数（与 SpatialVLA 一致）
TRAJECTORIES_PER_TASK = 50

# ===================== 输出路径 =====================
OUTPUT_DIR = "/root/data/xuyuan1/Codes/analysis/openpi_representation/outputs"
FEATURE_DIR = os.path.join(OUTPUT_DIR, "features")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# ===================== 从 SpatialVLA config 复用 task 定义 =====================
# 直接 import SpatialVLA config 中的 task 列表
import sys
sys.path.insert(0, "/root/data/xuyuan1/Codes/analysis")
from bridge_representation.config import (
    SELECTED_TASKS, SELECTED_TASK_INDICES, TASK_INDEX_TO_TEXT,
    HARD_GROUPS, TARGET_TASK_LANGS,
    get_canonical_task_label,
)
