"""
LIBERO Representation 分析配置

集中管理所有路径、checkpoint 列表、模型参数、提取超参。
包括:
  - pi0_libero_full/raw/: 全参数 fine-tune（无 alignment）的 checkpoint 序列
  - pi0_libero_align_lora_qwen/: alignment + LoRA 训练的 checkpoint（LoRA 已合并到 model.safetensors）
"""
import os

# ===================== 根目录 =====================
# 默认保持远端旧目录不变；本机通过环境变量复用同一套分析脚本。
PROJECT_ROOT = os.environ.get("MIRROR_PROJECT_ROOT", "/root/data/xuyuan1")
WORK_ROOT = os.environ.get(
    "MIRROR_WORK_ROOT",
    os.path.join(PROJECT_ROOT, "Codes/mirror_neuron"),
)
DATA_ROOT = os.environ.get(
    "MIRROR_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "dataset"),
)
OPENPI_ROOT = os.environ.get(
    "OPENPI_ROOT",
    os.path.join(WORK_ROOT, "vlas/openpi"),
)
ANALYSIS_ROOT = os.environ.get(
    "MIRROR_ANALYSIS_ROOT",
    os.path.join(PROJECT_ROOT, "Codes/analysis"),
)

# ===================== 模型 checkpoint 路径 =====================
CKPT_BASE = os.path.join(OPENPI_ROOT, "checkpoints")
PRETRAINED_PATH = os.environ.get(
    "PI0_PRETRAINED_PATH",
    os.path.join(DATA_ROOT, "pi0_base_pytorch"),
)

# pi0.5 LIBERO finetune (官方 JAX → 转 PyTorch 后路径)
PI05_LIBERO_JAX_PATH = os.path.join(PROJECT_ROOT, "dataset/pi05_libero_official")
PI05_LIBERO_PYTORCH_PATH = os.path.join(PROJECT_ROOT, "dataset/pi05_libero_pytorch")

# 全参数 fine-tune checkpoint 序列（无 alignment）
CHECKPOINTS = {
    "pretrained": {"path": PRETRAINED_PATH, "step": 0},
    "step_5k":    {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/5000"), "step": 5000},
    "step_10k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/10000"), "step": 10000},
    "step_15k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/15000"), "step": 15000},
    "step_20k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/20000"), "step": 20000},
    "step_25k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/25000"), "step": 25000},
    "step_30k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/30000"), "step": 30000},
    "step_35k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/35000"), "step": 35000},
    "step_40k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/40000"), "step": 40000},
    "step_45k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/45000"), "step": 45000},
    "step_50k":   {"path": os.path.join(CKPT_BASE, "pi0_libero_full/raw/50000"), "step": 50000},
}

# Alignment + LoRA 训练 checkpoint（LoRA 已合并到 model.safetensors，可直接加载）
# lq_sep: learnable_query + disentangle (pre_pool)
# lq_sep_new: 同上，第二轮实验
_ALIGN_BASE = os.path.join(CKPT_BASE, "pi0_libero_align_lora_qwen")
ALIGN_CHECKPOINTS = {
    "align_sep_20k":     {"path": os.path.join(_ALIGN_BASE, "lora_sigmoid_st01_lq_sep/20000"), "step": 20000, "run": "lq_sep"},
    "align_sep_25k":     {"path": os.path.join(_ALIGN_BASE, "lora_sigmoid_st01_lq_sep/25000"), "step": 25000, "run": "lq_sep"},
    "align_sep_30k":     {"path": os.path.join(_ALIGN_BASE, "lora_sigmoid_st01_lq_sep/30000"), "step": 30000, "run": "lq_sep"},
    "align_new_20k":     {"path": os.path.join(_ALIGN_BASE, "lora_sigmoid_st01_lq_sep_new/20000"), "step": 20000, "run": "lq_sep_new"},
    "align_new_25k":     {"path": os.path.join(_ALIGN_BASE, "lora_sigmoid_st01_lq_sep_new/25000"), "step": 25000, "run": "lq_sep_new"},
    "align_new_30k":     {"path": os.path.join(_ALIGN_BASE, "lora_sigmoid_st01_lq_sep_new/30000"), "step": 30000, "run": "lq_sep_new"},
    # 用户指定的当前 alignment-trained ckpt（pi0_libero_full_qwen/align_new/{20000,30000}）
    # 注意: 10000 step 实际没保存 (训练只在 20k 和 30k 存了 ckpt)
    "align_full_qwen_20k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/align_new/20000"), "step": 20000, "run": "align_full_qwen"},
    "align_full_qwen_30k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/align_new/30000"), "step": 30000, "run": "align_full_qwen"},
    # new_DSN 系列（DSN LayerNorm + hidden_dim bottleneck 优化后训出的 alignment ckpt）
    "new_dsn_30k":         {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN/30000"), "step": 30000, "run": "new_DSN"},
    # new_DSN_new 系列
    "new_dsn_new_5k":  {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/5000"),  "step": 5000,  "run": "new_DSN_new"},
    "new_dsn_new_10k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/10000"), "step": 10000, "run": "new_DSN_new"},
    "new_dsn_new_15k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/15000"), "step": 15000, "run": "new_DSN_new"},
    "new_dsn_new_20k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/20000"), "step": 20000, "run": "new_DSN_new"},
    "new_dsn_new_25k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/25000"), "step": 25000, "run": "new_DSN_new"},
    "new_dsn_new_30k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/30000"), "step": 30000, "run": "new_DSN_new"},
    "new_dsn_new_35k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/35000"), "step": 35000, "run": "new_DSN_new"},
    "new_dsn_new_40k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/40000"), "step": 40000, "run": "new_DSN_new"},
    "new_dsn_new_45k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/45000"), "step": 45000, "run": "new_DSN_new"},
    "new_dsn_new_50k": {"path": os.path.join(CKPT_BASE, "pi0_libero_full_qwen/new_DSN_new/50000"), "step": 50000, "run": "new_DSN_new"},
}

# pi0.5 LIBERO 官方 finetune checkpoint（JAX → PyTorch 转换后产物）
# 默认 model_variant="pi0"，pi05 系列必须显式标注以触发 Pi0Config(pi05=True)
PI05_CHECKPOINTS = {
    "pi05_libero_official": {
        "path": PI05_LIBERO_PYTORCH_PATH,
        "step": 0,
        "model_variant": "pi05",
    },
}

# 所有 checkpoint 的合集（用于 --checkpoint all 模式）
ALL_CHECKPOINTS = {**CHECKPOINTS, **ALIGN_CHECKPOINTS, **PI05_CHECKPOINTS}

# ===================== 数据路径 =====================
LIBERO_DATA_DIR = os.path.join(DATA_ROOT, "physical-intelligence/libero")
LIBERO_META_DIR = os.path.join(LIBERO_DATA_DIR, "meta")
TASKS_WITH_ID_PATH = os.path.join(LIBERO_META_DIR, "tasks_with_id.jsonl")

# LIBERO 评测结果根目录
EVAL_RESULTS_BASE = os.path.join(OPENPI_ROOT, "results/libero")

# ===================== Text Embedding 路径 =====================
EMBEDDING_DIR = os.path.join(DATA_ROOT, "embedding")

TEXT_EMBEDDINGS = {
    "qwen3": {
        "path": os.path.join(EMBEDDING_DIR, "libero_qwen3_text_features.npz"),
        "key": "sentence_embeddings",  # npz 中的 key
        "dim": 4096,
    },
    "egohod": {
        "path": os.path.join(EMBEDDING_DIR, "libero_egohod_text_features.npz"),
        "key": "sentence_embeddings",
        "dim": 512,
    },
}

# ===================== Pi0 模型配置 =====================
# 与训练时一致的参数（action_horizon=50 for LIBERO）
PI0_MODEL_CONFIG = {
    "action_dim": 32,
    "action_horizon": 50,
    "max_token_len": 48,
    "use_learnable_token": False,
    "dtype": "bfloat16",
}

# pi0.5: max_token_len=200, state 离散化进 prompt token, time 走 adaRMSNorm
PI05_MODEL_CONFIG = {
    "action_dim": 32,
    "action_horizon": 50,
    "max_token_len": 200,
    "use_learnable_token": False,
    "dtype": "bfloat16",
    "pi05": True,
}

# ===================== 提取参数 =====================
# action expert (gemma_300m) 共 18 层 (0-17)
# clean 模式（forward t=0,noise=0）默认提 4 层代表点：
#   0=输入  5=浅层  10=alignment 层  17=输出
LAYER_INDICES = [0, 5, 10, 17]
# rollout 模式（sample_actions 10 步去噪）需要更密集采样以覆盖深度方向：
# 偶数层 + 17，与 main.py 推理时 chunk feature 保存层一致
ROLLOUT_LAYER_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]

# ===================== 输出路径 =====================
OUTPUT_DIR = os.path.join(WORK_ROOT, "data_process/libero/outputs")
SPLIT_PATH = os.path.join(OUTPUT_DIR, "split.json")  # train/test 划分文件
FEATURE_DIR = os.environ.get(
    "MIRROR_FEATURE_DIR",
    os.path.join(OUTPUT_DIR, "features"),
)
PROBE_DIR = os.path.join(OUTPUT_DIR, "probes")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

# ===================== 归一化统计量路径 =====================
# LIBERO 训练时的 norm_stats（用于 state/actions 归一化）
NORM_STATS_PATH = os.path.join(
    CKPT_BASE, "pi0_libero_full/raw/30000/assets/libero/norm_stats.json"
)

# pi0.5 官方 ckpt 自带 norm_stats（state/actions 维度与 pi0 不同：8 vs 7）
PI05_NORM_STATS_PATH = os.path.join(
    PI05_LIBERO_JAX_PATH, "assets/physical-intelligence/libero/norm_stats.json"
)

# ===================== LIBERO 评测 suite 定义 =====================
LIBERO_SUITES = {
    "libero_spatial": {"task_indices": list(range(30, 40)), "n_tasks": 10},
    "libero_object":  {"task_indices": list(range(20, 30)), "n_tasks": 10},
    "libero_goal":    {"task_indices": list(range(10, 20)), "n_tasks": 10},
    "libero_10":      {"task_indices": list(range(0, 10)),  "n_tasks": 10},
}

# ===================== Mechanistic Analysis 输出路径 =====================
MECHANISTIC_DIR = os.path.join(RESULT_DIR, "mechanistic")
