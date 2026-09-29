import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import json
import pathlib  # 用于读取 scripts/intrinsics.json 路径
import shutil  # 用于复制配置文件到实验目录
import torch
import torch.distributed as dist
from tqdm import tqdm
import collections # 新增：导入collections模块，用于实现滑动平均
import random  # Python内置随机数生成器
import numpy as np  # numpy随机数生成器

# ================== Accelerate和DeepSpeed相关导入 ==================
from accelerate.utils import set_seed as accelerate_set_seed
import deepspeed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader

# ================== 导入utils模块中的函数 ==================
from train.utils.accelerator_utils import create_accelerator, update_deepspeed_scheduler_config
from train.utils.ckpt_utils import save_checkpoint, save_final_model, load_checkpoint

# ================== 自定义数据加载和训练相关导入 ==================
from train.dist_utils import init_dist
from train.monkey_patch import (
    concat_pad_data_collator,
    SaveProcessorCallback,
)
from train.utils.data_utils import (
    create_bridge_dataloader_and_sampler,  # 从utils模块导入数据加载函数
    create_train_sampler,
    create_train_dataloader
)
from train.utils.metric_utils import compute_action_metrics, evaluate_model
from train.utils.optim_utils import create_optimizer_and_scheduler, inspect_optimizer_param_groups

# ================== HuggingFace基础组件导入 ==================
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    TrainingArguments,  # 保留用于参数解析，但不用于Trainer
)
from peft import get_peft_model, LoraConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
from data.dataset import build_datasets
from model import (
    SpatialVLAConfig,
    SpatialVLAForConditionalGeneration,
    SpatialVLAProcessor,
    SpatialActionTokenizer,
    MIMICVLAConfig,
    MIMICVLAModel,
)
from egovlpv2.utils.model_data_init import (
    init_egovlpv2_training_components,
    init_egohod_training_components,
    init_egovideo_training_components,  # 新增：支持EgoVideo模型
    init_clip_components,  # 新增：支持CLIP模型
    init_alignment_model_components,
    init_embedding_components  # 新增：支持embedding模式
)

# === 不再使用trainer，改用accelerate+deepspeed直接训练 ===

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(default=None,
        metadata={"help": "Path to pretrained model or identifier for resume training."},
    )
    freeze_llm_embed: bool = field(
        default=True, metadata={"help": "Set to True to freeze the LLM embeddings."},
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    freeze_egovlpv2_model: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the EgoVLPv2 model parameters."},
    )
    lora: int = field(
        default=0,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."},
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 8."},
    )
    lora_target: Optional[str] = field(
        default="linear",
        metadata={"help": "Set the LoRA adapter target modules. Default is linear."},
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Set the modules to save for LoRA. Default is none."},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use gradient checkpointing."},
    )
    flash_attn: bool = field(
        default=True,
        metadata={"help": "Set to True to use Flash Attention 2.0."},
    )
    adapt_emb: Optional[str] = field(
        default=None,
        metadata={"help": "Path to adapt the spatial embeddings with new gaussian config."},
    )
    adpt_feature: bool = field(
        default=False,
        metadata={"help": "Set to True to adapt the feature embeddings."},
    )
    min_sigma: float = field(
        default=0.0,
        metadata={"help": "Set the minimum sigma for creating action grids."},
    )
    # MIMIC-VLA specific parameters - always enabled for alignment training
    use_egovlpv2: bool = field(
        default=True,
        metadata={"help": "Enable EgoVLPv2 co-training for alignment."},
    )
    use_alignment: bool = field(
        default=True,
        metadata={"help": "Enable alignment model training."},
    )
    vlm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for EgoVLPv2 loss in MIMIC-VLA training."},
    )
    alignment_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for alignment loss in MIMIC-VLA training."},
    )
    egovlpv2_config_path: str = field(
        default="/data/xuyuan/UniVLA_env/mirror_neuron/egovlpv2/egovlpv2/configs/ft/charades_a800_align_spatial_sigmoid.json",
        metadata={"help": "Path to EgoVLPv2 configuration file."},
    )
    vlm_mode: str = field(
        default="egovlpv2",
        metadata={"help": "VLM模型类型: 'egovlpv2', 'egohod', 'egovideo', 'qwen3', 'embedding', 'clip'. 默认'egovlpv2'."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_root_dir: Optional[str] = field(
        default="datasets/open-x-embodiment",
        metadata={"help": "The root directory of the dataset. Default is `datasets/open-x-embodiment`."},
    )
    data_mix: Optional[str] = field(
        default="bridge",
        metadata={"help": "The name of the dataset mixture. Default is `bridge`."},
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=1000_000,
        metadata={"help": "The shuffle buffer size for the dataset. Default is 1000000."},
    )
    tsfm_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds transform. Default is 1."},
    )
    read_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds reader. Default is 1."},
    )
    obs_backward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of backward steps in observation. 0 indicates current"},
    )
    obs_backward_delta: Optional[int] = field(
        default=1, metadata={"help": "Backward delta in observation."}
    )
    action_forward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of forward steps in action. 0 indicates current"},
    )
    fix_raw_length: Optional[int] = field(
        default=None, metadata={"help": "fix the iterable dataset iter length."}
    )
    use_raw_dataloader: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use raw dataloader"}
    )
    load_future_images: Optional[bool] = field(
        default=False, metadata={"help": "是否加载未来图像序列用于训练。设为True时会使用traj_transforms.py中的future_images功能"}
    )
    task_filename: Optional[str] = field(
        default=None, metadata={"help": "任务映射文件名（如 tasks.jsonl / tasks_with_id.jsonl），为空则默认 tasks.jsonl"}
    )




def main():
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 配置基础日志系统，使用更清晰的格式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # 获取用户指定的日志级别（通过--log_level参数）
    log_level = training_args.get_process_log_level()
    
    # 设置本模块的日志级别
    logger.setLevel(log_level)
    
    # 设置transformers库的日志级别，尊重用户设置
    # 注释掉强制INFO设置，改为使用用户指定的级别
    # if training_args.should_log: transformers.utils.logging.set_verbosity_info()  # 这行强制设置INFO，忽略用户参数
    set_verbosity(log_level)  # 使用用户指定的日志级别
    enable_default_handler()
    enable_explicit_format()
    
    # 设置deepspeed的日志级别（如果使用deepspeed）
    if training_args.deepspeed:
        import deepspeed
        # DeepSpeed使用标准的Python logging，设置其日志级别
        deepspeed_logger = logging.getLogger("DeepSpeed")
        deepspeed_logger.setLevel(log_level)
        accelerate_logger = logging.getLogger("accelerate")
        accelerate_logger.setLevel(log_level)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        ckpt_files = list(filter(lambda x: x.startswith("checkpoint"), os.listdir(training_args.output_dir)))
        if last_checkpoint is None and len(ckpt_files) > 0:
            ckpt_files = list(filter(lambda x: x.startswith("checkpoint"), os.listdir(training_args.output_dir)))
        if last_checkpoint is None and len(ckpt_files) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # ================== 1. 创建Accelerator实例（替换原有dist初始化） ==================
    logger.info("🚀 初始化Accelerator和DeepSpeed环境...")
    accelerator = create_accelerator(training_args, model_args)
    
    # 使用accelerate的seed设置确保分布式一致性
    accelerate_set_seed(training_args.seed, device_specific=True)

    # ================== 2. 模型和分词器初始化（保持原有逻辑） ==================
    logger.info("📖 加载处理器和分词器...")
    _processor = SpatialVLAProcessor.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    tokenizer = _processor.tokenizer
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    # ============ 关键 fix：注入 finetune 数据集的真实 intrinsics ============
    # base ckpt 的 processor_config.json 只有 bridge_orig + default 两条 K（都是 bridge 内参）。
    # 如果不在这里 overlay，finetune LIBERO 时 processor 会 fallback 到 default(=bridge K)，
    # 导致 Ego3D 反投影几何全错。pretrain 脚本通过 --intrinsic_config_path 注入；finetune
    # 这里必须做同样的事，把 scripts/intrinsics.json overlay 到现有 processor 上。
    # 用 print() 而非 logger.info：训练默认 --log_level warning 会屏蔽 INFO，
    # 但 K matrix 注入是必须可见的关键证据。
    _intrinsic_path = pathlib.Path("scripts/intrinsics.json")
    print(f"\n{'='*70}\n📐 [INTRINSIC FIX] start | path={_intrinsic_path.absolute()} exists={_intrinsic_path.exists()}\n{'='*70}", flush=True)
    print(f"  BEFORE keys = {list((_processor.intrinsic_config or {}).keys())}", flush=True)
    print(f"  BEFORE bridge_orig K[0,0] (after scale to img size) = {_processor.dataset_intrinsics.get('bridge_orig/1.0.0', torch.zeros(3,3))[0,0].item():.4f}", flush=True)
    if _intrinsic_path.exists():
        new_ic = json.load(open(_intrinsic_path))
        merged_ic = {**(_processor.intrinsic_config or {}), **new_ic}
        _processor.intrinsic_config = merged_ic
        _h = _processor.image_processor.size["height"]
        _w = _processor.image_processor.size["width"]
        new_di = {}
        for _k, _v in merged_ic.items():
            _K = torch.tensor(_v["intrinsic"]).float()
            _K[:2] *= torch.tensor([_w / _v["width"], _h / _v["height"]])[:, None]
            new_di[_k] = _K
        _processor.dataset_intrinsics = new_di
        print(f"  AFTER  keys = {list(merged_ic.keys())}", flush=True)
        print(f"  AFTER  image_processor.size = {_h}x{_w}", flush=True)
        for _k in merged_ic.keys():
            _K = _processor.dataset_intrinsics[_k]
            print(f"  AFTER  K['{_k}']: f=[{_K[0,0]:.3f}, {_K[1,1]:.3f}] cx={_K[0,2]:.3f} cy={_K[1,2]:.3f}", flush=True)
        # 关键 assert：libero_mix_no_noops/1.0.0 的 K 必须存在
        assert "libero_mix_no_noops/1.0.0" in _processor.dataset_intrinsics, "LIBERO K 注入失败!"
        print(f"✅ [INTRINSIC FIX] 成功：libero_mix_no_noops K = {_processor.dataset_intrinsics['libero_mix_no_noops/1.0.0'].tolist()}\n{'='*70}\n", flush=True)
    else:
        print(f"⚠ [INTRINSIC FIX] FAILED: {_intrinsic_path} 不存在！processor 将 fallback 到 bridge K\n{'='*70}\n", flush=True)
    
    logger.info("🏗️ 加载SpatialVLA配置...")
    config_spatialvla = SpatialVLAConfig.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, local_files_only=True)
    
    logger.info("🎭 加载MIMIC-VLA配置...")
    config = MIMICVLAConfig.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch_dtype, 
        local_files_only=True,
        use_egovlpv2=model_args.use_egovlpv2,
        use_alignment=model_args.use_alignment,
        vlm_loss_weight=model_args.vlm_loss_weight,
        alignment_loss_weight=model_args.alignment_loss_weight,
        egovlpv2_config_path=model_args.egovlpv2_config_path,
        vlm_mode=model_args.vlm_mode,
    )
    # 先创建SpatialVLA模型
    spatial_vla_model = SpatialVLAForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config_spatialvla,
        torch_dtype=torch_dtype,
        local_files_only=True
    )
    
    # 初始化多模型训练组件
    egovlpv2_components = None
    alignment_components = None
    
    # ⚠️ 关键修复：此时 spatial_vla_model 还在 CPU 上，需要使用正确的 GPU 设备
    # 使用 CUDA_VISIBLE_DEVICES 确定的第一个可用 GPU
    # 在分布式训练中，accelerator.prepare 后会移动到正确设备
    init_device = f"cuda:{training_args.local_rank}" if training_args.local_rank >= 0 else "cuda:0"
    logger.info(f"🔧 VLM组件将初始化到设备: {init_device}")
    
    # 判断是否需要初始化VLM组件来获取config
    # 条件：(1) use_egovlpv2=True 需要训练VLM，或 (2) freeze_egovlpv2_model=False 需要训练VLM参数（包括LoRA）
    # 或 (3) use_alignment=True 且 vlm_mode不是'ego'，alignment需要VLM的config
    vlm_mode = getattr(config, 'vlm_mode', 'ego')
    need_vlm_init = (config.use_egovlpv2 or 
                     not model_args.freeze_egovlpv2_model or config.use_alignment)
    
    if need_vlm_init:
        # 根据vlm_mode选择初始化函数
        if vlm_mode == 'egohod':
            # EgoHOD模式：use_egovlpv2控制是freeze还是training
            training_mode = config.use_egovlpv2  # use_egovlpv2=True → 训练VLM
            logger.info(f"初始化EgoHOD组件（{'训练' if training_mode else 'Freeze'}模式）...")
            egovlpv2_components = init_egohod_training_components(
                config_path=config.egovlpv2_config_path,
                device=init_device,
                infinite_dataloader=True,
                dtype=torch_dtype,
                training_mode=training_mode
            )
        elif vlm_mode == 'egovideo':
            logger.info("初始化EgoVideo推理组件（Freeze VLM模式）...")
            egovlpv2_components = init_egovideo_training_components(
                config_path=config.egovlpv2_config_path,
                device=init_device,  # 使用正确的GPU设备
                dtype=torch_dtype
            )
            logger.info("✅ EgoVideo组件初始化完成（参数已冻结，仅用于特征提取）")
        elif vlm_mode == 'embedding':
            # 新增：支持embedding模式（使用预计算的text-embedding特征）
            logger.info("初始化Embedding推理组件（使用预计算文本特征）...")
            egovlpv2_components = init_embedding_components(
                config_path=config.egovlpv2_config_path,
                device=init_device,  # ⚠️ 修复：使用正确的GPU设备
                dtype=torch_dtype
            )
            logger.info("✅ Embedding组件初始化完成（零计算开销，直接查表）")
        elif vlm_mode == 'clip':
            # 新增：支持CLIP模式（使用OpenAI CLIP模型）
            logger.info("初始化CLIP推理组件（Freeze VLM模式）...")
            egovlpv2_components = init_clip_components(
                config_path=config.egovlpv2_config_path,
                device=init_device,  # 使用正确的GPU设备
                dtype=torch_dtype
            )
            logger.info("✅ CLIP组件初始化完成（参数已冻结，仅用于特征提取）")
        else:
            logger.info(f"初始化EgoVLPv2训练组件（模式: {vlm_mode}）...")
            egovlpv2_components = init_egovlpv2_training_components(
                config_path=config.egovlpv2_config_path,
                device=init_device,  # ⚠️ 修复：使用正确的GPU设备
                infinite_dataloader=True,
                dtype=torch_dtype
            )
            logger.info("✅ EgoVLPv2组件初始化完成")
    
    if config.use_alignment:
        logger.info("初始化Alignment训练组件...")
        alignment_components = init_alignment_model_components(
            config_path=config.egovlpv2_config_path,
            device=init_device,  # ⚠️ 修复：使用正确的GPU设备
            dtype=torch_dtype
        )

        if alignment_components:
            logger.info("✅ Alignment组件初始化完成")
        else:
            logger.info("⚠️ Alignment组件初始化跳过（配置文件中未启用）")
    
    # 创建MIMIC-VLA包装器
    # 获取vlm_mode参数（默认为'ego'）
    vlm_mode = getattr(config, 'vlm_mode', 'ego')
    
    model = MIMICVLAModel(
        config=config, 
        vla_model=spatial_vla_model,
        egovlpv2_components=egovlpv2_components,
        alignment_components=alignment_components,
        vlm_mode=vlm_mode  # 传递vlm_mode参数
    )
    if model_args.flash_attn:
        model.spatial_vla.language_model.config._attn_implementation = model.spatial_vla.config.text_config._attn_implementation_internal = "flash_attention_2"
        model.spatial_vla.vision_tower.config._attn_implementation = model.spatial_vla.config.vision_config._attn_implementation_internal = "flash_attention_2"

    # 2. build datasets
    train_dataset, eval_dataset = build_datasets(
        data_args,
        training_args.output_dir,
        vla_processor=None,
    )

    # 3. build action tokenizer from current project
    action_tokenizer = SpatialActionTokenizer(
        tokenizer,
        num_bins=_processor.action_config["num_bins"],
        bin_policy=_processor.action_tokenizer.bin_policy,
        use_spherical=_processor.action_config["use_spherical"],
        min_sigma=_processor.action_config.get("min_sigma", 0.0),
    )
    
    if model_args.adapt_emb and config.use_spatial_token:
        logger.info(f"adapt spatial embeddings with guassian distribution {model_args.adapt_emb}")
        gs_params = json.load(open(model_args.adapt_emb))
        action_tokenizer.spatial_embedding_adaption(gs_params, model.spatial_vla.spatial_embed_tokens, model_args.min_sigma, model_args.adpt_feature)
        logger.info(f"new adaptation embedding {model.spatial_vla.spatial_embed_tokens.weight.data}")

        if model_args.adpt_feature:
            model_args.lora_target="linear"
            model_args.modules_to_save="spatial_embed_tokens"
            logger.info(f"reset lora_target to {model_args.lora_target} and modules_to_save {model_args.modules_to_save}")

    # overwrite attributes
    model.spatial_vla.action_token_begin_idx = model.spatial_vla.config.action_token_begin_idx = action_tokenizer.action_token_begin_idx
    model.spatial_vla.vision_tower.gradient_checkpointing = True

    if model_args.grad_checkpoint:
        model.spatial_vla.language_model._set_gradient_checkpointing()
    
    # set freeze params
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_llm_embed:
        model.spatial_vla.language_model.model.embed_tokens.weight.requires_grad = False

    if model_args.freeze_vision_tower:
        model.spatial_vla.vision_tower = model.spatial_vla.vision_tower.eval()
        _freeze_params(model.spatial_vla.vision_tower)
    
    # 冻结VLM模型参数（EgoVLPv2/EgoHOD等）：alignment-only训练时也必须冻结
    training_args.freeze_egovlpv2_model = model_args.freeze_egovlpv2_model
    if model_args.freeze_egovlpv2_model and hasattr(model, "egovlpv2_model") and model.egovlpv2_model is not None:
        model.egovlpv2_model = model.egovlpv2_model.eval()
        _freeze_params(model.egovlpv2_model)
        is_frozen = all(not p.requires_grad for p in model.egovlpv2_model.parameters())
        print(f"🔒 VLM模型参数已冻结。验证状态: {'成功' if is_frozen else '失败'}")

    model.spatial_vla.vision_zoe_model = model.spatial_vla.vision_zoe_model.eval()
    _freeze_params(model.spatial_vla.vision_zoe_model)

    # ================== SpatialVLA LoRA配置应用 ==================
    if model_args.lora:
        # peft https://github.com/huggingface/peft/blob/c1fe8105a5a4a612a6178699e1def5c66c2638d2/src/peft/tuners/tuners_utils.py#L1027
        if model_args.lora_target == "linear":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3" # ego3d
            ]
        elif model_args.lora_target == "linear+emb":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3", # ego3d
                "spatial_embed_tokens",
            ]
        elif model_args.lora_target == "linear+emb+h":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm_head", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3", # ego3d
                "spatial_embed_tokens",
            ]
        else:
            raise ValueError(f"don't support lora targets {model_args.lora_target}")
        
        # modules_to_save: https://github.com/huggingface/peft/issues/334#issuecomment-1786449397
        modules_to_save = model_args.modules_to_save.split("+") if model_args.modules_to_save else []
        lora_config = LoraConfig(
            r=model_args.lora,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            modules_to_save=modules_to_save,
        )
        model.spatial_vla = get_peft_model(model.spatial_vla, lora_config)
        logger.info(f"use SpatialVLA Lora ... with {model_args.lora_target} and modules {modules_to_save} ...")
        model.spatial_vla.print_trainable_parameters()

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad: logger.info(name)

    set_seed(training_args.seed)
    SpatialVLAConfig.register_for_auto_class() # register for auto save and map
    SpatialVLAForConditionalGeneration.register_for_auto_class()
    MIMICVLAConfig.register_for_auto_class() # register for auto save and map
    MIMICVLAModel.register_for_auto_class()
    SpatialVLAProcessor.register_for_auto_class()

    # ================== 3. 构建处理器（保持原有逻辑） ==================
    logger.info("🔧 构建SpatialVLA处理器...")
    statistic = train_dataset.ds_stats_pc
    _processor.statistics.update(statistic)
    processor = SpatialVLAProcessor(
        image_processor=_processor.image_processor,
        tokenizer=tokenizer,
        statistics=_processor.statistics,
        bin_policy=action_tokenizer.bin_policy,
        intrinsic_config=_processor.intrinsic_config,
        action_config=_processor.action_config,
        num_obs_steps=data_args.obs_backward_steps + 1,
        obs_delta=data_args.obs_backward_delta,
        action_chunk_size=data_args.action_forward_steps + 1,
    )

    # 设置模型相关属性
    model.spatial_vla.action_tokenizer = action_tokenizer
    train_dataset.vla_processor = processor
    model.tokenizer = tokenizer
    logger.info("✅ SpatialVLA处理器构建完成")
    
    # ================== 4. 创建数据加载器（使用utils函数） ==================
    logger.info("📊 创建数据加载器...")
    train_dataloader, eval_dataloader, train_sampler = create_bridge_dataloader_and_sampler(
        data_args=data_args,
        training_args=training_args,
        output_dir=training_args.output_dir,
        vla_processor=processor,
        tokenizer=tokenizer,
        accelerator=accelerator
    )
    
    # ================== 5. 创建优化器和调度器（使用utils函数） ==================
    logger.info("⚙️ 创建优化器和学习率调度器...")
    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    # The scheduler always follows the full epoch budget. ``max_steps`` is a
    # separate early-stop cap used by the Bridge reproduction runs, so a 10k
    # run remains the first 10k steps of the same two-epoch LR schedule.
    num_training_steps = num_update_steps_per_epoch * training_args.num_train_epochs
    
    # ================== 5.1. 更新DeepSpeed scheduler配置中的"auto"参数 ==================
    # logger.info("🔧 更新DeepSpeed scheduler配置...")
    # update_deepspeed_scheduler_config(accelerator, training_args, num_training_steps)
    
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,  # 传入原始模型用于参数分组，DeepSpeed需要在包装前获取优化器
        training_args=training_args,
        num_training_steps=num_training_steps,
        egovlpv2_config=egovlpv2_components['config'] if egovlpv2_components else None,
        alignment_config=alignment_components['config'] if alignment_components else None
    )
    logger.info("✅ 优化器和调度器创建完成")
    
    # ================== 检查优化器参数组（accelerator.prepare之前） ==================
    logger.info("🔍 检查optimizer参数组（accelerator.prepare之前）...")
    inspect_optimizer_param_groups(optimizer, "创建后", accelerator)
    
    # ================== 6. 初始化logging trackers（tensorboard等），确保只创建一个tracker ==================
    # 重要：只在主进程中初始化trackers，避免多进程环境下重复创建tensorboard文件
    tracker_config = getattr(accelerator, '_spatialvla_tracker_config', {})
    experiment_name = getattr(accelerator, '_spatialvla_experiment_name', 'spatialvla_experiment')
    
    accelerator.init_trackers(
        project_name="spatialvla",  # wandb项目名称
        config=tracker_config,     # 超参数配置字典
        init_kwargs={
            "wandb": {
                "entity": "x1141984720",
                "name": experiment_name,  # wandb实验run的名称
                "tags": ["spatial", "vla", "finetune"],  # wandb标签
            },
        }
    )
    logger.info(f"✅ Logging trackers初始化完成: {accelerator.log_with}")
    
    # ================== 7. 使用accelerate+deepspeed包装模型和优化器 ==================
    logger.info("🚀 使用accelerate和deepspeed包装模型和优化器...")
    # 重要：必须同时准备模型和优化器，让DeepSpeed能正确初始化ZeRO优化器
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    logger.info("✅ 模型和优化器包装完成")
    
    # ================== 关键修复：确保frozen的egovlpv2_model在正确设备上 ==================
    # DeepSpeed可能不会自动移动frozen参数的模块，需要手动确保
    if hasattr(model, 'module'):
        unwrapped_model = model.module
    else:
        unwrapped_model = model
    
    target_device = accelerator.device
    
    if hasattr(unwrapped_model, 'egovlpv2_model') and unwrapped_model.egovlpv2_model is not None:
        egovlpv2_device = next(unwrapped_model.egovlpv2_model.parameters()).device
        if egovlpv2_device != target_device:
            logger.warning(f"⚠️ egovlpv2_model在{egovlpv2_device}，需要移动到{target_device}")
            unwrapped_model.egovlpv2_model = unwrapped_model.egovlpv2_model.to(target_device)
            logger.info(f"✅ egovlpv2_model已移动到{target_device}")
        else:
            logger.info(f"✅ egovlpv2_model已在正确设备{target_device}")
    
    if hasattr(unwrapped_model, 'alignment_model') and unwrapped_model.alignment_model is not None:
        alignment_device = next(unwrapped_model.alignment_model.parameters()).device
        if alignment_device != target_device:
            logger.warning(f"⚠️ alignment_model在{alignment_device}，需要移动到{target_device}")
            unwrapped_model.alignment_model = unwrapped_model.alignment_model.to(target_device)
            logger.info(f"✅ alignment_model已移动到{target_device}")
        else:
            logger.info(f"✅ alignment_model已在正确设备{target_device}")
    
    # video_sampler只有buffers没有parameters，需要通过buffers检查设备
    if hasattr(unwrapped_model, 'video_sampler') and unwrapped_model.video_sampler is not None:
        sampler_buffers = list(unwrapped_model.video_sampler.buffers())
        if sampler_buffers:
            sampler_device = sampler_buffers[0].device
            if sampler_device != target_device:
                logger.warning(f"⚠️ video_sampler在{sampler_device}，需要移动到{target_device}")
                unwrapped_model.video_sampler = unwrapped_model.video_sampler.to(target_device)
                logger.info(f"✅ video_sampler已移动到{target_device}")
            else:
                logger.info(f"✅ video_sampler已在正确设备{target_device}")
    
    # ================== 检查优化器参数组（accelerator.prepare之后） ==================
    logger.info("🔍 检查optimizer参数组（accelerator.prepare之后）...")
    inspect_optimizer_param_groups(optimizer, "包装后", accelerator)
    
    # ================== 7. Resume Training：检查并加载checkpoint ==================
    resumed_training_state = None
    if last_checkpoint is not None:
        # 检测到checkpoint，尝试加载恢复训练状态
        logger.info(f"🔄 检测到checkpoint，准备恢复训练: {last_checkpoint}")
        
        # 使用新实现的load_checkpoint函数恢复模型权重、optimizer和训练状态
        resumed_training_state = load_checkpoint(
            checkpoint_path=last_checkpoint,
            model=model,  # 已经被accelerator包装过的模型
            accelerator=accelerator,
            processor=processor
        )
        
        if resumed_training_state is not None:
            logger.info(f"✅ 训练状态恢复成功，将从step {resumed_training_state.get('global_step', 1)}继续训练")
        else:
            logger.warning("⚠️ 训练状态恢复失败，将从头开始训练")
    else:
        logger.info("🆕 未检测到checkpoint，从头开始新训练")
    
    # ================== 8. 包装EgoVLPv2数据加载器 ==================
    egovlpv2_dataloader = None
    if egovlpv2_components and egovlpv2_components.get('train_dataloaders'):
        logger.info("📦 包装EgoVLPv2数据加载器...")
        raw_egovlpv2_dataloader = egovlpv2_components['train_dataloaders'][0]
        # 使用accelerate包装EgoVLPv2数据加载器，确保分布式训练同步
        egovlpv2_dataloader = accelerator.prepare(raw_egovlpv2_dataloader)
        logger.info("✅ EgoVLPv2数据加载器包装完成")
    
    # ================== 保存实验配置文件到输出目录 ==================
    # 将egovlpv2配置文件复制到实验目录，便于后续复现和分析实验设置
    if model_args.egovlpv2_config_path and os.path.exists(model_args.egovlpv2_config_path):
        # 确保输出目录存在
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # 构造目标配置文件路径：在输出目录下保存为egovlpv2_config.json
        config_filename = "egovlpv2_config.json"
        target_config_path = os.path.join(training_args.output_dir, config_filename)
        
        # 复制配置文件到实验输出目录
        shutil.copy2(model_args.egovlpv2_config_path, target_config_path)
        
        # 记录配置保存信息，包含源路径和目标路径
        logger.info(f"📋 EgoVLPv2配置已保存到实验目录:")
        logger.info(f"   源文件: {model_args.egovlpv2_config_path}")  
        logger.info(f"   目标文件: {target_config_path}")
    else:
        # 如果配置文件不存在，记录警告信息
        logger.warning(f"⚠️ EgoVLPv2配置文件未找到或路径为空: {model_args.egovlpv2_config_path}")
    
    if training_args.do_train:
        logger.info("🎯 开始MIMIC-VLA无trainer训练...")
        
        # 调用简化的训练循环，传入恢复的训练状态
        train_loop(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            training_args=training_args,
            egovlpv2_dataloader=egovlpv2_dataloader,
            processor=processor,
            action_tokenizer=action_tokenizer,
            resumed_training_state=resumed_training_state  # 新增：传入恢复的训练状态
        )
        
        logger.info("🎉 训练完成！")

# ================== 简化的训练循环实现 ==================

def train_loop(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    accelerator,
    training_args,
    egovlpv2_dataloader=None,
    processor=None,
    action_tokenizer=None,
    avg_window=100,
    resumed_training_state=None  # 新增：恢复的训练状态参数
):
    """
    简化的MIMIC-VLA训练循环，集成SpatialVLA、EgoVLPv2和Alignment三模型训练
    
    功能：
    1. 主训练循环：Bridge v2数据 + SpatialVLA训练
    2. 辅助训练：EgoVLPv2数据并行训练
    3. 对齐训练：Alignment模型损失计算
    4. 统一的梯度更新和学习率调整
    5. 定期保存检查点和指标记录
    6. 新增：支持从checkpoint恢复训练状态
    
    Args:
        model: MIMIC-VLA模型实例
        train_dataloader: Bridge v2训练数据加载器
        eval_dataloader: 验证数据加载器（可选）
        optimizer: 多模型统一优化器
        scheduler: 学习率调度器
        accelerator: Accelerator实例
        training_args: 训练参数配置
        egovlpv2_dataloader: EgoVLPv2数据加载器（可选）
        processor: SpatialVLA处理器
        action_tokenizer: 动作分词器
        resumed_training_state: 从checkpoint恢复的训练状态字典（可选）
    """
    
    # ================== 训练状态初始化：支持从checkpoint恢复 ==================
    # 从恢复的训练状态中获取值，如果没有则使用默认值
    if resumed_training_state is not None:
        # 从checkpoint恢复训练状态
        global_step = resumed_training_state.get('global_step', 1)
        total_loss = resumed_training_state.get('total_loss', 0.0)
        spatial_vla_loss_total = resumed_training_state.get('spatial_vla_loss_total', 0.0)
        egovlpv2_loss_total = resumed_training_state.get('egovlpv2_loss_total', 0.0)
        alignment_loss_total = resumed_training_state.get('alignment_loss_total', 0.0)
        accuracy_total = resumed_training_state.get('accuracy_total', 0.0)
        total_mini_batches = resumed_training_state.get('total_mini_batches', 0)
        accuracy_calculation_count = resumed_training_state.get('accuracy_calculation_count', 0)
        pos_logit_total = resumed_training_state.get('pos_logit_total', 0.0)
        neg_logit_total = resumed_training_state.get('neg_logit_total', 0.0)
        temp_total = resumed_training_state.get('temp_total', 0.0)
        pos_logit_count = resumed_training_state.get('pos_logit_count', 0)
        neg_logit_count = resumed_training_state.get('neg_logit_count', 0)
        temp_count = resumed_training_state.get('temp_count', 0)
        
        accelerator.print(f"🔄 训练状态已恢复：global_step={global_step}, total_mini_batches={total_mini_batches}")
    else:
        # 新训练：使用默认初始值
        global_step = 1
        total_loss = 0.0
        spatial_vla_loss_total = 0.0
        egovlpv2_loss_total = 0.0
        alignment_loss_total = 0.0
        accuracy_total = 0.0
        total_mini_batches = 0  # 总mini-batch次数，用于正确计算loss平均值
        accuracy_calculation_count = 0  # 准确率实际计算次数，用于正确计算accuracy平均值
        pos_logit_total = 0.0
        neg_logit_total = 0.0
        temp_total = 0.0
        pos_logit_count = 0
        neg_logit_count = 0
        temp_count = 0
        
        accelerator.print("🆕 新训练开始：所有状态从默认值初始化")

    # 新增：用于计算最近100步滑动平均的变量
    recent_losses = collections.deque(maxlen=avg_window) # 存储最近100个总损失的deque
    recent_spatial_vla_loss = collections.deque(maxlen=avg_window)
    recent_egovlpv2_loss = collections.deque(maxlen=avg_window)
    recent_alignment_loss = collections.deque(maxlen=avg_window)
    recent_accuracies = collections.deque(maxlen=avg_window) # 存储最近100个动作准确率的deque
    recent_pos_logits = collections.deque(maxlen=avg_window)
    recent_neg_logits = collections.deque(maxlen=avg_window)
    recent_temps = collections.deque(maxlen=avg_window)
    # 新增：正负样本数统计的滑动平均
    recent_pos_samples = collections.deque(maxlen=avg_window)
    recent_neg_samples = collections.deque(maxlen=avg_window)
    recent_total_samples = collections.deque(maxlen=avg_window)

    # 计算总训练步数和当前epoch信息
    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    schedule_training_steps = num_update_steps_per_epoch * training_args.num_train_epochs
    if training_args.max_steps > 0:
        run_training_steps = min(training_args.max_steps, schedule_training_steps)
    else:
        run_training_steps = schedule_training_steps
    
    # ================== 计算从哪个epoch开始：支持resume training ==================
    # 如果恢复训练，需要计算当前应该在哪个epoch
    if resumed_training_state is not None:
        # 从global_step推算当前epoch，确保从正确位置继续
        start_epoch = (global_step - 1) // num_update_steps_per_epoch
        # 确保start_epoch不超过总epoch数
        start_epoch = min(start_epoch, int(training_args.num_train_epochs) - 1)
        accelerator.print(f"🔄 Resume训练：从epoch {start_epoch}开始，global_step={global_step}")
    else:
        # 新训练从epoch 0开始
        start_epoch = 0
        accelerator.print(f"🆕 新训练：从epoch 0开始")
    
    # 创建EgoVLPv2数据迭代器（如果提供）
    egovlpv2_iter = iter(egovlpv2_dataloader) if egovlpv2_dataloader else None
    
    accelerator.print(
        f"🚀 开始训练：最多 {run_training_steps} 步；"
        f"学习率日程按 {schedule_training_steps} 步 / "
        f"{training_args.num_train_epochs} 轮计算"
    )
    
    # 创建tqdm进度条 - 基于总训练步数，只在主进程显示避免重复输出
    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=run_training_steps,
            desc="训练进度",
            unit="步",
            ncols=100,  # 设置进度条宽度
            leave=True,  # 训练完成后保留进度条
            dynamic_ncols=True  # 动态调整宽度适应终端
        )
    
    # ================== 主训练循环 ==================
    model.train()
    
    # 从计算出的start_epoch开始，支持resume training
    reached_step_limit = global_step >= run_training_steps
    for epoch in range(start_epoch, int(training_args.num_train_epochs)):
        if reached_step_limit:
            break
        # 设置分布式采样器的epoch，确保每轮数据分布的随机性（DDP训练必需）
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
            accelerator.print(f"🔄 设置训练采样器epoch={epoch}")
        
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # ==================== 设备转换：确保所有batch数据在正确设备上 ====================
                # accelerate的prepare()不会自动移动数据到GPU，需要手动转换
                # 获取模型所在的设备（通过accelerator获取）
                target_device = accelerator.device
                
                # 将batch中的所有张量移动到目标设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        # 使用non_blocking=True可以提高异步传输效率（需要pin_memory=True配合）
                        batch[key] = value.to(target_device, non_blocking=True,)
                
                # 获取EgoVLPv2批次数据（如果可用）
                egovlpv2_inputs = None
                if egovlpv2_iter is not None:
                    egovlpv2_inputs = next(egovlpv2_iter)
                    
                # 设置alignment的backbone_update_start_pct / stop_last_pct所需的step信息
                unwrapped = model.module if hasattr(model, 'module') else model
                if hasattr(unwrapped, '_global_step'):
                    unwrapped._global_step = global_step
                    unwrapped._num_train_steps = schedule_training_steps

                with accelerator.autocast():
                    outputs = model(
                        egovlpv2_inputs=egovlpv2_inputs,
                        **batch
                    )
                
                loss = outputs.loss
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if accelerator.sync_gradients and training_args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()  # 学习率调度器步骤
                    # Feature Bank Reset：在optimizer.step()后重置feature bank
                    # 清空gradient accumulation期间累积的features，开始新的累积周期
                    if hasattr(model, 'module'):
                        if hasattr(model.module, 'reset_feature_bank'):
                            model.module.reset_feature_bank()
                    else:
                        if hasattr(model, 'reset_feature_bank'):
                            model.reset_feature_bank()
                optimizer.zero_grad()  # 清零梯度，准备下一轮accumulation
                
                # 统计损失 - 修复avg计算：每个mini-batch都累积loss和计数器
                total_loss += loss.item()
                recent_losses.append(loss.item())
                epoch_loss += loss.item()
                total_mini_batches += 1  # 每次mini-batch都计数，确保loss平均值计算正确
                
                # 分别统计各个loss的有效步数，确保平均值计算正确
                spatial_loss_steps = 0
                egovlpv2_loss_steps = 0 
                alignment_loss_steps = 0
                
                # 检查并累积SpatialVLA损失
                if hasattr(outputs, 'spatial_vla_loss') and outputs.spatial_vla_loss is not None:
                    spatial_vla_loss_total += outputs.spatial_vla_loss.item()
                    recent_spatial_vla_loss.append(outputs.spatial_vla_loss.item())
                    spatial_loss_steps += 1
                    
                # 检查并累积EgoVLPv2损失 
                if hasattr(outputs, 'egovlpv2_loss') and outputs.egovlpv2_loss is not None:
                    egovlpv2_loss_total += outputs.egovlpv2_loss.item()
                    recent_egovlpv2_loss.append(outputs.egovlpv2_loss.item())
                    egovlpv2_loss_steps += 1
                    
                # 检查并累积Alignment损失
                if hasattr(outputs, 'alignment_loss') and outputs.alignment_loss is not None:
                    alignment_loss_total += outputs.alignment_loss.item()
                    recent_alignment_loss.append(outputs.alignment_loss.item())
                    alignment_loss_steps += 1
                
                # 累积alignment_loss_dict中的logit和temperature统计信息
                alignment_loss_dict = outputs.alignment_loss_dict if hasattr(outputs, 'alignment_loss_dict') and outputs.alignment_loss_dict else {}
                if 'pos_mean_logit' in alignment_loss_dict:
                    pos_logit_val = alignment_loss_dict['pos_mean_logit']
                    pos_logit_val = pos_logit_val.item() if isinstance(pos_logit_val, torch.Tensor) else pos_logit_val
                    pos_logit_total += pos_logit_val
                    recent_pos_logits.append(pos_logit_val)
                    pos_logit_count += 1
                
                if 'neg_mean_logit' in alignment_loss_dict:
                    neg_logit_val = alignment_loss_dict['neg_mean_logit']
                    neg_logit_val = neg_logit_val.item() if isinstance(neg_logit_val, torch.Tensor) else neg_logit_val
                    neg_logit_total += neg_logit_val
                    recent_neg_logits.append(neg_logit_val)
                    neg_logit_count += 1
                
                if 'temperature' in alignment_loss_dict:
                    temp_val = alignment_loss_dict['temperature']
                    temp_val = temp_val.item() if isinstance(temp_val, torch.Tensor) else temp_val
                    temp_total += temp_val
                    recent_temps.append(temp_val)
                    temp_count += 1
                
                # 收集正负样本数统计（用于滑动平均）
                if 'avg_pos_samples' in alignment_loss_dict:
                    pos_samples_val = alignment_loss_dict['avg_pos_samples']
                    pos_samples_val = pos_samples_val.item() if isinstance(pos_samples_val, torch.Tensor) else pos_samples_val
                    recent_pos_samples.append(pos_samples_val)
                if 'avg_neg_samples' in alignment_loss_dict:
                    neg_samples_val = alignment_loss_dict['avg_neg_samples']
                    neg_samples_val = neg_samples_val.item() if isinstance(neg_samples_val, torch.Tensor) else neg_samples_val
                    recent_neg_samples.append(neg_samples_val)
                if 'avg_total_samples' in alignment_loss_dict:
                    total_samples_val = alignment_loss_dict['avg_total_samples']
                    total_samples_val = total_samples_val.item() if isinstance(total_samples_val, torch.Tensor) else total_samples_val
                    recent_total_samples.append(total_samples_val)
                
                # 更新global_step：只在真正的梯度更新时递增，确保与梯度累计的大步对齐
                if accelerator.sync_gradients:
                    global_step += 1

                    # 更新tqdm进度条 - 显示当前损失信息，只在梯度同步时更新
                    if progress_bar is not None:
                        # 构建进度条描述信息，显示关键指标
                        current_total = loss.item()
                        progress_description = f"训练进度 | 总损失:{current_total:.4f}"
                        progress_bar.set_description(progress_description)
                        progress_bar.update(1)  # 更新进度条步数

                    # 定期记录指标 - 只在梯度更新后检查，确保与大步对齐
                    if global_step % training_args.logging_steps == 0 or global_step == 1:
                        # 计算平均损失 - 修复：使用total_mini_batches作为分母，解决gradient accumulation导致的放大问题
                        avg_loss = total_loss / total_mini_batches
                        avg_spatial_loss = spatial_vla_loss_total / total_mini_batches
                        avg_egovlpv2_loss = egovlpv2_loss_total / total_mini_batches 
                        avg_alignment_loss = alignment_loss_total / total_mini_batches
                        
                        # 计算alignment logit和temperature的平均值
                        avg_pos_logit = pos_logit_total / pos_logit_count if pos_logit_count > 0 else 0.0
                        avg_neg_logit = neg_logit_total / neg_logit_count if neg_logit_count > 0 else 0.0
                        avg_temp = temp_total / temp_count if temp_count > 0 else 0.0
                        
                        # 获取当前学习率 - 兼容DeepSpeed和传统PyTorch调度器
                        if scheduler is not None:
                            # 传统PyTorch调度器模式
                            current_lr = scheduler.get_last_lr()[0]
                        else:
                            # DeepSpeed自动管理调度器模式，从DeepSpeed引擎获取当前学习率
                            if hasattr(accelerator, 'deepspeed_engine') and accelerator.deepspeed_engine is not None:
                                current_lr = accelerator.deepspeed_engine.get_lr()[0]
                            else:
                                # 备用方案：从optimizer的param_groups获取学习率
                                current_lr = optimizer.param_groups[0]['lr']
                        
                        # 输出当前步的实时损失值（调试用）
                        current_total = loss.item()
                        current_spatial = outputs.spatial_vla_loss.item() if hasattr(outputs, 'spatial_vla_loss') and outputs.spatial_vla_loss is not None else 0.0
                        current_egovlp = outputs.egovlpv2_loss.item() if hasattr(outputs, 'egovlpv2_loss') and outputs.egovlpv2_loss is not None else 0.0
                        current_align = outputs.alignment_loss.item() if hasattr(outputs, 'alignment_loss') and outputs.alignment_loss is not None else 0.0
                        
                        # 提取alignment_loss_dict（包含细粒度loss和统计信息）
                        alignment_loss_dict = outputs.alignment_loss_dict if hasattr(outputs, 'alignment_loss_dict') and outputs.alignment_loss_dict else {}
                        
                        # 检查optimizer参数组状态（调试用）
                        if global_step % (training_args.logging_steps * 50) == 0 or global_step == 1:
                            inspect_optimizer_param_groups(optimizer, f"训练步{global_step}", accelerator)
                        
                        # 记录指标到配置的logging后端（tensorboard和/或wandb）
                        if accelerator.is_main_process:
                            # 计算梯度范数，用于监控训练稳定性
                            total_grad_norm = 0.0
                            if training_args.max_grad_norm > 0:
                                # 计算所有参数的梯度范数
                                total_norm = 0.0
                                param_count = 0
                                for p in model.parameters():
                                    if p.grad is not None:
                                        param_norm = p.grad.data.norm(2)
                                        total_norm += param_norm.item() ** 2
                                        param_count += 1
                                total_grad_norm = total_norm ** (1. / 2) if param_count > 0 else 0.0
                            
                            # 获取内存使用情况（GPU内存）
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                            
                            # spatial_outputs = getattr(outputs, 'spatial_outputs', outputs)
                            # metric_action = compute_action_metrics(model, spatial_outputs, batch, action_tokenizer, accelerator, global_step)
                            # # 修复准确率avg计算：每次计算accuracy时增加计数器，使用实际计算次数作为分母
                            # accuracy_total = accuracy_total + metric_action['train/accuracy']
                            # accuracy_calculation_count += 1  # 每次计算accuracy时计数
                            # avg_accuracy = accuracy_total / accuracy_calculation_count  # 使用实际计算次数作为分母
                            # recent_accuracies.append(metric_action['train/accuracy'])

                            # 新增：计算最近100步的滑动平均损失和准确率
                            recent_avg_loss = sum(recent_losses) / len(recent_losses) if len(recent_losses) > 0  else 0.0
                            recent_avg_accuracy = sum(recent_accuracies) / len(recent_accuracies) if len(recent_accuracies) > 0 else 0.0 
                            recent_avg_egovlpv2_loss = sum(recent_egovlpv2_loss) / len(recent_egovlpv2_loss) if len(recent_egovlpv2_loss) > 0 else 0.0
                            recent_avg_alignment_loss = sum(recent_alignment_loss) / len(recent_alignment_loss) if len(recent_alignment_loss) > 0 else 0.0
                            recent_avg_spatial_loss = sum(recent_spatial_vla_loss) / len(recent_spatial_vla_loss) if len(recent_spatial_vla_loss) > 0 else 0.0
                            # 计算正负样本数的滑动平均
                            recent_avg_pos_samples = sum(recent_pos_samples) / len(recent_pos_samples) if len(recent_pos_samples) > 0 else 0.0
                            recent_avg_neg_samples = sum(recent_neg_samples) / len(recent_neg_samples) if len(recent_neg_samples) > 0 else 0.0
                            recent_avg_total_samples = sum(recent_total_samples) / len(recent_total_samples) if len(recent_total_samples) > 0 else 0.0
                            # 准备要记录的详细指标字典
                            metrics_to_log = {
                                # 核心损失指标
                                "train/loss": current_total,
                                "train/spatial_vla_loss": current_spatial,
                                "train/egovlpv2_loss": current_egovlp,
                                "train/alignment_loss": current_align,
                                
                                # 优化器相关指标
                                "train/learning_rate": current_lr,
                                "train/grad_norm": total_grad_norm,
                                
                                # 训练进度指标
                                "train/epoch": epoch + 1,  # 从1开始计数，更直观
                                "train/global_step": global_step,
                                "train/progress": global_step / run_training_steps,  # 本次运行进度百分比
                                
                                # 系统资源指标
                                "system/gpu_memory_allocated_gb": memory_allocated,
                                "system/gpu_memory_reserved_gb": memory_reserved,
                                
                                # 当前步实时损失（调试用）
                                "debug/avg_total_loss": avg_loss,
                                "debug/avg_spatial_loss": avg_spatial_loss,
                                "debug/avg_egovlpv2_loss": avg_egovlpv2_loss,
                                "debug/avg_alignment_loss": avg_alignment_loss,
                                # "debug/avg_accuracy": avg_accuracy,
                                "debug/avg_pos_logit": avg_pos_logit,
                                "debug/avg_neg_logit": avg_neg_logit,
                                "debug/avg_temperature": avg_temp,

                                # 新增：最近100步的滑动平均指标
                                f"cur/loss_avg_{avg_window}_steps": recent_avg_loss,
                                f"cur/spatial_vla_loss_avg_{avg_window}_steps": recent_avg_spatial_loss,
                                f"cur/egovlpv2_loss_avg_{avg_window}_steps": recent_avg_egovlpv2_loss,
                                f"cur/alignment_loss_avg_{avg_window}_steps": recent_avg_alignment_loss,
                                f"cur/accuracy_avg_{avg_window}_steps": recent_avg_accuracy,
                                # 正负样本数统计（滑动平均）
                                f"cur/avg_pos_samples_{avg_window}_steps": recent_avg_pos_samples,
                                f"cur/avg_neg_samples_{avg_window}_steps": recent_avg_neg_samples,
                                f"cur/avg_total_samples_{avg_window}_steps": recent_avg_total_samples,
                            }
                            
                            # 添加细粒度对齐loss（as2ts_loss、as2tt_loss等）
                            for key, value in alignment_loss_dict.items():
                                if isinstance(value, torch.Tensor):
                                    metrics_to_log[f"train/align_{key}"] = value.item()
                                elif isinstance(value, dict):
                                    continue  # 跳过嵌套字典
                                else:
                                    metrics_to_log[f"train/align_{key}"] = value
                            
                            # 添加logit统计信息（sigmoid loss时）
                            if 'pos_mean_logit' in alignment_loss_dict:
                                pos_logit = alignment_loss_dict['pos_mean_logit']
                                neg_logit = alignment_loss_dict.get('neg_mean_logit', None)
                                temp = alignment_loss_dict.get('temperature', None)
                                
                                if isinstance(pos_logit, torch.Tensor):
                                    metrics_to_log['train/pos_mean_logit'] = pos_logit.item()
                                else:
                                    metrics_to_log['train/pos_mean_logit'] = pos_logit
                                
                                if neg_logit is not None:
                                    if isinstance(neg_logit, torch.Tensor):
                                        metrics_to_log['train/neg_mean_logit'] = neg_logit.item()
                                    else:
                                        metrics_to_log['train/neg_mean_logit'] = neg_logit
                                
                                if temp is not None:
                                    if isinstance(temp, torch.Tensor):
                                        metrics_to_log['train/alignment_temperature'] = temp.item()
                                    else:
                                        metrics_to_log['train/alignment_temperature'] = temp
                            
                            #metrics_to_log.update(metric_action)

                            # 输出训练进度和损失信息（压缩版）
                            # 提取各模式独立loss用于终端显示
                            as2ts_loss_val = alignment_loss_dict.get('as2ts_loss', 0.0)
                            as2vs_loss_val = alignment_loss_dict.get('as2vs_loss', 0.0)
                            as2ts_pos = alignment_loss_dict.get('as2ts_pos_logit', 0.0)
                            as2ts_neg = alignment_loss_dict.get('as2ts_neg_logit', 0.0)
                            as2vs_pos = alignment_loss_dict.get('as2vs_pos_logit', 0.0)
                            as2vs_neg = alignment_loss_dict.get('as2vs_neg_logit', 0.0)
                            if 'vicreg_invariance_loss' in alignment_loss_dict:
                                alignment_detail = (
                                    f"🧩VICReg inv={alignment_loss_dict['vicreg_invariance_loss']:.4f} "
                                    f"var={alignment_loss_dict['vicreg_variance_loss']:.4f} "
                                    f"cov={alignment_loss_dict['vicreg_covariance_loss']:.4f} "
                                    f"a_std={alignment_loss_dict['vicreg_action_std']:.3f} "
                                    f"t_std={alignment_loss_dict['vicreg_text_std']:.3f}"
                                )
                            else:
                                alignment_detail = (
                                    f"📝text(as2ts):{as2ts_loss_val:.4f} "
                                    f"pos={as2ts_pos:.3f} neg={as2ts_neg:.3f} | "
                                    f"🎥video(as2vs):{as2vs_loss_val:.4f} "
                                    f"pos={as2vs_pos:.3f} neg={as2vs_neg:.3f} | "
                                    f"🌡️T:{avg_temp:.4f}"
                                )
                            # DSN disentangle loss（如果启用）
                            dis_diff = alignment_loss_dict.get('disentangle_diff_loss', None)
                            dis_recon = alignment_loss_dict.get('disentangle_total_loss', None)
                            dsn_str = ""
                            if dis_diff is not None or dis_recon is not None:
                                _dd = f"{dis_diff:.4f}" if dis_diff is not None else "N/A"
                                _dr = f"{dis_recon:.4f}" if dis_recon is not None else "N/A"
                                dsn_str = f" | 🔀DSN diff={_dd} recon={_dr}"
                            # backbone detach状态
                            bb_detach = alignment_loss_dict.get('backbone_detached', 0.0)
                            bb_str = " [BB_DETACH]" if bb_detach > 0.5 else ""
                            accelerator.print(
                                f"🚀 Step {global_step}/{run_training_steps} | "
                                f"📊总损失:{avg_loss:.4f} | 🎯VLA_avg:{avg_spatial_loss:.4f} | 👁️Ego_avg:{avg_egovlpv2_loss:.4f} | 🔗Align_avg:{avg_alignment_loss:.4f} | "
                                f"📈lr:{current_lr:.2e} | 📋当前:总={current_total:.4f}, vla={current_spatial:.4f}, ego={current_egovlp:.4f} | "
                                f"{alignment_detail}{dsn_str}{bb_str}"
                            )
                            accelerator.log(metrics_to_log, step=global_step)

                    # 保存检查点 - 只在梯度更新后检查，确保与大步对齐
                    if global_step % training_args.save_steps == 0:
                        # 在保存前打印日志，帮助调试多卡同步问题
                        accelerator.print(f"🔄 准备保存检查点 - 步数: {global_step}")
                        
                        # ================== 保存前释放显存，降低保存阶段峰值 ==================
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        # ================== 构建当前训练状态字典 ==================
                        current_training_state = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "total_loss": total_loss,
                            "spatial_vla_loss_total": spatial_vla_loss_total,
                            "egovlpv2_loss_total": egovlpv2_loss_total,
                            "alignment_loss_total": alignment_loss_total,
                            "accuracy_total": accuracy_total,
                            "total_mini_batches": total_mini_batches,
                            "accuracy_calculation_count": accuracy_calculation_count,
                            "pos_logit_total": pos_logit_total,
                            "neg_logit_total": neg_logit_total,
                            "temp_total": temp_total,
                            "pos_logit_count": pos_logit_count,
                            "neg_logit_count": neg_logit_count,
                            "temp_count": temp_count,
                            # 其他训练配置信息
                            "training_args_num_train_epochs": training_args.num_train_epochs,
                            "num_update_steps_per_epoch": num_update_steps_per_epoch,
                            "save_timestamp": accelerator.state.device.type  # 简单的时间戳标记
                        }
                        
                        save_checkpoint(
                            model=model,
                            accelerator=accelerator,
                            training_args=training_args,
                            global_step=global_step,
                            processor=processor,
                            training_state=current_training_state  # 新增：传入当前训练状态
                        )
                        
                        # 保存完成后打印确认信息
                        accelerator.print(f"✅ 检查点保存完成 - 步数: {global_step}，继续训练...")

                    if global_step >= run_training_steps:
                        accelerator.print(
                            f"✅ 已达到训练步数上限 {run_training_steps}；"
                            f"LR 日程总长度仍为 {schedule_training_steps}"
                        )
                        reached_step_limit = True
                        break
        
        # 每轮结束时的验证和保存
        epoch_avg_loss = epoch_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch+1} 平均损失: {epoch_avg_loss:.4f}")
        if reached_step_limit:
            break
    
    # 关闭tqdm进度条
    if progress_bar is not None:
        progress_bar.close()
        accelerator.print("✅ 训练进度条已关闭")
    
    # 训练完成，保存最终模型
    # 在保存前确保所有进程都完成训练并到达此处，避免部分进程还在训练时开始保存
    accelerator.print("🎉 训练完成！等待所有进程同步...")
    accelerator.wait_for_everyone()  # 第一次同步：确保所有进程都退出训练循环
    
    accelerator.print("🔄 开始保存最终模型...")
    save_final_model(model, accelerator, training_args, processor)
    
    # 保存完成后，结束所有logging trackers
    # 注意：save_final_model内部已经包含了同步，这里不需要额外的wait_for_everyone
    accelerator.print("📊 结束logging trackers...")
    accelerator.end_training()
    accelerator.print("✅ 训练和保存流程完全结束")





if __name__ == "__main__":
    main()
