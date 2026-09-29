"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

新增功能：详细的训练步骤计时分析
- 实时监控每个训练步骤中各个环节的耗时
- 包括：数据加载、数据传输、前向传播、反向传播、梯度计算、优化器更新等
- 以毫秒为单位显示各步骤耗时和占比
- 计时信息会同时记录到swanlab中

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import copy
import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import deepspeed
import jax
import numpy as np
import safetensors.torch
import swanlab
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm

# ================== Accelerate和DeepSpeed相关导入 ==================
# accelerate: 统一的分布式训练管理框架，替代传统DDP
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed as accelerate_set_seed,
)
from transformers import set_seed as transformers_set_seed

# ================== LoRA相关导入 ==================  
# peft: Parameter-Efficient Fine-Tuning库，提供LoRA实现
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, set_peft_model_state_dict

# ================== OpenPI模型相关导入 ==================
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.optim_utils as optim_utils
from openpi.models_pytorch.pi0_pytorch_align import PI0PytorchAlign


def init_logging():
    """初始化日志系统，设置控制台输出"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def add_file_logging(checkpoint_dir, is_main_process=True):
    """
    添加文件日志handler，将日志同时输出到文件
    
    Args:
        checkpoint_dir: checkpoint目录路径
        is_main_process: 是否为主进程（只有主进程写入文件）
    """
    if not is_main_process:
        return
    
    logger = logging.getLogger()
    log_file = checkpoint_dir / "training.log"
    
    # 检查是否已经添加过文件handler（避免重复添加）
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == str(log_file.absolute()):
                logging.info(f"文件日志已配置: {log_file}")
                return
    
    # 创建FileHandler，使用append模式（支持resume）
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 使用与StreamHandler相同的formatter
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}
    
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)
    
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    fh.setFormatter(formatter)
    
    # 添加到logger
    logger.addHandler(fh)
    logging.info(f"✅ 日志将同时输出到文件: {log_file}")


def init_swanlab(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True, 
                 egovlpv2_config=None, alignment_config=None):
    """
    初始化swanlab日志记录，并保存完整配置到JSON文件
    
    Args:
        config: OpenPI训练配置
        resuming: 是否从checkpoint恢复
        enabled: 是否启用swanlab
        egovlpv2_config: EgoVLPv2配置（可选，从model中获取）
        alignment_config: Alignment配置（可选，从model中获取）
    """
    if not enabled:
        swanlab.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    # 合并所有配置：OpenPI config + EgoVLPv2 config + Alignment config
    merged_config = dataclasses.asdict(config)
    if egovlpv2_config is not None:
        merged_config['egovlpv2'] = egovlpv2_config
    if alignment_config is not None:
        merged_config['alignment'] = alignment_config
    
    # 保存完整配置到JSON文件（在checkpoint目录）
    import json
    config_save_path = ckpt_dir / "merged_config.json"
    with config_save_path.open('w') as f:
        json.dump(merged_config, f, indent=2, default=str)
    logging.info(f"✅ 已保存完整配置到: {config_save_path}")

    if resuming and os.path.exists(ckpt_dir / "swanlab_id.txt"):
        run_id = (ckpt_dir / "swanlab_id.txt").read_text().strip()
        swanlab.init(id=run_id, resume="must", project=config.project_name)
    else:
        swanlab.init(
            name=config.exp_name,
            config=merged_config,  # 使用合并后的配置
            project=config.project_name,
        )
        (ckpt_dir / "swanlab_id.txt").write_text(str(swanlab.get_run().public.json()))


def create_accelerator(config):
    """
    创建accelerator实例，替代传统DDP设置
    
    基于accelerator_utils.py但简化配置，只保留核心功能：
    1. 分布式训练环境配置
    2. 混合精度训练支持
    3. DeepSpeed集成（可选）
    
    Args:
        config: TrainConfig实例
        
    Returns:
        Accelerator: 配置好的accelerator实例
    """
    # 设置项目和日志目录
    logs_dir = config.checkpoint_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    project_config = ProjectConfiguration(
        project_dir=str(config.checkpoint_dir),
        logging_dir=str(logs_dir)
    )
    
    # DeepSpeed配置（如果提供配置文件）
    deepspeed_plugin = None
    if hasattr(config, 'deepspeed_config_file') and config.deepspeed_config_file:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=config.deepspeed_config_file)
        logging.info(f"💾 DeepSpeed启用: {config.deepspeed_config_file}")
    
    # 混合精度配置（基于pytorch_training_precision）
    mixed_precision = "no"
    if hasattr(config, 'pytorch_training_precision'):
        if config.pytorch_training_precision == "bfloat16":
            mixed_precision = "bf16"
        elif config.pytorch_training_precision == "float16":
            mixed_precision = "fp16"
    
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        gradient_as_bucket_view=True
    )

    # 梯度累积配置
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    
    # 3. 将其通过 kwargs_handlers 传入 Accelerator
    #    并移除 __init__ 中不被识别的参数
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        project_config=project_config,
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,  # ✅ 添加梯度累积配置
        kwargs_handlers=[ddp_kwargs]  # <-- 注意这里是关键改动
    )
    
    # 计算per_device_batch_size（如果未指定）
    # 注意：config是frozen dataclass，需要使用object.__setattr__来修改
    if hasattr(config, 'per_device_batch_size') and config.per_device_batch_size is None:
        total_batch_size = config.batch_size
        world_size = accelerator.num_processes
        
        if total_batch_size % (world_size * gradient_accumulation_steps) != 0:
            raise ValueError(
                f"❌ batch_size ({total_batch_size}) 必须能被 "
                f"(GPU数量 × 梯度累积步数) = ({world_size} × {gradient_accumulation_steps}) 整除！"
            )
        
        # 使用object.__setattr__修改frozen dataclass
        object.__setattr__(config, 'per_device_batch_size', 
                          total_batch_size // (world_size * gradient_accumulation_steps))
    
    # 计算实际的全局batch size
    actual_global_batch_size = (
        config.per_device_batch_size * accelerator.num_processes * gradient_accumulation_steps
        if hasattr(config, 'per_device_batch_size') and config.per_device_batch_size is not None
        else config.batch_size
    )
    
    logging.info("✅ Accelerator初始化完成:")
    logging.info(f"   - 混合精度: {mixed_precision}")
    logging.info(f"   - DeepSpeed: {'启用' if deepspeed_plugin else '禁用'}")
    logging.info(f"   - 设备: {accelerator.device}")
    logging.info(f"   - GPU数量: {accelerator.num_processes}")
    logging.info(f"   - 梯度累积步数: {gradient_accumulation_steps}")
    logging.info(f"   - 每GPU batch size: {config.per_device_batch_size if hasattr(config, 'per_device_batch_size') else 'N/A'}")
    logging.info(f"   - 全局总batch size: {actual_global_batch_size}")
    
    return accelerator

def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def apply_lora_to_model(model, lora_config):
    """
    ✅ 在训练脚本中统一应用LoRA包装
    
    这是标准的Hugging Face PEFT使用模式：
    1. 创建基础模型
    2. 加载预训练权重
    3. 应用PEFT（LoRA）
    4. 训练
    
    Args:
        model: PI0PytorchAlign模型实例（基础模型，未应用LoRA）
        lora_config: LoRA配置字典
    
    Returns:
        应用了LoRA的模型（子模块被PeftModel包装）
    """
    if lora_config is None:
        logging.info("🚫 未启用LoRA，将进行全参数微调")
        return model
    
    if not isinstance(model, PI0PytorchAlign):
        raise TypeError(f"❌ 期望PI0PytorchAlign模型，但得到: {type(model)}")
    
    # 配置PaliGemma的LoRA
    lora_config_paligemma = LoraConfig(
        r=lora_config['lora_rank_paligemma'],
        lora_alpha=lora_config['lora_alpha_paligemma'],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 配置Gemma Expert的LoRA
    lora_config_gemma_expert = LoraConfig(
        r=lora_config['lora_rank_gemma_expert'],
        lora_alpha=lora_config['lora_alpha_gemma_expert'],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # ✅ 应用LoRA到子模块
    logging.info("   - 应用LoRA到PaliGemma...")
    model.pi0_pytorch.paligemma_with_expert.paligemma = get_peft_model(
        model.pi0_pytorch.paligemma_with_expert.paligemma, 
        lora_config_paligemma
    )
    
    logging.info("   - 应用LoRA到Gemma Expert...")
    model.pi0_pytorch.paligemma_with_expert.gemma_expert = get_peft_model(
        model.pi0_pytorch.paligemma_with_expert.gemma_expert, 
        lora_config_gemma_expert
    )
    
    # 打印可训练参数
    logging.info("✅ PaliGemma LoRA配置:")
    model.pi0_pytorch.paligemma_with_expert.paligemma.print_trainable_parameters()
    logging.info("✅ Gemma Expert LoRA配置:")
    model.pi0_pytorch.paligemma_with_expert.gemma_expert.print_trainable_parameters()
    
    return model


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, accelerator=None, save_as_latest=False):
    """
    【修改-accelerate+lora】保存检查点，兼容accelerator和LoRA
    Save a checkpoint with model state, optimizer state, and metadata.
    
    Args:
        save_as_latest: 如果为True，保存到latest/目录；否则保存到{global_step}/目录
    """
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step or if saving as latest
    should_save = (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1 or save_as_latest
    
    if should_save:
        # Create temporary directory for atomic checkpoint saving
        if save_as_latest:
            final_ckpt_dir = config.checkpoint_dir / "latest"
            tmp_ckpt_dir = config.checkpoint_dir / "tmp_latest"
        else:
            final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
            tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 【修改-accelerate+lora】处理PI0模型中包含的多个LoRA模块
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        # 检查是否为PI0Pytorch模型，并且包含多个LoRA模块
        if isinstance(model_to_save, PI0PytorchAlign) and \
           hasattr(model_to_save.pi0_pytorch, 'paligemma_with_expert'):
            logging.info("检测到PI0PytorchAlign模型")
            
            # 检查是否使用了LoRA
            has_peft_paligemma = isinstance(model_to_save.pi0_pytorch.paligemma_with_expert.paligemma, PeftModel)
            has_peft_gemma_expert = isinstance(model_to_save.pi0_pytorch.paligemma_with_expert.gemma_expert, PeftModel)
            
            if has_peft_paligemma or has_peft_gemma_expert:
                # ✅ 最优方案：同时保存两份权重
                # 1. adapter权重（用于resume）
                # 2. 合并后的完整权重（用于推理）
                logging.info("✅ 检测到LoRA训练，保存adapter和合并权重...")
                
                # === 第1步：保存adapter权重（用于resume） ===
                adapter_dir = tmp_ckpt_dir / "adapter_model"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                
                if has_peft_paligemma:
                    paligemma_adapter_dir = adapter_dir / "paligemma_adapter"
                    model_to_save.pi0_pytorch.paligemma_with_expert.paligemma.save_pretrained(paligemma_adapter_dir)
                    logging.info(f"   ✅ 保存PaliGemma adapter（resume用）")
                    
                if has_peft_gemma_expert:
                    gemma_expert_adapter_dir = adapter_dir / "gemma_expert_adapter"
                    model_to_save.pi0_pytorch.paligemma_with_expert.gemma_expert.save_pretrained(gemma_expert_adapter_dir)
                    logging.info(f"   ✅ 保存Gemma Expert adapter（resume用）")
                
                # === 第2步：合并LoRA权重并保存完整模型（用于推理） ===
                # ⚠️ 绝对不能对原始模型做 .cpu()/.to(device)!
                # 那样会重置 AccumulateGrad 节点，摧毁 DDP 梯度钩子 → NCCL timeout。
                # 安全做法: merge/unmerge 只做 W.data += / -= ，不动 Parameter 对象。
                # ⚠️ 必须用 try-finally 保护 unmerge，否则 save 异常会导致 W 停留在 merged 状态。
                # ⚠️ 必须用 save_file 而非 save_model：save_model 会保存 PeftModel 的完整 state_dict
                #    （含 base_model.model. 前缀 + lora_A/B），推理加载到普通模型会失败。
                logging.info("   🔄 合并LoRA权重（in-place merge → clean state_dict → unmerge）...")
                with torch.no_grad():
                    # 1) 原地合并 LoRA 到 base weights
                    if has_peft_paligemma:
                        model_to_save.pi0_pytorch.paligemma_with_expert.paligemma.merge_adapter()
                    if has_peft_gemma_expert:
                        model_to_save.pi0_pytorch.paligemma_with_expert.gemma_expert.merge_adapter()
                    
                    # 2) 手动提取干净的 state_dict（过滤 LoRA 参数 + 清理 PEFT 前缀）
                    #    不能用 save_model：它会保存 PeftModel 的 state_dict，key 含 base_model.model. 前缀
                    #    和非零 lora_A/B 参数，推理时加载到普通模型会 key 不匹配或得到 W+2BA。
                    cpu_state_dict = {}
                    skipped_lora_keys = 0
                    for key, value in model_to_save.pi0_pytorch.state_dict().items():
                        # 跳过 LoRA 专属参数（已 merge 进 base 权重，不需要单独保存）
                        if 'lora_' in key or 'modules_to_save' in key:
                            skipped_lora_keys += 1
                            continue
                        
                        # 清理 PEFT 包装前缀：base_model.model. → 去掉
                        clean_key = key
                        for peft_module in [
                            'paligemma_with_expert.paligemma.',
                            'paligemma_with_expert.gemma_expert.',
                        ]:
                            peft_prefix = peft_module + 'base_model.model.'
                            if clean_key.startswith(peft_prefix):
                                clean_key = peft_module + clean_key[len(peft_prefix):]
                                break
                        # 清理 base_layer. 前缀（LoRA target 模块的原始权重）
                        clean_key = clean_key.replace('.base_layer.', '.')
                        cpu_state_dict[clean_key] = value.cpu()
                    
                    logging.info(f"   📊 保存 {len(cpu_state_dict)} 个参数，跳过 {skipped_lora_keys} 个LoRA参数")
                
                # 3) 用 save_file（不是 save_model！）保存干净的推理权重
                from safetensors.torch import save_file
                save_file(cpu_state_dict, str(tmp_ckpt_dir / "model.safetensors"))
                logging.info(f"   ✅ 保存PI0合并权重（推理用，干净key）")
                
                # 4) 原地还原 LoRA（恢复 base weights，继续训练）
                with torch.no_grad():
                    if has_peft_paligemma:
                        model_to_save.pi0_pytorch.paligemma_with_expert.paligemma.unmerge_adapter()
                    if has_peft_gemma_expert:
                        model_to_save.pi0_pytorch.paligemma_with_expert.gemma_expert.unmerge_adapter()
                
                del cpu_state_dict
                gc.collect()
                torch.cuda.empty_cache()
                
                logging.info(f"✅ PI0检查点保存完成：")
                logging.info(f"   - adapter_model/     -> resume时使用（预训练模型+adapter）")
                logging.info(f"   - model.safetensors  -> 推理时使用（合并后的完整权重，干净key）")
                logging.info(f"   - 原始模型保持不变，训练继续使用LoRA ✅")
            else:
                # 全参数训练 - 直接保存PI0部分
                logging.info("✅ 全参数训练，直接保存完整PI0模型")
                safetensors.torch.save_model(model_to_save.pi0_pytorch, tmp_ckpt_dir / "model.safetensors")
            
            # === 第3步：保存EgoVLPv2模型权重（如果启用） ===
            if model_to_save.use_egovlpv2:
                if save_as_latest:
                    egovlpv2_checkpoint_path = tmp_ckpt_dir / "egovlpv2_model.pth"
                else:
                    egovlpv2_checkpoint_path = tmp_ckpt_dir / f"egovlpv2_model_step{global_step}.pth"
                
                # 智能检测：DDP包装的模型有.module属性，否则直接访问
                if hasattr(model_to_save.egovlpv2_model, 'module'):
                    egovlpv2_state_dict = model_to_save.egovlpv2_model.module.state_dict()
                else:
                    egovlpv2_state_dict = model_to_save.egovlpv2_model.state_dict()
                
                # 构造egovlpv2原生checkpoint格式（参考SpatialVLA）
                egovlpv2_checkpoint = {
                    'arch': type(model_to_save.egovlpv2_model).__name__,
                    'epoch': global_step // config.save_interval,  # 使用step作为epoch近似
                    'state_dict': egovlpv2_state_dict,
                    'monitor_best': float('inf'),
                    'config': model_to_save.egovlpv2_config if hasattr(model_to_save, 'egovlpv2_config') else {}
                }
                torch.save(egovlpv2_checkpoint, egovlpv2_checkpoint_path)
                logging.info(f"   ✅ 保存EgoVLPv2模型权重: {egovlpv2_checkpoint_path}")
            
            # === 第4步：保存Alignment模型权重（如果启用） ===
            if model_to_save.use_alignment:
                if save_as_latest:
                    alignment_weight_path = tmp_ckpt_dir / "alignment_model.pth"
                else:
                    alignment_weight_path = tmp_ckpt_dir / f"alignment_model_step{global_step}.pth"
                
                # 智能检测DDP包装
                if hasattr(model_to_save.alignment_model, 'module'):
                    alignment_state_dict = model_to_save.alignment_model.module.state_dict()
                else:
                    alignment_state_dict = model_to_save.alignment_model.state_dict()
                
                torch.save(alignment_state_dict, alignment_weight_path)
                logging.info(f"   ✅ 保存Alignment模型权重: {alignment_weight_path}")
                
        elif accelerator is not None:
            # 使用accelerator保存模型，自动处理DDP/FSDP包装和标准LoRA权重
            accelerator.save_model(model, tmp_ckpt_dir)
            logging.info(f"使用accelerator保存模型到: {tmp_ckpt_dir}")
        else:
            # 备用方案：传统保存方式
            safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        accelerator.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        if save_as_latest:
            logging.info(f"Saved latest checkpoint at step {global_step} -> {final_ckpt_dir}")
        else:
            logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to swanlab
        if config.wandb_enabled:
            swanlab.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, accelerator=None):
    """Load the latest checkpoint and return the global step.
    
    支持从两种目录加载：
    - 数字目录（如 5000/）：save_interval 定期保存
    - latest/ 目录：每1000步保存的滚动检查点
    优先选择 global_step 更大的那个。
    """
    # 搜索数字目录
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    max_numbered_step = max(checkpoint_steps) if checkpoint_steps else -1
    
    # 检查 latest/ 目录（读取 metadata.pt 获取实际 step）
    latest_dir = checkpoint_dir / "latest"
    latest_dir_step = -1
    if latest_dir.exists() and (latest_dir / "metadata.pt").exists():
        try:
            meta = torch.load(latest_dir / "metadata.pt", map_location="cpu", weights_only=False)
            latest_dir_step = meta.get("global_step", -1)
            del meta
        except Exception as e:
            logging.warning(f"⚠️ 读取 latest/metadata.pt 失败，跳过: {e}")
    
    # 优先选择 latest/（它总是最近保存的滚动检查点）
    if latest_dir_step > 0:
        latest_step = latest_dir_step
        ckpt_dir = latest_dir
        if max_numbered_step > 0:
            logging.info(f"📂 优先从 latest/ 目录恢复 (step={latest_step}，数字目录最大 step={max_numbered_step})")
        else:
            logging.info(f"📂 从 latest/ 目录恢复 (step={latest_step})")
    elif max_numbered_step > 0:
        latest_step = max_numbered_step
        ckpt_dir = checkpoint_dir / f"{latest_step}"
        logging.info(f"📂 从数字目录恢复 (step={latest_step}，无 latest/ 目录)")
    else:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} (neither numbered dirs nor latest/)")

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading checkpoint for resume...")
        
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        # 检查是否有LoRA适配器权重
        adapter_path = ckpt_dir / "adapter_model"
        if adapter_path.exists():
            # ✅ 检测到LoRA检查点
            logging.info(f"✅ 检测到LoRA检查点")
            logging.info(f"   策略：只加载adapter权重（模型已从预训练权重初始化）")
            
            # 检查是否为PI0Pytorch模型
            if isinstance(model_to_load, PI0PytorchAlign):
                # PI0Pytorch模型包含两个使用LoRA的模块：paligemma和gemma_expert
                logging.info("检测到PI0PytorchAlign模型，加载adapter权重...")
                
                # 检查内部模型是否被PeftModel包装
                has_peft_paligemma = hasattr(model_to_load.pi0_pytorch, 'paligemma_with_expert') and \
                                     hasattr(model_to_load.pi0_pytorch.paligemma_with_expert, 'paligemma') and \
                                     isinstance(model_to_load.pi0_pytorch.paligemma_with_expert.paligemma, PeftModel)
                has_peft_gemma_expert = hasattr(model_to_load.pi0_pytorch, 'paligemma_with_expert') and \
                                        hasattr(model_to_load.pi0_pytorch.paligemma_with_expert, 'gemma_expert') and \
                                        isinstance(model_to_load.pi0_pytorch.paligemma_with_expert.gemma_expert, PeftModel)
                
                if not (has_peft_paligemma or has_peft_gemma_expert):
                    logging.warning("⚠️ 模型未应用LoRA，但检查点包含adapter权重")
                    logging.warning("   请确保训练脚本中apply_lora_to_model已执行")
                    return None
                
                # 只加载adapter权重（不加载model.safetensors）
                paligemma_adapter_path = adapter_path / "paligemma_adapter"
                gemma_expert_adapter_path = adapter_path / "gemma_expert_adapter"
                
                def _load_peft_weights_inplace(peft_model, adapter_dir, module_name):
                    """原地加载adapter权重到现有PeftModel，保持参数对象引用不变"""
                    # 加载adapter权重文件
                    adapter_safetensors = adapter_dir / "adapter_model.safetensors"
                    adapter_bin = adapter_dir / "adapter_model.bin"
                    
                    if adapter_safetensors.exists():
                        adapter_weights = safetensors.torch.load_file(str(adapter_safetensors))
                    elif adapter_bin.exists():
                        adapter_weights = torch.load(str(adapter_bin), map_location=device, weights_only=True)
                    else:
                        raise FileNotFoundError(
                            f"No adapter weights found in {adapter_dir} "
                            f"(expected adapter_model.safetensors or adapter_model.bin)"
                        )
                    
                    # 原地加载权重，不创建新参数对象
                    set_peft_model_state_dict(peft_model, adapter_weights)
                    logging.info(f"   ✅ 加载{module_name} adapter (原地加载，optimizer参数引用保持有效)")
                
                # 分别加载两个模块的LoRA权重（原地更新，不替换PeftModel对象）
                if has_peft_paligemma and paligemma_adapter_path.exists():
                    _load_peft_weights_inplace(
                        model_to_load.pi0_pytorch.paligemma_with_expert.paligemma,
                        paligemma_adapter_path, "PaliGemma"
                    )
                elif has_peft_paligemma:
                    raise FileNotFoundError(f"PaliGemma adapter not found: {paligemma_adapter_path}")
                    
                if has_peft_gemma_expert and gemma_expert_adapter_path.exists():
                    _load_peft_weights_inplace(
                        model_to_load.pi0_pytorch.paligemma_with_expert.gemma_expert,
                        gemma_expert_adapter_path, "Gemma Expert"
                    )
                elif has_peft_gemma_expert:
                    raise FileNotFoundError(f"Gemma Expert adapter not found: {gemma_expert_adapter_path}")
                
                logging.info("✅ 成功加载adapter权重")
                
                # ========== 修复：从model.safetensors恢复非LoRA可训练参数 ==========
                safetensors_path = ckpt_dir / "model.safetensors"
                if safetensors_path.exists():
                    logging.info("📦 从model.safetensors恢复非LoRA可训练参数...")
                    full_state_dict = safetensors.torch.load_file(str(safetensors_path), device=str(device))
                    
                    # 过滤：只保留不在paligemma和gemma_expert下的参数
                    # 这些参数（action_in_proj, action_out_proj, state_proj, learnable_token等）
                    # 在LoRA训练中也会被更新，但不在adapter保存范围内
                    lora_module_prefixes = (
                        "paligemma_with_expert.paligemma.",
                        "paligemma_with_expert.gemma_expert.",
                    )
                    non_lora_state_dict = {
                        k: v for k, v in full_state_dict.items()
                        if not any(k.startswith(prefix) for prefix in lora_module_prefixes)
                    }
                    
                    if non_lora_state_dict:
                        # 原地加载非LoRA参数到pi0_pytorch（strict=False因为只是部分key）
                        missing, unexpected = model_to_load.pi0_pytorch.load_state_dict(
                            non_lora_state_dict, strict=False
                        )
                        # missing keys是正常的（paligemma/gemma_expert相关的key被过滤了）
                        loaded_keys = list(non_lora_state_dict.keys())
                        logging.info(f"   ✅ 恢复了 {len(loaded_keys)} 个非LoRA参数: {loaded_keys}")
                        if unexpected:
                            logging.warning(f"   ⚠️ 意外的key: {unexpected}")
                    else:
                        logging.info("   ℹ️ 没有检测到需要恢复的非LoRA参数")
                    
                    # 清理内存
                    del full_state_dict, non_lora_state_dict
                    torch.cuda.empty_cache()
                else:
                    logging.warning("⚠️ 未找到model.safetensors，非LoRA参数（action_in_proj等）将保持预训练值！")
            else:
                logging.warning("⚠️ 非PI0Pytorch模型，跳过adapter加载")
        else:
            # 标准模型加载（非LoRA）
            safetensors_path = ckpt_dir / "model.safetensors"
            if safetensors_path.exists():
                model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                # 处理PeftModel情况
                if isinstance(model_to_load, PeftModel):
                    safetensors.torch.load_model(model_to_load.pi0_pytorch.base_model, safetensors_path, device=str(device))
                elif isinstance(model_to_load, PI0PytorchAlign):
                    # ✅ 修复：保存时保存的是pi0_pytorch子模块，加载时也应该加载到pi0_pytorch子模块
                    safetensors.torch.load_model(model_to_load.pi0_pytorch, safetensors_path, device=str(device))
                else:
                    safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
                logging.info("Loaded model state from safetensors format")
            else:
                raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_pi0_model")
        
        # ==================== 第2步：加载EgoVLPv2模型权重（如果启用） ====================
        if model_to_load.use_egovlpv2:
            # 兼容两种命名：数字目录用 egovlpv2_model_step{N}.pth，latest/ 用 egovlpv2_model.pth
            egovlpv2_checkpoint_path = ckpt_dir / f"egovlpv2_model_step{latest_step}.pth"
            if not egovlpv2_checkpoint_path.exists():
                egovlpv2_checkpoint_path = ckpt_dir / "egovlpv2_model.pth"
            
            logging.info(f"📦 加载EgoVLPv2模型权重: {egovlpv2_checkpoint_path}")
            egovlpv2_checkpoint = torch.load(egovlpv2_checkpoint_path, map_location=device, weights_only=False)
            
            # 提取state_dict
            if 'state_dict' in egovlpv2_checkpoint:
                egovlpv2_state_dict = egovlpv2_checkpoint['state_dict']
            else:
                egovlpv2_state_dict = egovlpv2_checkpoint
            
            # 智能检测DDP包装并加载
            if hasattr(model_to_load.egovlpv2_model, 'module'):
                model_to_load.egovlpv2_model.module.load_state_dict(egovlpv2_state_dict)
            else:
                model_to_load.egovlpv2_model.load_state_dict(egovlpv2_state_dict)
            
            logging.info(f"   ✅ 成功加载EgoVLPv2权重: {egovlpv2_checkpoint_path}")
            
            # 清理内存
            del egovlpv2_checkpoint, egovlpv2_state_dict
            torch.cuda.empty_cache()
            gc.collect()
        
        log_memory_usage(device, latest_step, "after_loading_egovlpv2_model")
        
        # ==================== 第3步：加载Alignment模型权重（如果启用） ====================
        if isinstance(model_to_load, PI0PytorchAlign) and model_to_load.use_alignment:
            # 兼容两种命名：数字目录用 alignment_model_step{N}.pth，latest/ 用 alignment_model.pth
            alignment_weight_path = ckpt_dir / f"alignment_model_step{latest_step}.pth"
            if not alignment_weight_path.exists():
                alignment_weight_path = ckpt_dir / "alignment_model.pth"
            
            logging.info(f"📦 加载Alignment模型权重...")
            
            # 加载state_dict
            alignment_state_dict = torch.load(alignment_weight_path, map_location=device, weights_only=False)
            
            # 智能检测DDP包装并加载
            if hasattr(model_to_load.alignment_model, 'module'):
                model_to_load.alignment_model.module.load_state_dict(alignment_state_dict)
            else:
                model_to_load.alignment_model.load_state_dict(alignment_state_dict)
            
            logging.info(f"   ✅ 成功加载Alignment权重: {alignment_weight_path}")
            
            # 清理内存
            del alignment_state_dict
            torch.cuda.empty_cache()
            gc.collect()
        
        log_memory_usage(device, latest_step, "after_loading_alignment_model")

        # 加载optimizer状态
        # 注意：accelerator.save()保存的文件应该用torch.load()加载
        # accelerator.save()只是确保在DDP环境下正确保存，但加载时仍使用torch.load()
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        # ==================== 总结：输出加载完成信息 ====================
        logging.info(f"✅ 检查点加载完成总结 (Step {latest_step}):")
        logging.info(f"✅ 训练元数据 (global_step={global_step})")
        
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


class TimingLogger:
    """精确的计时日志记录器，用于分析训练过程中各个环节的耗时"""
    
    def __init__(self, is_main=True, log_interval=10, detailed_log_interval=1):
        self.is_main = is_main
        self.log_interval = log_interval
        self.detailed_log_interval = detailed_log_interval  # 实时详细计时日志间隔
        self.timing_data = {}
        self.step_count = 0
        self.current_step_timings = {}  # 当前步骤的计时信息
        
    def start_timer(self, name):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保GPU操作完成
        self.timing_data[f"{name}_start"] = time.time()
        
    def end_timer(self, name):
        """结束计时并返回耗时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保GPU操作完成
        end_time = time.time()
        start_time = self.timing_data.get(f"{name}_start", end_time)
        duration = end_time - start_time
        
        # 记录到累积统计中
        if name not in self.timing_data:
            self.timing_data[name] = []
        self.timing_data[name].append(duration)
        
        # 记录当前步骤的计时
        self.current_step_timings[name] = duration
        
        return duration
        
    def log_step_timing(self, step, immediate_log=False):
        """记录当前步骤的时间统计"""
        self.step_count += 1
        
        # 实时输出详细计时信息
        if self.is_main and (step % self.detailed_log_interval == 0) and self.current_step_timings:
            self._print_detailed_step_timing(step)
        
        # 每隔一定步数或立即记录时输出平均耗时
        if immediate_log or (self.step_count % self.log_interval == 0):
            if self.is_main:
                self._print_timing_summary(step)
                
        # 清空当前步骤计时
        self.current_step_timings = {}
                
    def _print_timing_summary(self, step):
        """打印时间统计摘要"""
        if not self.timing_data:
            return
        
        # 定义总时间项和子步骤顺序
        total_time_keys = {"total_step", "total_initialization"}
        step_order = ["checkpoint_directory_setup", "swanlab_initialization", "data_loader_initialization", 
                     "sample_data_logging", "model_configuration", "model_creation", "ddp_setup", 
                     "pretrained_weights_loading", "optimizer_initialization", "checkpoint_loading",
                     "data_loading", "data_transfer", "lr_update", "forward_pass", 
                     "backward_pass", "gradient_clipping", "optimizer_step", ]
        
        # 分别收集子步骤和总时间项
        sub_step_summaries = []
        total_time_summaries = []
        sub_step_total = 0
        
        # 按顺序处理子步骤
        for name in step_order:
            if name in self.timing_data and self.timing_data[name]:
                times = self.timing_data[name]
                avg_time = sum(times) / len(times)
                latest_time = times[-1] if times else 0
                sub_step_total += latest_time
                sub_step_summaries.append(f"{name}: {latest_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
        
        # 处理其他未列出的子步骤
        for name, times in self.timing_data.items():
            if (name not in step_order and name not in total_time_keys and 
                not name.endswith('_start') and times):
                avg_time = sum(times) / len(times)
                latest_time = times[-1] if times else 0
                sub_step_total += latest_time
                sub_step_summaries.append(f"{name}: {latest_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
        
        # 处理总时间项
        for name in total_time_keys:
            if name in self.timing_data and self.timing_data[name]:
                times = self.timing_data[name]
                avg_time = sum(times) / len(times)
                latest_time = times[-1] if times else 0
                total_time_summaries.append(f"{name}: {latest_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
            
        if sub_step_summaries or total_time_summaries:
            logging.info(f"🕒 Step {step} 计时详情:")
            for summary in sub_step_summaries:
                logging.info(f"  ├─ {summary}")
            if total_time_summaries:
                logging.info(f"  ├─ 【总时间】")
                for summary in total_time_summaries:
                    logging.info(f"  ├─ {summary}")
            logging.info(f"  └─ 子步骤累计: {sub_step_total*1000:.2f}ms")
            
    def reset_stats(self):
        """重置统计数据"""
        for key in list(self.timing_data.keys()):
            if not key.endswith('_start'):
                self.timing_data[key] = []
                
    def _print_detailed_step_timing(self, step):
        """打印单个步骤的详细计时信息"""
        if not self.current_step_timings:
            return
        
        # 定义总时间项，这些不应该包含在百分比计算的分母中
        total_time_keys = {"total_step", "total_initialization"}
        
        # 计算实际子步骤的总时间（排除总时间项）
        sub_step_total = sum(time_val for name, time_val in self.current_step_timings.items() 
                           if name not in total_time_keys)
        
        # 按执行顺序排列计时项
        timing_order = ["data_loading", "data_transfer", "lr_update", "forward_pass", 
                       "backward_pass", "gradient_clipping", "optimizer_step", 
                       ]
        
        timing_info = []
        
        # 首先处理有序的子步骤
        for name in timing_order:
            if name in self.current_step_timings:
                time_ms = self.current_step_timings[name] * 1000
                percentage = (self.current_step_timings[name] / sub_step_total) * 100 if sub_step_total > 0 else 0
                timing_info.append(f"{name}: {time_ms:.1f}ms ({percentage:.1f}%)")
        
        # 添加其他子步骤（排除总时间项）
        for name, time_val in self.current_step_timings.items():
            if name not in timing_order and name not in total_time_keys:
                time_ms = time_val * 1000
                percentage = (time_val / sub_step_total) * 100 if sub_step_total > 0 else 0
                timing_info.append(f"{name}: {time_ms:.1f}ms ({percentage:.1f}%)")
        
        if timing_info:
            logging.info(f"⏱️  Step {step} 实时计时: {', '.join(timing_info)}")


def train_loop(config: _config.TrainConfig):
    # 创建accelerator替代DDP设置
    accelerator = create_accelerator(config)
    
    # 使用accelerator设置随机种子（确保分布式一致性）  # 
    transformers_set_seed(config.seed)
    accelerate_set_seed(config.seed, device_specific=True)
    is_main = accelerator.is_main_process
    
    # 初始化阶段的计时器
    init_timer = TimingLogger(is_main=is_main, log_interval=1, detailed_log_interval=1)
    
    if is_main:
        logging.info("🚀 开始训练初始化，详细计时如下：")
    
    init_timer.start_timer("total_initialization")

    # Initialize checkpoint directory and swanlab
    init_timer.start_timer("checkpoint_directory_setup")
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")
    
    init_timer.end_timer("checkpoint_directory_setup")
    
    # ========== 添加文件日志（在checkpoint目录创建后） ==========
    add_file_logging(config.checkpoint_dir, is_main_process=accelerator.is_main_process)

    # 注意：swanlab初始化已移至模型创建后，以便包含egovlpv2和alignment配置

    # Build data loader using the unified data loader
    init_timer.start_timer("data_loader_initialization")
    # ✅ 使用per_device_batch_size（已在create_accelerator中计算）
    # 注意：需要确保data loader使用正确的per_device_batch_size
    # 实际batch size = per_device_batch_size × num_gpus × gradient_accumulation_steps
    
    logging.info(f"📊 Batch Size配置:")
    logging.info(f"   - 每GPU batch size: {config.per_device_batch_size}")
    logging.info(f"   - GPU数量: {accelerator.num_processes}")
    logging.info(f"   - 梯度累积步数: {config.gradient_accumulation_steps}")
    logging.info(f"   - 实际全局batch size: {config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps}")

    # ✅ 临时修改config.batch_size为per_device_batch_size，因为DataLoader会使用这个值
    # Accelerator的prepare()会自动处理多GPU的数据分配
    # 注意：config是frozen dataclass，需要使用object.__setattr__来修改
    original_batch_size = config.batch_size
    object.__setattr__(config, 'batch_size', config.per_device_batch_size)
    loader, data_config = build_datasets(config)
    object.__setattr__(config, 'batch_size', original_batch_size)  # 恢复原始值
    init_timer.end_timer("data_loader_initialization")

    # 注意：sample_data_logging已移至swanlab初始化后，以便使用swanlab.log

    # Build model
    init_timer.start_timer("model_configuration")
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
            lora=config.lora_config  # 传递LoRA配置
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
        
        # 确保LoRA配置也被传递到模型配置中
        if hasattr(config, 'lora_config') and config.lora_config is not None:
            logging.info(f"Using LoRA configuration: {config.lora_config}")
            object.__setattr__(model_cfg, "lora", config.lora_config)
    
    # 打印LoRA配置信息
    if hasattr(model_cfg, "lora") and model_cfg.lora is not None:
        logging.info("LoRA配置已启用:")
        logging.info(f"  - PaliGemma LoRA rank: {model_cfg.lora.get('lora_rank_paligemma', 'N/A')}")
        logging.info(f"  - PaliGemma LoRA alpha: {model_cfg.lora.get('lora_alpha_paligemma', 'N/A')}")
        logging.info(f"  - Gemma Expert LoRA rank: {model_cfg.lora.get('lora_rank_gemma_expert', 'N/A')}")
        logging.info(f"  - Gemma Expert LoRA alpha: {model_cfg.lora.get('lora_alpha_gemma_expert', 'N/A')}")
    else:
        logging.info("LoRA配置未启用，将进行全参数微调")
    
    init_timer.end_timer("model_configuration")

    # ========== 步骤1: 创建基础模型 ==========
    init_timer.start_timer("model_creation")
     
    # 从config读取use_learnable_token，并将其设置到model_cfg中，以便PI0Pytorch在__init__时读取
    use_learnable_token = config.use_learnable_token
    object.__setattr__(model_cfg, "use_learnable_token", config.use_learnable_token)
    
    model = PI0PytorchAlign(
        config=model_cfg,
        use_egovlpv2=config.use_egovlpv2,
        use_alignment=config.use_alignment,
        egovlpv2_config_path=config.egovlpv2_config_path,
        vlm_loss_weight=config.vlm_loss_weight,
        alignment_loss_weight=config.alignment_loss_weight,
        freeze_vlm=config.freeze_vlm,
        vlm_mode=model_cfg.vlm_mode  # 传递vlm_mode参数
    ).to(accelerator.device)
    logging.info(f"✅ PI0PytorchAlign模型创建完成 (VLM模式: {model_cfg.vlm_mode})")
    
    # 如果启用freeze_vlm，冻结VLM模型参数并设置为eval模式以减少显存计算
    if config.freeze_vlm and hasattr(model, 'egovlpv2_model') and model.egovlpv2_model is not None:
        logging.info("🔒 冻结VLM模型参数...")
        for param in model.egovlpv2_model.parameters():
            param.requires_grad = False
        model.egovlpv2_model.eval()  # 设置为eval模式，禁用dropout和batch norm的更新，减少显存
        logging.info("✅ VLM模型参数已冻结（requires_grad=False, eval模式）")
    
    init_timer.end_timer("model_creation")
    
    # ========== 初始化swanlab（在模型创建后，以便包含egovlpv2和alignment配置） ==========
    init_timer.start_timer("swanlab_initialization")
    if accelerator.is_main_process:
        # 从模型中获取egovlpv2和alignment配置
        egovlpv2_config_for_logging = None
        alignment_config_for_logging = None
        
        if config.use_egovlpv2 and hasattr(model, 'egovlpv2_config'):
            # model.egovlpv2_config是ConfigParser对象，需要提取.config属性获取字典
            cfg = model.egovlpv2_config
            egovlpv2_config_for_logging = cfg.config if hasattr(cfg, 'config') else cfg
            
        if config.use_alignment and hasattr(model, 'alignment_config'):
            # model.alignment_config是ConfigParser对象，需要提取.config属性获取字典
            cfg = model.alignment_config
            alignment_config_for_logging = cfg.config if hasattr(cfg, 'config') else cfg
        
        init_swanlab(
            config, 
            resuming=resuming, 
            enabled=config.wandb_enabled,
            egovlpv2_config=egovlpv2_config_for_logging,
            alignment_config=alignment_config_for_logging
        )
    init_timer.end_timer("swanlab_initialization")
    
    # ========== 记录样本数据到swanlab（在swanlab初始化后） ==========
    init_timer.start_timer("sample_data_logging")
    if accelerator.is_main_process and config.wandb_enabled and not resuming:
        # 创建独立的data loader用于采样，避免消耗主loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # 转换observation和actions为torch张量
        if len(sample_batch) == 3:
            observation, actions, prompt = sample_batch
            task_index = None
            task_id = None
        elif len(sample_batch) == 4:
            observation, actions, prompt, task_id = sample_batch
            task_index = task_id  # 兼容旧格式
        elif len(sample_batch) == 5:
            observation, actions, prompt, task_index, task_id = sample_batch
        else:
            raise ValueError(f"sample_batch 长度错误: {len(sample_batch)}")
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # 创建样本图像用于swanlab记录
        images_to_log = []

        # 从第一个图像张量获取batch size
        batch_size = sample_batch['image']['base_0_rgb'].shape[0]
        for i in range(min(5, batch_size)):
            # 水平拼接所有相机视图
            img_concatenated = torch.cat([sample_batch['image']['base_0_rgb'][i], 
                                        sample_batch['image']['left_wrist_0_rgb'][i], 
                                        sample_batch['image']['right_wrist_0_rgb'][i]], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()

            # 转换为uint8格式（假设值在[0, 1]范围）
            img_for_pil = img_concatenated.transpose(1, 2, 0)
            
            if img_for_pil.max() <= 1.0 and img_for_pil.min() >= 0.0:
                img_for_pil = (img_for_pil * 255).astype('uint8')
            elif img_for_pil.max() <=1.0 and img_for_pil.min() < 0.0:
                img_for_pil = ((img_for_pil/2+0.5) * 255).astype('uint8')
            else:
                img_for_pil = img_for_pil.astype('uint8')
            
            images_to_log.append(swanlab.Image(img_for_pil))
            
        swanlab.log({"camera_views": images_to_log}, step=0)

        # 清理内存
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")
    init_timer.end_timer("sample_data_logging")

    # Log initial memory usage after model creation
    if accelerator.is_main_process and torch.cuda.is_available():
        log_memory_usage(accelerator.device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if accelerator.num_processes >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # ========== 步骤2: 加载预训练权重（简化！） ==========
    init_timer.start_timer("pretrained_weights_loading")
    if config.pytorch_weight_path is not None:
        logging.info(f"📦 加载预训练权重: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        
        # ✅ 简单直接的权重加载（因为还没有应用LoRA）
        if os.path.exists(model_path):
            missing_keys, unexpected_keys = safetensors.torch.load_model(model.pi0_pytorch, model_path, strict=False)
            if missing_keys:
                logging.warning(f"⚠️ 缺少以下键: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"⚠️ 意外找到以下键: {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                logging.info("✅ 成功加载预训练权重")
            else:
                logging.warning("⚠️ 加载预训练权重时存在缺失或意外键")
        else:
            logging.warning(f"⚠️ 未找到权重文件: {model_path}")
    init_timer.end_timer("pretrained_weights_loading")
    
    # ========== 步骤3: 应用LoRA（在加载权重之后！） ==========
    init_timer.start_timer("lora_application")
    model = apply_lora_to_model(model, config.lora_config)
    init_timer.end_timer("lora_application")
    
    # ========== 步骤4: 配置Gradient Checkpointing ==========
    init_timer.start_timer("gradient_checkpointing_setup")
    enable_gradient_checkpointing = config.enable_gradient_checkpointing
    if enable_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logging.info("✅ 已启用梯度检查点 - 节省显存，但会降低训练速度")
        else:
            logging.warning("⚠️ 模型不支持梯度检查点，将禁用此功能")
            enable_gradient_checkpointing = False
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            # Debug: 验证是否真的禁用了
            if hasattr(model, "is_gradient_checkpointing_enabled"):
                is_enabled = model.is_gradient_checkpointing_enabled()
                logging.info(f"🔍 DEBUG: gradient_checkpointing_enabled = {is_enabled}")
        logging.info("🚫 梯度检查点已禁用 - 使用更多显存以获得更快训练速度")
    init_timer.end_timer("gradient_checkpointing_setup")
    

    # ========== 优化器初始化：支持多模型参数分组 ==========
    init_timer.start_timer("optimizer_initialization")
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # 准备EgoVLPv2和Alignment配置（如果启用）
    egovlpv2_config = None
    alignment_config = None
    
    if config.use_egovlpv2 and hasattr(model, 'egovlpv2_config'):
        # model.egovlpv2_config是ConfigParser对象，需要提取.config属性获取字典
        cfg = model.egovlpv2_config
        egovlpv2_config = cfg.config if hasattr(cfg, 'config') else cfg
        
    if config.use_alignment and hasattr(model, 'alignment_config'):
        # model.alignment_config是ConfigParser对象，需要提取.config属性获取字典
        cfg = model.alignment_config
        alignment_config = cfg.config if hasattr(cfg, 'config') else cfg
    
    # 使用optim_utils创建多模型优化器
    # 如果freeze_vlm为True，则不将VLM参数加入优化器
    freeze_egovlpv2 = config.freeze_vlm if hasattr(config, 'freeze_vlm') else False
    
    if config.use_new_optimizer:
        optim = optim_utils.create_optimizer_and_scheduler(
            model=model,
            optimizer_config=config.optimizer,
            lr_schedule_config=config.lr_schedule,
            num_training_steps=config.num_train_steps,
            egovlpv2_config=egovlpv2_config,
            alignment_config=alignment_config,
            freeze_egovlpv2=freeze_egovlpv2
        )
    else:
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
    
    init_timer.end_timer("optimizer_initialization")

    # Load checkpoint if resuming  # 
    init_timer.start_timer("checkpoint_loading")
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, accelerator.device, accelerator)
        logging.info(f"Resumed training from step {global_step}")

        if config.lora_config is not None:
            model_param_ids = {id(p) for p in model.parameters() if p.requires_grad}
            optim_param_ids = {id(p) for pg in optim.param_groups for p in pg['params'] if p.requires_grad}
            orphaned = optim_param_ids - model_param_ids
            if orphaned:
                raise RuntimeError(
                    f"🚨 严重Bug：Resume后optimizer中有 {len(orphaned)} 个参数未被model引用！"
                    f"这意味着optimizer.step()会更新孤立参数，LoRA权重永远不会被更新。"
                    f"请检查load_checkpoint中是否错误地替换了PeftModel对象。"
                )
            else:
                logging.info("✅ Resume参数一致性验证通过：optimizer引用的参数 ⊆ model的参数")
    init_timer.end_timer("checkpoint_loading")

    def lr_schedule(step: int):
        """
        计算学习率的scale factor（0-1之间的值）
        返回的scale factor会与每个参数组的base_lr相乘，得到实际的学习率
        这样不同模型可以使用不同的base_lr，但共享相同的scheduler曲线
        """
        if step < warmup_steps:
            # Warmup阶段：从1/(warmup_steps+1)线性增长到1
            init_scale = 1.0 / (warmup_steps + 1)
            return init_scale + (1.0 - init_scale) * step / warmup_steps
        # Cosine decay阶段：从1衰减到end_lr/peak_lr
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        end_scale = end_lr / peak_lr  # 计算end_lr相对于peak_lr的比例
        return end_scale + (1.0 - end_scale) * cos

    # 使用accelerator包装模型、优化器和数据加载器
    # EgoHOD训练模式（freeze_vlm=False）也需要dataloader，与EgoVLPv2对齐
    need_vlm_dataloader = (
        config.use_egovlpv2 and not config.freeze_vlm and
        model_cfg.vlm_mode in ("egovlpv2", "egohod")  # 支持EgoHOD co-training
    )
    
    if need_vlm_dataloader:
        # co-training模式：需要额外的VLM数据加载器（EgoVLPv2或EgoHOD）
        raw_egovlpv2_dataloader = model.egovlpv2_components['train_dataloaders'][0]
        egovlpv2_dataloader = raw_egovlpv2_dataloader
        model, optim, loader, egovlpv2_dataloader = accelerator.prepare(
            model, optim, loader, egovlpv2_dataloader
        )
        # 创建VLM数据迭代器
        egovlpv2_dataloader_iter = iter(egovlpv2_dataloader)
        logging.info(f"✅ Accelerator包装完成（含{model_cfg.vlm_mode}数据加载器）")
    else:
        # 普通训练模式 或 freeze VLM模式
        egovlpv2_dataloader_iter = None
        model, optim, loader = accelerator.prepare(model, optim, loader)
        logging.info("✅ Accelerator包装完成")
    
    # 创建optimizer后立即检查参数组配置
    if config.use_new_optimizer:
        optim_utils.inspect_optimizer_param_groups(optim, "创建optimizer后", accelerator)
    
    if getattr(config, "egovlpv2_config_path", None):
        source_conf_path = config.egovlpv2_config_path
        import pathlib
        if isinstance(source_conf_path, str):
            source_conf_path = pathlib.Path(source_conf_path)
        if source_conf_path.exists():
            dest_conf_path = exp_checkpoint_dir / source_conf_path.name
            try:
                shutil.copy(str(source_conf_path), str(dest_conf_path))
                logging.info(f"✅ 已将 egovlpv2_config_path 复制到输出目录: {dest_conf_path}")
            except Exception as e:
                logging.warning(f"⚠️ 复制 egovlpv2_config_path 失败: {source_conf_path} 到 {dest_conf_path}，原因: {e}")
        else:
            logging.warning(f"⚠️ egovlpv2_config_path 配置文件不存在，跳过复制: {source_conf_path}")
    # -----------------------------------------------------------------------
    else:
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")
    
    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    # 使用accelerator.is_main_process替代is_main变量
    if accelerator.is_main_process:
        logging.info(
            f"Running on: {platform.node()} | world_size={accelerator.num_processes}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, per_device_batch_size={config.per_device_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")
    
    # 结束初始化总计时
    init_timer.end_timer("total_initialization")
    
    # 输出初始化阶段的详细计时汇总
    if is_main:
        init_timer.log_step_timing(0, immediate_log=True)
        logging.info("✅ 训练初始化完成，开始训练循环")

    
    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not accelerator.is_main_process)
        if accelerator.is_main_process
        else None
    )

    while global_step < config.num_train_steps:
        for batch in loader:
            # batch格式：(observation, actions, prompt, task_index, task_id) 或旧格式
            if len(batch)==3:
                observation, actions, prompt = batch
                task_index = None
                task_ids = None
            elif len(batch)==4:
                # 旧格式：只有task_id
                observation, actions, prompt, task_ids = batch
                task_index = task_ids  # 兼容旧格式
            elif len(batch)==5:
                # 新格式：同时有task_index和task_id
                observation, actions, prompt, task_index, task_ids = batch
            else:
                raise ValueError("batch的内容出现错误")
            
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(accelerator.device) if isinstance(x, torch.Tensor) else x, observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(accelerator.device)  # noqa: PLW2901

            # 使用accelerator进行梯度累积管理
            with accelerator.accumulate(model):
                # 更新学习率：根据每个参数组的base_lr和scheduler的scale factor计算
                scale_factor = lr_schedule(global_step)
                for pg in optim.param_groups:
                    base_lr = pg['base_lr']  # 如果没有base_lr，使用peak_lr作为默认值
                    pg["lr"] = base_lr * scale_factor  # 实际lr = base_lr * scale_factor
                
                # 准备输入：如果是对齐训练，获取EgoVLPv2 batch
                if not config.freeze_vlm and egovlpv2_dataloader_iter:
                    egovlpv2_batch = next(egovlpv2_dataloader_iter)
                else:
                    egovlpv2_batch = None
                # 统一的前向传播调用（兼容PI0PytorchAlign和PI0Pytorch）
                result = model(
                    observation, actions,
                    egovlpv2_batch=egovlpv2_batch,
                    prompt=prompt,
                    task_index=task_index,  # 原始任务索引
                    task_id=task_ids        # 聚类后的任务类别ID
                )
                
                # 统一的返回值解析（elegant & robust）
                # PI0PytorchAlign返回: (total_loss, (pi0_loss, vlm_loss, align_loss), alignment_loss_dict)
                # PI0Pytorch返回: (total_loss, (pi0_loss, vlm_loss, align_loss))
                # ✅ 修复：直接使用模型返回的loss_dict，避免默认值覆盖问题
                # 如果模型没有返回loss_dict，使用空字典（不会记录对齐loss，这是正确的）
                if len(result) == 3:
                    loss, (loss_pi0, loss_vlm, loss_align), result_loss_dict = result
                    # 直接使用模型返回的loss_dict，包含所有实际的loss值和统计信息
                    alignment_loss_dict = result_loss_dict.copy() if result_loss_dict else {}
                else:
                    loss, (loss_pi0, loss_vlm, loss_align) = result
                    # 如果没有返回loss_dict，使用空字典（不会记录对齐loss）
                    alignment_loss_dict = {}

                # === 动态alignment loss权重 ===
                # warmup期间线性从init_weight增加到alignment_loss_weight，避免早期梯度干扰PI0
                if config.alignment_loss_warmup and isinstance(loss_align, torch.Tensor) and loss_align.requires_grad:
                    align_warmup_steps = config.alignment_loss_warmup_steps or warmup_steps
                    if global_step < align_warmup_steps:
                        align_ratio = global_step / align_warmup_steps
                        dynamic_align_weight = config.alignment_loss_warmup_init + (config.alignment_loss_weight - config.alignment_loss_warmup_init) * align_ratio
                    else:
                        dynamic_align_weight = config.alignment_loss_weight
                    # 用动态权重重新计算total_loss（覆盖模型内的静态权重）
                    loss = loss_pi0 + config.vlm_loss_weight * loss_vlm + dynamic_align_weight * loss_align
                    alignment_loss_dict['dynamic_align_weight'] = dynamic_align_weight

                # 反向传播计时
                accelerator.backward(loss)

                # Log memory usage after backward pass
                if global_step < 5 and accelerator.is_main_process and torch.cuda.is_available():
                    log_memory_usage(accelerator.device, global_step, "after_backward")

                # 梯度裁剪（accelerator自动处理分布式同步）
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
                else:
                    grad_norm = 0.0

                # Optimizer step - 保持原有逻辑，accelerator.prepare()已经包装了优化器
                optim.step()
                optim.zero_grad(set_to_none=True)
                
                if accelerator.sync_gradients and getattr(config, 'reset_bank_every_step', True):
                    if hasattr(model, 'module'):
                        if hasattr(model.module, 'reset_feature_bank'):
                            model.module.reset_feature_bank()
                    else:
                        if hasattr(model, 'reset_feature_bank'):
                            model.reset_feature_bank()

            # Collect stats
            if accelerator.is_main_process:
                # 构建info字典，每步都记录loss和lr
                info = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "loss_pi0": loss_pi0.item() if isinstance(loss_pi0, torch.Tensor) else loss_pi0,
                    "loss_vlm": loss_vlm.item() if isinstance(loss_vlm, torch.Tensor) else loss_vlm,
                    "loss_align": loss_align.item() if isinstance(loss_align, torch.Tensor) else loss_align,
                }
                
                # 添加细粒度对齐loss（总是记录，保持日志一致性）
                for key, value in alignment_loss_dict.items():
                    # 将loss_dict中的值转换为标量
                    if isinstance(value, torch.Tensor):
                        info[f"align_{key}"] = value.item()
                    elif isinstance(value, dict):
                        # 跳过嵌套字典（如layer_losses）
                        continue
                    else:
                        info[f"align_{key}"] = value
                
                # 添加logit统计信息（正样本平均logit、负样本平均logit、temperature）
                if 'pos_mean_logit' in alignment_loss_dict:
                    pos_logit = alignment_loss_dict['pos_mean_logit']
                    neg_logit = alignment_loss_dict['neg_mean_logit']
                    temp = alignment_loss_dict['temperature']
                    
                    # 转换为标量（如果是Tensor）
                    if isinstance(pos_logit, torch.Tensor):
                        info['pos_mean_logit'] = pos_logit.item()
                    else:
                        info['pos_mean_logit'] = pos_logit
                    
                    if isinstance(neg_logit, torch.Tensor):
                        info['neg_mean_logit'] = neg_logit.item()
                    else:
                        info['neg_mean_logit'] = neg_logit
                    
                    info['temperature'] = temp
                
                # ✅ 只在梯度同步时记录grad_norm（避免记录0值导致平均值被稀释）
                if accelerator.sync_gradients:
                    info["grad_norm"] = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
                infos.append(info)

            if accelerator.sync_gradients and accelerator.is_main_process and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                # Loss和LR: 对所有记录求平均（包括梯度累积期间的步骤）
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                # Grad_norm: 只对包含grad_norm的记录求平均（即只统计梯度同步步骤）
                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                
                # 对齐训练的各部分loss平均值
                avg_loss_pi0 = sum(info.get("loss_pi0", 0) for info in infos) / len(infos)
                avg_loss_vlm = sum(info.get("loss_vlm", 0) for info in infos) / len(infos)
                avg_loss_align = sum(info.get("loss_align", 0) for info in infos) / len(infos)
                
                # 获取当前步骤的loss值（最新的info）
                current_loss_pi0 = infos[-1].get("loss_pi0", 0)
                current_loss_vlm = infos[-1].get("loss_vlm", 0)
                current_loss_align = infos[-1].get("loss_align", 0)
                
                # 计算细粒度对齐loss的平均值
                fine_grained_losses = {}
                for key in ["align_as2ts", "align_as2tt", "align_at2tt"]:
                    values = [info.get(key, 0) for info in infos if key in info]
                    if values:
                        fine_grained_losses[key] = sum(values) / len(values)
                
                # 计算logit统计信息的平均值
                avg_pos_logit = None
                avg_neg_logit = None
                avg_temperature = None
                if any("pos_mean_logit" in info for info in infos):
                    pos_logits = [info["pos_mean_logit"] for info in infos if "pos_mean_logit" in info]
                    neg_logits = [info["neg_mean_logit"] for info in infos if "neg_mean_logit" in info]
                    temps = [info["temperature"] for info in infos if "temperature" in info]
                    if pos_logits:
                        avg_pos_logit = sum(pos_logits) / len(pos_logits)
                        avg_neg_logit = sum(neg_logits) / len(neg_logits)
                        avg_temperature = sum(temps) / len(temps)
                
                # 获取当前步骤的logit统计信息
                current_pos_logit = infos[-1].get("pos_mean_logit", None)
                current_neg_logit = infos[-1].get("neg_mean_logit", None)
                current_temperature = infos[-1].get("temperature", None)
                
                # 统计正负样本数量（与SpatialVLA一致）
                avg_pos_samples = None
                avg_neg_samples = None
                avg_total_samples = None
                if any("align_avg_pos_samples" in info for info in infos):
                    pos_samples = [info["align_avg_pos_samples"] for info in infos if "align_avg_pos_samples" in info]
                    neg_samples = [info["align_avg_neg_samples"] for info in infos if "align_avg_neg_samples" in info]
                    total_samples = [info["align_avg_total_samples"] for info in infos if "align_avg_total_samples" in info]
                    if pos_samples:
                        avg_pos_samples = sum(pos_samples) / len(pos_samples)
                        avg_neg_samples = sum(neg_samples) / len(neg_samples)
                        avg_total_samples = sum(total_samples) / len(total_samples)
                
                # 获取当前步骤的正负样本统计
                current_pos_samples = infos[-1].get("align_avg_pos_samples", None)
                current_neg_samples = infos[-1].get("align_avg_neg_samples", None)
                current_total_samples = infos[-1].get("align_avg_total_samples", None)
                
                log_msg = (
                    f"step={global_step} total_loss={avg_loss:.4f} "
                    f"pi0={avg_loss_pi0:.4f} vlm={avg_loss_vlm:.4f} align={avg_loss_align:.4f} "
                    f"lr={avg_lr:.2e}"
                )
                # 显示动态alignment权重（如果启用）
                if config.alignment_loss_warmup and infos and 'align_dynamic_align_weight' in infos[-1]:
                    log_msg += f" aw={infos[-1]['align_dynamic_align_weight']:.4f}"
                
                # 添加细粒度对齐loss到日志（如果有）
                if fine_grained_losses:
                    fg_loss_str = " ".join([f"{k.replace('align_', '')}={v:.4f}" for k, v in fine_grained_losses.items()])
                    log_msg += f" [{fg_loss_str}]"
                
                # 添加logit统计信息到日志（如果有）
                if avg_pos_logit is not None:
                    log_msg += f" pos_logit_avg={avg_pos_logit:.4f} neg_logit_avg={avg_neg_logit:.4f} temp_avg={avg_temperature:.4f}"
                    log_msg += f" pos_logit_cur={current_pos_logit:.4f} neg_logit_cur={current_neg_logit:.4f} temp_cur={current_temperature:.4f}"
                
                # 添加正负样本数量到日志（如果有）
                if avg_pos_samples is not None:
                    log_msg += (
                        f" pos_samples_avg={avg_pos_samples:.2f} neg_samples_avg={avg_neg_samples:.2f} total_samples_avg={avg_total_samples:.2f}"
                    )
                    log_msg += (
                        f" pos_samples_cur={current_pos_samples:.2f} neg_samples_cur={current_neg_samples:.2f} total_samples_cur={current_total_samples:.2f}"
                    )
                
                if avg_grad_norm is not None:
                    log_msg += f" grad_norm={avg_grad_norm:.2f}"
                log_msg += f" time={elapsed:.1f}s"
                logging.info(log_msg)

                # Log to swanlab
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    
                    # 添加对齐训练的平均loss到日志
                    log_payload["loss_pi0"] = avg_loss_pi0
                    log_payload["loss_vlm"] = avg_loss_vlm
                    log_payload["loss_align"] = avg_loss_align
                    
                    # ✅ 新增：记录当前步骤的loss值
                    log_payload["loss_pi0_current"] = current_loss_pi0
                    log_payload["loss_vlm_current"] = current_loss_vlm
                    log_payload["loss_align_current"] = current_loss_align
                    
                    # ✅ 新增：记录logit统计信息（平均值和当前值）
                    if avg_pos_logit is not None:
                        log_payload["pos_mean_logit_avg"] = avg_pos_logit
                        log_payload["neg_mean_logit_avg"] = avg_neg_logit
                        log_payload["temperature_avg"] = avg_temperature
                        log_payload["pos_mean_logit_current"] = current_pos_logit
                        log_payload["neg_mean_logit_current"] = current_neg_logit
                        log_payload["temperature_current"] = current_temperature
                    
                    # ✅ 新增：记录正负样本数量统计（平均值和当前值）
                    if avg_pos_samples is not None:
                        log_payload["avg_pos_samples"] = avg_pos_samples
                        log_payload["avg_neg_samples"] = avg_neg_samples
                        log_payload["avg_total_samples"] = avg_total_samples
                        log_payload["avg_pos_samples_current"] = current_pos_samples
                        log_payload["avg_neg_samples_current"] = current_neg_samples
                        log_payload["avg_total_samples_current"] = current_total_samples
                    
                    swanlab.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            # 只在梯度同步时更新global_step，确保与梯度累积对齐
            if accelerator.sync_gradients:
                global_step += 1
                
                # 前30步中每5步检查一次optimizer参数组配置
                if config.use_new_optimizer and global_step <= 30 and global_step % 10 == 0:
                    optim_utils.inspect_optimizer_param_groups(optim, f"训练步骤{global_step}", accelerator)
                
                # 保存检查点，传入accelerator参数支持LoRA和分布式训练
                save_checkpoint(model, optim, global_step, config, accelerator.is_main_process, data_config, accelerator)
                # barrier：rank 0 保存耗时较长，其他 rank 需等待，避免 NCCL timeout
                if global_step % config.save_interval == 0 and global_step > 0:
                    accelerator.wait_for_everyone()
                
                # 每1000步保存latest checkpoint
                if global_step % 1000 == 0 and global_step > 0:
                    save_checkpoint(model, optim, global_step, config, accelerator.is_main_process, data_config, accelerator, save_as_latest=True)
                    accelerator.wait_for_everyone()
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                    )


    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish swanlab run
    # 使用accelerator.is_main_process替代is_main变量
    if accelerator.is_main_process and config.wandb_enabled:
        swanlab.finish()



def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()