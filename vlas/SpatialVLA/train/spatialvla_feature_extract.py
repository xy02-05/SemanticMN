"""
SpatialVLA 特征提取脚本（仅 VLA action 特征）
功能：从冻结的 VLA 中提取 action hidden states 并保存到 HDF5 文件
      文本特征不在此提取，只记录指令文本，后续通过 task_index 查表获取

使用方法：
    torchrun --standalone --nproc-per-node 1 train/spatialvla_feature_extract.py \
        --model_name_or_path /path/to/model \
        --output_dir /path/to/features \
        --do_train True
"""
import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.distributed as dist
from tqdm import tqdm

from transformers import HfArgumentParser, set_seed, TrainingArguments
from transformers.utils.logging import enable_default_handler, enable_explicit_format, set_verbosity

from data.dataset import build_datasets
from model import (
    SpatialVLAConfig, SpatialVLAForConditionalGeneration, SpatialVLAProcessor,
    SpatialActionTokenizer,
)
from egovlpv2.utils.model_forward import get_vla_features, get_layer_vla_features
from train.utils.accelerator_utils import create_accelerator
from train.dist_utils import init_dist
from train.monkey_patch import concat_pad_data_collator
from train.utils.data_utils import create_bridge_dataloader_and_sampler
from train.utils.h5_utils import H5FeatureWriter

import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    flash_attn: bool = field(default=True)
    alignment_config_path: str = field(
        default="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/alignment/configs/default_config.json",
        metadata={"help": "对齐配置文件路径，用于获取 layer_indices"}
    )
    # 特征提取专用参数
    extract_samples: int = field(default=-1, metadata={"help": "提取样本数，-1 表示全部"})
    # H5 写入参数
    h5_chunk_size: int = field(default=10000, metadata={"help": "HDF5 分块大小"})
    h5_compression: str = field(default="none", metadata={"help": "H5压缩方式: none(最快)/lzf(快)/gzip(小)"})
    h5_buffer_size: int = field(default=2000, metadata={"help": "内存缓冲大小，越大写入越快（推荐10000）"})


@dataclass
class DataTrainingArguments:
    data_root_dir: Optional[str] = field(default="datasets/open-x-embodiment")
    data_mix: Optional[str] = field(default="bridge")
    max_seq_length: Optional[int] = field(default=2048)
    shuffle_buffer_size: Optional[int] = field(default=1000_000)
    tsfm_thread_muti: Optional[int] = field(default=1)
    read_thread_muti: Optional[int] = field(default=1)
    obs_backward_steps: Optional[int] = field(default=0)
    obs_backward_delta: Optional[int] = field(default=1)
    action_forward_steps: Optional[int] = field(default=0)
    fix_raw_length: Optional[int] = field(default=None)
    use_raw_dataloader: Optional[bool] = field(default=True)
    task_filename: Optional[str] = field(default=None)
    use_eval_split: Optional[bool] = field(default=False)
    load_future_images: Optional[bool] = field(default=False)


def extract_loop(
    model, dataloader, accelerator, action_tokenizer, layer_indices,
    h5_path, vla_hidden_size, max_samples=-1,
    h5_chunk_size=10000, h5_compression="none", h5_buffer_size=10000
):
    """
    特征提取主循环（仅 VLA action 特征，不含文本特征）
    
    Args:
        model: SpatialVLA 模型（冻结）
        dataloader: 数据加载器
        accelerator: Accelerator 实例
        action_tokenizer: action tokenizer
        layer_indices: 要提取的层索引
        h5_path: 输出 HDF5 文件路径
        vla_hidden_size: VLA hidden dimension
        max_samples: 最大提取样本数（-1 表示全部）
        h5_chunk_size: HDF5 分块大小
        h5_compression: 压缩方式 (none/lzf/gzip)
        h5_buffer_size: 内存缓冲大小
        
    Returns:
        H5FeatureWriter 实例（需要调用者关闭）
    """
    model.eval()
    target_device = accelerator.device
    
    total_extracted = 0
    h5_writer = None  # 延迟创建，等第一个 batch 确定 action_chunk_size
    progress_bar = tqdm(dataloader, desc="提取特征", disable=not accelerator.is_main_process)
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in progress_bar:
            # 检查是否达到最大样本数
            if max_samples > 0 and total_extracted >= max_samples:
                break
            
            # 移动 batch 到设备
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(target_device, non_blocking=True)
            
            # ========== VLA 前向：获取 action hidden states ==========
            vla_outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                intrinsic=batch['intrinsic'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 提取多层 action 特征 [B, L, A, D]
            if hasattr(vla_outputs, 'action_hidden_states') and vla_outputs.action_hidden_states is not None:
                # SpatialVLA 已计算好 action_hidden_states
                action_features = get_layer_vla_features(
                    vla_outputs.action_hidden_states, layer_indices
                ).permute(1, 0, 2, 3)  # [L, B, A, D] -> [B, L, A, D]
            else:
                # 从 hidden_states 提取
                action_features = get_vla_features(
                    hidden_states=vla_outputs.hidden_states,
                    vision_patches_num=256,
                    batch=batch,
                    action_token_begin_idx=action_tokenizer.action_token_begin_idx,
                    layer_indices=layer_indices,
                ).permute(1, 0, 2, 3)
            
            # 第一个 batch 时创建 H5Writer（此时才知道实际的 action_chunk_size）
            if h5_writer is None:
                actual_action_chunk_size = action_features.shape[2]  # A 维度
                actual_num_layers = action_features.shape[1]         # L 维度
                accelerator.print(f"📐 实际维度: layers={actual_num_layers}, action_tokens={actual_action_chunk_size}, hidden={vla_hidden_size}")
                accelerator.print(f"📦 H5配置: chunk_size={h5_chunk_size}, compression={h5_compression}, buffer_size={h5_buffer_size}")
                h5_writer = H5FeatureWriter(
                    path=h5_path,
                    layer_indices=layer_indices[:actual_num_layers],  # 使用实际有效的层数
                    action_dim=vla_hidden_size,
                    action_chunk_size=actual_action_chunk_size,
                    chunk_size=h5_chunk_size,
                    compression=h5_compression,
                    buffer_size=h5_buffer_size,
                )
            
            # 获取文本指令（只记录，不编码）
            if 'lang' in batch and batch['lang']:
                batch_texts = batch['lang']
            else:
                # 从 input_ids 解码获取指令（备用方案）
                batch_texts = [""] * action_features.shape[0]
            
            # ========== 写入 HDF5 ==========
            h5_writer.append(
                action_features=action_features,
                task_ids=batch['task_ids'],
                task_index=batch['task_index'],
                traj_index=batch['traj_index'],
                timestep=batch['timestep'],
                progress=batch['task_progress'],
                instructions=batch_texts,
            )
            
            total_extracted += action_features.shape[0]
            progress_bar.set_postfix(extracted=total_extracted)
    
    accelerator.print(f"✅ Rank {accelerator.process_index}: 提取 {total_extracted} 样本完成")
    return h5_writer


def main():
    # ========== 初始化分布式 ==========
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # ========== 日志配置 ==========
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d %H:%M:%S")
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    set_seed(training_args.seed)

    # ========== Accelerator ==========
    accelerator = create_accelerator(training_args, model_args)
    
    # ========== 加载处理器和模型 ==========
    logger.info("📖 加载处理器...")
    _processor = SpatialVLAProcessor.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    tokenizer = _processor.tokenizer
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    
    config = SpatialVLAConfig.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, local_files_only=True)
    
    logger.info("🏗️ 加载 SpatialVLA 模型...")
    model = SpatialVLAForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, config=config, torch_dtype=torch_dtype, local_files_only=True
    )
    if model_args.flash_attn:
        model.language_model.config._attn_implementation = "flash_attention_2"
        model.vision_tower.config._attn_implementation = "flash_attention_2"
    
    # ========== 构建数据集 ==========
    train_dataset, _ = build_datasets(data_args, training_args.output_dir, vla_processor=None)
    
    action_tokenizer = SpatialActionTokenizer(
        tokenizer, num_bins=_processor.action_config["num_bins"],
        bin_policy=_processor.action_tokenizer.bin_policy,
        use_spherical=_processor.action_config["use_spherical"],
        min_sigma=_processor.action_config.get("min_sigma", 0.0),
    )
    
    model.action_token_begin_idx = action_tokenizer.action_token_begin_idx
    
    # 构建 processor
    statistic = train_dataset.ds_stats_pc
    _processor.statistics.update(statistic)
    processor = SpatialVLAProcessor(
        image_processor=_processor.image_processor, tokenizer=tokenizer,
        statistics=_processor.statistics, bin_policy=action_tokenizer.bin_policy,
        intrinsic_config=_processor.intrinsic_config, action_config=_processor.action_config,
        num_obs_steps=data_args.obs_backward_steps + 1,
        obs_delta=data_args.obs_backward_delta,
        action_chunk_size=data_args.action_forward_steps + 1,
    )
    train_dataset.vla_processor = processor
    
    # ========== 创建 DataLoader ==========
    train_dataloader, _, _ = create_bridge_dataloader_and_sampler(
        data_args=data_args, training_args=training_args, output_dir=training_args.output_dir,
        vla_processor=processor, tokenizer=tokenizer, accelerator=accelerator
    )
    
    # ========== 冻结模型并移动到设备 ==========
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(accelerator.device)
    logger.info("🔒 模型已冻结并移动到设备")
    
    # ========== 特征提取 ==========
    if training_args.do_train:  # 复用 do_train 参数作为 do_extract
        # 加载对齐配置获取 layer_indices
        with open(model_args.alignment_config_path, 'r') as f:
            align_config = json.load(f)
        layer_indices = align_config['layer_indices']
        
        # 验证 layer_indices 不超出模型层数
        num_hidden_layers = model.config.text_config.num_hidden_layers
        max_valid_index = num_hidden_layers
        layer_indices = [i for i in layer_indices if i <= max_valid_index]
        logger.info(f"📊 提取层: {layer_indices}")
        
        # 获取特征维度
        vla_hidden_size = model.config.text_config.hidden_size
        
        # 创建 H5 Writer（延迟到第一个 batch 后创建，因为需要知道实际 action_chunk_size）
        h5_path = os.path.join(training_args.output_dir, f"features_rank{accelerator.process_index}.h5")
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        logger.info(f"🎯 开始提取特征到: {h5_path}")
        h5_writer = extract_loop(
            model=model,
            dataloader=train_dataloader,
            accelerator=accelerator,
            action_tokenizer=action_tokenizer,
            layer_indices=layer_indices,
            h5_path=h5_path,
            vla_hidden_size=vla_hidden_size,
            max_samples=model_args.extract_samples,
            h5_chunk_size=model_args.h5_chunk_size,
            h5_compression=model_args.h5_compression,
            h5_buffer_size=model_args.h5_buffer_size,
        )
        
        if h5_writer is not None:
            h5_writer.close()
        accelerator.wait_for_everyone()
        logger.info("🎉 特征提取完成！")


if __name__ == "__main__":
    main()
