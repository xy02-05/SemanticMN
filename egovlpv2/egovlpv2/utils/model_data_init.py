# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import torch.nn as nn
import transformers
import argparse
import yaml
from egovlpv2.data_loader import data_loader as module_data
from egovlpv2.model import model as module_arch_standard
from egovlpv2.model import model_epic_charades as module_arch_epic
from egovlpv2.model import loss as module_loss
from egovlpv2.model.fg_alignment_model import AlignmentModel, create_alignment_model
from egovlpv2.model.video_sampler import create_video_sampler
from egovlpv2.utils.util import replace_nested_dict_item
from egovlpv2.trainer.trainer_charades import AllGather_multi
from egovlpv2.parse_config import ConfigParser
from egovlpv2.set_optim_schedule import set_schedule


def init_dataloaders(config, module_data):
    """
    Initialize data loaders from config
    
    Args:
        config: ConfigParser object with data loader configuration
        module_data: Data loader module
        
    Returns:
        list: List of data loaders
    """
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # Single dataloader
        data_loader = [config.initialize("data_loader", module_data)]
    elif isinstance(config["data_loader"], list):
        # Multiple dataloaders
        data_loader = [config.initialize('data_loader', module_data, index=idx) 
                      for idx in range(len(config['data_loader']))]
    else:
        raise ValueError("Check data_loader config, not correct format.")
    
    return data_loader


# ================== 新增训练组件初始化函数 ==================

def init_egovlpv2_model_for_training(config, device='cuda:0', use_lora=False, 
                                    lora_rank=32, lora_dropout=0.0, distributed=False, 
                                    device_id=None, dtype=torch.float32):
    """
    初始化EgoVLPv2模型用于训练，与OpenVLA训练流程对齐
    DDP包装由外部处理，保持与OpenVLA finetune.py一致的模式
    
    Args:
        config: ConfigParser object with model configuration
        device: Device to load model on
        use_lora: Whether to apply LoRA (interface ready, implementation pending)
        lora_rank: LoRA rank
        lora_dropout: LoRA dropout rate
        distributed: 是否准备分布式 (但DDP包装由外部处理)
        device_id: Device ID (用于日志显示)
        dtype: Model dtype (default: torch.float32, can be torch.bfloat16 for mixed precision)
        
    Returns:
        model: Model ready for training (未包装DDP)
    """
    
    # Initialize model based on model_type (same as multinode_train_charades.py)
    model_type = config.config['model_type']
    
    if model_type == 'epic_charades':
        model = config.initialize('arch', module_arch_epic)
    else:  # standard
        model = config.initialize('arch', module_arch_standard)
    
    # Convert model to specified dtype
    model = model.to(dtype)
    
    # Move model to device
    model = model.to(device)
    
    # Set to training mode (unlike inference version)
    model.train()
    
    # Apply LoRA if requested (interface ready, implementation pending)
    if use_lora:
        # TODO: Implement LoRA application
        # model = apply_lora_to_egovlpv2(model, lora_rank, lora_dropout)
        pass
    
    # 注意：DDP包装留给外部处理，与OpenVLA finetune.py保持一致
    # 外部可以这样包装：vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    
    return model


def init_egovlpv2_optimizer(model, config, data_loader):
    """
    初始化EgoVLPv2优化器，直接调用已有的set_schedule函数
    
    Args:
        model: Model to optimize
        config: ConfigParser object with optimizer configuration
        data_loader: Data loader for calculating max_steps
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    
    # Load config_yaml
    config_yaml_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'egovlpv2', 'configs', 'EgoNCE_MLM_ITM_Config.yml'
    )
    
    if os.path.exists(config_yaml_path):
        with open(config_yaml_path) as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError(f"Config file not found: {config_yaml_path}")
    
    
    max_steps = int(len(data_loader[0]) * config['trainer']['epochs'])
    if max_steps==0:
        max_steps = int(len(data_loader[0]) * 10)
    warmup_steps = config_yaml["warmup_steps"]
    if isinstance(config_yaml["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    optimizer, scheduler = set_schedule(model, config, config_yaml, max_steps, warmup_steps)
    
    return optimizer, scheduler


def init_egovlpv2_loss(config):
    """
    初始化EgoVLPv2损失函数
    
    Args:
        config: ConfigParser object with loss configuration
        
    Returns:
        loss_fn: Loss function
    """
    
    loss_fn = config.initialize(name="loss", module=module_loss)
    
    return loss_fn


def init_egovlpv2_dataloaders_for_training(config, distributed=False):
    """
    初始化EgoVLPv2数据加载器用于训练，与OpenVLA对齐
    
    Args:
        config: ConfigParser object with data loader configuration
        distributed: Whether to use distributed training
        
    Returns:
        tuple: (train_dataloaders, valid_dataloaders)
    """
    
    # Initialize train dataloaders
    train_dataloaders = init_dataloaders(config, module_data)
    
    # Initialize validation dataloaders
    # Save original config
    original_config = config._config.copy()
    
    # Modify config for validation
    if isinstance(config["data_loader"], list):
        new_cfg_li = []
        for dl_cfg in config['data_loader']:
            dl_cfg_copy = dl_cfg.copy()
            dl_cfg_copy['args'] = replace_nested_dict_item(dl_cfg_copy['args'], 'split', 'val')
            dl_cfg_copy['args'] = replace_nested_dict_item(dl_cfg_copy['args'], 'batch_size', 1)
            new_cfg_li.append(dl_cfg_copy)
        config._config['data_loader'] = new_cfg_li
        valid_dataloaders = [config.initialize('data_loader', module_data, index=idx) 
                           for idx in range(len(config['data_loader']))]
    else:
        # Single dataloader
        config['data_loader']['args'] = replace_nested_dict_item(
            config['data_loader']['args'], 'split', 'val'
        )
        config['data_loader']['args'] = replace_nested_dict_item(
            config['data_loader']['args'], 'batch_size', 1
        )
        valid_dataloaders = [config.initialize("data_loader", module_data)]

    # Restore original config
    config._config = original_config
    
    return train_dataloaders, valid_dataloaders



def init_egovlpv2_training_components(config_path, device='cuda:0', infinite_dataloader=False, dtype=torch.float32):
    """
    完整初始化EgoVLPv2训练组件，参考multinode_train_charades.py实现
    
    Args:
        config_path: 配置文件路径
        device: 设备 (默认cuda:0，分布式训练时由外部设置)
        infinite_dataloader: 是否使用无限数据加载器 (默认False)
        dtype: Model dtype (default: torch.float32, can be torch.bfloat16 for mixed precision)
        
    Returns:
        dict: 完整的训练组件
    """
    
    # Load config (same as multinode_train_charades.py)
    parser = argparse.ArgumentParser(description='EgoVLPv2 Training')
    # 只保留ConfigParser真正需要的参数
    parser.add_argument('-c', '--config', default=config_path, type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint') 
    parser.add_argument('-d', '--device', default=None, type=str, help='device to use')  # 设为None，避免干扰
    parser.add_argument('--save_dir', type=str, default='/tmp/egovlpv2_training', help="directory for model saving")
    
    # Set minimal sys.argv for ConfigParser (只包含必要参数)
    original_argv = sys.argv.copy()
    sys.argv = ['script_name', '--config', config_path, '--save_dir', '/tmp/egovlpv2_training']
    
    config = ConfigParser(parser, test=True)
    sys.argv = original_argv
    
    # Initialize tokenizer (same as multinode_train_charades.py)
    tokenizer_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'pretrain_weight', 
        config['arch']['args']['text_params']['model']
    )
    
    if os.path.exists(tokenizer_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path, TOKENIZERS_PARALLELISM=False
        )
    else:
        # Fallback to online download
        model_name = config['arch']['args']['text_params']['model']
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, TOKENIZERS_PARALLELISM=False
        )
    
    # Initialize data loaders (same as multinode_train_charades.py)
    train_dataloaders, valid_dataloaders = init_egovlpv2_dataloaders_for_training(
        config=config, distributed=False  # 分布式由外部DDP处理
    )
    
    training_config = config.config['trainer']
    use_lora = training_config["lora"]['use_lora']
    lora_rank = training_config["lora"]['lora_rank'] 
    lora_dropout = training_config["lora"]['lora_dropout'] 
    
    model = init_egovlpv2_model_for_training(
        config=config,
        device=device,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        distributed=False,  # DDP包装由外部处理，与OpenVLA对齐
        device_id=None,
        dtype=dtype
    )
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = init_egovlpv2_optimizer(
        model=model, config=config, data_loader=train_dataloaders
    )
    
    # Initialize loss function
    loss_fn = init_egovlpv2_loss(config)
    
    # Wrap dataloaders if infinite mode is requested
    if infinite_dataloader:
        wrapped_train_dataloaders = [create_egovlpv2_dataloader_wrapper(dl, infinite=True) 
                                   for dl in train_dataloaders]
    else:
        wrapped_train_dataloaders = train_dataloaders
    
    # Prepare training components
    components = {
        'model': model,
        'tokenizer': tokenizer,
        'train_dataloaders': wrapped_train_dataloaders,
        'valid_dataloaders': valid_dataloaders,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_fn': loss_fn,
        'config': config,
        'allgather': AllGather_multi.apply,
        'training_params': {
            'use_lora': use_lora,
            'lora_rank': lora_rank,
            'lora_dropout': lora_dropout,
            'infinite_dataloader': infinite_dataloader
        }
    }
    
    print("\n" + "=" * 60)
    print("✅ EgoVLPv2训练系统初始化完成!")
    print("=" * 60)
    print(f"🏗️  模型类型: {config.config['model_type']}")
    print(f"📊 数据集: {[dl.dataset_name for dl in train_dataloaders]} len:{len(train_dataloaders[0])}")
    print(f"⚙️  优化器: AdamW (分层学习率)")
    print(f"🎯 损失函数: {config['loss']['type']}")
    print(f"📱 设备: {device}")
    print(f"�� LoRA: {'启用' if use_lora else '禁用'}")
    print("=" * 60)
    
    return components


# === Alignment Model Initialization Functions ===

def init_alignment_model_components(config_path, device='cuda:0', dtype=torch.float32):
    """
    Initialize alignment model components for MIMIC-VLA training.
    
    Args:
        config_path: Path to alignment config file
        device: Device to use
        
    Returns:
        Dict containing alignment model and related components
    """
    
    # Load config
    parser = argparse.ArgumentParser(description='Alignment Model Training')
    parser.add_argument('-c', '--config', default=config_path, type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint') 
    parser.add_argument('-d', '--device', default=None, type=str, help='device to use')
    parser.add_argument('--save_dir', type=str, default='/tmp/alignment_training', help="directory for model saving")
    
    # Set minimal sys.argv for ConfigParser
    original_argv = sys.argv.copy()
    sys.argv = ['script_name', '--config', config_path, '--save_dir', '/tmp/alignment_training']
    
    config = ConfigParser(parser, test=True)
    sys.argv = original_argv
    
    # Create alignment model
    # Create alignment model from config
    if 'alignment' in config.config and config.config['alignment']['use_alignment']:
        alignment_config = config['alignment']['args']
        alignment_model = create_alignment_model(alignment_config, dtype=dtype)
        alignment_model = alignment_model.to(device)
        alignment_model.train()
        
        # Initialize optimizer for alignment model using alignment.optimizer config
        alignment_optimizer_config = config['alignment']['optimizer']
        alignment_optimizer = torch.optim.AdamW( 
            alignment_model.parameters(),
            lr=alignment_optimizer_config['args']['lr'],
            weight_decay=alignment_optimizer_config['args']['weight_decay']
        )

        # ============ 初始化VideoFeatureSampler（如果配置了as2vs + video_sampler） ============
        video_sampler = None
        video_sampler_config = config['alignment'].get('video_sampler', None)
        if video_sampler_config and video_sampler_config.get('enabled', False):
            video_sampler = create_video_sampler(
                config=video_sampler_config,
                device=device,
                dtype=dtype,
        )

        # before_proj: 对齐时是否使用EgoHOD投影前的文本特征（默认False，向后兼容）
        before_proj = alignment_config.get('before_proj', False)
        # vla_feature_type: Ablation用，控制使用哪种VLA token进行对齐
        # "action"(默认)=action token, "text"=文本指令token, "full"=所有有效token
        vla_feature_type = alignment_config.get('vla_feature_type', 'action')

        components = {
            'alignment_model': alignment_model,
            'alignment_optimizer': alignment_optimizer,
            'alignment_config': config['alignment'],
            'layer_indices': alignment_config['layer_indices'],
            'training_config': config['alignment']['training'],
            'video_sampler': video_sampler,  # None if not configured
            'before_proj': before_proj,  # 对齐时使用投影前特征
            'vla_feature_type': vla_feature_type,  # Ablation: VLA token类型
            'config': config
        }
        return components
    else:
        return None


def init_alignment_training_components(config_path, device='cuda:0', distributed=False):
    """
    Initialize complete alignment training setup.
    
    Args:
        config_path: Path to config file
        device: Device to use
        distributed: Whether using distributed training
        
    Returns:
        Dict with all training components
    """
    
    # Initialize alignment model components
    alignment_components = init_alignment_model_components(config_path, device)
    
    if alignment_components is None:
        return None
    
    # Initialize EgoVLPv2 components (for baseline comparison)
    egovlpv2_components = init_egovlpv2_training_components(config_path, device)
    alignment_components.update({
        'egovlpv2_model': egovlpv2_components['model'],
        'egovlpv2_optimizer': egovlpv2_components['optimizer'],
        'egovlpv2_tokenizer': egovlpv2_components['tokenizer'],
        'egovlpv2_loss_fn': egovlpv2_components['loss_fn'],
        'egovlpv2_allgather': egovlpv2_components['allgather'],
    })
    
    return alignment_components

# === Enhanced DataLoader Wrapper ===
class InfiniteEgoVLPDataset:
    """
    Enhanced wrapper for EgoVLPv2 dataloader with infinite parameter control.
    Provides seamless integration with OpenVLA training loop.
    """
    def __init__(self, base_dataloader, infinite=False):
        """
        Initialize the dataloader wrapper
        
        Args:
            base_dataloader: Original EgoVLPv2 dataloader
            infinite: If True, provides infinite iteration; if False, original behavior
        """
        self.base_dataloader = base_dataloader
        self.infinite = infinite
        self.dataset_length = len(base_dataloader.dataset) if hasattr(base_dataloader, 'dataset') else len(base_dataloader)
        
    def __iter__(self):
        """Iteration behavior controlled by infinite parameter"""
        if self.infinite:
            # Infinite iteration for training alignment with RLDSDataset
            while True:
                for batch in self.base_dataloader:
                    yield batch
        else:
            # Original behavior - single epoch iteration
            for batch in self.base_dataloader:
                yield batch
    
    def __len__(self):
        """Return the length of one epoch"""
        return self.dataset_length
    
    def __getitem__(self, idx):
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


def create_egovlpv2_dataloader_wrapper(dataloader, infinite=False):
    """
    Create EgoVLPv2 dataloader wrapper with infinite control
    
    Args:
        dataloader: Original EgoVLPv2 dataloader
        infinite: Whether to enable infinite iteration (default: False)
        
    Returns:
        InfiniteEgoVLPDataset: Wrapped dataloader
    """
    return InfiniteEgoVLPDataset(dataloader, infinite=infinite)

# TODO: Implement LoRA application for EgoVLPv2
def apply_lora_to_egovlpv2(model, lora_rank=32, lora_dropout=0.0):
    """
    将LoRA应用到EgoVLPv2模型 (待实现)
    
    Args:
        model: EgoVLPv2 model
        lora_rank: LoRA rank
        lora_dropout: LoRA dropout rate
        
    Returns:
        model: Model with LoRA applied
    """
    return model


# === EgoHOD Initialization Functions ===

def init_egohod_training_components(config_path, device='cuda:0', infinite_dataloader=False, dtype=torch.float32, training_mode=False):
    """
    初始化EgoHOD组件（统一接口，支持freeze和training两种模式）
    
    Args:
        config_path: 配置文件路径
        device: 设备
        infinite_dataloader: 是否使用无限数据加载器（仅training_mode=True时有效）
        dtype: 模型数据类型
        training_mode: True=训练模式（添加dataloader/loss），False=freeze模式（仅特征提取）
        
    Returns:
        dict: EgoHOD组件
    """
    import clip
    
    # 加载配置
    parser = argparse.ArgumentParser(description='EgoHOD Initialization')
    parser.add_argument('-c', '--config', default=config_path, type=str)
    parser.add_argument('-r', '--resume', default=None, type=str)
    parser.add_argument('-d', '--device', default=None, type=str)
    parser.add_argument('--save_dir', type=str, default='/tmp/egohod')
    
    original_argv = sys.argv.copy()
    sys.argv = ['script_name', '--config', config_path, '--save_dir', '/tmp/egohod']
    config = ConfigParser(parser, test=True)
    sys.argv = original_argv
    
    arch_args = config['arch']['args']
    
    # 检查LoRA配置（优先使用arch里的配置，保持与模型构造参数一致）
    # 说明：arch.args.lora_config / arch.args.video_lora_config 是EgoHOD官方式配置入口
    lora_config = arch_args.get('lora_config', None)
    if not (lora_config and lora_config.get('enabled', False)):
        # 兼容旧配置：允许从alignment.args读取（历史遗留）
        lora_config = None
        if 'alignment' in config.config and 'lora_config' in config.config['alignment']['args']:
            lora_config = config.config['alignment']['args']['lora_config']
            if not (lora_config and lora_config.get('enabled', False)):
                lora_config = None
    # Video LoRA配置仅在arch.args中定义，保持与EgoHOD模型一致
    video_lora_config = arch_args.get('video_lora_config', None)
    if not (video_lora_config and video_lora_config.get('enabled', False)):
        video_lora_config = None
    
    # XClip细粒度loss配置（可选，从arch.args.fine_grain_config读取）
    fine_grain_config = arch_args.get('fine_grain_config', None)
    if fine_grain_config and not fine_grain_config.get('enabled', False):
        fine_grain_config = None
        print("fine_grain_config is disabled")
    
    # 创建EgoHOD模型
    from egovlpv2.model.model_egohod import EgoHODModel
    model = EgoHODModel(
        video_params=arch_args['video_params'],
        text_params=arch_args['text_params'],
        projection_dim=arch_args['projection_dim'],
        load_checkpoint=arch_args['load_checkpoint'],
        num_frames=arch_args['num_frames'],
        project_embed_dim=arch_args['project_embed_dim'],
        use_fast_conv1=arch_args['use_fast_conv1'],
        use_flash_attn=arch_args['use_flash_attn'],
        context_length=arch_args['context_length'],
        vocab_size=arch_args['vocab_size'],
        freeze_temperature=arch_args['freeze_temperature'],
        egohod_checkpoint_path=arch_args['egohod_checkpoint_path'],
        lora_config=lora_config,
        video_lora_config=video_lora_config,
        fine_grain_config=fine_grain_config,
    )
    model = model.to(dtype).to(device)
    
    # 保持 logit_scale 为 float32（bfloat16 精度不足，优化器无法更新标量参数）
    if model.train_logit_scale and hasattr(model.clip_model, 'logit_scale'):
        model.clip_model.logit_scale = nn.Parameter(
            model.clip_model.logit_scale.data.float()
        )
    
    # 基础组件（freeze和training都需要）
    components = {
        'model': model,
        'config': config,
        'device': device,
        'dtype': dtype,
        'allgather': AllGather_multi.apply,
        'tokenizer': clip.tokenize,  # CLIP tokenizer
    }
    
    if training_mode:
        # 训练模式：设置train()，初始化dataloader和loss
        model.train()
        train_dataloaders = init_dataloaders(config, module_data)
        
        # 使用EgoHOD官方的ClipLoss（而不是EgoVLPv2的loss）
        # ClipLoss内部处理分布式gather，接口: (image_features, text_features, logit_scale)
        from egovlpv2.model.egohod.loss import ClipLoss
        loss_fn = ClipLoss(
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,  # 初始化时用0，运行时会更新
            world_size=1,  # 初始化时用1，运行时会更新
        )
        
        if infinite_dataloader:
            train_dataloaders = [create_egovlpv2_dataloader_wrapper(dl, infinite=True) 
                               for dl in train_dataloaders]
        
        components.update({
            'train_dataloaders': train_dataloaders,
            'loss_fn': loss_fn,
        })
        print(f"✅ EgoHOD初始化完成 [训练模式] | 数据集长度: {len(train_dataloaders[0])}")
    else:
        # Freeze模式：设置eval()，冻结参数
        model.eval()
        for name, param in model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        print(f"✅ EgoHOD初始化完成 [Freeze模式]")
    
    return components


def init_egovideo_training_components(config_path, device='cuda:0', dtype=torch.float32):
    """
    初始化EgoVideo组件用于freeze VLM场景（不训练EgoVideo，仅用于特征提取）
    
    说明：
    - EgoVideo模型用于freeze VLM场景，不需要训练，因此不初始化optimizer、scheduler、loss_fn
    - 仅初始化模型并设置为eval模式
    - 模型参数全部冻结，用于提取文本和视频特征
    - 返回allgather用于alignment训练（即使不训练VLM，alignment训练也需要）
    
    Args:
        config_path: 配置文件路径
        device: 设备 (默认cuda:0)
        dtype: Model dtype (default: torch.float32, can be torch.bfloat16 for mixed precision)
        
    Returns:
        dict: EgoVideo组件
            - model: EgoVideo模型（已设置为eval模式，参数已冻结）
            - tokenizer: BERT tokenizer
            - config: 配置对象
            - allgather: 分布式AllGather函数（alignment训练需要）
            - device: 设备
            - dtype: 数据类型
    """
    
    # 加载配置
    parser = argparse.ArgumentParser(description='EgoVideo Initialization')
    parser.add_argument('-c', '--config', default=config_path, type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint') 
    parser.add_argument('-d', '--device', default=None, type=str, help='device to use')
    parser.add_argument('--save_dir', type=str, default='/tmp/egovideo_inference', help="directory for saving")
    
    # 设置最小sys.argv供ConfigParser使用
    original_argv = sys.argv.copy()
    sys.argv = ['script_name', '--config', config_path, '--save_dir', '/tmp/egovideo_inference']
    
    config = ConfigParser(parser, test=True)
    sys.argv = original_argv
    
    # 从配置中提取EgoVideo模型参数
    arch_args = config['arch']['args']
    
    # 创建EgoVideo模型
    from egovlpv2.model.model_egovideo import EgoVideoWrapper
    
    model = EgoVideoWrapper(
        video_params=arch_args['video_params'],
        text_params=arch_args['text_params'],
        projection_dim=arch_args['projection_dim'],
        load_checkpoint=arch_args['load_checkpoint'],  # EgoVideo预训练权重路径
    )
    
    # 转换数据类型并移动到设备
    model = model.to(dtype).to(device)
    
    # 设置为eval模式
    model.eval()
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 获取tokenizer
    tokenizer = model.tokenizer
    
    # 准备组件字典
    # 注意：即使EgoVideo不训练，allgather也是alignment训练的必需组件
    components = {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'device': device,
        'dtype': dtype,
        'allgather': AllGather_multi.apply,  # alignment训练需要
    }
    
    print("\n" + "=" * 60)
    print("✅ EgoVideo推理系统初始化完成!")
    print("=" * 60)
    print(f"🏗️  模型类型: EgoVideo (Freeze VLM)")
    print(f"📱 设备: {device}")
    print(f"🔒 参数状态: 全部冻结")
    print(f"🎯 特征维度: {arch_args.get('projection_dim', 512)}")
    if arch_args.get('load_checkpoint'):
        print(f"📂 EgoVideo权重: {arch_args.get('load_checkpoint')}")
    print("=" * 60)
    
    return components


# === CLIP Initialization Functions ===

def init_clip_components(config_path, device='cuda:0', dtype=torch.float32):
    """
    初始化CLIP组件用于freeze VLM场景（不训练CLIP，仅用于特征提取）
    
    说明：
    - CLIP模型用于freeze VLM场景，不需要训练，因此不初始化optimizer、scheduler、loss_fn
    - 仅初始化模型并设置为eval模式
    - 模型参数全部冻结，用于提取文本特征
    - 返回allgather用于alignment训练（即使不训练VLM，alignment训练也需要）
    
    Args:
        config_path: 配置文件路径
        device: 设备 (默认cuda:0)
        dtype: Model dtype (default: torch.float32, can be torch.bfloat16 for mixed precision)
        
    Returns:
        dict: CLIP组件
            - model: CLIPModel（已设置为eval模式，参数已冻结）
            - config: 配置对象
            - allgather: 分布式AllGather函数（alignment训练需要）
            - device: 设备
            - dtype: 数据类型
    """
    
    # 加载配置
    parser = argparse.ArgumentParser(description='CLIP Initialization')
    parser.add_argument('-c', '--config', default=config_path, type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint') 
    parser.add_argument('-d', '--device', default=None, type=str, help='device to use')
    parser.add_argument('--save_dir', type=str, default='/tmp/clip_inference', help="directory for saving")
    
    # 设置最小sys.argv供ConfigParser使用
    original_argv = sys.argv.copy()
    sys.argv = ['script_name', '--config', config_path, '--save_dir', '/tmp/clip_inference']
    
    config = ConfigParser(parser, test=True)
    sys.argv = original_argv
    
    # 从配置中提取CLIP模型参数
    arch_args = config['arch']['args']
    
    # 创建CLIPModel
    from egovlpv2.model.model_clip import CLIPModel
    
    model = CLIPModel(
        video_params=arch_args['video_params'],
        text_params=arch_args['text_params'],
        projection_dim=arch_args['projection_dim'],
        model_name=arch_args.get('model_name', 'ViT-B/16'),  # 默认ViT-B/16
        device=device,
    )
    
    # 转换数据类型并移动到设备
    model = model.to(dtype).to(device)
    
    # 设置为eval模式
    model.eval()
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 准备组件字典
    # 注意：即使CLIP不训练，allgather也是alignment训练的必需组件
    components = {
        'model': model,
        'config': config,
        'device': device,
        'dtype': dtype,
        'allgather': AllGather_multi.apply,  # alignment训练需要
    }
    
    print("\n" + "=" * 60)
    print("✅ CLIP推理系统初始化完成!")
    print("=" * 60)
    print(f"🏗️  模型类型: CLIP (Freeze VLM)")
    print(f"📱 设备: {device}")
    print(f"🔒 参数状态: 全部冻结")
    print(f"🎯 特征维度: {model.clip_dim}")
    print(f"📂 模型名称: {arch_args.get('model_name', 'ViT-B/16')}")
    print("=" * 60)
    
    return components


# === Qwen3-Embedding Initialization Functions ===

def init_qwen3_components(config_path, device='cuda:0', dtype=torch.bfloat16):
    """
    初始化Qwen3-Embedding组件用于freeze VLM场景（不训练Qwen3，仅用于特征提取）
    
    说明：
    - Qwen3-Embedding模型用于freeze VLM场景，不需要训练，因此不初始化optimizer、scheduler、loss_fn
    - 仅初始化模型并设置为eval模式
    - 模型参数全部冻结，用于提取文本特征
    - 返回allgather用于alignment训练（即使不训练VLM，alignment训练也需要）
    
    Args:
        config_path: 配置文件路径
        device: 设备 (默认cuda:0)
        dtype: Model dtype (default: torch.bfloat16)
        
    Returns:
        dict: Qwen3组件
            - model: Qwen3Model（已设置为eval模式，参数已冻结）
            - config: 配置对象
            - allgather: 分布式AllGather函数（alignment训练需要）
            - device: 设备
            - dtype: 数据类型
    """
    # 加载配置
    parser = argparse.ArgumentParser(description='Qwen3-Embedding Initialization')
    parser.add_argument('-c', '--config', default=config_path, type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint') 
    parser.add_argument('-d', '--device', default=None, type=str, help='device to use')
    parser.add_argument('--save_dir', type=str, default='/tmp/qwen3_inference', help="directory for saving")
    
    # 设置最小sys.argv供ConfigParser使用
    original_argv = sys.argv.copy()
    sys.argv = ['script_name', '--config', config_path, '--save_dir', '/tmp/qwen3_inference']
    
    config = ConfigParser(parser, test=True)
    sys.argv = original_argv
    
    # 从配置中提取Qwen3模型参数
    arch_args = config['arch']['args']
    
    # 创建Qwen3Model
    from egovlpv2.model.model_qwen3 import create_qwen3_model
    
    model = create_qwen3_model(
        model_path=arch_args['model_path'],
        device=device,
        dtype=dtype,
        use_flash_attention=arch_args.get('use_flash_attention', False),
        normalize_embeddings=arch_args.get('normalize_embeddings', True),
    )
    
    # 准备组件字典
    components = {
        'model': model,
        'config': config,
        'allgather': AllGather_multi.apply,
        'device': device,
        'dtype': dtype,
    }
    
    print("\n" + "=" * 60)
    print("✅ Qwen3-Embedding系统初始化完成!")
    print("=" * 60)
    print(f"🏗️  模型类型: Qwen3-Embedding (Freeze VLM)")
    print(f"📱 设备: {device}")
    print(f"🔒 参数状态: 全部冻结")
    print(f"🎯 输出维度: {model.output_dim}")
    print(f"📂 模型路径: {arch_args['model_path']}")
    print(f"🔧 Flash Attention: {'启用' if arch_args.get('use_flash_attention', False) else '禁用'}")
    print(f"🎯 L2归一化: {'启用' if arch_args.get('normalize_embeddings', True) else '禁用'}")
    print("=" * 60)
    
    return components


# === 统一VLM组件初始化函数 ===

def init_vlm_components(vlm_mode, config_path, device='cuda:0', dtype=torch.float32, infinite_dataloader=False, training_mode=False):
    """
    根据vlm_mode统一初始化VLM组件
    
    Args:
        vlm_mode: VLM模型类型 ("egovlpv2", "egohod", "egovideo", "qwen3", "embedding", "clip", "roboticclip")
        config_path: 配置文件路径
        device: 设备 (默认cuda:0)
        dtype: Model dtype (default: torch.float32)
        training_mode: 是否为训练模式（目前只对egohod生效）
        infinite_dataloader: 是否使用无限数据加载器，仅用于egovlpv2/egohod_cotraining模式 (默认False)
        
    Returns:
        dict: VLM组件字典
    """
    
    # 根据vlm_mode选择对应的初始化函数
    if vlm_mode == "egohod":
        # EgoHOD freeze模式：仅用于特征提取
        # EgoHOD模式：training_mode参数控制freeze/training
        components = init_egohod_training_components(
            config_path=config_path,
            device=device,
            infinite_dataloader=infinite_dataloader,
            dtype=dtype,
            training_mode=training_mode  # 由调用方控制
        )
    elif vlm_mode == "egovideo":
        components = init_egovideo_training_components(
            config_path=config_path,
            device=device,
            dtype=dtype
        )
    elif vlm_mode == "qwen3":
        components = init_qwen3_components(
            config_path=config_path,
            device=device,
            dtype=dtype
        )
    elif vlm_mode == "embedding":
        components = init_embedding_components(
            config_path=config_path,
            device=device,
            dtype=dtype
        )
    elif vlm_mode == "clip":
        components = init_clip_components(
            config_path=config_path,
            device=device,
            dtype=dtype
        )
    elif vlm_mode == "roboticclip":
        # RoboticCLIP模式：基于AlphaCLIP的图像-文本模型
        components = init_roboticclip_components(
            config_path=config_path,
            device=device,
            dtype=dtype
        )
    else:
        # 默认使用egovlpv2
        components = init_egovlpv2_training_components(
            config_path=config_path,
            device=device,
            infinite_dataloader=infinite_dataloader,
            dtype=dtype
        )
    
    return components


# === Embedding (预计算特征) Initialization Functions ===

def init_embedding_components(config_path, device='cuda:0', dtype=torch.float32):
    """
    初始化Embedding组件用于freeze VLM场景（使用预计算的text-embedding-3-large特征）
    
    说明：
    - Embedding模型用于freeze VLM场景，加载预计算的文本特征，无需训练
    - 直接从.npz文件加载特征，无需forward pass（零计算开销）
    - 模型参数全部冻结（实际上没有可训练参数），仅用于特征查找
    - 返回allgather用于alignment训练
    
    Args:
        config_path: 配置文件路径
        device: 设备 (默认cuda:0)
        dtype: Model dtype (default: torch.float32, can be torch.bfloat16 for mixed precision)
        
    Returns:
        dict: Embedding组件
            - model: EmbeddingModel（已设置为eval模式）
            - config: 配置对象
            - allgather: 分布式AllGather函数（alignment训练需要）
            - device: 设备
            - dtype: 数据类型
    """
    # 加载配置
    parser = argparse.ArgumentParser(description='Embedding Initialization')
    parser.add_argument('-c', '--config', default=config_path, type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint') 
    parser.add_argument('-d', '--device', default=None, type=str, help='device to use')
    parser.add_argument('--save_dir', type=str, default='/tmp/embedding_inference', help="directory for saving")
    
    # 设置最小sys.argv供ConfigParser使用
    original_argv = sys.argv.copy()
    sys.argv = ['script_name', '--config', config_path, '--save_dir', '/tmp/embedding_inference']
    
    config = ConfigParser(parser, test=True)
    sys.argv = original_argv
    
    # 从配置中提取Embedding模型参数
    arch_args = config['arch']['args']
    
    # 创建EmbeddingModel
    from egovlpv2.model.model_embedding import create_embedding_model
    
    model = create_embedding_model(
        embeddings_path=arch_args['embeddings_path'],
        index_path=arch_args['index_path'],
        device=device,
        dtype=dtype,
        normalize_embeddings=arch_args.get('normalize_embeddings', True),
        preload_to_gpu=arch_args.get('preload_to_gpu', True)  # 新增：从配置读取（默认True）
    )
    
    # 原子级对齐：加载 atomic embeddings（可选，配置中有 atomic_embeddings_path 时启用）
    atomic_embeddings_path = arch_args.get('atomic_embeddings_path', None)
    if atomic_embeddings_path:
        model.load_atomic_embeddings(atomic_embeddings_path)

    # chunk-level video 对齐：加载预计算的 chunk video features（可选）
    chunk_video_features_path = arch_args.get('chunk_video_features_path', None)
    if chunk_video_features_path:
        model.load_chunk_video_features(chunk_video_features_path)
    
    # 准备组件字典
    components = {
        'model': model,
        'config': config,
        'device': device,
        'dtype': dtype,
        'allgather': AllGather_multi.apply,  # alignment训练需要
    }
    
    print("\n" + "=" * 60)
    print("✅ Embedding推理系统初始化完成!")
    print("=" * 60)
    print(f"🏗️  模型类型: Embedding (Freeze VLM, 预计算特征)")
    print(f"📱 设备: {device}")
    print(f"🔒 参数状态: 无可训练参数（预计算特征）")
    print(f"🎯 特征维度: {model.output_dim}")
    print(f"📂 特征文件: {arch_args['embeddings_path']}")
    print(f"📂 索引文件: {arch_args['index_path']}")
    print(f"🎯 L2归一化: {'是' if arch_args.get('normalize_embeddings', True) else '否'}")
    print("=" * 60)
    
    return components


# === RoboticCLIP Initialization Functions ===

def init_roboticclip_components(config_path, device='cuda:0', dtype=torch.float32):
    """
    初始化RoboticCLIP组件用于freeze VLM场景（不训练RoboticCLIP，仅用于特征提取）
    
    说明：
    - RoboticCLIP基于AlphaCLIP，支持alpha channel（物体mask）
    - 是图像-文本模型，用于文本特征提取和图像-文本对齐
    - 仅初始化模型并设置为eval模式，参数全部冻结
    
    Args:
        config_path: 配置文件路径
        device: 设备 (默认cuda:0)
        dtype: Model dtype (default: torch.float32)
        
    Returns:
        dict: RoboticCLIP组件
            - model: RoboticCLIPModel（已设置为eval模式，参数已冻结）
            - config: 配置对象
            - allgather: 分布式AllGather函数
            - device: 设备
            - dtype: 数据类型
    """
    # 加载配置
    parser = argparse.ArgumentParser(description='RoboticCLIP Initialization')
    parser.add_argument('-c', '--config', default=config_path, type=str, help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='device to use')
    parser.add_argument('--save_dir', type=str, default='/tmp/roboticclip_inference', help="directory for saving")
    
    original_argv = sys.argv.copy()
    sys.argv = ['script_name', '--config', config_path, '--save_dir', '/tmp/roboticclip_inference']
    
    config = ConfigParser(parser, test=True)
    sys.argv = original_argv
    
    # 从配置中提取RoboticCLIP模型参数
    arch_args = config['arch']['args']
    
    # 创建RoboticCLIPModel
    from egovlpv2.model.model_roboticclip import RoboticCLIPModel
    
    model = RoboticCLIPModel(
        video_params=arch_args.get('video_params', {}),
        text_params=arch_args.get('text_params', {}),
        projection_dim=arch_args.get('projection_dim', 768),
        model_name=arch_args.get('model_name', 'ViT-L/14@336px'),
        alpha_vision_ckpt_pth=arch_args.get('alpha_vision_ckpt_pth', None),
        roboticclip_checkpoint_path=arch_args.get('roboticclip_checkpoint_path', None),
        device=device,
    )
    
    # 转换数据类型并移动到设备
    model = model.to(dtype).to(device)
    
    # 设置为eval模式
    model.eval()
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 准备组件字典
    components = {
        'model': model,
        'config': config,
        'device': device,
        'dtype': dtype,
        'allgather': AllGather_multi.apply,
    }
    
    print("\n" + "=" * 60)
    print("✅ RoboticCLIP推理系统初始化完成!")
    print("=" * 60)
    print(f"🏗️  模型类型: RoboticCLIP (Freeze VLM)")
    print(f"📱 设备: {device}")
    print(f"🔒 参数状态: 全部冻结")
    print(f"🎯 特征维度: {model.clip_dim}")
    print(f"📂 模型名称: {arch_args.get('model_name', 'ViT-L/14@336px')}")
    if arch_args.get('roboticclip_checkpoint_path'):
        print(f"📂 RoboticCLIP权重: {arch_args.get('roboticclip_checkpoint_path')}")
    print("=" * 60)
    
    return components
