# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import transformers
import argparse
import yaml
import clip  # EgoHOD使用OpenAI CLIP的tokenizer
from egovlpv2.data_loader import data_loader as module_data
from egovlpv2.model import model as module_arch_standard
from egovlpv2.model import model_epic_charades as module_arch_epic
from egovlpv2.model import loss as module_loss
from egovlpv2.model.fg_alignment_model import AlignmentModel, create_alignment_model
from egovlpv2.utils.util import replace_nested_dict_item
from egovlpv2.trainer.trainer_charades import AllGather_multi
from egovlpv2.parse_config import ConfigParser
from egovlpv2.set_optim_schedule import set_schedule


# ================== 新增训练组件初始化函数 ==================


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
    print(f"🔧 LoRA: {'启用' if use_lora else '禁用'}")
    print("=" * 60)
    
    return components


def extract_content_vla(input_str):    
    # 从固定位置45开始截取"<s> In: What action should the robot take to "之后的内容
    after_in_part = input_str[45:]        
    # 查找问号的位置
    question_mark_index = after_in_part.find('?')
    # 提取问号前的内容并去除首尾空格
    extracted_content = after_in_part[:question_mark_index].strip()
    return extracted_content

def text_to_token(tokenizer, text, device_id, return_tensors='pt', padding='max_length', max_length=77, truncation=True):
    """
    统一的tokenization函数
    
    Args:
        tokenizer: EgoVLPv2 tokenizer (roberta-base)
        text: 输入文本
        device_id: 设备ID
        return_tensors: 返回tensor格式
        padding: padding策略
        max_length: 最大长度
            - 15: EgoVLPv2训练时的标准短文本长度
            - 30: EgoVLPv2分类任务的中等文本长度  
            - 77: 对齐任务的标准长度（CLIP兼容）
        truncation: 是否截断
        
    Returns:
        dict: tokenized结果，已移动到指定设备
    """
    token = tokenizer(
        text, 
        return_tensors=return_tensors, 
        padding=padding, 
        max_length=max_length, 
        truncation=truncation
    )
    token = {k: v.to(device_id, non_blocking=True) for k, v in token.items()}
    return token

def token_to_text(tokenizer, input_ids, extract_content=extract_content_vla):
    """
    将input_ids转换为文本并可选地提取内容
    
    Args:
        tokenizer: 分词器
        input_ids: token ID序列
        extract_content: 内容提取函数，如果为None则直接返回原始文本
        
    Returns:
        list: 文本列表（提取内容后或原始文本）
    """
    # 转文字
    texts = [tokenizer.decode(ids) for ids in input_ids]
    if extract_content is not None:
        texts_extracted = [extract_content(t) for t in texts]
        return texts_extracted
    else:
        return texts

def extract_egovlpv2_text_embeddings(
    egovlpv2_model, 
    egovlpv2_tokenizer, 
    batch_texts, 
    device='cuda:0'
):
    """
    Extract EgoVLPv2 text embeddings from OpenVLA batch texts.
    
    Args:
        egovlpv2_model: EgoVLPv2 model instance
        egovlpv2_tokenizer: EgoVLPv2 tokenizer
        batch_texts: List of text strings from OpenVLA (extracted using get_texts)
        device: Device to use for computation
        
    Returns:
        torch.Tensor: Text embeddings with shape [batch_size, embedding_dim]
    """
    # Tokenize texts using EgoVLPv2 tokenizer
    text_tokens = text_to_token(egovlpv2_tokenizer, batch_texts, device)
    
    # Get model dtype (handle both DDP and non-DDP cases)
    if hasattr(egovlpv2_model, 'module'):
        model_dtype = next(egovlpv2_model.module.parameters()).dtype
    else:
        model_dtype = next(egovlpv2_model.parameters()).dtype
    
    with torch.no_grad():
        # Use the same dtype as the model for autocast
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if hasattr(egovlpv2_model, 'module'):
                    text_embeddings = egovlpv2_model.module.compute_text(text_tokens)
                else:
                    text_embeddings = egovlpv2_model.compute_text(text_tokens)
        else:
            # For float32 or float16 models, let default autocast handle it
            with torch.cuda.amp.autocast():
                if hasattr(egovlpv2_model, 'module'):
                    text_embeddings = egovlpv2_model.module.compute_text(text_tokens)
                else:
                    text_embeddings = egovlpv2_model.compute_text(text_tokens)

    # Return embeddings in the same dtype as the model
    return text_embeddings.to(model_dtype)

def extract_egovlpv2_text_token_embeddings(
    egovlpv2_model, 
    egovlpv2_tokenizer, 
    batch_texts, 
    device='cuda:0'
):
    """
    提取EgoVLPv2的token级别文本特征（用于细粒度对齐）
    
    与extract_egovlpv2_text_embeddings区别：
    - extract_egovlpv2_text_embeddings: 返回句子级特征 [B, D]（使用compute_text，取CLS token）
    - extract_egovlpv2_text_token_embeddings: 返回token级特征 [B, T-1, D]（使用compute_text_tokens，排除CLS）
    
    RoBERTa token序列: [CLS, token1, token2, ..., SEP, PAD, ...]
    - compute_text_tokens返回: [token1, token2, ..., SEP, PAD, ...] (排除CLS)
    - attention_mask也需要排除第一个位置（对应CLS）
    
    Args:
        egovlpv2_model: EgoVLPv2模型实例
        egovlpv2_tokenizer: EgoVLPv2 tokenizer
        batch_texts: 文本列表（从OpenVLA batch中提取）
        device: 计算设备
        
    Returns:
        tuple: (text_token_embeddings, attention_mask)
            - text_token_embeddings: Token级文本特征 [batch_size, max_tokens-1, embedding_dim]（排除CLS）
            - attention_mask: 注意力mask [batch_size, max_tokens-1] (1=valid, 0=padding)（排除CLS位置）
    """
    # Tokenize texts (包含input_ids和attention_mask)
    text_tokens = text_to_token(egovlpv2_tokenizer, batch_texts, device)
    
    # 提取attention_mask并排除CLS位置（与compute_text_tokens的[:, 1:]对应）
    # 原始mask: [B, T]，包含CLS位置
    # 需要的mask: [B, T-1]，排除CLS位置
    attention_mask = text_tokens['attention_mask'][:, 1:]  # [B, T-1] 排除第一个位置（CLS）
    
    # 获取模型dtype（处理DDP和非DDP情况）
    if hasattr(egovlpv2_model, 'module'):
        model_dtype = next(egovlpv2_model.module.parameters()).dtype
    else:
        model_dtype = next(egovlpv2_model.parameters()).dtype
    
    with torch.no_grad():
        # 使用与模型相同的dtype进行autocast
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if hasattr(egovlpv2_model, 'module'):
                    text_token_embeddings = egovlpv2_model.module.compute_text_tokens(text_tokens)
                else:
                    text_token_embeddings = egovlpv2_model.compute_text_tokens(text_tokens)
        else:
            # 对于float32或float16模型，让默认autocast处理
            with torch.cuda.amp.autocast():
                if hasattr(egovlpv2_model, 'module'):
                    text_token_embeddings = egovlpv2_model.module.compute_text_tokens(text_tokens)
                else:
                    text_token_embeddings = egovlpv2_model.compute_text_tokens(text_tokens)
    
    # 返回与模型相同dtype的embeddings和attention_mask
    # text_token_embeddings: [B, T-1, D]
    # attention_mask: [B, T-1]
    return text_token_embeddings.to(model_dtype), attention_mask


def extract_egohod_text_embeddings(
    egohod_model,
    batch_texts,
    device="cuda:0",
    context_length=77,
    enable_grad=False,  # co-training时设为True，允许alignment loss梯度流回EgoHOD文本编码器
    before_proj=False,  # True=返回投影前特征（对齐用），False=返回投影后特征（训练用）
):
    """
    使用EgoHOD模型提取文本句子级特征。

    说明：
    - 与extract_egovlpv2_text_embeddings接口类似，但使用OpenAI CLIP的tokenizer；
    - batch_texts 为 list[str]，通常来自OpenPI的prompt列表；
    - before_proj=False时返回投影后特征 [B, project_embed_dim]（默认512）
    - before_proj=True时返回投影前特征 [B, transformer.width]（如768），用于对齐
    - enable_grad=True时不使用no_grad，允许梯度流回EgoHOD（co-training场景）
    """
    # 使用CLIP自带的tokenize函数，将批量文本编码为token id张量
    text_tokens = clip.tokenize(batch_texts, context_length=context_length)
    text_tokens = text_tokens.to(device)

    # EgoHODModel.compute_text 期望的输入格式：dict，至少包含 'input_ids'
    text_data = {"input_ids": text_tokens}

    # 获取模型dtype（支持DDP包装）
    if hasattr(egohod_model, "module"):
        model_dtype = next(egohod_model.module.parameters()).dtype
        model_ref = egohod_model.module
    else:
        model_dtype = next(egohod_model.parameters()).dtype
        model_ref = egohod_model

    def _compute(model_ref, text_data, model_dtype):
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                return model_ref.compute_text(text_data, before_proj=before_proj)
        else:
            with torch.cuda.amp.autocast():
                return model_ref.compute_text(text_data, before_proj=before_proj)

    # co-training时允许梯度流过，freeze时使用no_grad节省显存
    if enable_grad:
        text_embeddings = _compute(model_ref, text_data, model_dtype)
    else:
        with torch.no_grad():
            text_embeddings = _compute(model_ref, text_data, model_dtype)

    return text_embeddings.to(model_dtype)


def extract_egohod_text_token_embeddings(
    egohod_model,
    batch_texts,
    device="cuda:0",
    context_length=77,
    enable_grad=False,  # co-training时设为True，允许alignment loss梯度流回EgoHOD文本编码器
):
    """
    使用EgoHOD模型提取文本token级特征（用于细粒度对齐）
    
    与extract_egovlpv2_text_token_embeddings的区别：
    1. 使用CLIP tokenizer而非RoBERTa tokenizer
    2. Token特征维度由transformer.width决定（512），未经过text_projection
    3. Mask生成基于EOT token位置（由TextTransformer.forward_tokens生成）
    
    说明：
    - batch_texts 为 list[str]，通常来自OpenPI的prompt列表
    - Token特征不经过text_projection（projection是为EOT token的句子级特征训练的）
    - AlignmentModel会有自己的投影层来处理这些未投影的token特征
    - enable_grad=True时不使用no_grad，允许梯度流回EgoHOD（co-training场景）
    
    参数：
        egohod_model: EgoHODModel实例
        batch_texts: 文本列表
        device: 计算设备
        context_length: CLIP的上下文长度（默认77）
        enable_grad: 是否允许梯度流过（co-training时为True）
        
    返回：
        tuple: (text_token_embeddings, attention_mask)
            - text_token_embeddings: Token级文本特征 [B, T, D]（未经过text_projection）
            - attention_mask: 注意力mask [B, T] (1=valid, 0=padding)
    """
    # 使用CLIP自带的tokenize函数
    # CLIP会自动填充到context_length，并添加BOS和EOT token
    text_tokens = clip.tokenize(batch_texts, context_length=context_length)
    text_tokens = text_tokens.to(device)
    
    # EgoHODModel.compute_text_tokens 期望的输入格式：dict，至少包含 'input_ids'
    text_data = {"input_ids": text_tokens}
    
    # 获取模型dtype（支持DDP包装）
    if hasattr(egohod_model, "module"):
        model_dtype = next(egohod_model.module.parameters()).dtype
        model_ref = egohod_model.module
    else:
        model_dtype = next(egohod_model.parameters()).dtype
        model_ref = egohod_model
    
    def _compute(model_ref, text_data, model_dtype):
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                return model_ref.compute_text_tokens(text_data)
        else:
            with torch.cuda.amp.autocast():
                return model_ref.compute_text_tokens(text_data)

    # co-training时允许梯度流过，freeze时使用no_grad节省显存
    if enable_grad:
        text_token_embeddings, attention_mask = _compute(model_ref, text_data, model_dtype)
    else:
        with torch.no_grad():
            text_token_embeddings, attention_mask = _compute(model_ref, text_data, model_dtype)
    
    # 返回与模型相同dtype的embeddings和attention_mask
    # attention_mask由TextTransformer.forward_tokens生成，基于EOT token位置
    return text_token_embeddings.to(model_dtype), attention_mask


def extract_egovideo_text_embeddings(
    egovideo_model,
    batch_texts,
    device="cuda:0",
    max_length=77,
):
    """
    使用EgoVideo模型提取文本句子级特征
    
    说明：
    - 与extract_egovlpv2_text_embeddings和extract_egohod_text_embeddings接口类似
    - batch_texts为list[str]，通常来自OpenPI的prompt列表
    - 使用BERT tokenizer进行tokenization（HuggingFace BertTokenizer）
    - 返回[B, D]的文本嵌入，其中D=512（projection_dim）
    
    参数：
        egovideo_model: EgoVideoWrapper实例
        batch_texts: 文本列表
        device: 计算设备
        max_length: BERT的最大长度（默认40）
    
    返回：
        text_embeddings: [B, D] 文本特征（已L2归一化）
    """
    # 获取EgoVideo的tokenizer（BERT tokenizer）
    if hasattr(egovideo_model, "module"):
        tokenizer = egovideo_model.module.tokenizer
        model_ref = egovideo_model.module
        model_dtype = next(egovideo_model.module.parameters()).dtype
    else:
        tokenizer = egovideo_model.tokenizer
        model_ref = egovideo_model
        model_dtype = next(egovideo_model.parameters()).dtype
    
    # 使用BERT tokenizer编码文本
    text_tokens = tokenizer(
        batch_texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    
    # EgoVideoWrapper.compute_text期望的输入格式：dict，包含'input_ids'和'attention_mask'
    text_data = text_tokens
    
    with torch.no_grad():
        # 使用与模型相同的dtype进行autocast
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                text_embeddings = model_ref.compute_text(text_data)
        else:
            with torch.cuda.amp.autocast():
                text_embeddings = model_ref.compute_text(text_data)
    
    return text_embeddings.to(model_dtype)


def extract_egovideo_text_token_embeddings(
    egovideo_model,
    batch_texts,
    device="cuda:0",
    max_length=40,
):
    """
    使用EgoVideo模型提取文本token级特征（用于细粒度对齐）
    
    与extract_egohod_text_token_embeddings的区别：
    1. 使用BERT tokenizer而非CLIP tokenizer
    2. Token特征维度为1024（BERT hidden_size），未经过text_projection
    3. Mask基于BERT的padding策略
    
    说明：
    - batch_texts为list[str]，通常来自OpenPI的prompt列表
    - Token特征不经过text_projection（保持原始BERT输出）
    - AlignmentModel会有自己的投影层来处理这些未投影的token特征
    
    参数：
        egovideo_model: EgoVideoWrapper实例
        batch_texts: 文本列表
        device: 计算设备
        max_length: BERT的最大长度（默认40）
        
    返回：
        tuple: (text_token_embeddings, attention_mask)
            - text_token_embeddings: Token级文本特征 [B, T, D]（未经过text_projection，D=1024）
            - attention_mask: 注意力mask [B, T] (1=valid, 0=padding)
    """
    # 获取EgoVideo的tokenizer（BERT tokenizer）
    if hasattr(egovideo_model, "module"):
        tokenizer = egovideo_model.module.tokenizer
        model_ref = egovideo_model.module
        model_dtype = next(egovideo_model.module.parameters()).dtype
    else:
        tokenizer = egovideo_model.tokenizer
        model_ref = egovideo_model
        model_dtype = next(egovideo_model.parameters()).dtype
    
    # 使用BERT tokenizer编码文本
    text_tokens = tokenizer(
        batch_texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    attention_mask = text_tokens['attention_mask']
    
    # EgoVideoWrapper.compute_text_tokens期望的输入格式：dict
    text_data = text_tokens
    
    with torch.no_grad():
        # 使用与模型相同的dtype进行autocast
        # compute_text_tokens返回(text_token_embeddings, attention_mask)
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                text_token_embeddings, attention_mask = model_ref.compute_text_tokens(text_data)
        else:
            with torch.cuda.amp.autocast():
                text_token_embeddings, attention_mask = model_ref.compute_text_tokens(text_data)
    
    # 返回与模型相同dtype的embeddings和attention_mask
    return text_token_embeddings.to(model_dtype), attention_mask


def extract_clip_text_embeddings(
    clip_model,
    batch_texts,
    device="cuda:0",
    context_length=77,
):
    """
    使用CLIP模型提取文本句子级特征
    
    说明：
    - 与extract_egovlpv2_text_embeddings和extract_egohod_text_embeddings接口类似
    - batch_texts为list[str]，通常来自OpenPI的prompt列表
    - 使用OpenAI CLIP的tokenizer进行tokenization
    - 返回[B, D]的文本嵌入，其中D由CLIP模型决定（ViT-B/16: 512维，ViT-L/14: 768维）
    
    参数：
        clip_model: CLIPModel实例
        batch_texts: 文本列表
        device: 计算设备
        context_length: CLIP的上下文长度（默认77）
    
    返回：
        text_embeddings: [B, D] 文本特征
    """
    # 使用CLIP自带的tokenize函数，将批量文本编码为token id张量
    text_tokens = clip.tokenize(batch_texts, context_length=context_length)
    text_tokens = text_tokens.to(device)
    
    # CLIPModel.compute_text期望的输入格式：dict，至少包含'input_ids'
    text_data = {"input_ids": text_tokens}
    
    # 获取模型dtype（支持DDP包装）
    if hasattr(clip_model, "module"):
        model_dtype = next(clip_model.module.parameters()).dtype
        model_ref = clip_model.module
    else:
        model_dtype = next(clip_model.parameters()).dtype
        model_ref = clip_model
    
    with torch.no_grad():
        # 使用与模型相同的dtype进行autocast
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                text_embeddings = model_ref.compute_text(text_data)
        else:
            with torch.cuda.amp.autocast():
                text_embeddings = model_ref.compute_text(text_data)
    
    return text_embeddings.to(model_dtype)


def extract_clip_text_token_embeddings(
    clip_model,
    batch_texts,
    device="cuda:0",
    context_length=77,
):
    """
    使用CLIP模型提取文本token级特征（用于细粒度对齐）
    
    与extract_clip_text_embeddings的区别：
    1. 返回所有token的特征序列，而不是只返回EOT token
    2. Token特征维度由transformer.width决定（ViT-B/16: 512维），未经过text_projection
    3. Mask基于padding token位置（CLIP使用0作为padding）
    
    说明：
    - batch_texts为list[str]，通常来自OpenPI的prompt列表
    - Token特征不经过text_projection（projection是为EOT token的句子级特征训练的）
    - AlignmentModel会有自己的投影层来处理这些未投影的token特征
    
    参数：
        clip_model: CLIPModel实例
        batch_texts: 文本列表
        device: 计算设备
        context_length: CLIP的上下文长度（默认77）
        
    返回：
        tuple: (text_token_embeddings, attention_mask)
            - text_token_embeddings: Token级文本特征 [B, T, D]（未经过text_projection）
            - attention_mask: 注意力mask [B, T] (1=valid, 0=padding)
    """
    # 使用CLIP自带的tokenize函数
    # CLIP会自动填充到context_length，并添加BOS和EOT token
    text_tokens = clip.tokenize(batch_texts, context_length=context_length)
    text_tokens = text_tokens.to(device)
    
    # CLIPModel.compute_text_tokens期望的输入格式：dict，至少包含'input_ids'
    text_data = {"input_ids": text_tokens}
    
    # 获取模型dtype（支持DDP包装）
    if hasattr(clip_model, "module"):
        model_dtype = next(clip_model.module.parameters()).dtype
        model_ref = clip_model.module
    else:
        model_dtype = next(clip_model.parameters()).dtype
        model_ref = clip_model
    
    with torch.no_grad():
        # 使用与模型相同的dtype进行autocast
        # compute_text_tokens返回(text_token_embeddings, attention_mask)
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                text_token_embeddings, attention_mask = model_ref.compute_text_tokens(text_data)
        else:
            with torch.cuda.amp.autocast():
                text_token_embeddings, attention_mask = model_ref.compute_text_tokens(text_data)
    
    # 返回与模型相同dtype的embeddings和attention_mask
    return text_token_embeddings.to(model_dtype), attention_mask


def extract_qwen3_text_embeddings(
    qwen3_model,
    batch_texts,
    device="cuda:0",
):
    """
    使用Qwen3-Embedding模型提取文本句子级特征
    
    说明：
    - 与extract_egovlpv2_text_embeddings/extract_egohod_text_embeddings接口类似
    - batch_texts为list[str]，通常来自OpenPI的prompt列表
    - 使用SentenceTransformer的encode方法进行编码
    - 返回[B, D]的文本嵌入，其中D由Qwen3-Embedding模型决定（4B版本: 4096维）
    
    核心特点：
    - 简洁实现：直接调用Qwen3Model.compute_text，无需手动tokenization
    - 自动归一化：默认进行L2归一化，提升检索性能
    - 支持DDP：自动处理DDP包装
    
    参数：
        qwen3_model: Qwen3Model实例（封装了SentenceTransformer）
        batch_texts: 文本列表 [B]
        device: 计算设备
    
    返回：
        text_embeddings: [B, D] 文本特征（已归一化）
    """
    # 构建输入数据（与其他模型保持一致的接口）
    text_data = {"text": batch_texts}
    
    # 获取模型dtype（支持DDP包装）
    if hasattr(qwen3_model, "module"):
        model_dtype = next(qwen3_model.module.parameters()).dtype
        model_ref = qwen3_model.module
    else:
        model_dtype = next(qwen3_model.parameters()).dtype
        model_ref = qwen3_model
    
    with torch.no_grad():
        # Qwen3Model内部已处理autocast，这里直接调用
        # SentenceTransformer不需要显式的autocast context
        text_embeddings = model_ref.compute_text(text_data)
    
    # 确保返回的dtype与模型一致
    return text_embeddings.to(model_dtype)


def extract_qwen3_text_token_embeddings(
    qwen3_model,
    batch_texts,
    device="cuda:0",
):
    """
    使用Qwen3-VL-Embedding提取 token 级文本特征。

    返回值语义与 EgoHOD / CLIP 路线保持一致：
    - text_token_embeddings: [B, T, D]
    - attention_mask: [B, T]
    """
    text_data = {"text": batch_texts}

    if hasattr(qwen3_model, "module"):
        model_dtype = next(qwen3_model.module.parameters()).dtype
        model_ref = qwen3_model.module
    else:
        model_dtype = next(qwen3_model.parameters()).dtype
        model_ref = qwen3_model

    with torch.no_grad():
        text_token_embeddings, attention_mask = model_ref.compute_text_tokens(text_data)

    return text_token_embeddings.to(model_dtype), attention_mask


def extract_embedding_text_embeddings(
    embedding_model,
    batch_texts,
    device="cuda:0",
):
    """
    使用预计算的Embedding模型提取文本句子级特征（文本字符串匹配模式）
    
    参数：
        embedding_model: EmbeddingModel实例（加载了预计算特征）
        batch_texts: 文本列表 [B]
        device: 计算设备
    
    返回：
        text_embeddings: [B, D] 文本特征
    """
    text_data = {"text": batch_texts}
    
    if hasattr(embedding_model, "module"):
        model_ref = embedding_model.module
    else:
        model_ref = embedding_model
    
    with torch.no_grad():
        text_embeddings = model_ref.compute_text(text_data)
    
    return text_embeddings


def extract_embedding_text_embeddings_by_index(
    embedding_model,
    task_index,
    device="cuda:0",
):
    """
    通过task_index直接查表获取预计算embedding（高效模式，跳过字符串匹配）
    
    前提：npz中的embeddings已按task_index排序，embeddings[i] = task_index为i的文本embedding
    
    参数：
        embedding_model: EmbeddingModel实例
        task_index: [B] 任务索引张量（来自dataset batch）
        device: 计算设备
    
    返回：
        text_embeddings: [B, D] 文本特征
    """
    if hasattr(embedding_model, "module"):
        model_ref = embedding_model.module
    else:
        model_ref = embedding_model
    
    with torch.no_grad():
        text_embeddings = model_ref.compute_text_by_index(task_index)
    
    return text_embeddings


def extract_embedding_text_token_embeddings(
    embedding_model,
    batch_texts,
    device="cuda:0",
):
    """
    通过文本匹配从预计算 embedding 中取 token 级特征。
    """
    if hasattr(embedding_model, "module"):
        model_ref = embedding_model.module
    else:
        model_ref = embedding_model

    with torch.no_grad():
        text_token_embeddings, attention_mask = model_ref.compute_text_tokens({"text": batch_texts})

    return text_token_embeddings, attention_mask


def extract_embedding_text_token_embeddings_by_index(
    embedding_model,
    task_index,
    device="cuda:0",
):
    """
    通过 task_index 直接查预计算 token 级特征。
    """
    if hasattr(embedding_model, "module"):
        model_ref = embedding_model.module
    else:
        model_ref = embedding_model

    with torch.no_grad():
        text_token_embeddings, attention_mask = model_ref.compute_text_tokens_by_index(task_index)

    return text_token_embeddings, attention_mask


def extract_roboticclip_text_embeddings(
    roboticclip_model,
    batch_texts,
    device="cuda:0",
    context_length=77,
):
    """
    使用RoboticCLIP模型提取文本句子级特征
    
    说明：
    - 与extract_egohod_text_embeddings接口类似，但使用alpha_clip.tokenize
    - batch_texts为list[str]，通常来自OpenPI的prompt列表
    - 返回[B, D]的文本嵌入（ViT-L/14@336px: 768维）
    
    参数：
        roboticclip_model: RoboticCLIPModel实例
        batch_texts: 文本列表
        device: 计算设备
        context_length: 上下文长度（默认77）
    
    返回：
        text_embeddings: [B, D] 文本特征
    """
    # 导入alpha_clip的tokenize
    from egovlpv2.model.RoboticCLIP.PromptGD.alpha_clip import alpha_clip
    
    # 使用alpha_clip的tokenize函数
    text_tokens = alpha_clip.tokenize(batch_texts, context_length=context_length)
    text_tokens = text_tokens.to(device)
    
    # 构建输入数据（与其他模型保持一致的接口）
    text_data = {"input_ids": text_tokens}
    
    # 获取模型dtype（支持DDP包装）
    if hasattr(roboticclip_model, "module"):
        model_dtype = next(roboticclip_model.module.parameters()).dtype
        model_ref = roboticclip_model.module
    else:
        model_dtype = next(roboticclip_model.parameters()).dtype
        model_ref = roboticclip_model
    
    with torch.no_grad():
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                text_embeddings = model_ref.compute_text(text_data)
        else:
            with torch.cuda.amp.autocast():
                text_embeddings = model_ref.compute_text(text_data)
    
    return text_embeddings.to(model_dtype)


def extract_roboticclip_text_token_embeddings(
    roboticclip_model,
    batch_texts,
    device="cuda:0",
    context_length=77,
):
    """
    使用RoboticCLIP模型提取文本token级特征（用于细粒度对齐）
    
    参数：
        roboticclip_model: RoboticCLIPModel实例
        batch_texts: 文本列表
        device: 计算设备
        context_length: 上下文长度（默认77）
        
    返回：
        tuple: (text_token_embeddings, attention_mask)
            - text_token_embeddings: Token级文本特征 [B, T, D]
            - attention_mask: 注意力mask [B, T] (1=valid, 0=padding)
    """
    from egovlpv2.model.RoboticCLIP.PromptGD.alpha_clip import alpha_clip
    
    text_tokens = alpha_clip.tokenize(batch_texts, context_length=context_length)
    text_tokens = text_tokens.to(device)
    
    text_data = {"input_ids": text_tokens}
    
    if hasattr(roboticclip_model, "module"):
        model_dtype = next(roboticclip_model.module.parameters()).dtype
        model_ref = roboticclip_model.module
    else:
        model_dtype = next(roboticclip_model.parameters()).dtype
        model_ref = roboticclip_model
    
    with torch.no_grad():
        if model_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                text_token_embeddings, attention_mask = model_ref.compute_text_tokens(text_data)
        else:
            with torch.cuda.amp.autocast():
                text_token_embeddings, attention_mask = model_ref.compute_text_tokens(text_data)
    
    return text_token_embeddings.to(model_dtype), attention_mask
