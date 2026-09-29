# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from egovlpv2.utils.text_feature_encode import (
    text_to_token,
    token_to_text,
    extract_content_vla,
    extract_egovlpv2_text_embeddings,
    extract_egovlpv2_text_token_embeddings,
    extract_egohod_text_embeddings,
    extract_egohod_text_token_embeddings,
    extract_egovideo_text_embeddings,
    extract_egovideo_text_token_embeddings,
    extract_qwen3_text_embeddings,
    extract_qwen3_text_token_embeddings,
    extract_embedding_text_embeddings,
    extract_embedding_text_embeddings_by_index,  # 新增：通过task_index直接查表
    extract_embedding_text_token_embeddings,
    extract_embedding_text_token_embeddings_by_index,
    extract_clip_text_embeddings,
    extract_clip_text_token_embeddings,
    extract_roboticclip_text_embeddings,
    extract_roboticclip_text_token_embeddings,
)


# === Alignment Model Forward Pass Functions ===

def alignment_forward_pass(
    alignment_model,
    openvla_features,
    egovlpv2_features,
    openvla_global_features=None,
    egovlpv2_text_tokens=None,
    egovlpv2_text_mask=None,
    task_index=None,  # [B] 原始任务索引（与tasks.jsonl对应）
    task_ids=None,  # [B] 聚类后的任务类别ID，用于多正样本对比学习
    langs=None,  # List[str] 指令文本列表，用于检测空指令
    allgather_fn=None,
    n_gpu=1,
    args=None,
    device_id=0,
    mode: str = "egovlpv2",
    video_features=None,      # [M, D] 去重后的 sampled video anchors（as2vs模式）
    video_sim_weights=None,    # [B, M] dense query-to-anchor text 相似度矩阵（as2vs模式）
    atomic_text_features=None,  # [B, D] 原子级text embedding（替换egovlpv2_features用于as2ts）
    chunk_video_features=None,   # [B, D_video] 每条样本对应的 chunk-level video embedding（as2cv）
    chunk_video_task_ids=None,   # [B] task_id 副本（仅 as2cv trajectory 用，None=回退到 task_index）
    chunk_video_idx=None,        # [B] chunk 全局 id，as2cv chunk head 用（同 id = 正）
):
    """
    Alignment model forward pass wrapper with autocast support.
    只支持simple模式（batch内对角线为正样本）
    
    Args:
        alignment_model: AlignmentModel实例
        openvla_features: OpenVLA特征 [B, L, A, D1]
        egovlpv2_features: 文本句子特征 [B, D2]
        egovlpv2_text_tokens: 文本token特征 [B, T, D2]（as2tt/at2tt模式需要）
        egovlpv2_text_mask: 文本attention mask [B, T] (1=valid, 0=padding)
        task_ids: [B] task_id向量，用于多正样本对比学习
        allgather_fn: 分布式gather函数
        n_gpu: GPU数量
        args: 额外参数
        device_id: 设备ID
        mode: 文本后端类型标记（'egovlpv2' 或 'egohod' 等）
        
    Returns:
        Dict: 包含alignment loss和特征的字典
    """
    # Move features to correct device
    if torch.is_tensor(openvla_features):
        openvla_features = openvla_features.to(device_id, non_blocking=True)
    if openvla_global_features is not None and torch.is_tensor(openvla_global_features):
        openvla_global_features = openvla_global_features.to(device_id, non_blocking=True)
    if torch.is_tensor(egovlpv2_features):
        egovlpv2_features = egovlpv2_features.to(device_id, non_blocking=True)
    if egovlpv2_text_tokens is not None and torch.is_tensor(egovlpv2_text_tokens):
        egovlpv2_text_tokens = egovlpv2_text_tokens.to(device_id, non_blocking=True)
    if egovlpv2_text_mask is not None and torch.is_tensor(egovlpv2_text_mask):
        egovlpv2_text_mask = egovlpv2_text_mask.to(device_id, non_blocking=True)
    if task_index is not None and torch.is_tensor(task_index):
        task_index = task_index.to(device_id, non_blocking=True)
    if task_ids is not None and torch.is_tensor(task_ids):
        task_ids = task_ids.to(device_id, non_blocking=True)
    
    # Get alignment model dtype for proper autocast configuration
    if hasattr(alignment_model, 'module'):
        model_dtype = next(alignment_model.module.parameters()).dtype
    else:
        model_dtype = next(alignment_model.parameters()).dtype
    
    # 只在模型本身已经是半精度时启用autocast。
    # 如果模型是float32，不应强行降到fp16，否则会改变训练数值行为。
    if model_dtype == torch.bfloat16:
        autocast_enabled = True
        autocast_dtype = torch.bfloat16
    elif model_dtype == torch.float16:
        autocast_enabled = True
        autocast_dtype = torch.float16
    else:
        autocast_enabled = False
        autocast_dtype = torch.float16
    
    # 将video features移到正确设备
    if video_features is not None and torch.is_tensor(video_features):
        video_features = video_features.to(device_id, non_blocking=True)
    if video_sim_weights is not None and torch.is_tensor(video_sim_weights):
        video_sim_weights = video_sim_weights.to(device_id, non_blocking=True)
    
    # 原子级对齐：atomic_text_features 作为独立的 as2atomic 分支并行传入
    if atomic_text_features is not None and torch.is_tensor(atomic_text_features):
        atomic_text_features = atomic_text_features.to(device_id, non_blocking=True)

    # chunk-level video 对齐：chunk_video_features 作为 as2cv 独立分支
    if chunk_video_features is not None and torch.is_tensor(chunk_video_features):
        chunk_video_features = chunk_video_features.to(device_id, non_blocking=True)
    if chunk_video_task_ids is not None and torch.is_tensor(chunk_video_task_ids):
        chunk_video_task_ids = chunk_video_task_ids.to(device_id, non_blocking=True)
    if chunk_video_idx is not None and torch.is_tensor(chunk_video_idx):
        chunk_video_idx = chunk_video_idx.to(device_id, non_blocking=True)
    
    # Alignment Forward Pass with autocast
    with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
        alignment_loss, loss_dict, ret_dict = alignment_model(
            data=None,
            openvla_features=openvla_features,
            openvla_global_features=openvla_global_features,
            egovlpv2_features=egovlpv2_features,
            egovlpv2_text_tokens=egovlpv2_text_tokens,
            egovlpv2_text_mask=egovlpv2_text_mask,
            task_index=task_index,
            task_ids=task_ids,
            langs=langs,  # 传递指令文本用于检测空指令
            video_features=video_features,          # as2vs: 采样的video features
            video_sim_weights=video_sim_weights,     # as2vs: soft positive mask
            atomic_text_features=atomic_text_features,  # as2atomic: 原子级text embedding
            chunk_video_features=chunk_video_features,  # as2cv: chunk-level video embedding
            chunk_video_task_ids=chunk_video_task_ids,   # as2cv: task_id 用于 trajectory mask
            chunk_video_idx=chunk_video_idx,             # as2cv: chunk_id 用于 chunk mask
            allgather=allgather_fn,
            n_gpu=n_gpu,
            args=args,
            config=None,
            loss_fn=None,
            gpu=device_id,
            return_embeds=True,
            task_names='Alignment',
        )
    
    return {
        'loss': alignment_loss,
        'loss_dict': loss_dict,
        'aligned_features': ret_dict,
        'egovlpv2_aligned': ret_dict.get('egovlpv2_aligned'),
        'openvla_aligned': ret_dict.get('openvla_aligned')
    }


# === EgoHOD Forward Pass Wrapper ===
def egohod_forward_pass(model, vlm_batch, tokenizer, config, args, loss_fn, 
                        allgather_fn, device_id, task_names='Dual'):
    """
    EgoHOD训练前向传播（接口与egovlpv2_forward_pass对齐）
    
    说明：
    - 保持与EgoVLPv2相同的接口签名，确保MIMICVLAModel可统一调用
    - EgoHOD使用ClipLoss（内部处理分布式gather），allgather_fn参数保留但不使用
    - ClipLoss的rank/world_size在model.forward中更新
    
    Args:
        model: EgoHOD模型实例
        vlm_batch: {'video': [B,T,C,H,W], 'text': List[str]}
        tokenizer: clip.tokenize函数（用于文本tokenize）
        config: 配置对象（保留兼容）
        args: 包含rank和world_size（用于更新ClipLoss）
        loss_fn: ClipLoss实例
        allgather_fn: 分布式gather函数（保留兼容，ClipLoss内部处理）
        device_id: GPU设备ID
        task_names: 任务名称
    """
    import clip
    
    # 准备视频数据
    video_data = vlm_batch['video'].to(device_id, non_blocking=True)
    
    # 准备文本数据：使用CLIP tokenize
    text_list = vlm_batch['text']
    if isinstance(text_list, torch.Tensor):
        text_tokens = text_list.to(device_id, non_blocking=True)
    else:
        text_tokens = clip.tokenize(text_list, truncate=True).to(device_id)
    
    # 组装输入数据
    egohod_data = {
        'video': video_data,
        'text': {'input_ids': text_tokens}
    }
    
    # 将ClipLoss移到正确设备
    #print(f"video data.shape: {video_data.shape}")
    loss_fn = loss_fn.to(device_id)
    
    # 获取模型dtype用于autocast
    model_dtype = next(model.parameters()).dtype
    autocast_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
    
    # 前向传播（接口与EgoVLPv2对齐：allgather, n_gpu, args, config, loss_dual, gpu）
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        vlm_loss, vlm_loss_dict, vlm_ret = model(
            egohod_data, 
            allgather_fn,      # 保留兼容（ClipLoss内部处理gather）
            args.world_size,   # n_gpu
            args, 
            config, 
            loss_fn,           # loss_dual (实际是ClipLoss)
            device_id,         # gpu
            task_names=task_names,
            dataset_name='fho'
        )
    
    return {
        'loss': vlm_loss,
        'loss_dict': vlm_loss_dict,
        'ret': vlm_ret
    }


# === EgoVLPv2 Forward Pass Wrapper ===
def egovlpv2_forward_pass(model, vlm_batch, tokenizer, config, args, loss_fn, 
                         allgather_fn, device_id, task_names, vlm_loss_weight=1.0):
    """
    Streamlined EgoVLPv2 forward pass wrapper
    
    Args:
        model: EgoVLPv2 model
        vlm_batch: EgoVLPv2 batch data
        tokenizer: EgoVLPv2 tokenizer
        config: EgoVLPv2 config
        args: EgoVLPv2 args
        loss_fn: Loss function
        allgather_fn: AllGather function
        device_id: GPU device ID
        task_names: Task names for the model
        vlm_loss_weight: Weight for VLM loss (for logging purposes)
        
    Returns:
        dict: {
            'loss': computed loss tensor,
            'loss_dict': detailed loss dictionary,
            'ret': model return values
        }
    """
    if vlm_batch is None:
        # Return zero loss if no batch
        return {
            'loss': torch.tensor(0.0, device=device_id),
            'loss_dict': {},
            'ret': {}
        }
    
    # Prepare EgoVLPv2 data (直接使用原始batch，避免copy增加显存)
    egovlpv2_data = vlm_batch
    
    # Tokenize and move to GPU
    egovlpv2_data['text'] = text_to_token(tokenizer, egovlpv2_data['text'], device_id)
    
    egovlpv2_data['video'] = egovlpv2_data['video'].to(device_id, non_blocking=True)
    
    # Extract n_embeds and v_embeds if available
    n_embeds = None
    v_embeds = None
    if 'noun_vec' in vlm_batch and 'verb_vec' in vlm_batch:
        n_embeds = vlm_batch['noun_vec'].to(device_id)
        v_embeds = vlm_batch['verb_vec'].to(device_id)
    
    # EgoVLPv2 Forward Pass (与trainer_charades.py保持一致)
    # Get model dtype for proper autocast configuration
    if hasattr(model, 'module'):
        model_dtype = next(model.module.parameters()).dtype
    else:
        model_dtype = next(model.parameters()).dtype
    
    # Use appropriate autocast based on model dtype
    if model_dtype == torch.bfloat16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16  # Default for standard autocast
    
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        if config['model_type'] == 'epic_charades':
            vlm_loss, vlm_loss_dict, vlm_ret = model(
                egovlpv2_data, allgather_fn, args.world_size,
                args, config, loss_fn, device_id, 
                task_names=task_names, dataset_name='charades'
            )
        else:
            vlm_loss, vlm_loss_dict, vlm_ret = model(
                egovlpv2_data, n_embeds, v_embeds, allgather_fn, 
                args.world_size, args, config, loss_fn, 
                device_id, task_names=task_names
            )
    
    return {
        'loss': vlm_loss,
        'loss_dict': vlm_loss_dict,
        'ret': vlm_ret
    }


# === VLA Feature Extraction Functions ===

def get_layer_vla_features(hidden_states_stacked, layer_indices=None):
    """
    从VLA隐藏状态中选择特定层的特征
    
    Args:
        hidden_states_stacked: 堆叠的隐藏状态 [num_layers, batch_size, seq_len, hidden_size]
        layer_indices: 要选择的层索引列表，如果为None则使用最后8层
        
    Returns:
        选择后的隐藏状态
    """
    # Select specific layers
    if layer_indices is not None:
        hidden_states_stacked = hidden_states_stacked[layer_indices]
    else:
        # Default: use last 8 layers if not specified
        total_layers = hidden_states_stacked.shape[0]
        default_num_layers = min(8, total_layers)
        layer_indices = list(range(total_layers - default_num_layers, total_layers))
        hidden_states_stacked = hidden_states_stacked[layer_indices]
    return hidden_states_stacked


def get_vla_features(hidden_states, vision_patches_num, batch, action_token_begin_idx, layer_indices=None):
    """
    从VLA模型的隐藏状态中提取action相关特征
    
    Args:
        hidden_states (tuple[torch.Tensor]): 每层的隐藏状态元组
        vision_patches_num (int): 视觉token数量（256）
        batch (dict): VLA batch，包含用于生成mask的labels
        action_token_begin_idx (int): action token开始的索引
        layer_indices (list, optional): 要提取的层索引列表（例如[5, 33]）
                                       如果为None，则提取所有层的特征
    
    Returns:
        torch.Tensor: 形状为 (selected_layer_num, batch_size, num_actions, hidden_size)
    """
    
    # Generate action mask
    action_gt = batch["labels"][:, 1:].to(hidden_states[0].device)
    action_mask = action_gt > action_token_begin_idx

    # Stack hidden states from all layers
    hidden_states_stacked = torch.stack(hidden_states, dim=0)

    hidden_states_stacked = get_layer_vla_features(hidden_states_stacked, layer_indices)

    layer_num, batch_size, _, hidden_size = hidden_states_stacked.shape

    # Remove vision-related features
    text_action_features = hidden_states_stacked[:, :, vision_patches_num : -1, :]

    # Apply action mask
    expanded_mask = action_mask.unsqueeze(0).unsqueeze(-1)
    extracted_features = text_action_features[expanded_mask.expand_as(text_action_features)].view(
        layer_num, batch_size, -1, hidden_size
    )
    
    return extracted_features


# === Complete Alignment Forward Pass ===

def alignment_forward_pass_complete(
    alignment_model,
    vla_output,
    vla_batch,
    vla_tokenizer,
    egovlpv2_model,
    egovlpv2_tokenizer,
    action_token_begin_idx=None,
    num_vision_patches=None,
    layer_indices=None,
    device_id='cuda:0',
    allgather_fn=None,
    n_gpu=1,
    args=None,
    task_index=None,  # 原始任务索引（与tasks.jsonl对应）
    task_ids=None,  # 聚类后的任务类别ID
    mode: str = "egovlpv2",
    video_sampler=None,        # VideoFeatureSampler实例（as2vs模式），传入后内部完成采样
    video_features=None,       # [M, D] 也可直接传入去重后的 sampled video anchors
    video_sim_weights=None,    # [B, M] 也可直接传入 dense query-to-anchor 相似度矩阵
    enable_grad=False,         # co-training时设为True，允许alignment loss梯度流回VLM
    before_proj=False,         # True=对齐时使用投影前的EgoHOD文本特征
    vla_feature_type: str = "action",  # Ablation: "action"(默认), "vision", "text"(文本token), "full"(所有token)
    image_token_index: int = 256000,   # SpatialVLA的image token index，用于vision/text/full特征提取
    ignore_index: int = -100,          # label中的ignore index，用于区分action和非action token
    atomic_label_idx=None,     # [B] 原子级标签索引（用于 chunk-level 细粒度对齐）
):
    """
    完整的对齐模型前向传播（只支持simple模式）
    
    Args:
        alignment_model: AlignmentModel实例
        vla_output: OpenVLA模型输出（包含hidden_states）
        vla_batch: OpenVLA batch数据
        vla_tokenizer: OpenVLA processor的tokenizer
        egovlpv2_model: EgoVLPv2模型用于文本嵌入
        egovlpv2_tokenizer: EgoVLPv2 tokenizer
        action_token_begin_idx: action token开始的索引
        num_vision_patches: 视觉patch数量（默认256）
        layer_indices: 要提取的层索引列表
        device_id: 设备
        allgather_fn: 分布式gather函数
        n_gpu: GPU数量
        args: 额外参数
        task_ids: 可选的task id张量，用于多正样本对齐
        mode: 文本后端类型（'egovlpv2' 或 'egohod'）
        
    Returns:
        包含对齐loss和特征的字典
    """
    
    # ============ VideoFeatureSampler采样（as2vs模式） ============
    # 如果传入了video_sampler且有task_index，在此处完成采样
    # 这样模型侧只需传入sampler实例，不需要关心采样逻辑
    if video_sampler is not None and task_index is not None and video_features is None:
        video_features, video_sim_weights, _ = video_sampler.sample(task_index)
    
    # Extract texts from batch - support both old and new methods
    if "lang" in vla_batch:
        batch_texts = vla_batch["lang"]
    else:
        # 旧方式：从input_ids中提取并解析
        batch_texts = token_to_text(vla_tokenizer, vla_batch["input_ids"], extract_content=extract_content_vla)
    
    # ============ 根据 vla_feature_type 提取不同类型的 VLA 特征 ============
    if vla_feature_type in ("vision", "text", "full"):
        # === Ablation: 提取 vision / text / full token 的 hidden states ===
        # 从 hidden_states 中选层，然后 masked mean pooling → [B, L, 1, D]
        hidden_states_stacked = torch.stack(vla_output.hidden_states, dim=0)  # [all_layers+1, B, seq_len, D]
        hidden_states_stacked = get_layer_vla_features(hidden_states_stacked, layer_indices)  # [L, B, seq_len, D]
        
        input_ids = vla_batch['input_ids']   # [B, seq_len]
        labels = vla_batch['labels']         # [B, seq_len]
        attn_mask = vla_batch.get('attention_mask', None)
        
        if vla_feature_type == "vision":
            # 视觉token = image_token_index 且有效位置（不包含padding）
            token_mask = (input_ids == image_token_index)
            if attn_mask is not None:
                token_mask = token_mask & attn_mask.bool()
        elif vla_feature_type == "text":
            # 文本token = 不含vision(image_token_index)、不含action(labels!=-100)、需要valid(attention_mask)
            vision_mask = (input_ids == image_token_index)  # [B, seq_len]
            action_mask = (labels != ignore_index)          # [B, seq_len]
            token_mask = ~vision_mask & ~action_mask        # [B, seq_len]
            if attn_mask is not None:
                token_mask = token_mask & attn_mask.bool()
        else:  # full
            # 所有有效token（含vision、text、action，排除padding）
            if attn_mask is not None:
                token_mask = attn_mask.bool()  # [B, seq_len]
            else:
                token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Masked mean pooling: [L, B, seq_len, D] → [L, B, D]
        mask_exp = token_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, B, seq_len, 1]
        masked_sum = (hidden_states_stacked * mask_exp).sum(dim=2)  # [L, B, D]
        token_count = token_mask.sum(dim=1, keepdim=True).unsqueeze(0).float().clamp(min=1)  # [1, B, 1]
        pooled = masked_sum / token_count  # [L, B, D]
        
        # 转换为 AlignmentModel 期望的格式: [B, L, 1, D]（1个token = 已mean pooled）
        selected_layer_num = pooled.shape[0]
        batch_size = pooled.shape[1]
        openvla_features = pooled.permute(1, 0, 2).unsqueeze(2)  # [B, L, 1, D]
    
    elif hasattr(vla_output, 'action_hidden_states') and vla_output.action_hidden_states is not None:
        # === 默认: 使用 action_hidden_states（新方法） ===
        action_hidden_states = vla_output.action_hidden_states
        action_hidden_states = get_layer_vla_features(action_hidden_states, layer_indices)
        selected_layer_num, batch_size, num_actions, hidden_size = action_hidden_states.shape
        openvla_features = action_hidden_states.permute(1, 0, 2, 3)
    else:
        # Old method: extract from hidden_states
        openvla_action_features = get_vla_features(
            hidden_states=vla_output.hidden_states,
            vision_patches_num=num_vision_patches,
            batch=vla_batch,
            action_token_begin_idx=action_token_begin_idx,
            layer_indices=layer_indices
        )
        
        # Reshape to match AlignmentModel expected input: [batch_size, selected_layers, num_actions, hidden_size]
        selected_layer_num, batch_size, num_actions, hidden_size = openvla_action_features.shape
        openvla_features = openvla_action_features.permute(1, 0, 2, 3)
    
    # ============ Simple 模式：标准编码 ============
    # 根据不同的VLM后端提取文本特征
    if mode == "embedding":
        # Embedding后端：优先通过task_index直接查表（跳过字符串匹配，更高效）
        if task_index is not None:
            egovlpv2_text_embeds = extract_embedding_text_embeddings_by_index(
                embedding_model=egovlpv2_model,
                task_index=task_index,
                device=device_id,
            )
        else:
            # 兼容无task_index的场景，回退到文本匹配
            egovlpv2_text_embeds = extract_embedding_text_embeddings(
                embedding_model=egovlpv2_model,
                batch_texts=batch_texts,
                device=device_id,
            )
        egovlpv2_text_tokens = None
        egovlpv2_text_mask = None
        if hasattr(alignment_model, 'mode_config') or (hasattr(alignment_model, 'module') and hasattr(alignment_model.module, 'mode_config')):
            mode_config = alignment_model.mode_config if hasattr(alignment_model, 'mode_config') else alignment_model.module.mode_config
            if mode_config.get('as2tt', {}).get('enabled', False) or mode_config.get('at2tt', {}).get('enabled', False):
                if task_index is not None:
                    egovlpv2_text_tokens, egovlpv2_text_mask = extract_embedding_text_token_embeddings_by_index(
                        embedding_model=egovlpv2_model,
                        task_index=task_index,
                        device=device_id,
                    )
                else:
                    egovlpv2_text_tokens, egovlpv2_text_mask = extract_embedding_text_token_embeddings(
                        embedding_model=egovlpv2_model,
                        batch_texts=batch_texts,
                        device=device_id,
                    )
    elif mode == "qwen3":
        # Qwen3-Embedding后端
        egovlpv2_text_embeds = extract_qwen3_text_embeddings(
            qwen3_model=egovlpv2_model,
            batch_texts=batch_texts,
            device=device_id,
        )
        egovlpv2_text_tokens = None
        egovlpv2_text_mask = None
        if hasattr(alignment_model, 'mode_config') or (hasattr(alignment_model, 'module') and hasattr(alignment_model.module, 'mode_config')):
            mode_config = alignment_model.mode_config if hasattr(alignment_model, 'mode_config') else alignment_model.module.mode_config
            if mode_config.get('as2tt', {}).get('enabled', False) or mode_config.get('at2tt', {}).get('enabled', False):
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_qwen3_text_token_embeddings(
                    qwen3_model=egovlpv2_model,
                    batch_texts=batch_texts,
                    device=device_id,
                )
    elif mode == "egohod":
        # EgoHOD后端：使用CLIP tokenizer + EgoHODModel.compute_text
        # enable_grad控制：freeze VLM时no_grad，co-training时允许梯度流回EgoHOD
        # before_proj控制：True=返回投影前特征（对齐用），False=返回投影后特征
        egovlpv2_text_embeds = extract_egohod_text_embeddings(
            egohod_model=egovlpv2_model,
            batch_texts=batch_texts,
            device=device_id,
            enable_grad=enable_grad,
            before_proj=before_proj,
        )
        egovlpv2_text_tokens = None
        egovlpv2_text_mask = None
        # 检查是否需要token级特征
        if hasattr(alignment_model, 'mode_config') or (hasattr(alignment_model, 'module') and hasattr(alignment_model.module, 'mode_config')):
            mode_config = alignment_model.mode_config if hasattr(alignment_model, 'mode_config') else alignment_model.module.mode_config
            if mode_config.get('as2tt', {}).get('enabled', False) or mode_config.get('at2tt', {}).get('enabled', False):
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_egohod_text_token_embeddings(
                    egohod_model=egovlpv2_model,
                    batch_texts=batch_texts,
                    device=device_id,
                    enable_grad=enable_grad,
                )
    elif mode == "clip":
        # CLIP后端
        egovlpv2_text_embeds = extract_clip_text_embeddings(
            clip_model=egovlpv2_model,
            batch_texts=batch_texts,
            device=device_id,
        )
        egovlpv2_text_tokens = None
        egovlpv2_text_mask = None
        # 检查是否需要token级特征
        if hasattr(alignment_model, 'mode_config') or (hasattr(alignment_model, 'module') and hasattr(alignment_model.module, 'mode_config')):
            mode_config = alignment_model.mode_config if hasattr(alignment_model, 'mode_config') else alignment_model.module.mode_config
            if mode_config.get('as2tt', {}).get('enabled', False) or mode_config.get('at2tt', {}).get('enabled', False):
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_clip_text_token_embeddings(
                    clip_model=egovlpv2_model,
                    batch_texts=batch_texts,
                    device=device_id
                )
    elif mode == "egovideo":
        # EgoVideo后端
        egovlpv2_text_embeds = extract_egovideo_text_embeddings(
            egovideo_model=egovlpv2_model,
            batch_texts=batch_texts,
            device=device_id,
        )
        egovlpv2_text_tokens = None
        egovlpv2_text_mask = None
        # 检查是否需要token级特征
        if hasattr(alignment_model, 'mode_config') or (hasattr(alignment_model, 'module') and hasattr(alignment_model.module, 'mode_config')):
            mode_config = alignment_model.mode_config if hasattr(alignment_model, 'mode_config') else alignment_model.module.mode_config
            if mode_config.get('as2tt', {}).get('enabled', False) or mode_config.get('at2tt', {}).get('enabled', False):
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_egovideo_text_token_embeddings(
                    egovideo_model=egovlpv2_model,
                    batch_texts=batch_texts,
                    device=device_id
                )
    elif mode == "roboticclip":
        # RoboticCLIP后端：基于AlphaCLIP，使用alpha_clip.tokenize
        egovlpv2_text_embeds = extract_roboticclip_text_embeddings(
            roboticclip_model=egovlpv2_model,
            batch_texts=batch_texts,
            device=device_id,
        )
        egovlpv2_text_tokens = None
        egovlpv2_text_mask = None
        # 检查是否需要token级特征
        if hasattr(alignment_model, 'mode_config') or (hasattr(alignment_model, 'module') and hasattr(alignment_model.module, 'mode_config')):
            mode_config = alignment_model.mode_config if hasattr(alignment_model, 'mode_config') else alignment_model.module.mode_config
            if mode_config.get('as2tt', {}).get('enabled', False) or mode_config.get('at2tt', {}).get('enabled', False):
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_roboticclip_text_token_embeddings(
                    roboticclip_model=egovlpv2_model,
                    batch_texts=batch_texts,
                    device=device_id
                )
    else:
        # 默认EgoVLPv2后端：使用roberta tokenizer + EgoVLPv2.compute_text
        egovlpv2_text_embeds = extract_egovlpv2_text_embeddings(
            egovlpv2_model=egovlpv2_model,
            egovlpv2_tokenizer=egovlpv2_tokenizer,
            batch_texts=batch_texts,
            device=device_id
        )
        egovlpv2_text_tokens = None
        egovlpv2_text_mask = None
        # 检查是否需要token级特征（as2tt/at2tt模式需要）
        if hasattr(alignment_model, 'mode_config') or (hasattr(alignment_model, 'module') and hasattr(alignment_model.module, 'mode_config')):
            mode_config = alignment_model.mode_config if hasattr(alignment_model, 'mode_config') else alignment_model.module.mode_config
            if mode_config.get('as2tt', {}).get('enabled', False) or mode_config.get('at2tt', {}).get('enabled', False):
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_egovlpv2_text_token_embeddings(
                    egovlpv2_model=egovlpv2_model,
                    egovlpv2_tokenizer=egovlpv2_tokenizer,
                    batch_texts=batch_texts,
                    device=device_id
                )
    # 原子级对齐：如果提供了 atomic_label_idx 且 embedding model 支持，查表获取原子 text embedding
    atomic_text_features = None
    if atomic_label_idx is not None and mode == "embedding" and hasattr(egovlpv2_model, 'atomic_embed_layer') and egovlpv2_model.atomic_embed_layer is not None:
        atomic_text_features = egovlpv2_model.compute_atomic_text_by_index(atomic_label_idx)

    # Compute alignment loss
    alignment_result = alignment_forward_pass(
        alignment_model=alignment_model,
        openvla_features=openvla_features,
        egovlpv2_features=egovlpv2_text_embeds,
        egovlpv2_text_tokens=egovlpv2_text_tokens,
        egovlpv2_text_mask=egovlpv2_text_mask,
        allgather_fn=allgather_fn,
        n_gpu=n_gpu,
        args=args,
        device_id=device_id,
        task_index=task_index,
        task_ids=task_ids,
        langs=batch_texts,  # 传递指令文本用于检测空指令
        mode=mode,
        video_features=video_features,          # as2vs: 采样的video features
        video_sim_weights=video_sim_weights,     # as2vs: soft positive mask
        atomic_text_features=atomic_text_features,  # as2atomic: 原子级text embedding（与task-level并行）
    )
    
    # 添加额外信息到结果字典
    alignment_result.update({
        'batch_texts': batch_texts,
        'openvla_action_features_shape': tuple(openvla_features.shape),
        'selected_layer_indices': layer_indices,
        'num_selected_layers': selected_layer_num,
    })
    
    return alignment_result
