# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import clip
from easydict import EasyDict
from typing import Optional, Dict, Any

from egovlpv2.base import BaseModel
from egovlpv2.model.egohod.clip import CLIP_VITB16, CLIP_VITL14_336PX
from egovlpv2.model.egohod_peft_lora import (
    validate_lora_config,
    create_lora_config,
    apply_lora_to_text_model,
    print_lora_info,
)


class CLIPModel(BaseModel):
    """
    OpenAI CLIP模型封装，兼容EgoVLPv2接口
    
    该模型直接使用OpenAI CLIP模型，提供与EgoVLPv2相同的特征提取接口：
    - compute_text: 提取文本特征 [B, D]
    - compute_text_tokens: 提取token级文本特征 [B, T, D]
    - compute_video: 提取视频特征 [B, D]
    
    Args:
        video_params: 视频参数配置（用于兼容接口，实际不使用）
        text_params: 文本参数配置（用于兼容接口，实际不使用）
        projection_dim: 统一投影维度（默认512，与CLIP一致）
        model_name: CLIP模型名称（如'ViT-B/16', 'ViT-L/14'等）
        device: 设备（默认'cpu'，会在加载时自动设置）
    """
    
    def __init__(
        self,
        video_params,
        text_params,
        projection_dim=512,
        model_name='ViT-B/16',  # OpenAI CLIP模型名称
        device='cpu',
        **kwargs,
    ):
        """
        初始化 CLIPModel
        
        说明：
        - 直接使用OpenAI CLIP模型（通过clip.load加载）
        - 支持所有OpenAI CLIP预训练模型（ViT-B/16, ViT-L/14等）
        - 文本/视频特征的维度由CLIP模型决定（ViT-B/16: 512维，ViT-L/14: 768维）
        """
        super().__init__()
        
        self.video_params = video_params
        self.text_params = text_params
        self.projection_dim = projection_dim
        self.model_name = model_name
        self.device = device
        
        # 加载OpenAI CLIP模型
        # clip.load返回(model, preprocess)，我们只需要model
        self.clip_model, _ = clip.load(model_name, device=device, jit=False)
        
        # 获取特征维度（从text_projection的维度推断）
        if hasattr(self.clip_model, 'text_projection'):
            self.clip_dim = self.clip_model.text_projection.shape[1]
        else:
            # 如果没有text_projection，使用transformer的width
            self.clip_dim = self.clip_model.transformer.width
        
        # 设置为eval模式（用于特征提取，不训练）
        self.clip_model.eval()
        
        # ✅ 确保所有参数是contiguous的（分布式训练必须）
        for param in self.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    def compute_text(self, text_data):
        """
        文本句子级特征提取接口
        
        参数：
        - text_data: dict，至少包含键 'input_ids'，形状为 [B, L]
        
        返回：
        - text_embeddings: [B, D] 的句子级文本特征（D = self.clip_dim）
        
        说明：
        - 使用CLIP的encode_text方法，提取EOT token的特征
        - 与 EgoVLPv2 的 compute_text 接口保持一致
        """
        input_ids = text_data["input_ids"]
        
        # 确保input_ids与模型在同一设备上
        if input_ids.device != next(self.clip_model.parameters()).device:
            input_ids = input_ids.to(next(self.clip_model.parameters()).device)
        
        # OpenAI CLIP的encode_text返回句子级特征 [B, D]
        text_embeddings = self.clip_model.encode_text(input_ids)
        
        return text_embeddings
    
    def compute_text_tokens(self, text_data):
        """
        文本 token 级别特征提取接口（用于细粒度对齐）
        
        与compute_text的区别：
        1. 返回所有token的特征序列，而不是只返回EOT token
        2. 返回attention mask用于屏蔽padding token
        3. Token特征不经过text_projection（保持原始transformer输出）
        
        参数：
        - text_data: dict，至少包含键 'input_ids'，形状为 [B, L]
          其中每个元素为 CLIP 词表中的 token id
        
        返回：
        - tuple: (text_token_embeddings, attention_mask)
            - text_token_embeddings: [B, L, D] 的token级文本特征（未经过text_projection）
            - attention_mask: [B, L] 的注意力掩码（1=valid, 0=padding）
        
        说明：
        - 从CLIP transformer输出中提取token级特征（在ln_final之前，不经过text_projection）
        - 与EgoVLPv2的compute_text_tokens接口保持一致
        - AlignmentModel会有自己的投影层来处理这些token特征
        """
        input_ids = text_data["input_ids"]
        
        # 确保input_ids与模型在同一设备上
        if input_ids.device != next(self.clip_model.parameters()).device:
            input_ids = input_ids.to(next(self.clip_model.parameters()).device)
        
        # 手动提取token级特征（参考CLIP的encode_text实现）
        # 1. Token embedding + Positional embedding
        x = self.clip_model.token_embedding(input_ids).type(self.clip_model.dtype)  # [B, L, D]
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        
        # 2. Transformer forward
        x = x.permute(1, 0, 2)  # [L, B, D] (NLD -> LND)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, D] (LND -> NLD)
        
        # 3. Layer norm（不应用text_projection）
        # 注意：这里使用ln_final，但不应用text_projection
        # text_token_embeddings: [B, L, transformer.width]
        text_token_embeddings = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        # 4. 生成attention mask（基于padding token，CLIP使用0作为padding）
        # attention_mask: [B, L]，1表示valid token，0表示padding
        attention_mask = (input_ids != 0).long()
        
        return text_token_embeddings, attention_mask
    
    def compute_video(self, video_data):
        """
        视频特征提取接口
        
        参数：
        - video_data: 视频数据张量，形状为 [B, T, C, H, W]
          其中 B=batch size, T=num_frames, C=channels, H=height, W=width
        
        返回：
        - video_embeddings: [B, D] 的视频特征（D = self.clip_dim）
        
        说明：
        - 对于多帧视频，取平均pooling
        - 与 EgoVLPv2 的 compute_video 接口保持一致
        """
        B, T, C, H, W = video_data.shape
        
        # 将视频帧展平为 [B*T, C, H, W]
        video_frames = video_data.view(B * T, C, H, W)
        
        # CLIP的encode_image期望输入 [B*T, C, H, W]
        # 返回 [B*T, D]
        frame_embeddings = self.clip_model.encode_image(video_frames)
        
        # 重新reshape为 [B, T, D]
        frame_embeddings = frame_embeddings.view(B, T, -1)
        
        # 对时间维度取平均，得到 [B, D]
        video_embeddings = frame_embeddings.mean(dim=1)
        
        return video_embeddings
    
    def reset_feature_bank(self):
        """
        重置Feature Bank（用于梯度累积后的清理）
        
        说明：
        - CLIP model不使用feature bank
        - 此方法仅为兼容EgoVLPv2接口而保留
        """
        pass  # CLIP不使用feature bank，无需实现
    
    def save_checkpoint(self, save_path, **kwargs):
        """
        保存CLIP模型checkpoint
        
        参数：
            save_path: 保存路径（.pth文件）
        """
        checkpoint = {
            'arch': 'CLIPModel',
            'state_dict': self.clip_model.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"✓ CLIP模型已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path, **kwargs):
        """
        加载CLIP模型checkpoint
        
        参数：
            checkpoint_path: checkpoint文件路径（.pth文件）
        """
        print(f"=> 加载CLIP checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 移除DDP的module.前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '') if k.startswith('module.') else k] = v
        
        self.clip_model.load_state_dict(new_state_dict, strict=False)
        
        # ✅ 确保所有参数是contiguous的（分布式训练必须）
        for param in self.clip_model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        print(f"✅ CLIP权重加载完成")