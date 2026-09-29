# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
EgoVideo 模型适配器 - 简化版
直接使用 EgoVideo 原生模型，最小化改动
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from egovlpv2.base import BaseModel
from egovlpv2.model.egovideo.setup_model import build_model


class EgoVideoWrapper(BaseModel):
    """
    EgoVideo 模型的 egovlpv2 适配器 - 直接使用原生 build_model
    
    参数:
        video_params (dict): 视频参数配置
            - num_frames (int): 视频帧数, 默认 4
        text_params (dict): 文本参数配置 (兼容性保留)
        projection_dim (int): 投影维度, 默认 512
        load_checkpoint (str): 预训练权重路径
    """
    
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=512,
                 load_checkpoint=None,
                 **kwargs):
        super().__init__()
        
        self.video_params = video_params
        self.text_params = text_params
        self.projection_dim = projection_dim
        
        # 从 video_params 获取帧数
        num_frames = video_params.get('num_frames', 4)
        
        # 直接使用 EgoVideo 原生的 build_model
        self.model, self.tokenizer = build_model(
            embed_dim=projection_dim,
            ckpt_path=load_checkpoint,
            num_frames=num_frames,
            vision_width=768,
            text_width=1024
        )
        
        self.model.train()
        
        # ✅ 确保所有参数是contiguous的（分布式训练必须）
        for param in self.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    def compute_video(self, video_data):
        """
        计算视频特征 - egovlpv2 格式转 EgoVideo 格式
        
        参数:
            video_data: torch.Tensor [B, T, C, H, W] - egovlpv2 格式
        
        返回:
            video_embeds: torch.Tensor [B, D] - L2归一化的视频特征
        """
        # 格式转换: [B, T, C, H, W] -> [B, C, T, H, W]
        video_data = video_data.permute(0, 2, 1, 3, 4)
        
        # EgoVideo encode_image 返回未归一化的特征
        video_embeds = self.model.encode_image(video_data)
        
        # 手动归一化
        video_embeds = F.normalize(video_embeds, dim=-1)
        
        return video_embeds
    
    def compute_text(self, text_data):
        """
        计算文本特征（句子级）
        
        参数:
            text_data: dict 包含 input_ids 和 attention_mask
        
        返回:
            text_embeds: torch.Tensor [B, D] - L2归一化的文本特征
        """
        input_ids = text_data['input_ids']           # [B, seq_len]
        attention_mask = text_data['attention_mask']  # [B, seq_len]
        
        # EgoVideo encode_text 需要 [B, 1, seq_len]
        input_ids = input_ids.unsqueeze(1)
        
        # EgoVideo encode_text 返回未归一化的特征
        text_embeds = self.model.encode_text(input_ids, attention_mask)
        
        return text_embeds
    
    def compute_text_tokens(self, text_data):
        """
        计算文本 token 级别特征（用于细粒度对齐）
        
        参数:
            text_data: dict 包含 input_ids 和 attention_mask
        
        返回:
            tuple: (text_token_embeds, attention_mask)
                - text_token_embeds: [B, L-1, D] token级文本特征（剔除[CLS]）
                - attention_mask: [B, L-1] 注意力掩码（剔除[CLS]位置）
        
        说明:
            - EgoVideo 的 BERT 输出维度为 1024 (text_width)
            - Token特征不经过 text_projection，保持原始 BERT 输出
            - BERT的last_hidden_state包含[CLS] token（位置0）
            - 为了与EgoHOD保持一致，需要剔除[CLS] token
            - 返回的是实际内容token的特征（位置1开始）
            - Mask中1=valid content token, 0=padding/[SEP]
        """
        input_ids = text_data['input_ids']           # [B, seq_len]
        attention_mask = text_data['attention_mask']  # [B, seq_len]
        
        # EgoVideo 的 textual 就是 BERT text_encoder
        # 调用 BERT 的 forward 获取所有 token 的特征
        text_encoder = self.model.textual
        
        # BERT forward: 输出包含 last_hidden_state
        # last_hidden_state: [B, seq_len, text_width=1024]
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            mode="text"
        )
        
        # 获取所有 token 的特征（包含[CLS]）
        text_token_embeds = outputs.last_hidden_state  # [B, seq_len, 1024]
        
        # ⚠️ 剔除 [CLS] token（位置0）
        # BERT的token序列：[CLS] token1 token2 ... tokenN [SEP] [PAD] [PAD] ...
        # 我们只需要 token1 到 [SEP] 之间的内容token
        text_token_embeds = text_token_embeds[:, 1:, :]  # [B, seq_len-1, 1024]
        attention_mask = attention_mask[:, 1:]  # [B, seq_len-1]
        
        return text_token_embeds, attention_mask
    
    def reset_feature_bank(self):
        """
        重置 Feature Bank（兼容接口）
        
        说明:
            - EgoVideo 模型不使用 feature bank
            - 此方法仅为兼容 egovlpv2 接口而保留
        """
        pass  # EgoVideo 不使用 feature bank
    
    def save_checkpoint(self, save_path, **kwargs):
        """
        保存EgoVideo模型checkpoint
        
        参数：
            save_path: 保存路径（.pth文件）
        """
        checkpoint = {
            'arch': 'EgoVideoWrapper',
            'state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"✓ EgoVideo模型已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path, **kwargs):
        """
        加载EgoVideo模型checkpoint
        
        参数：
            checkpoint_path: checkpoint文件路径（.pth文件）
        """
        print(f"=> 加载EgoVideo checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 移除DDP的module.前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '') if k.startswith('module.') else k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # ✅ 确保所有参数是contiguous的（分布式训练必须）
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        print(f"✅ EgoVideo权重加载完成")
    
    def infer(self, data, video_only=False, return_embeds=True, task_names=None, ret={}):
        """推理接口 - 兼容 egovlpv2"""
        video_data = data['video']
        video_embeds = self.compute_video(video_data)
        
        if return_embeds:
            ret.update({'video_embeds': video_embeds})
        
        if not video_only:
            text_data = data['text']
            text_embeds = self.compute_text(text_data)
            if return_embeds:
                ret.update({'text_embeds': text_embeds})
        
        return ret
    
    def forward(self, data, allgather=None, n_gpu=None, args=None, config=None, 
                loss_dual=None, gpu=None, return_embeds=True, task_names='Dual', 
                dataset_name=None):
        """前向传播 - 兼容 egovlpv2 训练接口"""
        ret = {}
        loss_dict = {}
        
        if 'Dual' in task_names:
            ret = self.infer(data, task_names='Dual', return_embeds=True)
            video_embeds = ret['video_embeds']
            text_embeds = ret['text_embeds']
            
            # 分布式训练
            if allgather is not None and n_gpu is not None and n_gpu > 1:
                video_embeds = allgather(video_embeds, n_gpu, args)
                text_embeds = allgather(text_embeds, n_gpu, args)
            
            batch_size = video_embeds.shape[0]
            
            # 计算对比学习损失
            if loss_dual is not None:
                logits = torch.matmul(video_embeds, text_embeds.T)
                labels = torch.arange(batch_size, device=video_embeds.device)
                
                loss_v2t = loss_dual(logits, labels)
                loss_t2v = loss_dual(logits.T, labels)
                
                dual_loss = (loss_v2t + loss_t2v) / 2
                loss_dict['dual_loss'] = dual_loss
                
                ret.update({
                    'logit_scale': 1.0,
                    'similarity_matrix': logits
                })
        
        return ret, loss_dict


# 兼容性别名
FrozenInTimeEgoVideo = EgoVideoWrapper
