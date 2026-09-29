"""
Feature Extractor - 特征提取器

复用EgoHOD和SpatialVLA提取特征，用于CKA分析和alignment实验。
冻结所有主模型参数，只提取特征。

特征格式:
- EgoHOD text: [B, D_text] 句子级特征
- SpatialVLA action: [layer_num, B, A, D_action] 多层action token
"""

import torch
import torch.nn as nn
import clip
from typing import Dict, List, Optional, Tuple


class EgoHODTextExtractor:
    """
    EgoHOD文本特征提取器
    
    复用EgoHOD的compute_text方法提取文本特征。
    模型参数冻结，只做推理。
    """
    
    def __init__(
        self,
        model_path: str = 'ViT-L/14@336px',
        checkpoint_path: str = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            model_path: CLIP模型名称
            checkpoint_path: EgoHOD预训练权重路径
            device: 设备
        """
        from egovlpv2.model.model_egohod import EgoHODModel
        
        self.device = device
        self.dtype = dtype
        
        # 创建EgoHOD模型
        self.model = EgoHODModel(
            video_params={'model': 'SpaceTimeTransformer', 'num_frames': 4},
            text_params={'model': 'clip'},
            projection_dim=512,
            load_checkpoint=model_path,
            project_embed_dim=512,
            num_frames=4,
            freeze_temperature=True,
            egohod_checkpoint_path=checkpoint_path,
        )
        
        self.model = self.model.to(dtype).to(device)
        self.model.eval()
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def extract(self, texts: List[str]) -> torch.Tensor:
        """
        提取文本特征
        
        Args:
            texts: 文本列表
            
        Returns:
            [B, D] 文本特征
        """
        # CLIP tokenize
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        
        # 提取特征
        text_features = self.model.compute_text({'input_ids': text_tokens})
        
        return text_features
    
    @torch.no_grad()
    def extract_tokens(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取token级特征（用于细粒度分析）
        
        Returns:
            (text_tokens, attention_mask): [B, T, D], [B, T]
        """
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        token_features, attention_mask = self.model.compute_text_tokens({'input_ids': text_tokens})
        return token_features, attention_mask


class SpatialVLAActionExtractor:
    """
    SpatialVLA Action特征提取器
    
    提取SpatialVLA各层的action hidden states。
    模型参数冻结，只做推理。
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            model_path: SpatialVLA模型路径
            device: 设备
        """
        from model import SpatialVLAForConditionalGeneration, SpatialVLAProcessor
        
        self.device = device
        self.dtype = dtype
        
        # 加载模型和处理器
        self.processor = SpatialVLAProcessor.from_pretrained(model_path)
        self.model = SpatialVLAForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def extract(
        self,
        batch: Dict,
        layer_indices: List[int] = None,
    ) -> torch.Tensor:
        """
        提取action hidden states
        
        Args:
            batch: VLA batch，需要包含pixel_values, input_ids, labels等
            layer_indices: 要提取的层索引
            
        Returns:
            [B, L, A, D] 多层action特征
        """
        # 确保输出hidden states
        outputs = self.model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 获取action hidden states: [layer_num, B, A, D]
        action_hidden_states = outputs.action_hidden_states
        
        if action_hidden_states is None:
            raise ValueError("Model did not return action_hidden_states. Check if labels are provided.")
        
        # 选择指定层
        if layer_indices is not None:
            action_hidden_states = action_hidden_states[layer_indices]
        
        # 转换为 [B, L, A, D] 格式
        action_features = action_hidden_states.permute(1, 0, 2, 3)
        
        return action_features


class FeatureExtractor:
    """
    统一的特征提取器
    
    封装EgoHOD和SpatialVLA的特征提取。
    """
    
    def __init__(
        self,
        egohod_model_path: str = 'ViT-L/14@336px',
        egohod_checkpoint_path: str = None,
        spatialvla_model_path: str = None,
        device: str = 'cuda',
    ):
        """
        Args:
            egohod_model_path: EgoHOD模型名称
            egohod_checkpoint_path: EgoHOD预训练权重
            spatialvla_model_path: SpatialVLA模型路径
            device: 设备
        """
        self.device = device
        
        # 初始化EgoHOD提取器
        self.text_extractor = EgoHODTextExtractor(
            model_path=egohod_model_path,
            checkpoint_path=egohod_checkpoint_path,
            device=device,
        )
        
        # 初始化SpatialVLA提取器（如果提供了路径）
        self.action_extractor = None
        if spatialvla_model_path is not None:
            self.action_extractor = SpatialVLAActionExtractor(
                model_path=spatialvla_model_path,
                device=device,
            )
    
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """提取文本特征 [B, D]"""
        return self.text_extractor.extract(texts)
    
    def extract_action_features(
        self, 
        batch: Dict, 
        layer_indices: List[int] = None,
    ) -> torch.Tensor:
        """提取action特征 [B, L, A, D]"""
        if self.action_extractor is None:
            raise ValueError("SpatialVLA extractor not initialized")
        return self.action_extractor.extract(batch, layer_indices)
