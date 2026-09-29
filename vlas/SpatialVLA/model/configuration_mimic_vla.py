# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MIMIC-VLA三模型并行训练包装器配置类

该配置类继承并扩展SpatialVLA配置，添加多模型训练控制参数。
支持灵活地启用/禁用EgoVLPv2和Alignment模型组件。
"""
import os
import warnings
from typing import Optional, List, Union, Dict
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from .configuration_spatialvla import SpatialVLAConfig

logger = logging.get_logger(__name__)


class MIMICVLAConfig(SpatialVLAConfig):
    """
    MIMIC-VLA三模型并行训练包装器的配置类
    
    该配置类继承SpatialVLA的完整配置，并添加了多模型训练的控制参数。
    保持与SpatialVLA的完全兼容性，同时支持EgoVLPv2和Alignment模型的集成。
    
    Args:
        # === SpatialVLA原有参数 (完全继承) ===
        vision_config, text_config, vision_zoe_config: 继承自SpatialVLAConfig
        ignore_index, image_token_index, vocab_size: 继承自SpatialVLAConfig
        projection_dim, hidden_size: 继承自SpatialVLAConfig
        action_token_begin_idx, spatial_token_num, use_spatial_token: 继承自SpatialVLAConfig
        ego3d_patch_reso, n_freqs, use_vision_zoe: 继承自SpatialVLAConfig
        
        # === 多模型控制参数 ===
        use_egovlpv2 (bool, optional): 
            是否启用EgoVLPv2模型进行辅助训练。默认为False。
        use_alignment (bool, optional): 
            是否启用Alignment模型进行特征对齐。默认为False。
        
        # === 损失权重配置 ===
        vlm_loss_weight (float, optional): 
            EgoVLPv2损失的权重。默认为1.0。
        alignment_loss_weight (float, optional): 
            Alignment损失的权重。默认为1.0。
        
        # === VLM模型选择 ===
        vlm_mode (str, optional):
            VLM模型类型。可选值：
            - 'ego'（EgoVLPv2标准模式）
            - 'egohod'（EgoHOD模式）
            - 'qwen3'（Qwen3-Embedding模式）
            - 'embedding'（预计算文本特征模式，使用text-embedding-3-large）
            - 'lt'（EgoVLPv2 with learnable token）
            默认为'ego'。
        
        # === EgoVLPv2配置 ===
        egovlpv2_config_path (str, optional): 
            EgoVLPv2或EgoHOD模型配置文件的路径。
        egovlpv2_task_names (str, optional): 
            EgoVLPv2模型的任务名称。默认为'EgoNCE_ITM_MLM'。仅在vlm_mode='ego'或'lt'时使用。
        
        # === Alignment配置 ===
        alignment_layer_indices (List[int], optional): 
            用于对齐的SpatialVLA层索引。默认为[26]（最后一层）。
        
        # === 运行时配置 ===
        device (str, optional): 
            模型运行设备。默认为'cuda:0'。
        infinite_dataloader (bool, optional): 
            是否使用无限数据加载器用于EgoVLPv2。默认为True。
    """
    
    model_type = "mimic_vla"
    
    def __init__(
        self,
        # === 多模型控制开关 ===
        use_egovlpv2: bool = False,
        use_alignment: bool = False,
        
        # === 损失权重配置 ===
        vlm_loss_weight: float = 1.0,
        alignment_loss_weight: float = 1.0,
        
        # === VLM模型选择 ===
        vlm_mode: str = 'ego',  # 'ego', 'egohod', 'qwen3', 'embedding', 'lt'
        
        # === EgoVLPv2/EgoHOD配置 ===
        egovlpv2_config_path: Optional[str] = "/data/xuyuan/UniVLA_env/mirror_neuron/egovlpv2/egovlpv2/configs/ft/egofho_align.json",
        egovlpv2_task_names: str = 'EgoNCE_ITM_MLM',
        
        # === Alignment配置 ===
        alignment_layer_indices: Optional[List[int]] = None,
        
        # === 运行时配置 ===
        device: str = 'cuda:0',
        infinite_dataloader: bool = True,
        
        # === SpatialVLA原有参数 (完全透传) ===
        **kwargs
    ):
        
        # 先调用父类初始化，保持SpatialVLA的完整功能
        super().__init__(**kwargs)
        
        # === 多模型控制参数 ===
        self.use_egovlpv2 = use_egovlpv2
        self.use_alignment = use_alignment
        
        # === 损失权重 ===
        self.vlm_loss_weight = vlm_loss_weight
        self.alignment_loss_weight = alignment_loss_weight
        
        # === VLM模型选择 ===
        self.vlm_mode = vlm_mode
        
        # === EgoVLPv2/EgoHOD配置 ===
        self.egovlpv2_config_path = egovlpv2_config_path
        self.egovlpv2_task_names = egovlpv2_task_names
        
        # === Alignment配置 ===
        if alignment_layer_indices is None:
            # 默认使用最后一层进行对齐，根据text_config动态确定
            if hasattr(self.text_config, 'num_hidden_layers'):
                last_layer = self.text_config.num_hidden_layers - 1
                self.alignment_layer_indices = [last_layer]
            else:
                # fallback到常见的Gemma2-2B配置
                self.alignment_layer_indices = [26]
        else:
            self.alignment_layer_indices = alignment_layer_indices
            
        # === 运行时配置 ===
        self.device = device
        self.infinite_dataloader = infinite_dataloader
    
    def validate_config(self):
        """
        验证配置的合理性
        
        Returns:
            bool: 配置是否有效
            
        Raises:
            ValueError: 当配置不合理时抛出异常
        """
        # 检查层索引是否合理
        if self.use_alignment:
            max_layers = getattr(self.text_config, 'num_hidden_layers', 18)
            for layer_idx in self.alignment_layer_indices:
                if layer_idx >= max_layers or layer_idx < 0:
                    raise ValueError(
                        f"alignment_layer_indices contains invalid layer index {layer_idx}. "
                        f"Valid range: [0, {max_layers - 1}]"
                    )
        
        # 检查权重是否为正数
        if self.vlm_loss_weight < 0:
            raise ValueError(f"vlm_loss_weight must be non-negative, got {self.vlm_loss_weight}")
        
        if self.alignment_loss_weight < 0:
            raise ValueError(f"alignment_loss_weight must be non-negative, got {self.alignment_loss_weight}")
        
        # 检查配置文件路径 (仅警告，允许运行时配置)
        if self.use_egovlpv2 and self.egovlpv2_config_path:
            import os
            if not os.path.exists(self.egovlpv2_config_path):
                logger.warning(
                    f"EgoVLPv2 config path {self.egovlpv2_config_path} does not exist. "
                    "Please ensure the path is correct when initializing the model."
                )
        
        return True
    
    def to_dict(self):
        """
        将配置转换为字典格式，保持与SpatialVLA兼容
        """
        output = super().to_dict()
        
        # 添加MIMIC-VLA特有配置
        mimic_config = {
            "use_egovlpv2": self.use_egovlpv2,
            "use_alignment": self.use_alignment,
            "vlm_loss_weight": self.vlm_loss_weight,
            "alignment_loss_weight": self.alignment_loss_weight,
            "vlm_mode": self.vlm_mode,
            "egovlpv2_config_path": self.egovlpv2_config_path,
            "egovlpv2_task_names": self.egovlpv2_task_names,
            "alignment_layer_indices": self.alignment_layer_indices,
            "device": self.device,
            "infinite_dataloader": self.infinite_dataloader,
        }
        
        output.update(mimic_config)
        return output
    
    @classmethod
    def from_spatial_vla_config(
        cls, 
        spatial_vla_config: SpatialVLAConfig, 
        use_egovlpv2: bool = False, 
        use_alignment: bool = False,
        **kwargs
    ):
        """
        从现有的SpatialVLA配置创建MIMIC-VLA配置
        
        Args:
            spatial_vla_config (SpatialVLAConfig): 现有的SpatialVLA配置
            use_egovlpv2 (bool): 是否启用EgoVLPv2
            use_alignment (bool): 是否启用Alignment
            **kwargs: 其他MIMIC-VLA配置参数
            
        Returns:
            MIMICVLAConfig: 新的MIMIC-VLA配置实例
        """
        # 提取SpatialVLA的所有配置参数
        spatial_config_dict = spatial_vla_config.to_dict()
        
        # 移除model_type避免冲突
        spatial_config_dict.pop("model_type", None)
        
        # 合并MIMIC-VLA特有参数
        mimic_config = {
            "use_egovlpv2": use_egovlpv2,
            "use_alignment": use_alignment,
            **kwargs
        }
        
        # 创建新配置
        return cls(**spatial_config_dict, **mimic_config)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs
    ):
        """
        从预训练模型路径加载配置，支持从SpatialVLA配置自动转换
        """
        try:
            # 首先尝试直接加载MIMIC-VLA配置
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            # 如果失败，尝试从SpatialVLA配置转换
            logger.info("Loading SpatialVLA config and converting to MIMIC-VLA config")
            spatial_config = SpatialVLAConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
            # 提取MIMIC-VLA特有参数
            mimic_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['use_egovlpv2', 'use_alignment', 'vlm_loss_weight', 
                                   'alignment_loss_weight', 'vlm_mode', 'egovlpv2_config_path', 
                                   'egovlpv2_task_names', 'alignment_layer_indices',
                                   'device', 'infinite_dataloader']}
            
            return cls.from_spatial_vla_config(spatial_config, **mimic_kwargs)
