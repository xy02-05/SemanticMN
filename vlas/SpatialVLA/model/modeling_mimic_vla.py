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
MIMIC-VLA三模型并行训练包装器实现

该模块实现了SpatialVLA、EgoVLPv2和Alignment模型的统一包装器，
提供与标准HuggingFace Trainer完全兼容的接口。

核心设计原则：
1. 完全兼容SpatialVLA的原有功能和接口
2. 内部封装多模型协调逻辑
3. 保持与HuggingFace生态的标准兼容性
4. 最小化对现有代码的修改需求
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, HybridCache, StaticCache
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput
from easydict import EasyDict as edict

# 导入SpatialVLA相关组件
from .configuration_mimic_vla import MIMICVLAConfig
from .modeling_spatialvla import (
    SpatialVLAPreTrainedModel, 
    SpatialVLAForConditionalGeneration,
    SpatialVLACausalLMOutputWithPast
)
from egovlpv2.utils.model_data_init import (
    init_vlm_components,  # 统一VLM组件初始化函数
    init_alignment_model_components
)
from egovlpv2.utils.model_forward import (
    egovlpv2_forward_pass, 
    egohod_forward_pass,
    alignment_forward_pass_complete
)
from egovlpv2.utils.util import state_dict_data_parallel_fix
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)
@dataclass
class MIMICVLACausalLMOutputWithPast(SpatialVLACausalLMOutputWithPast):
    """
    MIMIC-VLA模型输出，扩展SpatialVLA输出以包含多模型损失信息
    
    继承SpatialVLA的所有输出字段，并添加多模型训练的损失分解信息。
    这样可以保持与现有代码的完全兼容，同时提供额外的调试信息。
    """
    # 继承所有SpatialVLA字段：loss, logits, past_key_values, hidden_states, 
    # attentions, image_hidden_states, action_hidden_states
    
    # 添加多模型损失分解 (用于调试和监控)
    spatial_outputs: Optional[SpatialVLACausalLMOutputWithPast] = None
    spatial_vla_loss: Optional[torch.FloatTensor] = None
    egovlpv2_loss: Optional[torch.FloatTensor] = None
    alignment_loss: Optional[torch.FloatTensor] = None
    alignment_loss_dict: Optional[Dict[str, Any]] = None


class MIMICVLAModel(SpatialVLAPreTrainedModel, GenerationMixin):
    """
    MIMIC-VLA三模型并行训练包装器
    
    该类将SpatialVLA、EgoVLPv2和Alignment模型封装为一个统一的训练单元，
    对外提供与SpatialVLA完全兼容的接口，内部协调三个模型的训练。
    
    核心功能：
    1. 作为SpatialVLA的完全兼容包装器
    2. 内部集成EgoVLPv2和Alignment模型（可选）
    3. 协调三个模型的前向传播和损失计算
    4. 保持所有SpatialVLA的特有功能（action tokenizer、3D感知等）
    
    设计思路：
    - 继承SpatialVLAPreTrainedModel确保完全兼容
    - 主要的SpatialVLA实例通过组合方式集成
    - 辅助模型通过注入方式动态加载
    - forward方法协调所有模型的计算
    """
    
    config_class = MIMICVLAConfig
    
    def __init__(self, config: MIMICVLAConfig, vla_model: SpatialVLAForConditionalGeneration, egovlpv2_components: Optional[Any] = None, alignment_components: Optional[Any] = None, vlm_mode: str = "egovlpv2"):
        """
        初始化MIMIC-VLA包装器
        
        Args:
            config: MIMIC-VLA配置，包含SpatialVLA配置和多模型控制参数
            vla_model: 已初始化的SpatialVLA模型
            egovlpv2_components: VLM组件（可选，如果为None且use_egovlpv2=True会自动初始化）
            alignment_components: Alignment组件（可选，如果为None且use_alignment=True会自动初始化）
            vlm_mode: VLM模型类型 ("egovlpv2", "egohod", "egovideo", "qwen3" 或 "embedding")
        """
        super().__init__(config)
        
        # === 核心组件：SpatialVLA模型 ===
        self.spatial_vla = vla_model
        self.use_egovlpv2 = config.use_egovlpv2
        self.use_alignment = config.use_alignment
        self.vlm_loss_weight = config.vlm_loss_weight
        self.alignment_loss_weight = config.alignment_loss_weight
        self.vlm_mode = vlm_mode  # 保存VLM模型类型

        # 自动初始化EgoVLPv2组件（如果需要且未提供）
        # 注意：即使 use_egovlpv2=False，如果 use_alignment=True 也需要初始化 egovlpv2 组件
        # 因为 alignment 需要使用 egovlpv2_model 进行文本编码
        if self.use_egovlpv2 or self.use_alignment:
            if egovlpv2_components:
                self._setup_egovlpv2_components(egovlpv2_components)
            else:
                self._auto_init_egovlpv2_components()
                
        # 自动初始化Alignment组件（如果需要且未提供）
        if self.use_alignment:
            if alignment_components:
                self._setup_alignment_components(alignment_components)
            else:
                self._auto_init_alignment_components()
        self._distributed_args_updated = False
    
    def _setup_egovlpv2_components(self, egovlpv2_components):
        """
        设置VLM相关组件（EgoVLPv2、EgoHOD、EgoVideo等）
        
        Args:
            egovlpv2_components: VLM组件字典
        
        说明：
        - EgoVLPv2/egohod_cotraining模式：包含model, tokenizer, config, loss_fn, allgather
        - EgoHOD/EgoVideo/Qwen3/Embedding模式：包含model, config, allgather（allgather用于alignment训练）
        """
        self.egovlpv2_model = egovlpv2_components['model']
        self.egovlpv2_config = egovlpv2_components['config']
        
        # allgather是通用分布式工具，与VLM模型类型无关，alignment训练需要
        self.egovlpv2_allgather = egovlpv2_components['allgather']
        
        if self.vlm_mode in ["egovideo", "qwen3", "embedding", "clip"]:
            # freeze VLM模式：不需要tokenizer、loss_fn、task_names（这些只用于VLM训练）
            self.egovlpv2_tokenizer = None
            self.egovlpv2_loss_fn = None
            self.egovlpv2_task_names = None
        elif self.vlm_mode == "egohod":
            # EgoHOD既可能freeze也可能训练，按组件是否包含loss_fn判断
            self.egovlpv2_tokenizer = egovlpv2_components.get('tokenizer', None)
            self.egovlpv2_loss_fn = egovlpv2_components.get('loss_fn', None)
            self.egovlpv2_task_names = 'Dual' if self.egovlpv2_loss_fn is not None else None
        elif self.vlm_mode == "egohod_cotraining":
            # EgoHOD co-training模式：需要完整的训练组件
            self.egovlpv2_tokenizer = egovlpv2_components['tokenizer']
            self.egovlpv2_loss_fn = egovlpv2_components['loss_fn']
            self.egovlpv2_task_names = 'Dual'  # EgoHOD使用Dual任务（视频-文本对比学习）
        else:
            # EgoVLPv2模式：需要完整的训练组件
            self.egovlpv2_tokenizer = egovlpv2_components['tokenizer']
            self.egovlpv2_loss_fn = egovlpv2_components['loss_fn']
            self.egovlpv2_task_names = self.egovlpv2_config['trainer']['task_names']
        
        # 设置默认分布式参数，将在运行时更新
        self.egovlpv2_args = edict()
        
        # 从config中读取LoRA状态（用于save_pretrained时判断保存格式）
        # alignment.lora_config.enabled 表示是否对EgoHOD text encoder应用了LoRA
        self.has_egohod_lora = False
        if 'alignment' in self.egovlpv2_config.config:
            lora_cfg = self.egovlpv2_config.config['alignment'].get('args', {}).get('lora_config', {})
            self.has_egohod_lora = lora_cfg.get('enabled', False)
    
    def _setup_alignment_components(self, alignment_components):
        """
        设置Alignment相关组件，将forward需要的组件赋给self
        
        Args:
            alignment_components: Alignment组件字典，来自init_alignment_model_components
        """
        self.alignment_model = alignment_components['alignment_model']
        self.alignment_config = alignment_components['alignment_config']
        self.alignment_layer_indices = alignment_components['layer_indices']
        # VideoFeatureSampler（由init_alignment_model_components创建，可能为None）
        self.video_sampler = alignment_components.get('video_sampler', None)
        # before_proj: 对齐时是否使用投影前的EgoHOD文本特征（从alignment config中读取）
        self.alignment_before_proj = alignment_components.get('before_proj', False)
        # vla_feature_type: Ablation用，控制使用哪种VLA token进行对齐（默认action）
        self.vla_feature_type = alignment_components.get('vla_feature_type', 'action')
        # backbone_update_start_pct: 前N%步detach backbone特征，只训练alignment head
        # stop_last_pct: 最后N%步关闭alignment loss
        training_cfg = alignment_components.get('training_config', {})
        self.alignment_backbone_update_start_pct = float(training_cfg.get('backbone_update_start_pct', 0.0))
        self.alignment_stop_last_pct = float(training_cfg.get('stop_last_pct', 0.0))
        # 运行时由训练脚本设置
        self._global_step = 0
        self._num_train_steps = None
    
    def _auto_init_egovlpv2_components(self):
        """
        自动初始化VLM组件（统一入口）
        
        说明：调用model_data_init中的统一初始化函数，根据vlm_mode自动选择对应的VLM模型
        """
        # 获取模型dtype以确保一致性
        model_dtype = next(self.spatial_vla.parameters()).dtype
        
        # 调用统一初始化函数（封装了所有vlm_mode的分支逻辑）
        egovlpv2_components = init_vlm_components(
            vlm_mode=self.vlm_mode,
            config_path=self.config.egovlpv2_config_path,
            device=str(self.device),
            dtype=model_dtype,
            infinite_dataloader=True,  # 仅用于egovlpv2模式
            training_mode=self.use_egovlpv2
        )
        
        # 设置组件
        self._setup_egovlpv2_components(egovlpv2_components)
    
    def _auto_init_alignment_components(self):
        """
        自动初始化Alignment组件，使用model_data_init中的初始化函数
        """
        # 获取模型dtype以确保一致性
        model_dtype = next(self.spatial_vla.parameters()).dtype
        
        # 使用配置中的路径
        config_path = self.config.egovlpv2_config_path
        
        # 调用model_data_init的初始化函数
        alignment_components = init_alignment_model_components(
            config_path=config_path,
            device=str(self.device),
            dtype=model_dtype
        )
        
        # 设置组件
        self._setup_alignment_components(alignment_components)
        logger.info("✅ Alignment组件自动初始化完成")
    
    def _update_egovlpv2_distributed_args(self):
        """在运行时更新EgoVLPv2的分布式参数"""
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        # 更新分布式参数
        self.egovlpv2_args.rank = rank
        self.egovlpv2_args.world_size = world_size
        self._distributed_args_updated = True
        logger.info(f"✅ EgoVLPv2分布式参数已更新: rank={rank}, world_size={world_size}")
    
    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.spatial_vla.get_input_embeddings()

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        return self.spatial_vla.set_input_embeddings(value)

    def get_output_embeddings(self):
        """获取输出嵌入层"""
        return self.spatial_vla.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        return self.spatial_vla.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        """设置解码器"""
        return self.spatial_vla.set_decoder(decoder)

    def get_decoder(self):
        """获取解码器"""
        return self.spatial_vla.get_decoder()

    def tie_weights(self):
        """绑定权重"""
        return self.spatial_vla.tie_weights()
    
    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        """调整token嵌入大小"""
        return self.spatial_vla.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
    
    def get_image_features(self, pixel_values: torch.FloatTensor, intrinsic: torch.FloatTensor):
        """获取图像特征"""
        return self.spatial_vla.get_image_features(pixel_values, intrinsic)
    
    def backproject_patch(self, K: torch.Tensor, depth: torch.Tensor, patch_size=14, reso=2) -> torch.Tensor:
        """反投影深度图到3D点云"""
        return self.spatial_vla.backproject_patch(K, depth, patch_size, reso)
    
    @property
    def vocab_size(self):
        """词汇表大小"""
        return self.spatial_vla.vocab_size
    
    @property
    def pad_token_id(self):
        return self.spatial_vla.pad_token_id
    
    @property 
    def action_tokenizer(self):
        """动作分词器"""
        return self.spatial_vla.action_tokenizer
    
    @property
    def action_token_begin_idx(self):
        """动作token起始索引"""
        return self.spatial_vla.action_token_begin_idx
    
    def reset_feature_bank(self):
        """
        重置所有层的feature bank（在optimizer.step()后调用）
        
        该方法在训练循环中，每次optimizer.step()后调用，用于清空feature bank中累积的特征，
        开始新的gradient accumulation周期。
        
        说明：
        - 只在use_alignment=True且alignment模型启用feature bank时有效
        - 训练脚本需要在accelerator.sync_gradients时调用此方法
        - 确保在DDP环境下正确处理（model.module.reset_feature_bank()）
        """
        if self.use_alignment and hasattr(self, 'alignment_model'):
            # 检查alignment_model是否有reset_feature_bank方法（使用feature bank时才有）
            if hasattr(self.alignment_model, 'reset_feature_bank'):
                self.alignment_model.reset_feature_bank()
    
    # ===== 核心前向传播方法 =====
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        actions: Optional[torch.FloatTensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        lang: Optional[List[str]] = None,
        egovlpv2_inputs: Optional[Dict[str, Any]] = None,
        task_index: Optional[torch.Tensor] = None,  # 原始任务索引（与tasks.jsonl对应）
        task_ids: Optional[torch.Tensor] = None,  # 聚类后的任务类别ID
        **kwargs,
    ) -> Union[Tuple, MIMICVLACausalLMOutputWithPast]:
        """
        MIMIC-VLA的核心前向传播方法
        
        协调SpatialVLA、EgoVLPv2和Alignment三个模型的计算，
        聚合损失并返回标准的模型输出格式。
        
        Args:
            vla_inputs: SpatialVLA模型的输入参数字典
            egovlpv2_inputs: EgoVLPv2模型的输入参数字典（可选）
        """
        # 确保返回字典格式和隐藏状态
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True  # 强制设置为True
        
        # 构建SpatialVLA的输入参数
        vla_inputs = {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'actions': actions,
            'intrinsic': intrinsic,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'token_type_ids': token_type_ids,
            'cache_position': cache_position,
            'inputs_embeds': inputs_embeds,
            'labels': labels,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
            'num_logits_to_keep': num_logits_to_keep,
        }
        
        # === 1. SpatialVLA前向传播 (主模型) ===
        spatial_outputs = self._forward_spatial_vla(vla_inputs, **kwargs)

        vla_inputs['lang'] = lang
        
        # 推理时只使用SpatialVLA，训练时使用3个模型
        if not self.training:
            return spatial_outputs
        
        # 提取主模型损失
        spatial_vla_loss = spatial_outputs.loss
        total_loss = spatial_vla_loss
        
        # === 2. EgoVLPv2前向传播 (可选) ===
        egovlpv2_loss = torch.tensor(0.0, device=spatial_vla_loss.device, requires_grad=True)
        results_egovlp = None
        if not self._distributed_args_updated and (self.use_egovlpv2 or self.use_alignment):
            self._update_egovlpv2_distributed_args()
        if self.use_egovlpv2 and self.training:
            # Lazy initialization: 确保组件已初始化
            results_egovlp = self._forward_egovlpv2(egovlpv2_inputs)
            egovlpv2_loss = results_egovlp['loss']
            total_loss = total_loss + self.vlm_loss_weight * egovlpv2_loss
        
        # === 3. Alignment前向传播 (可选) ===
        alignment_loss = torch.tensor(0.0, device=spatial_vla_loss.device, requires_grad=True)
        alignment_loss_dict: Optional[Dict[str, Any]] = None
        results_align = None
        if self.use_alignment and self.training:
            # stop_last_pct: 最后N%步关闭alignment loss
            stop_alignment = self._should_stop_alignment()
            if not stop_alignment:
                results_align = self._forward_alignment(spatial_outputs, vla_inputs, task_index=task_index, task_ids=task_ids)
                alignment_loss = results_align['loss']
                alignment_loss_dict = results_align.get('loss_dict', {})
                # backbone_update_start_pct状态记录（detach已在_forward_alignment中完成）
                detach_backbone = self._should_detach_alignment_backbone()
                if alignment_loss_dict is None:
                    alignment_loss_dict = {}
                alignment_loss_dict['backbone_detached'] = float(detach_backbone)
                alignment_loss_dict['alignment_stopped'] = 0.0
                total_loss = total_loss + self.alignment_loss_weight * alignment_loss
            else:
                alignment_loss_dict = {'alignment_stopped': 1.0, 'backbone_detached': 0.0}
        # === 4. 构造输出 ===
        if not return_dict:
            # 构造包含所有三个模型输出的元组
            output_items = [spatial_outputs.logits]
            
            # 添加SpatialVLA的其他输出（跳过loss）
            if len(spatial_outputs) > 2:
                output_items.extend(spatial_outputs[2:])
                
            # 添加EgoVLPv2输出（如果启用且有输出）
            if self.use_egovlpv2:
                # 添加EgoVLPv2的详细输出
                output_items.append(results_egovlp)
                
            # 添加Alignment输出（如果启用且有输出）
            if self.use_alignment:
                # 添加Alignment的详细信息
                output_items.append(results_align)
            
            output = tuple(output_items)
            return (total_loss,) + output if vla_inputs['labels'] is not None else output
        
        return MIMICVLACausalLMOutputWithPast(
            loss=total_loss if vla_inputs['labels'] is not None else None,
            logits=spatial_outputs.logits,
            past_key_values=spatial_outputs.past_key_values,
            hidden_states=spatial_outputs.hidden_states,
            attentions=spatial_outputs.attentions,
            image_hidden_states=spatial_outputs.image_hidden_states,
            action_hidden_states=spatial_outputs.action_hidden_states,
            vision_hidden_states=spatial_outputs.vision_hidden_states,
            # 损失分解 (调试用)
            spatial_outputs=spatial_outputs,
            spatial_vla_loss=spatial_vla_loss,
            egovlpv2_loss=egovlpv2_loss,
            alignment_loss=alignment_loss,
            alignment_loss_dict=alignment_loss_dict,
        )
    
    # ===== 三个模型的独立前向传播包装函数 =====
    
    def _forward_spatial_vla(self, vla_inputs: Dict[str, Any], **kwargs) -> Any:
        """
        SpatialVLA模型的前向传播包装函数
        
        Args:
            vla_inputs: SpatialVLA的输入参数字典
            
        Returns:
            SpatialVLA的输出结果
        """
        # 直接使用**vla_inputs展开所有参数，简化代码
        return self.spatial_vla(**vla_inputs, **kwargs)
    
    def _forward_egovlpv2(self, egovlpv2_inputs: Dict[str, Any]) -> torch.FloatTensor:
        """
        VLM模型的前向传播包装函数
        
        说明：
        - 是否训练VLM由 use_egovlpv2 参数控制（在forward中判断）
        - egohod模式：调用egohod_forward_pass
        - egovlpv2模式：调用egovlpv2_forward_pass
        - 其他模式（egovideo/qwen3/embedding/clip）：返回零loss（这些模式暂不支持训练）
        """
        # EgoHOD模式：接口与EgoVLPv2对齐（ClipLoss内部处理分布式gather）
        if self.vlm_mode == "egohod":
            return egohod_forward_pass(
                model=self.egovlpv2_model,
                vlm_batch=egovlpv2_inputs,
                tokenizer=self.egovlpv2_tokenizer,
                config=self.egovlpv2_config,
                args=self.egovlpv2_args,
                loss_fn=self.egovlpv2_loss_fn,
                allgather_fn=self.egovlpv2_allgather,
                device_id=self.device,
                task_names='Dual'
            )
        
        # EgoVLPv2模式：正常执行前向传播
        if self.vlm_mode == "egovlpv2":
            return egovlpv2_forward_pass(
                model=self.egovlpv2_model,
                vlm_batch=egovlpv2_inputs,
                tokenizer=self.egovlpv2_tokenizer,
                config=self.egovlpv2_config,
                args=self.egovlpv2_args,
                loss_fn=self.egovlpv2_loss_fn,
                allgather_fn=self.egovlpv2_allgather,
                device_id=self.device,
                task_names=self.egovlpv2_task_names
            )
        
        # 其他模式暂不支持训练
        return {'loss': torch.tensor(0.0, device=self.device)}
    
    def _forward_alignment(
        self,
        spatial_outputs: Any,
        vla_inputs: Dict[str, Any],
        task_index: Optional[torch.Tensor] = None,  # 原始任务索引
        task_ids: Optional[torch.Tensor] = None  # 聚类后的任务类别ID
    ) -> Dict[str, Any]:
        """
        Alignment模型的前向传播包装函数
        
        Args:
            spatial_outputs: SpatialVLA的输出结果
            vla_inputs: VLA输入参数字典，alignment从中获取所需内容
            task_index: 原始任务索引
            task_ids: 聚类后的任务类别ID
            video_cls_features: [B, D] 预提取的video CLS特征
            video_frame_features: [B, F, D] 预提取的video帧级特征
            pair_is_positive: [B] 是否为正样本
            pair_similarity: [B] pair相似度
            
        Returns:
            Alignment的损失值
        
        说明：
        - 根据vlm_mode选择文本编码后端（EgoVLPv2、EgoHOD、EgoVideo等）
        """
        
        # 使用导入的alignment_forward_pass_complete函数
        # video_sampler由init_alignment_model_components创建，采样逻辑在forward_pass_complete内部完成
        # enable_grad: co-training（use_egovlpv2=True）时允许alignment loss梯度流回VLM
        #              freeze VLM（use_egovlpv2=False）时使用no_grad节省显存
        # backbone_update_start_pct: 前N%步detach backbone特征
        if self._should_detach_alignment_backbone():
            import copy
            spatial_outputs = copy.copy(spatial_outputs)
            if hasattr(spatial_outputs, 'action_hidden_states') and spatial_outputs.action_hidden_states is not None:
                spatial_outputs.action_hidden_states = spatial_outputs.action_hidden_states.detach()
        result = alignment_forward_pass_complete(
            alignment_model=self.alignment_model,
            vla_output=spatial_outputs,
            vla_batch=vla_inputs,
            vla_tokenizer=self.spatial_vla.action_tokenizer.tokenizer,
            egovlpv2_model=self.egovlpv2_model,
            egovlpv2_tokenizer=self.egovlpv2_tokenizer,
            layer_indices=self.alignment_layer_indices,
            device_id=self.device,
            allgather_fn=self.egovlpv2_allgather,
            args=self.egovlpv2_args,
            n_gpu=self.egovlpv2_args.world_size,
            task_index=task_index,
            task_ids=task_ids,
            mode=self.vlm_mode,
            video_sampler=self.video_sampler,  # 传入sampler实例，采样在内部完成
            enable_grad=self.use_egovlpv2,     # VLM训练时允许梯度流过
            before_proj=self.alignment_before_proj,  # 对齐时是否使用投影前特征
            # Ablation参数：控制使用哪种VLA token进行对齐
            vla_feature_type=getattr(self, 'vla_feature_type', 'action'),
            image_token_index=self.config.image_token_index,
        )

        
        return result
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        intrinsic=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        """为生成准备输入，委托给SpatialVLA"""
        return self.spatial_vla.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            intrinsic=intrinsic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            labels=labels,
            **kwargs,
        )
    
    @torch.no_grad()
    def predict_action(self, model_inputs) -> torch.Tensor:
        """预测动作，委托给SpatialVLA"""
        return self.spatial_vla.predict_action(model_inputs)
    
    def reset_feature_bank(self):
        """重置对齐/文本模型中的feature bank，保持与OpenPI一致的训练节奏"""
        if self.use_alignment and hasattr(self, 'alignment_model'):
            alignment_model = self.alignment_model
            if hasattr(alignment_model, 'module'):
                alignment_model = alignment_model.module
            if hasattr(alignment_model, 'reset_feature_bank'):
                alignment_model.reset_feature_bank()
        if self.use_egovlpv2 and hasattr(self, 'egovlpv2_model'):
            egovlpv2_model = self.egovlpv2_model
            if hasattr(egovlpv2_model, 'module'):
                egovlpv2_model = egovlpv2_model.module
            if hasattr(egovlpv2_model, 'reset_feature_bank'):
                egovlpv2_model.reset_feature_bank()
    

    def _should_detach_alignment_backbone(self) -> bool:
        if self.alignment_backbone_update_start_pct <= 0:
            return False
        if self._num_train_steps is None:
            return False
        detach_until = int(self._num_train_steps * self.alignment_backbone_update_start_pct + 0.5)
        return self._global_step < detach_until

    def _should_stop_alignment(self) -> bool:
        if self.alignment_stop_last_pct <= 0:
            return False
        if self._num_train_steps is None:
            return False
        stop_start = int(self._num_train_steps * (1.0 - self.alignment_stop_last_pct))
        return self._global_step >= stop_start

    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        # 获取SpatialVLA的设备
        return next(self.spatial_vla.parameters()).device

    # ===== SpatialVLA兼容的类方法 =====
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        *model_args,
        egovlpv2_components: Optional[Any] = None,
        alignment_components: Optional[Any] = None,
        **kwargs,
    ):
        """
        从预训练模型加载MIMIC-VLA包装器
        
        Args:
            pretrained_model_name_or_path: 预训练模型路径
            egovlpv2_components: EgoVLPv2组件（可选）
            alignment_components: Alignment组件（可选）
            **kwargs: 其他参数传递给SpatialVLA
        """
        # 首先加载SpatialVLA模型
        spatial_vla_model = SpatialVLAForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        
        # 加载MIMIC配置
        config = kwargs.get('config')
        if config is None:
            from .configuration_mimic_vla import MIMICVLAConfig
            config = MIMICVLAConfig.from_pretrained(pretrained_model_name_or_path)
        
        # 注意：即使 use_egovlpv2=False，如果 use_alignment=True 也需要初始化 egovlpv2 组件
        # 因为 alignment 需要使用 egovlpv2_model 进行文本编码
        if config.use_egovlpv2 or config.use_alignment:
            # 获取VLM模式
            vlm_mode = kwargs.get('vlm_mode', 'egovlpv2')
            
            # 使用统一初始化函数（调用model_data_init中的封装函数）
            egovlpv2_components = init_vlm_components(
                vlm_mode=vlm_mode,
                config_path=config.egovlpv2_config_path,
                device=str(spatial_vla_model.device),
                dtype=torch_dtype,
                infinite_dataloader=False  # from_pretrained不需要infinite
            )
            
            # 检查是否存在预训练的VLM权重，如果存在则使用模型统一接口加载
            egovlpv2_checkpoint_path = os.path.join(pretrained_model_name_or_path, "egovlpv2_model.pth")
            if os.path.exists(egovlpv2_checkpoint_path):
                vlm_model = egovlpv2_components['model']
                if hasattr(vlm_model, 'load_checkpoint'):
                    vlm_model.load_checkpoint(egovlpv2_checkpoint_path, load_lora_adapter=True)
                    logger.info(f"✅ 已加载预训练{vlm_mode.upper()}权重: {egovlpv2_checkpoint_path}")
                else:
                    logger.warning(f"⚠️ VLM模型不支持load_checkpoint接口，使用随机初始化")
            else:
                logger.info(f"⚠️ 未找到预训练{vlm_mode.upper()}权重，使用随机初始化")
        if config.use_alignment:
            # 检查是否存在预训练的Alignment权重
            alignment_weight_path = os.path.join(pretrained_model_name_or_path, "alignment_model.bin")
            if os.path.exists(alignment_weight_path):
                # 先初始化组件，然后加载权重
                alignment_components = init_alignment_model_components(config.egovlpv2_config_path, device=str(spatial_vla_model.device), dtype=torch_dtype)
                # 加载预训练权重
                alignment_state_dict = torch.load(alignment_weight_path, map_location=spatial_vla_model.device)
                alignment_components['alignment_model'].load_state_dict(alignment_state_dict)
                logger.info(f"✅ 已加载预训练Alignment权重: {alignment_weight_path}")
            else:
                # 回退到训练时初始化
                alignment_components = init_alignment_model_components(config.egovlpv2_config_path, device=str(spatial_vla_model.device), dtype=torch_dtype)
                logger.info("⚠️ 未找到预训练Alignment权重，使用随机初始化")
        
        # 创建MIMIC-VLA包装器
        vlm_mode = kwargs.get('vlm_mode', 'egovlpv2')
        model = cls(
            config=config,
            vla_model=spatial_vla_model,
            egovlpv2_components=egovlpv2_components,
            alignment_components=alignment_components,
            vlm_mode=vlm_mode
        )
        
        return model
    
    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs
    ):
        """
        保存MIMIC-VLA模型的预训练权重，包含三个子模型
        
        Args:
            save_directory: 保存目录路径
            safe_serialization: 是否使用安全序列化（safetensors格式）
            **kwargs: 其他传递给SpatialVLA save_pretrained的参数
        """
        # 创建保存目录
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存SpatialVLA模型（使用其原生方法）
        self.spatial_vla.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)
        logger.info(f"✅ 已保存SpatialVLA模型到: {save_directory}")
        
        # 2. 保存VLM模型权重（如果存在）- 使用模型统一接口
        if hasattr(self, 'egovlpv2_model') and self.egovlpv2_model is not None:
            actual_model = self.egovlpv2_model.module if hasattr(self.egovlpv2_model, 'module') else self.egovlpv2_model
            
            # 调用模型自己的save_checkpoint方法（各模型已实现统一接口）
            egovlpv2_checkpoint_path = os.path.join(save_directory, "egovlpv2_model.pth")
            
            # 对于EgoHOD，自动保存LoRA adapter；其他模型使用默认保存逻辑
            if hasattr(actual_model, 'save_checkpoint'):
                actual_model.save_checkpoint(egovlpv2_checkpoint_path, save_lora_adapter=True)
                logger.info(f"✅ 已保存VLM模型到: {egovlpv2_checkpoint_path}")
            else:
                logger.warning(f"⚠️ VLM模型不支持save_checkpoint接口，跳过保存")
        
        # 3. 保存Alignment模型权重（如果存在）
        if self.use_alignment and hasattr(self, 'alignment_model'):
            alignment_weight_path = os.path.join(save_directory, "alignment_model.bin")
            torch.save(self.alignment_model.state_dict(), alignment_weight_path)
            logger.info(f"✅ 已保存Alignment权重到: {alignment_weight_path}")
        
        # 4. 保存MIMIC配置（覆盖SpatialVLA的config.json）
        config_path = os.path.join(save_directory, "config.json")
        self.config.save_pretrained(save_directory)
        logger.info(f"✅ 已保存MIMIC-VLA配置到: {config_path}")
        
        logger.info(f"🎉 MIMIC-VLA模型完整保存完成: {save_directory}")