# coding=utf-8
"""
OpenPI对齐训练包装器

该模块实现了OpenPI、EgoVLPv2和Alignment模型的统一包装器，
参考SpatialVLA的modeling_mimic_vla.py设计。

核心设计原则：
1. 保持PI0Pytorch的原有功能和接口不变
2. 通过组合方式集成多模型协调逻辑
3. 最小化对训练脚本的修改需求
4. 完全兼容原有训练流程（use_alignment=False时）
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

# 导入基础PI0Pytorch模型
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

# 导入EgoVLPv2相关工具
from egovlpv2.utils.model_data_init import (
    init_egovlpv2_training_components,
    init_egohod_training_components,
    init_qwen3_components,
    init_embedding_components,
    init_alignment_model_components,
)
from egovlpv2.utils.model_forward import (
    egovlpv2_forward_pass,
    egohod_forward_pass,  # EgoHOD训练前向传播（与SpatialVLA对齐）
    alignment_forward_pass
)
from egovlpv2.utils.text_feature_encode import (
    extract_egovlpv2_text_embeddings,
    extract_egovlpv2_text_token_embeddings,
    extract_egohod_text_embeddings,
    extract_egohod_text_token_embeddings,
    extract_qwen3_text_embeddings,
    extract_qwen3_text_token_embeddings,
    extract_embedding_text_embeddings,
    extract_embedding_text_embeddings_by_index,
    extract_embedding_text_token_embeddings,
    extract_embedding_text_token_embeddings_by_index,
)

logger = logging.getLogger(__name__)


def convert_openpi_to_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    将范围在[-1, 1]的图像张量转换为符合ImageNet标准化参数的图像
    
    转换步骤：
    1. 将[-1, 1]范围转换为[0, 1]范围
    2. 使用ImageNet的均值和标准差进行标准化处理
    
    参数:
        x: 输入图像张量，形状为 (B, C, H, W)，像素值范围为[-1, 1]
        
    返回:
        符合ImageNet标准化参数的图像张量，形状保持不变
    """
    # 步骤1：将[-1, 1]转换为[0, 1]
    # 公式：x_01 = (x + 1) / 2
    x_01 = (x + 1.0) / 2.0
    
    # 步骤2：应用ImageNet标准化
    # ImageNet均值和标准差（RGB通道）
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    
    # 调整形状以匹配输入张量 (1, C, 1, 1)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    
    # 标准化计算
    x_imagenet = (x_01 - mean) / std
    
    return x_imagenet


@dataclass
class PI0AlignOutput:
    """
    OpenPI对齐训练输出，包含多模型损失信息
    
    类似SpatialVLA的MIMICVLACausalLMOutputWithPast
    """
    total_loss: torch.FloatTensor
    pi0_loss: torch.FloatTensor
    egovlpv2_loss: Optional[torch.FloatTensor] = None
    alignment_loss: Optional[torch.FloatTensor] = None
    action_hidden_states: Optional[torch.FloatTensor] = None


class PI0PytorchAlign(nn.Module):
    """
    OpenPI对齐训练包装器
    
    该类将PI0Pytorch、EgoVLPv2和Alignment模型封装为一个统一的训练单元，
    对外提供与PI0Pytorch兼容的接口，内部协调三个模型的训练。
    
    核心功能：
    1. 作为PI0Pytorch的完全兼容包装器
    2. 内部集成EgoVLPv2和Alignment模型（可选）
    3. 协调三个模型的前向传播和损失计算
    
    设计思路：
    - 通过组合方式集成PI0Pytorch（不修改原代码）
    - 辅助模型通过注入方式动态加载
    - forward方法协调所有模型的计算
    """
    
    def __init__(
        self, 
        config,
        use_egovlpv2: bool = False,
        use_alignment: bool = False,
        egovlpv2_config_path: Optional[str] = None,
        vlm_loss_weight: float = 1.0,
        alignment_loss_weight: float = 1.0,
        freeze_vlm: bool = False,
        vlm_mode: str = "egovlpv2"
    ):
        """
        初始化OpenPI对齐训练包装器
        
        Args:
            config: PI0Config配置
            use_egovlpv2: 是否启用VLM训练（EgoVLPv2、EgoHOD、LIV或Qwen3）
            use_alignment: 是否启用对齐训练
            egovlpv2_config_path: VLM配置文件路径
            vlm_loss_weight: VLM损失权重
            alignment_loss_weight: 对齐损失权重
            freeze_vlm: 是否冻结VLM模型参数
            vlm_mode: VLM模型类型 ("egovlpv2", "egohod", "liv" 或 "qwen3")
        """
        super().__init__()
        
        # === 核心组件：PI0Pytorch模型 ===
        self.pi0_pytorch = PI0Pytorch(config)
        self.config = config
        
        # 对齐训练配置
        self.use_egovlpv2 = use_egovlpv2
        self.use_alignment = use_alignment
        self.vlm_loss_weight = vlm_loss_weight
        self.alignment_loss_weight = alignment_loss_weight
        self.freeze_vlm = freeze_vlm
        self.vlm_mode = vlm_mode  # 保存VLM模型类型
        
        # EgoVLPv2/EgoHOD和Alignment组件
        self.egovlpv2_components = None
        self.alignment_components = None
        self._distributed_args_updated = False
        # 前若干训练步内，仅让 alignment 头更新；0 表示从头到尾都允许 backbone 接收 alignment 梯度。
        self.alignment_backbone_update_start_pct = 0.0
        # 后若干训练步内，直接关闭 alignment loss；0 表示从头到尾都保留 alignment。
        self.alignment_stop_last_pct = 0.0
        
        # 初始化对齐训练组件
        if (use_egovlpv2 or use_alignment) and egovlpv2_config_path:
            self._init_alignment_components(egovlpv2_config_path)
    
    def _init_alignment_components(self, egovlpv2_config_path: str):
        """
        初始化VLM和Alignment组件
        
        说明：
        - 根据vlm_mode选择初始化EgoVLPv2、EgoHOD或LIV
        - EgoHOD和LIV模式下不需要tokenizer、loss_fn等训练组件
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16 if hasattr(self.config, 'dtype') and self.config.dtype == 'bfloat16' else torch.float32
        
        # 初始化VLM组件（EgoVLPv2、EgoHOD或LIV）
        if self.use_egovlpv2 or self.use_alignment:
            if self.vlm_mode == "egohod":
                # EgoHOD模式：根据freeze_vlm决定是freeze模式还是training模式
                # training_mode=True时：初始化dataloader、ClipLoss，支持co-training
                # training_mode=False时：仅用于特征提取（freeze模式）
                # 参考SpatialVLA的_setup_egovlpv2_components实现
                egohod_training_mode = not self.freeze_vlm
                logger.info(f"🔧 初始化EgoHOD组件（training_mode={egohod_training_mode}）...")
                self.egovlpv2_components = init_egohod_training_components(
                    config_path=egovlpv2_config_path,
                    device=device,
                    infinite_dataloader=True,  # co-training需要无限迭代
                    dtype=dtype,
                    training_mode=egohod_training_mode  # 关键：由freeze_vlm控制
                )
                
                # 提取关键组件（通用）
                self.egovlpv2_model = self.egovlpv2_components['model']
                self.egovlpv2_config = self.egovlpv2_components['config']
                self.egovlpv2_allgather = self.egovlpv2_components['allgather']
                
                # EgoHOD的CLIP tokenizer（training和freeze模式都提供）
                self.egovlpv2_tokenizer = self.egovlpv2_components.get('tokenizer', None)
                
                if egohod_training_mode:
                    # 训练模式：提取ClipLoss和dataloader
                    self.egovlpv2_loss_fn = self.egovlpv2_components['loss_fn']
                    self.egovlpv2_task_names = None  # EgoHOD不使用task_names
                    logger.info("✅ EgoHOD组件初始化完成（Co-training模式：VLM可训练）")
                else:
                    # Freeze模式：不需要loss_fn和dataloader
                    self.egovlpv2_loss_fn = None
                    self.egovlpv2_task_names = None
                    logger.info("✅ EgoHOD组件初始化完成（Freeze VLM + Alignment训练模式）")
            elif self.vlm_mode == "qwen3":
                logger.info("🔧 初始化Qwen3-Embedding推理组件...")
                self.egovlpv2_components = init_qwen3_components(
                    config_path=egovlpv2_config_path,
                    device=device,
                    dtype=dtype
                )
                
                # 提取关键组件
                self.egovlpv2_model = self.egovlpv2_components['model']
                self.egovlpv2_config = self.egovlpv2_components['config']
                
                # Qwen3模式下不训练VLM，但alignment训练需要allgather
                self.egovlpv2_allgather = self.egovlpv2_components['allgather']
                
                # Qwen3不需要tokenizer、loss_fn、task_names（只用于特征提取）
                self.egovlpv2_tokenizer = None
                self.egovlpv2_loss_fn = None
                self.egovlpv2_task_names = None
                
                logger.info("✅ Qwen3-Embedding组件初始化完成（Freeze VLM + Alignment训练模式）")
            elif self.vlm_mode == "embedding":
                logger.info("🔧 初始化预计算Embedding推理组件...")
                self.egovlpv2_components = init_embedding_components(
                    config_path=egovlpv2_config_path,
                    device=device,
                    dtype=dtype
                )

                self.egovlpv2_model = self.egovlpv2_components['model']
                self.egovlpv2_config = self.egovlpv2_components['config']
                self.egovlpv2_allgather = self.egovlpv2_components['allgather']
                self.egovlpv2_tokenizer = None
                self.egovlpv2_loss_fn = None
                self.egovlpv2_task_names = None

                logger.info("✅ Embedding组件初始化完成（Freeze VLM + Alignment训练模式）")
            else:
                logger.info("🔧 初始化EgoVLPv2训练组件...")
                self.egovlpv2_components = init_egovlpv2_training_components(
                    config_path=egovlpv2_config_path,
                    device=device,
                    infinite_dataloader=True,
                    dtype=dtype
                )
                
                # 提取关键组件到self以便访问
                self.egovlpv2_model = self.egovlpv2_components['model']
                self.egovlpv2_tokenizer = self.egovlpv2_components['tokenizer']
                self.egovlpv2_config = self.egovlpv2_components['config']
                self.egovlpv2_loss_fn = self.egovlpv2_components['loss_fn']
                self.egovlpv2_allgather = self.egovlpv2_components['allgather']
                self.egovlpv2_task_names = self.egovlpv2_config['trainer']['task_names']
                
                logger.info("✅ EgoVLPv2组件初始化完成")
            
            # 初始化分布式参数占位符
            self.egovlpv2_args = type('Args', (), {})()  # 创建简单对象
        
        # 初始化Alignment组件
        if self.use_alignment:
            logger.info("🔧 初始化Alignment训练组件...")
            self.alignment_components = init_alignment_model_components(
                config_path=egovlpv2_config_path,
                device=device,
                dtype=dtype
            )
            
            self.alignment_model = self.alignment_components['alignment_model']
            self.alignment_config = self.alignment_components['config']
            self.alignment_layer_indices = self.alignment_components['layer_indices']
            # before_proj: 对齐时是否使用投影前的EgoHOD文本特征（从alignment config中读取）
            self.alignment_before_proj = self.alignment_components.get('before_proj', False)
            # VideoFeatureSampler: as2vs 模式下从预计算的 video pool 中采样 anchor
            self.video_sampler = self.alignment_components.get('video_sampler', None)
            training_cfg = self.alignment_components.get('training_config', {})
            self.alignment_backbone_update_start_pct = float(training_cfg.get('backbone_update_start_pct', 0.0))
            self.alignment_stop_last_pct = float(training_cfg.get('stop_last_pct', 0.0))
            logger.info("✅ Alignment组件初始化完成")
            logger.info(
                "🔧 alignment backbone 延迟更新比例: %.4f",
                self.alignment_backbone_update_start_pct,
            )
            logger.info(
                "🔧 alignment 后段停止比例: %.4f",
                self.alignment_stop_last_pct,
            )
            inner_alignment_model = self.alignment_model.module if hasattr(self.alignment_model, 'module') else self.alignment_model
            logger.info(
                "🔧 alignment 标量dtype: temperature=%s, sigmoid_bias=%s",
                inner_alignment_model._get_temperature().dtype,
                inner_alignment_model.sigmoid_bias.dtype,
            )
    
    def _update_egovlpv2_distributed_args(self):
        """在运行时更新EgoVLPv2的分布式参数"""
        if not hasattr(self, 'egovlpv2_args'):
            return
            
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        self.egovlpv2_args.rank = rank
        self.egovlpv2_args.world_size = world_size
        self._distributed_args_updated = True
        logger.debug(f"✅ EgoVLPv2分布式参数已更新: rank={rank}, world_size={world_size}")
    
    # ===== 委托PI0Pytorch的方法，确保完全兼容 =====

    def gradient_checkpointing_enable(self):
        """启用梯度检查点"""
        return self.pi0_pytorch.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        return self.pi0_pytorch.gradient_checkpointing_disable()
    
    def is_gradient_checkpointing_enabled(self):
        """检查梯度检查点状态"""
        return self.pi0_pytorch.is_gradient_checkpointing_enabled()
    
    def sample_actions(self, device, observation, noise=None, num_steps=10, generator=None):
        """推理时采样动作"""
        return self.pi0_pytorch.sample_actions(device, observation, noise, num_steps, generator=generator)
    
    def get_base_model(self):
        """
        获取内部的PI0Pytorch基础模型
        
        这个方法用于训练脚本中需要访问PI0Pytorch模型的场景，
        例如：LoRA应用、权重保存/加载等
        
        Returns:
            PI0Pytorch: 内部的PI0Pytorch模型实例
        """
        return self.pi0_pytorch
    
    def reset_feature_bank(self):
        """
        重置feature banks（包括alignment model和egovlpv2 model）
        
        在gradient accumulation完成并执行optimizer.step()后调用，
        清空累积的features，开始新的累积周期。
        """
        # 重置alignment model的feature bank
        if self.use_alignment and hasattr(self, 'alignment_model'):
            if hasattr(self.alignment_model, 'reset_feature_bank'):
                self.alignment_model.reset_feature_bank()
        
        # 重置egovlpv2 model的feature bank
        if self.use_egovlpv2 and hasattr(self, 'egovlpv2_model'):
            if hasattr(self.egovlpv2_model, 'reset_feature_bank'):
                self.egovlpv2_model.reset_feature_bank()
    
    # ===== 核心前向传播方法 =====
    
    def forward(
        self, 
        observation, 
        actions, 
        noise=None, 
        time=None,
        egovlpv2_batch: Optional[Dict[str, Any]] = None,
        prompt: Optional[list] = None,
        task_index: Optional[torch.Tensor] = None,  # [B] 原始任务索引（与tasks.jsonl对应）
        task_id: Optional[torch.Tensor] = None,  # [B] 聚类后的任务类别ID，用于多正样本对比学习
        global_step: Optional[int] = None,
        num_train_steps: Optional[int] = None,
        atomic_label_idx: Optional[torch.Tensor] = None,  # [B] 原子级标签索引（可选）
        chunk_video_idx: Optional[torch.Tensor] = None,   # [B] chunk-level video 索引（可选）
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        OpenPI对齐训练的核心前向传播方法
        
        ✅ 统一返回格式：始终返回 (total_loss, (pi0_loss, vlm_loss, align_loss))
        未启用的部分自动置为0，减少条件分支复杂度
        
        Args:
            observation: 观测数据
            actions: 目标动作
            noise: 噪声（可选）
            time: 时间步（可选）
            egovlpv2_batch: EgoVLPv2数据batch（可选）
            prompt: Bridge数据集的文本指令列表（可选）
        
        Returns:
            total_loss: 标量张量（加权总损失）
            (pi0_loss, vlm_loss, align_loss): 三个loss的元组（未启用的为0）
        """

        # print("####### VLA 的图像 ##########")
        # print(type(observation)) # class 'openpi.models.model.Observation'>
        # print(type(observation.images["base_0_rgb"])) # <class 'torch.Tensor'>
        # print(observation.images["base_0_rgb"].shape) # torch.Size([8, 3, 224, 224])
        # print(observation.images["base_0_rgb"].max(), observation.images["base_0_rgb"].min(), ) # torch.Size([8, 3, 224, 224]) (B_1 C H W)
        # print("######### 把 VLA 的 openpi 格式图像转换为 imgenet 标准化后的 和 egovlp 对齐 ###########")
        # imgs = convert_openpi_to_imagenet(observation.images["base_0_rgb"])
        # print(imgs.shape, imgs.max(), imgs.min()) # torch.Size([8, 3, 224, 224]) tensor(2.6400, device='cuda:1') tensor(-2.1179, device='cuda:1')
        # print("######### 以下是egovlp的图像 ###########")
        # print(egovlpv2_batch.keys())
        # print(egovlpv2_batch["video"].shape) # (12,4,3,224,224)     (B_2 4 C H W)
        # print(egovlpv2_batch["video"].max(), egovlpv2_batch["video"].min())

        # === 1. PI0Pytorch前向传播（主模型，必需） ===
        # 传入层索引列表而非 True，只保存 alignment 需要的层，大幅节省显存
        need_hidden_states = self.use_alignment and self.training
        hidden_layers_arg = self.alignment_layer_indices if need_hidden_states else False
        pi0_result = self.pi0_pytorch.forward(
            observation, actions, noise, time,
            output_hidden_states=hidden_layers_arg
        )
        
        # 解包PI0Pytorch返回值
        if need_hidden_states:
            # 支持learnable token：返回值可能是(loss, hidden_states)或(loss, hidden_states, learnable_token_hidden_states)
            if isinstance(pi0_result, tuple) and len(pi0_result) == 3:
                pi0_loss, action_hidden_states, learnable_token_hidden_states = pi0_result
            else:
                pi0_loss, action_hidden_states = pi0_result
                learnable_token_hidden_states = None
            
            if isinstance(pi0_loss, list | tuple):
                print(f"pi0_loss is a list or tuple: {type(pi0_loss)}")
                pi0_loss = torch.stack(pi0_loss)
            elif not isinstance(pi0_loss, torch.Tensor):
                print(f"pi0_loss is not a tensor: {type(pi0_loss)}")
                pi0_loss = torch.tensor(pi0_loss, device=pi0_loss.device, dtype=torch.float32)
        else:
            pi0_loss = pi0_result
            action_hidden_states = None
            learnable_token_hidden_states = None
        
        # 计算主损失（batch内平均）
        pi0_loss_scalar = pi0_loss.mean()
        
        # === 2. EgoVLPv2前向传播（可选，未启用时为0） ===
        egovlpv2_loss = torch.tensor(0.0, device=pi0_loss_scalar.device)
        if self.use_egovlpv2 and self.training:
            # Lazy initialization：确保分布式参数已更新
            if not self._distributed_args_updated:
                self._update_egovlpv2_distributed_args()
            
            # 如果VLM被冻结，使用no_grad上下文减少显存计算
            if not self.freeze_vlm and egovlpv2_batch is not None:
                egovlpv2_result = self._forward_egovlpv2(egovlpv2_batch)
                egovlpv2_loss = egovlpv2_result['loss']
        
        # === 3. Alignment前向传播（可选，未启用时为0） ===
        alignment_loss = torch.tensor(0.0, device=pi0_loss_scalar.device)
        alignment_loss_dict = {}  # 细粒度对齐loss字典（as2ts, as2tt等）
        stop_alignment = self._should_stop_alignment(global_step, num_train_steps)
        if self.use_alignment and self.training and action_hidden_states is not None and prompt is not None and not stop_alignment:
            alignment_result = self._forward_alignment(
                action_hidden_states,
                prompt,
                task_index=task_index,  # 原始任务索引
                task_id=task_id,        # 聚类后的任务类别ID
                learnable_token_hidden_states=learnable_token_hidden_states,
                global_step=global_step,
                num_train_steps=num_train_steps,
                atomic_label_idx=atomic_label_idx,
                chunk_video_idx=chunk_video_idx,
            )
            alignment_loss = alignment_result['loss']
            # 提取细粒度loss_dict（fg_alignment_model会返回，旧模型可能没有）
            alignment_loss_dict = alignment_result.get('loss_dict', {})
        elif self.use_alignment:
            stop_start_step = 0
            if self.alignment_stop_last_pct > 0 and num_train_steps is not None:
                stop_start_step = int(math.floor(num_train_steps * (1.0 - self.alignment_stop_last_pct)))
            alignment_loss_dict = {
                "alignment_stopped": float(stop_alignment),
                "alignment_stop_last_pct": self.alignment_stop_last_pct,
                "alignment_stop_start_step": stop_start_step,
            }
        
        # === 4. 聚合损失 ===
        total_loss = (
            pi0_loss_scalar + 
            self.vlm_loss_weight * egovlpv2_loss + 
            self.alignment_loss_weight * alignment_loss
        )
        
        # ✅ 返回格式：(total_loss, (pi0_loss, vlm_loss, align_loss), alignment_loss_dict)
        # alignment_loss_dict包含细粒度loss（as2ts、as2tt、at2tt等）
        return total_loss, (pi0_loss_scalar, egovlpv2_loss, alignment_loss), alignment_loss_dict
    
    # ===== 三个模型的独立前向传播包装函数 =====
    
    def _forward_egovlpv2(self, egovlpv2_batch: Dict[str, Any]) -> Dict:
        """
        VLM模型的前向传播包装函数
        
        Args:
            egovlpv2_batch: VLM的输入数据batch
        
        Returns:
            包含loss的字典
        
        说明：
        - EgoHOD模式：如果freeze_vlm=False，调用egohod_forward_pass进行训练（与SpatialVLA对齐）
        - LIV和Qwen3模式：不支持VLM训练，返回零loss
        - EgoVLPv2模式：正常执行前向传播
        """
        if self.vlm_mode == "egohod":
            # EgoHOD模式：调用egohod_forward_pass进行训练
            # 参考SpatialVLA的_forward_egovlpv2实现
            result = egohod_forward_pass(
                model=self.egovlpv2_model,
                vlm_batch=egovlpv2_batch,
                tokenizer=self.egovlpv2_tokenizer,
                config=self.egovlpv2_config,
                args=self.egovlpv2_args,
                loss_fn=self.egovlpv2_loss_fn,
                allgather_fn=self.egovlpv2_allgather,
                device_id=next(self.pi0_pytorch.parameters()).device,
                task_names='Dual'
            )
            return result
        
        if self.vlm_mode in ["liv", "qwen3", "embedding"]:
            # LIV、Qwen3和离线Embedding模式下不训练VLM，返回零loss
            return {'loss': torch.tensor(0.0, device=next(self.pi0_pytorch.parameters()).device)}
        
        # EgoVLPv2模式下正常执行
        result = egovlpv2_forward_pass(
            model=self.egovlpv2_model,
            vlm_batch=egovlpv2_batch,
            tokenizer=self.egovlpv2_tokenizer,
            config=self.egovlpv2_config,
            args=self.egovlpv2_args,
            loss_fn=self.egovlpv2_loss_fn,
            allgather_fn=self.egovlpv2_allgather,
            device_id=next(self.pi0_pytorch.parameters()).device,
            task_names=self.egovlpv2_task_names
        )
        return result
    
    def _forward_alignment(
        self, 
        action_hidden_states: torch.Tensor,
        prompt: list,
        task_index: Optional[torch.Tensor] = None,  # [B] 原始任务索引（与tasks.jsonl对应）
        task_id: Optional[torch.Tensor] = None,  # [B] 聚类后的任务类别ID，用于多正样本对比学习
        learnable_token_hidden_states: Optional[torch.Tensor] = None,  # learnable token的特征
        global_step: Optional[int] = None,
        num_train_steps: Optional[int] = None,
        atomic_label_idx: Optional[torch.Tensor] = None,  # [B] 原子级标签索引
        chunk_video_idx: Optional[torch.Tensor] = None,   # [B] chunk-level video 索引
    ) -> Dict:
        """
        Alignment模型的前向传播包装函数
        
        Args:
            action_hidden_states: OpenPI的action特征
                格式: [num_layers, batch_size, suffix_seq_len, hidden_size]
                其中 suffix_seq_len 包含 state + action tokens（如果使用learnable token，则包含state + learnable + actions）
            prompt: Bridge数据集的文本指令列表
            task_index: [B] 原始任务索引（与tasks.jsonl对应）
            task_id: [B] 聚类后的任务类别ID（如果没有则等于task_index）
            learnable_token_hidden_states: learnable token的特征
                格式: tuple of [batch_size, 1, hidden_size] for each layer（如果启用learnable token）
        
        Returns:
            包含loss的字典
        """
        device = next(self.pi0_pytorch.parameters()).device

        # 统一 hidden states 表示：
        # - PI0Pytorch 主干通常返回 [num_layers, B, T, D]
        # - learnable token 特征当前实现返回 tuple([B, 1, D], ...)
        # 这里统一堆叠，后续索引逻辑保持一致。
        if isinstance(action_hidden_states, (tuple, list)):
            action_hidden_states = torch.stack(list(action_hidden_states), dim=0)
        if isinstance(learnable_token_hidden_states, (tuple, list)):
            learnable_token_hidden_states = torch.stack(list(learnable_token_hidden_states), dim=0)
        
        # 0. 确保prompt是字符串列表格式
        if isinstance(prompt, (np.ndarray, torch.Tensor)):
            prompt = prompt.tolist()

        # 前若干步仅训练 alignment 头：通过 detach backbone 特征实现。
        # 这样不会改动优化器分组，也不会影响 PI0 loss / VLM loss 的正常训练。
        detach_alignment_backbone = self._should_detach_alignment_backbone(global_step, num_train_steps)
        
        # 1. 获取alignment model配置
        alignment_model = self.alignment_model.module if hasattr(self.alignment_model, 'module') else self.alignment_model
        mode_config = alignment_model.mode_config
        need_text_tokens = mode_config['as2tt']['enabled'] or mode_config['at2tt']['enabled']

        # ============ Simple模式：标准编码 ============
        if self.vlm_mode == "egohod":
            # co-training时允许alignment loss梯度流回EgoHOD文本编码器
            egohod_enable_grad = (not self.freeze_vlm) and (not detach_alignment_backbone)
            # 使用EgoHOD提取文本特征
            # before_proj: 对齐时使用投影前特征（从alignment config中读取）
            egovlpv2_text_embeds = extract_egohod_text_embeddings(
                egohod_model=self.egovlpv2_model,
                batch_texts=prompt,
                device=device,
                enable_grad=egohod_enable_grad,
                before_proj=self.alignment_before_proj,
            )
            
            # 3. 如果需要，提取token级别的文本特征
            egovlpv2_text_tokens = None
            egovlpv2_text_mask = None
            if need_text_tokens:
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_egohod_text_token_embeddings(
                    egohod_model=self.egovlpv2_model,
                    batch_texts=prompt,
                    device=device,
                    enable_grad=egohod_enable_grad,
                )
        elif self.vlm_mode == "qwen3":
            # 使用Qwen3-Embedding提取文本特征
            egovlpv2_text_embeds = extract_qwen3_text_embeddings(
                qwen3_model=self.egovlpv2_model,
                batch_texts=prompt,
                device=device
            )
            
            egovlpv2_text_tokens = None
            egovlpv2_text_mask = None
            if need_text_tokens:
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_qwen3_text_token_embeddings(
                    qwen3_model=self.egovlpv2_model,
                    batch_texts=prompt,
                    device=device
                )
        elif self.vlm_mode == "embedding":
            if task_index is not None:
                egovlpv2_text_embeds = extract_embedding_text_embeddings_by_index(
                    embedding_model=self.egovlpv2_model,
                    task_index=task_index,
                    device=device
                )
            else:
                egovlpv2_text_embeds = extract_embedding_text_embeddings(
                    embedding_model=self.egovlpv2_model,
                    batch_texts=prompt,
                    device=device
                )

            egovlpv2_text_tokens = None
            egovlpv2_text_mask = None
            if need_text_tokens:
                if task_index is not None:
                    egovlpv2_text_tokens, egovlpv2_text_mask = extract_embedding_text_token_embeddings_by_index(
                        embedding_model=self.egovlpv2_model,
                        task_index=task_index,
                        device=device
                    )
                else:
                    egovlpv2_text_tokens, egovlpv2_text_mask = extract_embedding_text_token_embeddings(
                        embedding_model=self.egovlpv2_model,
                        batch_texts=prompt,
                        device=device
                    )
        else:
            # 使用EgoVLPv2提取文本特征
            egovlpv2_text_embeds = extract_egovlpv2_text_embeddings(
                egovlpv2_model=self.egovlpv2_model,
                egovlpv2_tokenizer=self.egovlpv2_tokenizer,
                batch_texts=prompt,
                device=device
            )
            
            # 3. 如果需要，提取token级别的文本特征
            egovlpv2_text_tokens = None
            egovlpv2_text_mask = None
            if need_text_tokens:
                egovlpv2_text_tokens, egovlpv2_text_mask = extract_egovlpv2_text_token_embeddings(
                    egovlpv2_model=self.egovlpv2_model,
                    egovlpv2_tokenizer=self.egovlpv2_tokenizer,
                    batch_texts=prompt,
                    device=device
                )
        
        # 4. 准备 OpenPI 特征
        # action_hidden_states: [num_selected_layers, B, suffix_seq_len, D]
        # 已经在 gemma forward 中只保存了 alignment_layer_indices 指定的层
        selected_action_features = action_hidden_states.permute(1, 0, 2, 3)  # -> [B, layers, seq, D]

        selected_global_action_features = None
        if learnable_token_hidden_states is not None:
            # [selected_layers, B, 1, D] -> [B, selected_layers, D]
            selected_global_action_features = learnable_token_hidden_states.permute(1, 0, 2, 3).squeeze(2)

        if detach_alignment_backbone:
            selected_action_features = selected_action_features.detach()
            egovlpv2_text_embeds = egovlpv2_text_embeds.detach()
            if selected_global_action_features is not None:
                selected_global_action_features = selected_global_action_features.detach()
            if egovlpv2_text_tokens is not None:
                egovlpv2_text_tokens = egovlpv2_text_tokens.detach()

        # 原子级对齐：embedding 模式下用 atomic_label_idx 查表
        atomic_text_features = None
        if (atomic_label_idx is not None and self.vlm_mode == "embedding"
                and hasattr(self.egovlpv2_model, 'atomic_embed_layer')
                and self.egovlpv2_model.atomic_embed_layer is not None):
            atomic_text_features = self.egovlpv2_model.compute_atomic_text_by_index(
                atomic_label_idx.to(device))

        # 5.5 as2vs: 如果配置了 video_sampler，在此处完成采样
        video_features = None
        video_sim_weights = None
        if self.video_sampler is not None and task_index is not None:
            video_features, video_sim_weights, _ = self.video_sampler.sample(task_index)

        # 5.6 as2cv: chunk-level video features 离线查表
        # EmbeddingModel 在初始化时已通过 load_chunk_video_features 加载好；
        # 这里仅按 chunk_video_idx 查表得到 frozen video embedding。
        chunk_video_features = None
        if (chunk_video_idx is not None and self.vlm_mode == "embedding"
                and getattr(self.egovlpv2_model, 'chunk_video_embed_layer', None) is not None):
            chunk_video_features = self.egovlpv2_model.compute_chunk_video_by_index(
                chunk_video_idx.to(device))

        # 6. 计算对齐损失
        alignment_result = alignment_forward_pass(
            alignment_model=self.alignment_model,
            openvla_features=selected_action_features,
            openvla_global_features=selected_global_action_features,
            egovlpv2_features=egovlpv2_text_embeds,
            egovlpv2_text_tokens=egovlpv2_text_tokens,
            egovlpv2_text_mask=egovlpv2_text_mask,
            task_index=task_index,  # 原始任务索引
            task_ids=task_id,       # 聚类后的任务类别ID
            langs=prompt,           # 传入文本，便于缺失ID时兜底生成并做对齐打印
            allgather_fn=self.egovlpv2_allgather,
            n_gpu=self.egovlpv2_args.world_size if hasattr(self.egovlpv2_args, 'world_size') else 1,
            args=self.egovlpv2_args,
            device_id=device,
            mode=self.vlm_mode,
            video_features=video_features,
            video_sim_weights=video_sim_weights,
            atomic_text_features=atomic_text_features,
            chunk_video_features=chunk_video_features,
            chunk_video_task_ids=task_id,  # as2cv trajectory 监督仍按 task_id 聚合
            chunk_video_idx=(chunk_video_idx.to(device) if chunk_video_idx is not None else None),
        )
        if alignment_result.get("loss_dict") is not None:
            detach_until_step = 0
            if self.alignment_backbone_update_start_pct > 0 and num_train_steps is not None:
                detach_until_step = int(math.ceil(num_train_steps * self.alignment_backbone_update_start_pct))
            alignment_result["loss_dict"]["backbone_detached"] = float(detach_alignment_backbone)
            alignment_result["loss_dict"]["backbone_update_start_pct"] = self.alignment_backbone_update_start_pct
            alignment_result["loss_dict"]["backbone_update_start_step"] = detach_until_step
        
        return alignment_result

    def _should_detach_alignment_backbone(
        self,
        global_step: Optional[int],
        num_train_steps: Optional[int],
    ) -> bool:
        """判断当前步是否只训练 alignment 头。"""
        if self.alignment_backbone_update_start_pct <= 0:
            return False
        if global_step is None or num_train_steps is None:
            return False
        detach_until_step = int(math.ceil(num_train_steps * self.alignment_backbone_update_start_pct))
        return global_step < detach_until_step

    def _should_stop_alignment(
        self,
        global_step: Optional[int],
        num_train_steps: Optional[int],
    ) -> bool:
        """判断当前步是否进入训练尾段，直接关闭 alignment loss。"""
        if self.alignment_stop_last_pct <= 0:
            return False
        if global_step is None or num_train_steps is None:
            return False
        stop_start_step = int(math.floor(num_train_steps * (1.0 - self.alignment_stop_last_pct)))
        return global_step >= stop_start_step
    
    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        return next(self.pi0_pytorch.parameters()).device
