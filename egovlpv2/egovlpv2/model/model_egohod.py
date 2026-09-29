# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from easydict import EasyDict
from typing import Optional, Dict, Any
from peft import PeftModel

from egovlpv2.base import BaseModel
from egovlpv2.model.egohod.clip import CLIP_VITB16, CLIP_VITL14_336PX
from egovlpv2.model.egohod_peft_lora import (
    validate_lora_config,
    validate_video_lora_config,
    create_lora_config,
    apply_lora_to_text_model,
    apply_lora_to_vision_model,
    print_lora_info,
    convert_text_projection_to_linear,
    convert_image_projection_to_linear,
)


class EgoHODModel(BaseModel):
    """
    EgoHOD模型封装，兼容EgoVLPv2接口
    
    该模型封装EgoHOD的CLIP模型，提供与EgoVLPv2相同的特征提取接口：
    - compute_text: 提取文本特征 [B, D]
    - compute_text_tokens: 提取token级文本特征 [B, T, D]
    - compute_video: 提取视频特征 [B, D]
    
    Args:
        video_params: 视频参数配置
        text_params: 文本参数配置（用于兼容接口，实际不使用）
        projection_dim: 统一投影维度（默认512，与EgoHOD一致）
        load_checkpoint: 检查点路径（可选）
        num_frames: 视频帧数（默认4）
        project_embed_dim: EgoHOD的投影维度（默认512）
    """
    
    def __init__(
        self,
        video_params,
        text_params,
        projection_dim=512,
        load_checkpoint=None,  # OpenAI CLIP权重路径（ViT-B-16.pt）
        num_frames=4,
        project_embed_dim=512,
        use_fast_conv1=True,
        use_flash_attn=True,
        context_length=77,
        vocab_size=49408,
        freeze_temperature=True,
        egohod_checkpoint_path=None,  # EgoVideo预训练权重路径（base_best.pt）
        lora_config: Optional[Dict[str, Any]] = None,  # Text LoRA配置（可选）
        video_lora_config: Optional[Dict[str, Any]] = None,  # Video LoRA配置（可选）
        fine_grain_config: Optional[Dict[str, Any]] = None,  # XClip细粒度loss配置（可选）
        **kwargs,
    ):
        """
        初始化 EgoHODModel

        说明：
        - `load_checkpoint`: OpenAI CLIP权重路径（必需，用于初始化CLIP模型结构）
        - `egohod_checkpoint_path`: EgoVideo预训练权重路径（可选，用于加载fine-tuned权重）
        - `lora_config`: LoRA配置字典（可选），用于对text model应用参数高效微调
        - 文本/视频特征的维度由 `project_embed_dim` 决定（默认 512）
        
        权重加载顺序：
        1. 首先通过load_checkpoint加载OpenAI CLIP权重（在CLIP_VITB16内部完成）
        2. 然后通过egohod_checkpoint_path加载EgoVideo fine-tuned权重（覆盖OpenAI权重）
        3. 最后应用LoRA（如果配置启用）到text model
        """
        super().__init__()
        
        self.video_params = video_params
        self.text_params = text_params
        self.projection_dim = projection_dim
        self.egohod_dim = project_embed_dim  # EgoHOD 默认特征维度
        self.egohod_checkpoint_path = egohod_checkpoint_path  # 保存EgoVideo权重路径
        self.train_logit_scale = not freeze_temperature  # 是否训练 logit_scale
        
        # 创建简单的配置对象，供 CLIP 函数使用
        config = EasyDict()
        config.ckpt_path = load_checkpoint
        config.lavila_path = "dont_use"

        # 根据load_checkpoint选择CLIP模型架构
        # ViT-B/16: base模型
        # ViT-L/14 或 ViT-L/14@336px: large模型
        is_large_model = (isinstance(load_checkpoint, str) and 
                         ("ViT-L" in load_checkpoint or "large" in load_checkpoint.lower()))
        
        if is_large_model:
            # Large模型：使用CLIP_VITL14_336PX (1024维, 24层)
            self.clip_model = CLIP_VITL14_336PX(
                config=config,
                freeze_temperature=freeze_temperature,
                use_grad_checkpointing=False,
                use_bidirectional_lm=False,
                context_length=context_length,
                vocab_size=vocab_size,
                patch_dropout=0.0,
                drop_path_rate=0.0,
                num_frames=num_frames,
                use_fast_conv1=use_fast_conv1,
                use_flash_attn=use_flash_attn,
                project_embed_dim=project_embed_dim,
                pretrain_zoo="openai",
                pretrain_path=None,
            )
        else:
            # Base模型：使用CLIP_VITB16 (768维, 12层)
            self.clip_model = CLIP_VITB16(
                config=config,
                freeze_temperature=freeze_temperature,
                use_grad_checkpointing=False,
                use_bidirectional_lm=False,
                context_length=context_length,
                patch_dropout=0.0,
                drop_path_rate=0.0,
                num_frames=num_frames,
                use_fast_conv1=use_fast_conv1,
                use_flash_attn=use_flash_attn,
                project_embed_dim=project_embed_dim,
                pretrain_zoo="openai",
                pretrain_path=None,
            )
        
        # 加载EgoVideo预训练权重（如果提供）
        # 这会覆盖OpenAI CLIP权重，使用EgoVideo在Ego4D上fine-tuned的权重
        if egohod_checkpoint_path is not None:
            self.load_egovideo_checkpoint(egohod_checkpoint_path)
        
        # 应用LoRA到text model（如果配置启用）
        # 必须在加载所有预训练权重之后执行
        self.text_lora_enabled = False
        self.lora_enabled = False  # 保持向后兼容
        if lora_config is not None:
            validated_config = validate_lora_config(lora_config)
            if validated_config is not None:
                self._apply_lora_to_text_model(validated_config)
                self.text_lora_enabled = True
                self.lora_enabled = True
        
        # 应用LoRA到vision model（如果配置启用）
        self.video_lora_enabled = False
        if video_lora_config is not None:
            validated_video_config = validate_video_lora_config(video_lora_config)
            if validated_video_config is not None:
                self._apply_lora_to_vision_model(validated_video_config)
                self.video_lora_enabled = True
        
        # ============ XClip细粒度loss配置（可选） ============
        # 在全局video↔text对比之外，增加video↔text_tokens的XClip风格细粒度loss
        # fine_grain_config 示例:
        # {'enabled': True, 'weight': 0.5, 'token_temperature': 0.01, 'projection_dim': 256}
        self.fine_grain_enabled = False
        if fine_grain_config is not None and fine_grain_config.get('enabled', False):
            self.fine_grain_enabled = True
            self.fg_weight = fine_grain_config.get('weight', 0.5)
            self.fg_token_temperature = fine_grain_config.get('token_temperature', 0.01)
            fg_proj_dim = fine_grain_config.get('projection_dim', self.egohod_dim)
            # TextTransformer的宽度（投影前token特征的维度）
            text_width = self.clip_model.textual.width
            # XClip投影层：将video和text_tokens投影到公共空间
            self.fg_video_proj = nn.Linear(self.egohod_dim, fg_proj_dim)
            self.fg_text_token_proj = nn.Linear(text_width, fg_proj_dim)
            print(f"📊 Fine-Grain XClip: weight={self.fg_weight}, token_temp={self.fg_token_temperature}, "
                  f"proj={self.egohod_dim}/{text_width}→{fg_proj_dim}")
        
        # ✅ 确保所有参数是contiguous的（分布式训练必须）
        # 在初始化完成后立即执行，防止DeepSpeed广播时出错
        for param in self.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        # 打印 logit_scale (温度参数) 训练状态
        if hasattr(self.clip_model, 'logit_scale'):
            logit_scale_value = self.clip_model.logit_scale.exp().item()
            temp_status = "🔥 可训练" if self.train_logit_scale else "❄️ 冻结"
            print(f"📊 Logit Scale 状态: {temp_status} | 初始值: {logit_scale_value:.4f} (温度={1/logit_scale_value:.4f})")
    
    def compute_text(self, text_data, before_proj=False):
        """
        文本句子级特征提取接口
        
        参数：
        - text_data: dict，至少包含键 'input_ids'，形状为 [B, L]
        - before_proj: 如果为True，返回text_projection之前的特征（用于对齐任务）
        
        返回：
        - text_embeddings: [B, D] 的句子级文本特征
          - before_proj=False: D = self.egohod_dim（投影后，用于训练）
          - before_proj=True: D = transformer.width（投影前，用于对齐）
        """
        input_ids = text_data["input_ids"]
        
        # 确保input_ids与模型在同一设备上
        if input_ids.device != next(self.clip_model.parameters()).device:
            input_ids = input_ids.to(next(self.clip_model.parameters()).device)
        
        # EgoHOD的CLIP文本编码，before_proj控制是否跳过text_projection
        text_embeddings = self.clip_model.encode_text(input_ids, before_proj=before_proj)
        
        return text_embeddings
    
    def compute_text_tokens(self, text_data):
        """
        文本 token 级别特征提取接口（用于细粒度对齐）
        
        与compute_text的区别：
        1. 返回所有token的特征序列，而不是只返回句子级特征
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
        - 与EgoVLPv2的compute_text_tokens接口保持一致
        - Token特征使用transformer的原始输出，不经过text_projection
        - AlignmentModel会有自己的投影层来处理这些token特征
        """
        input_ids = text_data["input_ids"]
        # 确保input_ids与模型在同一设备上，避免device不一致报错
        if input_ids.device != next(self.clip_model.parameters()).device:
            input_ids = input_ids.to(next(self.clip_model.parameters()).device)
        # 调用 CLIP 的 encode_text_tokens 方法获取 token 级特征和mask
        # 返回的是未经过text_projection的原始transformer输出
        text_token_embeddings, attention_mask = self.clip_model.encode_text_tokens(input_ids)
        return text_token_embeddings, attention_mask
    
    def compute_video(self, video_data):
        """
        视频特征提取接口
        
        参数：
        - video_data: 视频数据张量，形状为 [B, T, C, H, W]
          其中 B=batch size, T=num_frames, C=channels, H=height, W=width
        
        返回：
        - video_embeddings: [B, D] 的视频特征（D = self.egohod_dim）
        
        说明：
        - 与 EgoVLPv2 的 compute_video 接口保持一致
        - 输入格式从 [B, T, C, H, W] 转换为 [B, C, T, H, W] (CLIP/TimeSformer标准格式)
        - 返回pooled video特征，用于与text特征对齐
        """
        
        # 转换输入格式：[B, T, C, H, W] -> [B, C, T, H, W]
        # EgoHOD/CLIP 期望 Channel 在 Time 之前
        video_data = video_data.permute(0, 2, 1, 3, 4)
        
        # 处理PEFT包装：get_peft_model会创建PeftModelForFeatureExtraction，
        # 其forward()将第一个位置参数映射为input_ids关键字参数传给base_model，
        # 而VisionTransformer.forward(x)不接受input_ids。
        # 解决：直接调用base_model.model.forward()（与encode_text中的处理方式一致）
        visual = self.clip_model.visual
        if hasattr(visual, 'base_model'):
            x_pooling, _ = visual.base_model.model.forward(video_data)
        else:
            x_pooling, _ = visual(video_data)
        
        return x_pooling
    
    def reset_feature_bank(self):
        """
        重置Feature Bank（用于梯度累积后的清理）
        
        说明：
        - EgoHOD model目前不使用feature bank (ClipLoss在当前Batch内计算)
        - 此方法仅为兼容EgoVLPv2接口而保留
        - 如果未来需要支持Feature Bank，需在此处实现重置逻辑
        """
        pass
    
    def save_checkpoint(self, save_path, save_lora_adapter=True):
        """
        保存EgoHOD模型checkpoint（支持text和video LoRA分离存储）
        
        参数：
            save_path: 保存路径（.pth文件）
            save_lora_adapter: 是否单独保存LoRA adapter（默认True）
        """
        
        has_any_lora = self.text_lora_enabled or self.video_lora_enabled
        
        if has_any_lora and save_lora_adapter:
            # 构建base model的state_dict（不包含LoRA权重）
            clip_state = {}
            
            # 处理textual部分
            text_model = self.clip_model.textual
            if isinstance(text_model, PeftModel):
                base_text = text_model.get_base_model()
                for k, v in base_text.state_dict().items():
                    clip_state[f'textual.{k}'] = v
            else:
                for k, v in text_model.state_dict().items():
                    clip_state[f'textual.{k}'] = v
            
            # 处理visual部分
            vision_model = self.clip_model.visual
            if isinstance(vision_model, PeftModel):
                base_vision = vision_model.get_base_model()
                for k, v in base_vision.state_dict().items():
                    clip_state[f'visual.{k}'] = v
            else:
                for k, v in vision_model.state_dict().items():
                    clip_state[f'visual.{k}'] = v
            
            # 添加其他参数（logit_scale等）
            for k, v in self.clip_model.state_dict().items():
                if not k.startswith('textual.') and not k.startswith('visual.'):
                    clip_state[k] = v
            
            checkpoint = {
                'arch': 'EgoHODModel',
                'state_dict': clip_state,
                'has_text_lora': self.text_lora_enabled,
                'has_video_lora': self.video_lora_enabled,
            }
            torch.save(checkpoint, save_path)
            print(f"✓ EgoHOD base模型已保存到: {save_path}")
            
            # 保存text LoRA adapter
            if self.text_lora_enabled and isinstance(text_model, PeftModel):
                adapter_dir = save_path.replace('.pth', '_text_lora_adapter')
                os.makedirs(adapter_dir, exist_ok=True)
                text_model.save_pretrained(adapter_dir)
                print(f"✓ Text LoRA adapter已保存到: {adapter_dir}")
            
            # 保存video LoRA adapter
            if self.video_lora_enabled and isinstance(vision_model, PeftModel):
                adapter_dir = save_path.replace('.pth', '_video_lora_adapter')
                os.makedirs(adapter_dir, exist_ok=True)
                vision_model.save_pretrained(adapter_dir)
                print(f"✓ Video LoRA adapter已保存到: {adapter_dir}")
        else:
            checkpoint = {
                'arch': 'EgoHODModel',
                'state_dict': self.clip_model.state_dict(),
                'has_text_lora': False,
                'has_video_lora': False,
            }
            torch.save(checkpoint, save_path)
            print(f"✓ EgoHOD模型已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path, load_lora_adapter=True):
        """
        加载EgoHOD模型checkpoint（支持text和video LoRA）
        
        参数：
            checkpoint_path: checkpoint文件路径（.pth文件）
            load_lora_adapter: 是否加载LoRA adapter（默认True）
        """
        from collections import OrderedDict
        
        print(f"=> 加载EgoHOD checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 移除DDP的module.前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '') if k.startswith('module.') else k] = v
        
        # 加载权重
        missing_keys, unexpected_keys = self.clip_model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  缺失的权重键: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"⚠️  未预期的权重键: {unexpected_keys[:5]}...")
        
        # 确保参数contiguous
        for param in self.clip_model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        print(f"✅ EgoHOD base权重加载完成")
        
        if load_lora_adapter:
            # 加载text LoRA adapter
            text_adapter_dir = checkpoint_path.replace('.pth', '_text_lora_adapter')
            if os.path.exists(text_adapter_dir):
                text_model = self.clip_model.textual
                if not isinstance(text_model, PeftModel):
                    self.clip_model.textual = PeftModel.from_pretrained(text_model, text_adapter_dir)
                    for param in self.clip_model.textual.parameters():
                        if not param.is_contiguous():
                            param.data = param.data.contiguous()
                    print(f"✅ Text LoRA adapter加载完成: {text_adapter_dir}")
                    self.text_lora_enabled = True
                    self.lora_enabled = True
            
            # 加载video LoRA adapter
            video_adapter_dir = checkpoint_path.replace('.pth', '_video_lora_adapter')
            if os.path.exists(video_adapter_dir):
                vision_model = self.clip_model.visual
                if not isinstance(vision_model, PeftModel):
                    self.clip_model.visual = PeftModel.from_pretrained(vision_model, video_adapter_dir)
                    for param in self.clip_model.visual.parameters():
                        if not param.is_contiguous():
                            param.data = param.data.contiguous()
                    print(f"✅ Video LoRA adapter加载完成: {video_adapter_dir}")
                    self.video_lora_enabled = True
    
    def load_egovideo_checkpoint(self, checkpoint_path):
        """
        加载EgoVideo预训练权重
        
        参数：
        - checkpoint_path: EgoVideo检查点路径（base_best.pt或large_best.pt）
        """
        print(f"=> 加载EgoVideo预训练权重: {checkpoint_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 提取state_dict（处理不同的checkpoint格式）
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 移除module.前缀（如果存在DDP包装）
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k.replace('module.', '')] = v
            else:
                new_state_dict[k] = v
        
        # 处理temporal_embedding尺寸不匹配
        temporal_key = 'visual.temporal_embedding'
        if temporal_key in new_state_dict:
            ckpt_temporal = new_state_dict[temporal_key]  # [T_ckpt, D]
            model_temporal = self.clip_model.visual.temporal_embedding  # [T_model, D]
            if ckpt_temporal.shape[0] != model_temporal.shape[0]:
                ckpt_frames = ckpt_temporal.shape[0]
                model_frames = model_temporal.shape[0]
                print(f"🔄 temporal_embedding尺寸不匹配: checkpoint={ckpt_frames}帧, model={model_frames}帧")
                print(f"   EgoHOD原生支持forward时自动插值，调整模型temporal_embedding以匹配checkpoint")
                # 重新初始化模型的temporal_embedding为checkpoint的尺寸
                self.clip_model.visual.temporal_embedding = torch.nn.Parameter(
                    torch.zeros(ckpt_frames, model_temporal.shape[1], device=model_temporal.device, dtype=model_temporal.dtype)
                )
                self.clip_model.visual.num_frames = ckpt_frames
                print(f"✅ 已调整模型temporal_embedding: {model_frames}帧 -> {ckpt_frames}帧 (forward时会自动插值到实际输入帧数)")
        
        # 加载权重（strict=True）
        missing_keys, unexpected_keys = self.clip_model.load_state_dict(new_state_dict, strict=True)
        
        if missing_keys:
            print(f"⚠️  警告：缺失的权重键: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  警告：未预期的权重键: {unexpected_keys}")
        
        # ✅ 确保所有参数是contiguous的（分布式训练必须）
        for param in self.clip_model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        print(f"✅ 成功加载EgoVideo预训练权重（strict=True）")
        
        # 打印checkpoint信息（如果有）
        if 'epoch' in checkpoint:
            print(f"   - Epoch: {checkpoint['epoch']}")
        if 'best_score' in checkpoint:
            print(f"   - Best Score: {checkpoint['best_score']}")
    
    def _apply_lora_to_text_model(self, lora_config_dict: Dict[str, Any]):
        """
        将LoRA应用到text model（TextTransformer）
        """
        print(f"\n{'='*60}")
        print(f"应用 LoRA 到 Text Model")
        print(f"{'='*60}")
        print(f"配置: {lora_config_dict}")
        
        # 如果target_modules包含text_projection，需要先转换为Linear层
        target_modules = lora_config_dict.get("target_modules", None)
        if target_modules and "text_projection" in target_modules:
            print("✓ 检测到text_projection，转换为Linear层...")
            self.clip_model.textual = convert_text_projection_to_linear(self.clip_model.textual)
        
        # 创建LoraConfig
        peft_config = create_lora_config(
            r=lora_config_dict["r"],
            lora_alpha=lora_config_dict["lora_alpha"],
            lora_dropout=lora_config_dict["lora_dropout"],
            target_modules=target_modules,
            layers_to_transform=lora_config_dict["layers_to_transform"],
            bias=lora_config_dict["bias"],
        )
        
        # 应用LoRA到textual模块
        self.clip_model.textual = apply_lora_to_text_model(
            self.clip_model.textual,
            peft_config
        )
        
        # 冻结vision model的所有参数（确保只训练text LoRA）
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False
        
        # 根据配置决定是否冻结 logit_scale
        if hasattr(self.clip_model, 'logit_scale'):
            self.clip_model.logit_scale.requires_grad = self.train_logit_scale
        if hasattr(self.clip_model, 'image_projection') and self.clip_model.image_projection is not None:
            if isinstance(self.clip_model.image_projection, nn.Parameter):
                self.clip_model.image_projection.requires_grad = False
            else:
                for param in self.clip_model.image_projection.parameters():
                    param.requires_grad = False
        if hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
            if isinstance(self.clip_model.text_projection, nn.Parameter):
                self.clip_model.text_projection.requires_grad = False
            else:
                for param in self.clip_model.text_projection.parameters():
                    param.requires_grad = False
        
        # 打印LoRA应用信息和参数统计
        lora_info = print_lora_info(self.clip_model.textual)
        
        print(f"✅ LoRA 成功应用到 Text Model")
        print(f"   - 应用了 LoRA 的层数: {len(lora_info['lora_layers'])}")
        print(f"   - LoRA 参数量: {lora_info['lora_params']:,}")
        print(f"   - 参数效率: {100 * lora_info['trainable_params'] / lora_info['all_params']:.4f}%")
        print(f"{'='*60}\n")
    
    def _apply_lora_to_vision_model(self, lora_config_dict: Dict[str, Any]):
        """
        将LoRA应用到vision model（VisionTransformer）
        """
        print(f"\n{'='*60}")
        print(f"应用 LoRA 到 Vision Model")
        print(f"{'='*60}")
        print(f"配置: {lora_config_dict}")
        
        # 如果target_modules包含image_projection，需要先转换为Linear层
        target_modules = lora_config_dict.get("target_modules", None)
        if target_modules and "image_projection" in target_modules:
            print("✓ 检测到image_projection，转换为Linear层...")
            self.clip_model.visual = convert_image_projection_to_linear(self.clip_model.visual)
        
        # 创建LoraConfig
        peft_config = create_lora_config(
            r=lora_config_dict["r"],
            lora_alpha=lora_config_dict["lora_alpha"],
            lora_dropout=lora_config_dict["lora_dropout"],
            target_modules=target_modules,
            layers_to_transform=lora_config_dict["layers_to_transform"],
            bias=lora_config_dict["bias"],
        )
        
        # 应用LoRA到visual模块
        self.clip_model.visual = apply_lora_to_vision_model(
            self.clip_model.visual,
            peft_config
        )
        
        # 如果text model未启用LoRA，需要冻结text model
        if not self.text_lora_enabled:
            for param in self.clip_model.textual.parameters():
                param.requires_grad = False
        
        # 根据配置决定是否冻结 logit_scale
        if hasattr(self.clip_model, 'logit_scale'):
            self.clip_model.logit_scale.requires_grad = self.train_logit_scale
        
        # 打印LoRA应用信息
        lora_info = print_lora_info(self.clip_model.visual)
        
        print(f"✅ LoRA 成功应用到 Vision Model")
        print(f"   - 应用了 LoRA 的层数: {len(lora_info['lora_layers'])}")
        print(f"   - LoRA 参数量: {lora_info['lora_params']:,}")
        print(f"   - 参数效率: {100 * lora_info['trainable_params'] / lora_info['all_params']:.4f}%")
        print(f"{'='*60}\n")
    
    @staticmethod
    def _gather_tensor(tensor, rank, world_size):
        """
        分布式gather：将各GPU上的tensor拼接，保持local rank的梯度
        
        原理：与ClipLoss中gather_features一致 —— 
        用dist.all_gather收集所有GPU的tensor（无梯度），
        然后将local rank位置替换为原始tensor（有梯度），确保梯度只回传到当前GPU。
        """
        if world_size <= 1:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)
        gathered[rank] = tensor  # 保持local rank的梯度流
        return torch.cat(gathered, dim=0)
    
    def _compute_xclip_loss(self, video_embeds, text_data, rank=0, world_size=1):
        """
        XClip风格细粒度loss：video全局特征对text token序列做注意力加权对比学习
        
        原理（参考fg_alignment_model._compute_token_contrastive_loss）：
        1. 将video_embeds和text_tokens投影到公共空间
        2. 分布式gather：将所有GPU的特征拼接，构建全局对比矩阵
        3. 对每个(video_i, text_j)对，计算video_i对text_j所有token的attention相似度
        4. softmax加权得到该对的最终相似度
        5. 在全局batch维度做标准双向InfoNCE loss
        
        参数：
        - video_embeds: [B, D_proj] 全局视频特征（已在forward中计算）
        - text_data: dict，包含 'input_ids' [B, L]
        - rank: 当前GPU rank（分布式训练）
        - world_size: GPU总数（分布式训练）
        
        返回：
        - fg_loss: 标量，细粒度对比loss
        - fg_stats: dict，包含pos/neg logit统计
        """
        # 提取text token级特征（复用已有接口，不经过text_projection）
        text_token_embeds, text_mask = self.compute_text_tokens(text_data)
        # text_token_embeds: [B, T, D_text_width], text_mask: [B, T]
        
        B = video_embeds.shape[0]
        device = video_embeds.device
        
        # 投影到公共空间并L2归一化
        video_proj = F.normalize(self.fg_video_proj(video_embeds), dim=-1)          # [B, fg_dim]
        text_flat = text_token_embeds.reshape(-1, text_token_embeds.shape[-1])       # [B*T, D_text]
        text_proj = F.normalize(
            self.fg_text_token_proj(text_flat).reshape(B, -1, video_proj.shape[-1]),
            dim=-1
        )  # [B, T, fg_dim]
        
        # ============ 分布式gather（与ClipLoss.gather_features一致） ============
        # 将所有GPU的投影特征和mask拼接，构建全局对比矩阵
        # 梯度只流过local rank的特征，其他GPU的特征无梯度（作为额外负样本）
        if world_size > 1:
            all_video_proj = self._gather_tensor(video_proj, rank, world_size)  # [B*W, fg_dim]
            all_text_proj = self._gather_tensor(text_proj, rank, world_size)    # [B*W, T, fg_dim]
            all_text_mask = self._gather_tensor(text_mask, rank, world_size) if text_mask is not None else None
        else:
            all_video_proj, all_text_proj, all_text_mask = video_proj, text_proj, text_mask
        
        B_all = all_video_proj.shape[0]  # B * world_size
        
        # XClip注意力相似度：text_proj @ video_proj.T → [B_all, T, B_all] → permute → [B_all, B_all, T]
        sim_matrix = torch.matmul(all_text_proj, all_video_proj.t()).permute(2, 0, 1)  # [B_all, B_all, T]
        
        # Mask掉padding token位置
        if all_text_mask is not None:
            mask_exp = all_text_mask.unsqueeze(0).expand_as(sim_matrix)
            sim_matrix = sim_matrix.masked_fill(mask_exp == 0, -1e9)
        
        # softmax加权（token_temperature控制token注意力的锐利度）
        weights = F.softmax(sim_matrix / self.fg_token_temperature, dim=-1)  # [B_all, B_all, T]
        logits = (weights * sim_matrix).sum(dim=-1)  # [B_all, B_all] 每对的最终相似度
        
        # 用全局logit_scale作为温度参数，与Dual loss一致
        logit_scale = self.clip_model.logit_scale.exp()
        scaled_logits = logits * logit_scale
        
        # 标准双向InfoNCE loss（全局对比矩阵，对角线为正样本）
        labels = torch.arange(B_all, device=device, dtype=torch.long)
        loss_v2t = F.cross_entropy(scaled_logits, labels)
        loss_t2v = F.cross_entropy(scaled_logits.t(), labels)
        fg_loss = (loss_v2t + loss_t2v) / 2
        
        # 统计正负样本logit（用于监控训练状态）
        with torch.no_grad():
            diag = scaled_logits.diag()
            mask = ~torch.eye(B_all, dtype=torch.bool, device=device)
            fg_stats = {
                'fg_pos_logit': diag.mean().item(),
                'fg_neg_logit': scaled_logits[mask].mean().item(),
            }
        
        return fg_loss, fg_stats
    
    def forward(self, data, allgather, n_gpu, args, config, loss_dual, gpu, 
                return_embeds=True, task_names='Dual', dataset_name='fho'):
        """
        EgoHOD训练前向传播（保持与EgoVLPv2/model_epic_charades接口一致）
        
        接口说明（与EgoVLPv2对齐，即使部分参数不使用也保留）：
        - data: dict，包含 'video' 和 'text'
        - allgather: 分布式gather函数（EgoHOD的ClipLoss内部处理，此参数不使用）
        - n_gpu: GPU数量（ClipLoss内部使用world_size）
        - args: 包含rank和world_size
        - config: 配置对象（保留兼容）
        - loss_dual: ClipLoss实例（EgoHOD使用ClipLoss而非EgoVLPv2的loss）
        - gpu: GPU id（保留兼容）
        - task_names: 任务名称
        - dataset_name: 数据集名称
        
        返回：(loss, loss_dict, ret)
        """
        ret = {}
        loss_dict = {}
        
        if 'Dual' in task_names:
            # 更新ClipLoss的分布式参数（关键：确保分布式gather正确）
            # ClipLoss内部使用self.rank和self.world_size来处理gather
            if hasattr(loss_dual, 'rank'):
                loss_dual.rank = args.rank
            if hasattr(loss_dual, 'world_size'):
                loss_dual.world_size = args.world_size
            
            # 获取输入数据
            text_data = data['text']
            video_data = data['video']
            
            # 提取特征
            text_embeds = self.compute_text(text_data)    # [B, D]
            video_embeds = self.compute_video(video_data)  # [B, D]
            
            text_embeds = F.normalize(text_embeds, dim=-1)
            video_embeds = F.normalize(video_embeds, dim=-1)
            
            logit_scale = self.clip_model.logit_scale.exp()
            
            # 调用ClipLoss计算损失（全局video↔text对比）
            loss_result = loss_dual(video_embeds, text_embeds, logit_scale)
            
            # 提取全局loss
            loss = loss_result['loss']
            
            # 更新loss_dict（全局loss部分）
            loss_dict.update({
                'Dual': loss, 
                'clip_acc': loss_result.get('clip_acc', 0.0),
                'vlp_loss': loss_result.get('vlp_loss', 0.0)
            })
            
            # ============ XClip细粒度loss（可选） ============
            # video全局特征对text token序列做attention加权对比学习
            # 与ClipLoss一样，需要gather所有GPU的特征构建全局对比矩阵
            if self.fine_grain_enabled:
                fg_loss, fg_stats = self._compute_xclip_loss(
                    video_embeds, text_data,
                    rank=args.rank, world_size=args.world_size,
                )
                loss = loss + fg_loss * self.fg_weight
                loss_dict['fg_loss'] = fg_loss.item()
                loss_dict.update(fg_stats)
            
            # 构建返回字典
            ret.update({
                "video_embeds": video_embeds,
                "text_embeds": text_embeds,
                "sim_v2t": None,  
                "sim_t2v": None,
            })
        
        return loss, loss_dict, ret
