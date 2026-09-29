# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import torch
import torch.utils.checkpoint
from torch import nn
from torch.linalg import inv
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from transformers.cache_utils import Cache, HybridCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.utils import (
    ModelOutput,
    logging,
)
from .configuration_spatialvla import SpatialVLAConfig
from .modeling_gemma2 import Gemma2ForCausalLM
from transformers import AutoModel, ZoeDepthForDepthEstimation

SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
ZOE_MEAN, ZOE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

logger = logging.get_logger(__name__)

class Ego3DPositionEmbeddingMLP(nn.Module):
    """Absolute pos embedding, learned.
    https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/models/nerf.py#L4
    """

    def __init__(self, in_channels=3, num_pos_feats=768, n_freqs=8, logscale=True):
        super(Ego3DPositionEmbeddingMLP, self).__init__()
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)
        
        center = torch.tensor([0., 0., 2.]).repeat(in_channels // 3)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """init with small weights to maintain stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def frequency_encoding(self, xyz):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        x \in [-2, 2]
        y \in [-2, 2]
        z \in [0., 4]
        Inputs:
            x: (b n m)
        Outputs:
            out: (b n o)
        """
        xyz_n = ((xyz - self.center) / 2.0).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)  # (b n m nf)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        freq_encoding = self.frequency_encoding(xyz)
        position_embedding = self.position_embedding_head(freq_encoding)
        return position_embedding

def process_zoe(pixel_values, pad_mode="reflect", output_size=(384, 512)):
    """https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/zoedepth/image_processing_zoedepth.py"""
    # h, w = images.shape[-2:]
    # pad
    ph, pw = 31, 31  # int((h / 2)**0.5 * 3), int((w / 2)**0.5 * 3) # 32, 31
    images = F.pad(pixel_values, (pw, pw, ph, ph), mode=pad_mode)
    # resize
    size = (384, 384)  # get_resize_output_image_size
    images = F.interpolate(images, size=size, mode="bicubic", align_corners=True)
    # zoe: padding -> resize -> nomalize. we follow `nomalize -> padding -> resize` from siglip
    images = TF.normalize(images, mean=ZOE_MEAN, std=ZOE_STD)
    return images, ph, pw

@dataclass
class SpatialVLACausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None  # 进入语言模型前的图像特征
    action_hidden_states: Optional[torch.FloatTensor] = None
    vision_hidden_states: Optional[torch.FloatTensor] = None  # 图像tokens在语言模型每层的hidden states

class SpatialVLAMultiModalProjector(nn.Module):
    def __init__(self, config: SpatialVLAConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states

class SpatialVLAPreTrainedModel(PreTrainedModel):
    config_class = SpatialVLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SpatialVLAMultiModalProjector", "ZoeDepthForDepthEstimation", "Ego3DPositionEmbeddingMLP"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class SpatialVLAForConditionalGeneration(SpatialVLAPreTrainedModel, GenerationMixin):
    def __init__(self, config: SpatialVLAConfig, vision_model=None, vision_zoe_model=None, projector_model=None, language_model=None):
        super().__init__(config)

        self.vision_tower = vision_model or AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = projector_model or SpatialVLAMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        if language_model is None:
            language_model = Gemma2ForCausalLM(config=config.text_config)
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model

        if config.use_vision_zoe:
            self.vision_zoe_model = vision_zoe_model or ZoeDepthForDepthEstimation(config.vision_zoe_config)
            self.position_embedding_3d = Ego3DPositionEmbeddingMLP(
                config.ego3d_patch_reso**2 * 3, num_pos_feats=config.vision_config.hidden_size, n_freqs=config.n_freqs
            )
            # register buffer
            patch_size, reso, image_size = config.vision_config.patch_size, config.ego3d_patch_reso, config.vision_config.image_size
            y, x = torch.meshgrid(torch.arange(0, image_size, patch_size // reso), torch.arange(0, image_size, patch_size // reso), indexing="ij")  # (h//sp w//sp)
            y, x = y + patch_size / reso / 2, x + patch_size / reso / 2
            uv_h = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)  # (3 hw)
            self.register_buffer("uv_h", uv_h, persistent=False)

        # shared spatial embeddings for <ACTION> <IMG>
        if config.use_spatial_token:
            self.spatial_embed_tokens = nn.Embedding(self.config.spatial_token_num, config.text_config.hidden_size)
        else:
            self.spatial_embed_tokens = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1


    def backproject_patch(self, K: torch.Tensor, depth: torch.Tensor, patch_size=14, reso=2) -> torch.Tensor:
        """
        Backproject depth map to 3D points in camera coordinate.
        Args:
            K: camera intrinsic matrix (b 3 3)
            depth: depth map (b 1 h w)
            patch_size: patch size for siglip
            reso: reso^2 -> sample points in each patch
        patch sz = 14  ......          
        ┌────────┬────────┐       
        │ ─    ─ │ ─    ─ │       
        │ points │        ├─ ─ ─ 
        │ ─    ─ │ ─    ─ │       
        ├────────┼────────┤       
        │ ─    ─ │ ─    ─ │       
        │        │        │       
        │ ─    ─ │ ─    ─ │       
        └────────┴────────┘       
        reso=2───►points=4
            │                    
            │                        
        """
        b, c, h, w = depth.shape
        hp, wp = h // patch_size, w // patch_size
        sub_hp = sub_wp = reso
        patch_depth = F.interpolate(depth, size=(hp * reso, wp * reso), mode="area").reshape(b, c, -1)
        p_cam = (inv(K.float()) @ self.uv_h.float()) * patch_depth  # (b 3 3) @ (3 hw) -> (b 3 hw) * (b 1 hw) -> (b 3 hw)
        patch_p_cam = p_cam.reshape(b, 3, hp, sub_hp, wp, sub_wp).permute(0, 2, 4, 3, 5, 1).reshape(b, hp * wp, -1)
        return patch_p_cam

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        vocab_size = model_embeds.weight.shape[0]
        self.config.text_config.vocab_size = self.vocab_size = self.config._vocab_size = vocab_size
        self.tie_weights()
        return model_embeds
    
    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_ids=None,
        inputs_embeds=None,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device)
        if sequence_length != 1:
            if is_training: causal_mask = torch.triu(causal_mask, diagonal=1)
            else: causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0)
        return causal_mask

    def get_image_features(self, pixel_values: torch.FloatTensor, intrinsic: torch.FloatTensor):
        siglip_pixel_values = TF.normalize(pixel_values, mean=SIGLIP_MEAN, std=SIGLIP_STD)
        image_outputs = self.vision_tower(siglip_pixel_values)

        # ego3d position encoding
        if self.config.use_vision_zoe:
            zoe_pixel_values, ph, pw = process_zoe(pixel_values, pad_mode="reflect")
            with torch.no_grad():
                pvh, pvw = pixel_values.shape[-2:]
                # Ensure input dtype matches model dtype to avoid mixed precision issues
                zoe_input = zoe_pixel_values.to(dtype=next(self.vision_zoe_model.parameters()).dtype)
                depth = self.vision_zoe_model(pixel_values=zoe_input).predicted_depth
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(pvh+2*ph, pvw+2*pw),
                    mode="bicubic",
                    align_corners=True,
                )[..., ph:-ph, pw:-pw]
                xyz = self.backproject_patch(
                    intrinsic, depth, patch_size=self.config.vision_config.patch_size, reso=self.config.ego3d_patch_reso
                )  # (b, n, 3*4)
            pos_embed_3d = self.position_embedding_3d(xyz)
            selected_image_feature = image_outputs.last_hidden_state + pos_embed_3d
        else:
            selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.text_config.hidden_size**0.5)
        return image_features

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
        **kwargs,  # 新增：接收额外参数（如task_progress）
    ) -> Union[Tuple, SpatialVLACausalLMOutputWithPast]:

        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        return_dict = return_dict or self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None
        
        # 调试：检查input_ids和embedding权重的设备分配
        if inputs_embeds is None: 
            embedding_weight_device = self.get_input_embeddings().weight.device
            input_ids_device = input_ids.device
            if input_ids_device != embedding_weight_device:
                input_ids = input_ids.to(embedding_weight_device)
            inputs_embeds = self.get_input_embeddings()(input_ids).clone() # avoid checkpint grad True

        if self.config.use_spatial_token:
            spatial_selected = (input_ids >= self.config.action_token_begin_idx) & (input_ids < self.config.action_token_begin_idx + self.config.spatial_token_num)
            inputs_embeds[spatial_selected] = inputs_embeds[spatial_selected] * 0.0 + self.spatial_embed_tokens(input_ids[spatial_selected] - self.config.action_token_begin_idx)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # merge
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, intrinsic)
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. ",
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, input_ids, inputs_embeds, is_training
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs.logits
        loss = None
        action_hidden_states = None
        vision_hidden_states = None  # 初始化vision_hidden_states变量，用于存储图像tokens在每层的hidden states
        
        # Extract vision hidden states - 提取融合了3D信息的图像tokens在每层的hidden states
        if outputs.hidden_states is not None and pixel_values is not None:
            # 1. 堆叠所有层的hidden states: (layer_num, batch_size, seq_len, hidden_dim)
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            
            # 2. 创建vision token mask - 定位图像tokens在序列中的位置
            vision_mask = (input_ids == self.config.image_token_index)  # (batch_size, seq_len)
            
            # 3. 展平batch和sequence维度，方便索引提取
            flat_hidden_states = hidden_states.contiguous().view(hidden_states.shape[0], -1, hidden_states.shape[-1])  # (layer_num, batch*seq, hidden_dim)
            flat_vision_mask = vision_mask.view(-1)  # (batch*seq,)
            
            # 4. 提取vision tokens对应的hidden states
            vision_hidden_states = flat_hidden_states[:, flat_vision_mask]  # (layer_num, num_vision_tokens, hidden_dim)
            
            # 5. 重新整形为 (layer_num, batch_size, num_vision_tokens_per_batch, hidden_dim)
            if flat_vision_mask.sum() > 0:  # 确保有vision tokens
                num_vision_tokens_per_batch = vision_mask.sum(dim=1)[0].item()  # 假设每个batch的vision token数量相同
                vision_hidden_states = vision_hidden_states.contiguous().view(
                    vision_hidden_states.shape[0], 
                    input_ids.shape[0], 
                    num_vision_tokens_per_batch, 
                    vision_hidden_states.shape[-1]
                )
        
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
            
            # Extract action hidden states - simple version aligned with loss computation
            if outputs.hidden_states is not None:
                # Stack all layers: (layer_num, batch_size, seq_len, hidden_dim)
                hidden_states = torch.stack(outputs.hidden_states, dim=0)
                shift_hidden_states = hidden_states[..., :-1, :]  # Align with shift_labels
                
                # Create action mask same as loss computation
                action_mask = shift_labels != self.config.ignore_index
                if attention_mask is not None:
                    action_mask = action_mask & shift_attention_mask.to(action_mask.device)
                
                # Extract action tokens: simplified since action_mask already combines both conditions
                # Flatten batch and sequence dimensions
                flat_hidden_states = shift_hidden_states.contiguous().view(shift_hidden_states.shape[0], -1, shift_hidden_states.shape[-1])  # (layer_num, batch*seq, hidden_dim)
                flat_action_mask = action_mask.view(-1)  # (batch*seq,)
                action_hidden_states = flat_hidden_states[:, flat_action_mask]  # (layer_num, num_action_tokens, hidden_dim)
                action_hidden_states = action_hidden_states.contiguous().view(action_hidden_states.shape[0], shift_labels.shape[0], -1, action_hidden_states.shape[-1])
                # action_hidden_states = action_hidden_states[:, :, :-1, :]
            
            if attention_mask is not None:
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return SpatialVLACausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            action_hidden_states=action_hidden_states,
            vision_hidden_states=vision_hidden_states,
        )

    # AR inference
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
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            causal_mask = self._update_causal_mask(attention_mask, token_type_ids, past_key_values, cache_position, input_ids, inputs_embeds, is_training)
            model_inputs["attention_mask"] = causal_mask
        model_inputs["intrinsic"] = intrinsic
        return model_inputs

    @torch.no_grad()
    def predict_action(
        self,
        model_inputs,
    ) -> torch.Tensor:
        model_inputs = model_inputs.to(torch.bfloat16).to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]
        generation_outputs = self.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        return generation_outputs[:,input_len:]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        if model.config.use_spatial_token: 
            model.language_model.model.embed_tokens.weight.data[-model.config.spatial_token_num:] = model.spatial_embed_tokens.weight.data
        return model