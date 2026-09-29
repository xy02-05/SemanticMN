from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


class PaliGemmaWithExpertModel(nn.Module):
    """
    PaliGemma with Expert模型 - 纯模型定义，不包含训练策略（如LoRA）
    
    训练策略（LoRA、量化等）应该在训练脚本中通过PEFT库统一处理。
    """
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        # ✅ 移除lora参数 - 模型定义不应该知道训练策略
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # ✅ 创建基础模型（不应用LoRA - LoRA将在训练脚本中统一处理）
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _get_paligemma_model(self):
        """
        获取PaliGemma的实际模型
        
        如果在训练脚本中应用了PEFT包装（如LoRA），此方法会返回基础模型。
        这样可以访问原始模型的属性和方法（如gradient_checkpointing）。
        """
        if isinstance(self.paligemma, PeftModel):
            return self.paligemma.base_model.model
        return self.paligemma
    
    def _get_gemma_expert_model(self):
        """
        获取Gemma Expert的实际模型
        
        如果在训练脚本中应用了PEFT包装（如LoRA），此方法会返回基础模型。
        这样可以访问原始模型的属性和方法（如gradient_checkpointing）。
        """
        if isinstance(self.gemma_expert, PeftModel):
            return self.gemma_expert.base_model.model
        return self.gemma_expert

    def embed_image(self, image: torch.Tensor):
        paligemma_model = self._get_paligemma_model()
        return paligemma_model.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        paligemma_model = self._get_paligemma_model()
        return paligemma_model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        output_hidden_states: bool | list[int] = False,  # bool或层索引列表，只保存指定层的hidden states
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        # 解析 output_hidden_states：支持 bool 或层索引列表
        if isinstance(output_hidden_states, list):
            target_layers = set(output_hidden_states)
            do_output_hidden = True
        else:
            target_layers = None  # None 表示保存所有层
            do_output_hidden = output_hidden_states
        
        # ✅ 重要说明：
        # _get_*_model()返回base_model，但LoRA参数已经注入到base_model的层中
        # 所以即使我们使用base_model.forward()，LoRA参数仍然会被使用
        # 这里主要是为了访问gradient_checkpointing等属性
        paligemma_model = self._get_paligemma_model()
        gemma_expert_model = self._get_gemma_expert_model()
        
        # ✅ 统一初始化：suffix_hidden_states_stacked始终存在，未使用时为None
        suffix_hidden_states_stacked = None
        
        if inputs_embeds[1] is None:
            prefix_output = paligemma_model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_raw_output = gemma_expert_model.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                output_hidden_states=do_output_hidden,
            )
            suffix_output = suffix_raw_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
            # HF GemmaModel.hidden_states = (embedding_output, layer0_out, ..., layerN_out)
            if do_output_hidden and suffix_raw_output.hidden_states is not None:
                all_hs = suffix_raw_output.hidden_states[1:]  # skip embedding
                if target_layers is not None:
                    selected = [all_hs[i] for i in sorted(target_layers) if i < len(all_hs)]
                else:
                    selected = list(all_hs)
                suffix_hidden_states_stacked = torch.stack(selected, dim=0)
        else:
            models = [paligemma_model.language_model, gemma_expert_model.model]
            num_layers = paligemma_model.config.text_config.num_hidden_layers

            # ✅ 简化的gradient checkpointing检测逻辑
            # 检查是否应该使用gradient checkpointing（训练时且模型支持）
            use_gradient_checkpointing = (
                self.training 
                and hasattr(gemma_expert_model.model, "gradient_checkpointing")
                and gemma_expert_model.model.gradient_checkpointing
            )

            # 初始化hidden states收集器（用于对齐训练）
            all_suffix_hidden_states = [] if do_output_hidden else None
            
            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                # 使用已获取的实际模型
                models = [paligemma_model.language_model, gemma_expert_model.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = paligemma_model.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = paligemma_model.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    paligemma_model.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = paligemma_model.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if layer_idx % 6 == 0 and torch.cuda.is_available():
                    import torch.distributed as _dist
                    if _dist.is_initialized() and _dist.get_rank() == 0:
                        _m = torch.cuda.memory_allocated() / 1024**3
                        print(f"[MEM] layer {layer_idx}/{num_layers}: {_m:.2f}GB")
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )
                
                # 收集suffix的hidden states（只保存目标层）
                if do_output_hidden and (target_layers is None or layer_idx in target_layers):
                    all_suffix_hidden_states.append(inputs_embeds[1].clone())

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None
            
            if do_output_hidden and all_suffix_hidden_states:
                suffix_hidden_states_stacked = torch.stack(all_suffix_hidden_states, dim=0)

        # ✅ 统一返回格式：始终返回三元组，suffix_hidden_states_stacked未使用时为None
        return [prefix_output, suffix_output], prefix_past_key_values, suffix_hidden_states_stacked
