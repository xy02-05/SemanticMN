import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # ✅ 创建基础模型（不传递lora参数 - LoRA将在训练脚本中统一处理）
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # 使用 config.action_dim 而非硬编码值，与官方 JAX 版 pi0.py 对齐
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # 可学习的intention token（用于镜像神经元对齐）
        self.use_learnable_token = config.use_learnable_token
        if self.use_learnable_token:
            # 创建learnable token embedding，维度与action expert的hidden size一致
            self.learnable_token = nn.Parameter(torch.randn(1, 1, action_expert_config.width) * 0.02)
            logging.info("✅ 已启用Learnable Token（镜像神经元对齐）")

        torch.set_float32_matmul_precision("high")
        # NOTE: torch.compile with max-autotune can cause non-deterministic behavior
        # Commenting out for reproducible evaluation
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for memory optimization.
        
        ✅ 自动处理PEFT包装的模型（如LoRA）- 通过_get_*_model()方法获取基础模型
        """
        self.gradient_checkpointing_enabled = True
        
        # 使用辅助方法获取实际模型（处理PEFT包装）
        paligemma_model = self.paligemma_with_expert._get_paligemma_model()
        gemma_expert_model = self.paligemma_with_expert._get_gemma_expert_model()
        
        paligemma_model.language_model.gradient_checkpointing = True
        paligemma_model.vision_tower.gradient_checkpointing = True
        gemma_expert_model.model.gradient_checkpointing = True

        logging.info("✅ 已启用梯度检查点（支持PEFT包装的模型）")

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing.
        
        ✅ 自动处理PEFT包装的模型（如LoRA）- 通过_get_*_model()方法获取基础模型
        """
        self.gradient_checkpointing_enabled = False
        
        # 使用辅助方法获取实际模型（处理PEFT包装）
        paligemma_model = self.paligemma_with_expert._get_paligemma_model()
        gemma_expert_model = self.paligemma_with_expert._get_gemma_expert_model()
        
        paligemma_model.language_model.gradient_checkpointing = False
        paligemma_model.vision_tower.gradient_checkpointing = False
        gemma_expert_model.model.gradient_checkpointing = False

        logging.info("🚫 已禁用梯度检查点")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            logging.debug(f"🔍 DEBUG: Applying checkpoint to {func.__name__}")
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        logging.debug(f"🔍 DEBUG: NOT applying checkpoint to {func.__name__} (enabled={self.gradient_checkpointing_enabled}, training={self.training})")
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device, generator=None):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            # 如果mask全为False，跳过该图像的embedding（例如use_only_cam_high=True时）
            # 这样可以避免对不需要的图像进行embedding计算
            if not img_mask.any():
                continue

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]
            
            # 插入learnable token（在state之后，action之前）
            if self.use_learnable_token:
                # learnable token embedding，扩展到batch size
                learnable_token_emb = self.learnable_token.expand(bsize, -1, -1)  # [B, 1, D]
                embs.append(learnable_token_emb)
                
                # learnable token的pad mask（始终有效）
                learnable_token_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
                pad_masks.append(learnable_token_mask)
                # learnable token的attention mask = 1（与state相同，能看到prefix）
                att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks for action tokens
        # 如果使用learnable token，action tokens的mask全部设为0（与learnable token共享cumsum，能看到prefix和learnable token）
        # 如果不使用learnable token，保持原逻辑：[1] + [0]*(action_horizon-1)
        if self.use_learnable_token:
            # action tokens的mask全部为0，这样它们与learnable token有相同的cumsum，可以互相看到
            att_masks += [0] * self.config.action_horizon
        else:
            # 原始逻辑：第一个action token的mask=1，后续为0
            att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None, output_hidden_states=False, train=True):
        """
        Do a full training forward pass and compute the loss
        
        Args:
            observation: 观测数据
            actions: 目标动作
            noise: 噪声（可选）
            time: 时间步（可选）
            output_hidden_states: 是否返回action hidden states（用于对齐训练）
            train: 是否使用训练时的数据增强（推理特征提取时应为False）
        
        Returns:
            如果 output_hidden_states=False: 返回 loss 张量 (batch_size x num_steps x num_motors)
            如果 output_hidden_states=True: 返回 (loss, action_hidden_states)
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=train)
        # print(f"images shape: {images[0].shape}, img_masks shape: {img_masks}, lang_tokens shape: {lang_tokens.shape}, lang_masks shape: {lang_masks.shape}, state shape: {state.shape}")

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
            _mem = torch.cuda.memory_allocated() / 1024**3
            print(f"[MEM] after embed: {_mem:.2f}GB, prefix={prefix_embs.shape}, suffix={suffix_embs.shape}")
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
            _mem = torch.cuda.memory_allocated() / 1024**3
            print(f"[MEM] after att_mask: {_mem:.2f}GB, att_2d_masks_4d={att_2d_masks_4d.shape}")

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, output_hidden_states):
            (_, suffix_out), _, suffix_hidden_states = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=output_hidden_states,
            )
            return suffix_out, suffix_hidden_states

        suffix_out, suffix_hidden_states = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, output_hidden_states
        )
        if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
            _mem = torch.cuda.memory_allocated() / 1024**3
            _peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[MEM] after transformer: {_mem:.2f}GB, peak={_peak:.2f}GB")

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss = F.mse_loss(u_t, v_t, reduction="none")
        
        # ✅ 统一返回格式：如果需要hidden states，返回(loss, hidden_states, learnable_token_hidden_states)，否则只返回loss
        if output_hidden_states:
            # 提取learnable token的特征（如果启用）
            learnable_token_hidden_states = None
            if self.use_learnable_token and suffix_hidden_states is not None:
                # suffix序列结构：state (pos 0), learnable_token (pos 1), actions (pos 2...)
                # 从每一层的hidden states中提取learnable token位置的特征
                # suffix_hidden_states: tuple of [B, suffix_len, D] for each layer
                learnable_token_hidden_states = []
                for layer_hidden in suffix_hidden_states:
                    # layer_hidden: [B, suffix_len, D]
                    # learnable token在suffix中的位置是1（state=0, learnable=1, action1=2...）
                    learnable_token_feat = layer_hidden[:, 1:2, :]  # [B, 1, D]
                    learnable_token_hidden_states.append(learnable_token_feat)
                # 转换为tuple格式，与suffix_hidden_states保持一致
                learnable_token_hidden_states = tuple(learnable_token_hidden_states)
            return loss, suffix_hidden_states, learnable_token_hidden_states
        else:
            return loss

    # 与 analysis/openpi_representation 保持一致的层索引，gemma_expert 共18层取10个代表层
    ACTION_FEATURE_LAYER_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, generator=None,
                       output_action_features=False) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)
        当 output_action_features=True 时，在最后一步去噪提取 action hidden states，
        mean pool 后返回 (actions, action_feature)，其中 action_feature: [B, L, D]。
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device, generator=generator)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        action_feature = None
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            # 在最后一步（time ≈ 0）提取 hidden states，语义与 analysis 中 time=0 一致
            is_last_step = (time + dt < -dt / 2)
            if output_action_features and is_last_step:
                v_t, suffix_hs = self.denoise_step(
                    state, prefix_pad_masks, past_key_values, x_t, expanded_time,
                    output_hidden_states=True,
                )
                # suffix_hs: [num_layers, B, suffix_len, D]
                # action tokens 在 suffix 的最后 action_horizon 个位置
                action_hs = suffix_hs[:, :, -self.config.action_horizon:, :]
                # 选取指定层 → mean pool action tokens → [L, B, D]
                selected = action_hs[self.ACTION_FEATURE_LAYER_INDICES]  # [L, B, ah, D]
                action_feature = selected.mean(dim=2)                    # [L, B, D]
                action_feature = action_feature.permute(1, 0, 2).float().cpu().numpy()  # [B, L, D]
            else:
                v_t = self.denoise_step(
                    state, prefix_pad_masks, past_key_values, x_t, expanded_time,
                )

            x_t = x_t + dt * v_t
            time += dt

        if output_action_features:
            return x_t, action_feature
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        output_hidden_states=False,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep.
        当 output_hidden_states=True 时，额外返回 suffix_hidden_states 用于 feature 提取。
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _, suffix_hidden_states = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            output_hidden_states=output_hidden_states,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        if output_hidden_states:
            return v_t, suffix_hidden_states
        return v_t
