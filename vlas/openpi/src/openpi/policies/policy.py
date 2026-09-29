from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        seed: int | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            # PyTorch diffusion 单独维护一个生成器，避免全局 RNG 被其它路径消耗后打乱 rollout。
            self._torch_generator = None
            if seed is not None:
                self._torch_generator = torch.Generator(device=pytorch_device)
                self._torch_generator.manual_seed(seed)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # 评测时允许单独指定 policy seed；未指定时保持原来的 key(0) 行为。
            self._rng = rng if rng is not None else jax.random.key(seed if seed is not None else 0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # 提取客户端传入的 per-inference noise_seed（在 transform 之前 pop 掉，避免被当成模型输入）
        # noise_seed 使每次推理的扩散噪声仅取决于 (seed, task, episode, step)，
        # 不同 episode 数量的评测在相同 (task, episode) 上也能得到一致结果
        noise_seed = obs.pop("noise_seed", None)
        # 客户端可选请求提取 action feature（pop 掉避免进入 transform）
        extract_action_features = obs.pop("_extract_action_features", False)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # TokenizePrompt 保留了 prompt（np.str_，无法 torch.from_numpy），需要 pop；
        # TokenizeFASTInputs 已经 pop 了 prompt，所以这里用 default=None 兼容两种模型。
        inputs.pop("prompt", None)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        elif noise_seed is not None and self._is_pytorch_model:
            # 客户端提供了 per-inference seed → 创建独立 generator，确保噪声只与 seed 有关
            gen = torch.Generator(device=self._pytorch_device)
            gen.manual_seed(int(noise_seed))
            sample_kwargs["generator"] = gen
        elif self._is_pytorch_model and getattr(self, "_torch_generator", None) is not None:
            sample_kwargs["generator"] = self._torch_generator

        observation = _model.Observation.from_dict(inputs)

        # 临时诊断：前几次推理打印转换后模型输入的hash，用于跨server对比
        if not hasattr(self, "_debug_infer_count"):
            self._debug_infer_count = 0
        self._debug_infer_count += 1
        if self._debug_infer_count <= 3 and self._is_pytorch_model:
            import hashlib as _hl
            _parts = []
            for k in sorted(inputs.keys()):
                v = inputs[k]
                if isinstance(v, torch.Tensor):
                    _parts.append(f"{k}={_hl.md5(v.cpu().float().numpy().tobytes()).hexdigest()}")
                elif isinstance(v, dict):
                    for kk in sorted(v.keys()):
                        vv = v[kk]
                        if isinstance(vv, torch.Tensor):
                            _parts.append(f"{k}/{kk}={_hl.md5(vv.cpu().float().numpy().tobytes()).hexdigest()}")
            _gen_state = ""
            if "generator" in sample_kwargs:
                _g = sample_kwargs["generator"]
                _gen_state = f" gen_seed_check={_g.initial_seed()}"
            print(f"[policy.infer#{self._debug_infer_count}] {' '.join(_parts)}{_gen_state}", flush=True)

        # 如果客户端请求 action feature 且是 PyTorch 模型，则传递 output_action_features
        if extract_action_features and self._is_pytorch_model:
            sample_kwargs["output_action_features"] = True

        start_time = time.monotonic()
        sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        # sample_actions 在 output_action_features=True 时返回 (actions, feature_numpy)
        action_feature_np = None
        if isinstance(sample_result, tuple):
            raw_actions, action_feature_np = sample_result
        else:
            raw_actions = sample_result

        outputs = {
            "state": inputs["state"],
            "actions": raw_actions,
        }
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        # action_feature 已经是 numpy，batch 维度取 [0]
        if action_feature_np is not None:
            outputs["action_feature"] = action_feature_np[0]  # [L, D]
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
