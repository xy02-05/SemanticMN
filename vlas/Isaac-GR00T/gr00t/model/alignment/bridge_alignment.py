import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F


def _load_source_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_source_package(name: str, path: Path) -> ModuleType:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(path)]
    sys.modules[name] = package
    return package


def _load_alignment_factory():
    """绕过 EgoVLPv2 聚合式 __init__，只加载 alignment 的直接源码依赖。"""
    if "egovlpv2.model.fg_alignment_model" in sys.modules:
        return sys.modules["egovlpv2.model.fg_alignment_model"].create_alignment_model

    import egovlpv2

    source_root = Path(next(iter(egovlpv2.__path__)))
    base_package = _make_source_package("egovlpv2.base", source_root / "base")
    base_model = _load_source_module(
        "egovlpv2.base.base_model", source_root / "base/base_model.py"
    )
    base_package.BaseModel = base_model.BaseModel

    _make_source_package("egovlpv2.model", source_root / "model")
    _make_source_package("egovlpv2.model.egohod", source_root / "model/egohod")
    if "ipdb" not in sys.modules:
        ipdb_stub = ModuleType("ipdb")

        def unsupported_set_trace(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("ipdb.set_trace is unavailable in the minimal alignment environment")

        ipdb_stub.set_trace = unsupported_set_trace
        sys.modules["ipdb"] = ipdb_stub

    dependency_files = (
        ("egovlpv2.model.egohod.loss", "model/egohod/loss.py"),
        ("egovlpv2.model.feature_bank", "model/feature_bank.py"),
        ("egovlpv2.model.utils", "model/utils.py"),
        ("egovlpv2.model.loss", "model/loss.py"),
        ("egovlpv2.model.action_pooling", "model/action_pooling.py"),
        ("egovlpv2.model.disentangle", "model/disentangle.py"),
        ("egovlpv2.model.info_theory", "model/info_theory.py"),
    )
    for module_name, relative_path in dependency_files:
        _load_source_module(module_name, source_root / relative_path)

    alignment_module = _load_source_module(
        "egovlpv2.model.fg_alignment_model",
        source_root / "model/fg_alignment_model.py",
    )
    return alignment_module.create_alignment_model


class _AllGather(torch.autograd.Function):
    """在 batch 维合并各 rank 特征，并只回传当前 rank 对应的梯度。"""

    @staticmethod
    def forward(ctx, tensor, world_size, args):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.rank * ctx.batch_size
        end = start + ctx.batch_size
        return grad_output[start:end], None, None


class BridgeAlignmentAdapter(nn.Module):
    """把 GR00T DiT action hidden states 接到现有 Mirror Neuron AlignmentModel。"""

    def __init__(self, config_path: str, loss_weight: float):
        super().__init__()
        if loss_weight < 0:
            raise ValueError(f"alignment_loss_weight must be non-negative, got {loss_weight}")

        config_path = Path(config_path)
        config = json.loads(config_path.read_text())
        if not config["alignment"]["use_alignment"]:
            raise ValueError("alignment config must set alignment.use_alignment=true")

        alignment_args = config["alignment"]["args"]
        if alignment_args.get("vla_mode") != "gr00t":
            raise ValueError("GR00T alignment config must set alignment.args.vla_mode='gr00t'")

        # 延迟加载保证 use_alignment=false 时原 GR00T 环境不依赖 EgoVLPv2。
        create_alignment_model = _load_alignment_factory()
        self.alignment_model = create_alignment_model(alignment_args, dtype=torch.float32)
        self.layer_indices = tuple(alignment_args["layer_indices"])
        self.loss_weight = float(loss_weight)
        self.temperature = float(alignment_args["temperature"])
        self.sigmoid_bias = float(alignment_args["sigmoid_bias"])
        self.backbone_update_start_pct = float(
            config["alignment"]["training"].get("backbone_update_start_pct", 0.0)
        )
        if not 0.0 <= self.backbone_update_start_pct <= 1.0:
            raise ValueError(
                "alignment.training.backbone_update_start_pct must be in [0, 1]"
            )

        arch_args = config["arch"]["args"]
        embeddings_path = Path(arch_args["embeddings_path"])
        index_path = Path(arch_args["index_path"]) if arch_args.get("index_path") else None
        task_meta_path = Path(arch_args["task_meta_path"])
        dataset_task_path = Path(arch_args["dataset_task_path"])
        embedding_data = np.load(embeddings_path, allow_pickle=True)
        embeddings = torch.from_numpy(
            np.asarray(embedding_data["sentence_embeddings"], dtype=np.float32)
        )
        task_indices = torch.from_numpy(
            np.asarray(embedding_data["task_index"], dtype=np.int64)
        )
        embedding_texts = embedding_data["texts"] if "texts" in embedding_data.files else None
        embedding_data.close()

        expected_indices = torch.arange(task_indices.numel(), dtype=torch.long)
        if not torch.equal(task_indices, expected_indices):
            raise ValueError("embedding task_index must be contiguous and start from zero")
        if embeddings.shape[1] != alignment_args["egovlpv2_dim"]:
            raise ValueError(
                f"embedding dim {embeddings.shape[1]} does not match "
                f"egovlpv2_dim {alignment_args['egovlpv2_dim']}"
            )

        task_meta = [json.loads(line) for line in task_meta_path.open()]
        if len(task_meta) != embeddings.shape[0]:
            raise ValueError(
                f"task meta rows {len(task_meta)} do not match embedding rows {embeddings.shape[0]}"
            )
        for row_index, row in enumerate(task_meta):
            if row["task_id"] == -1:
                if row["task_index"] != -1:
                    raise ValueError(
                        f"invalid task row {row_index} must set task_index and task_id to -1"
                    )
            elif row["task_index"] != row_index:
                raise ValueError(
                    f"task meta row {row_index} has mismatched task_index={row['task_index']}"
                )
        task_ids = torch.tensor([row["task_id"] for row in task_meta], dtype=torch.long)

        # 有文本的 embedding 额外核对语义行；task_id=-1 的原始乱码允许被离线编码阶段清空。
        if embedding_texts is not None:
            for row, embedding_text in zip(task_meta, embedding_texts):
                if row["task_id"] != -1 and row["task"] != str(embedding_text):
                    raise ValueError(
                        f"task text mismatch at task_index={row['task_index']}: "
                        f"meta={row['task']!r}, embedding={str(embedding_text)!r}"
                    )
        if index_path is not None:
            index_meta = json.loads(index_path.read_text())
            if len(index_meta) != embeddings.shape[0]:
                raise ValueError(
                    f"embedding index rows {len(index_meta)} do not match "
                    f"embedding rows {embeddings.shape[0]}"
                )
            for row_index, (row, index_row) in enumerate(zip(task_meta, index_meta)):
                if index_row["idx"] != row_index:
                    raise ValueError(
                        f"embedding idx mismatch at row={row_index}"
                    )
                if index_row.get("task_index", index_row["idx"]) != row_index:
                    raise ValueError(
                        f"embedding task_index mismatch at row={row_index}"
                    )
                if row["task_id"] != -1 and index_row["task_id"] != row["task_id"]:
                    raise ValueError(
                        f"embedding task_id mismatch at row={row_index}"
                    )
                if row["task_id"] != -1 and index_row["text"] != row["task"]:
                    raise ValueError(
                        f"embedding index text mismatch at row={row_index}"
                    )

        # GR00T 的 LeRobot tasks.jsonl 已去重并重新编号，必须映射回 RLDS embedding 行。
        # strip+lower 在当前 Bridge 全量任务表上经过验证为 0 缺失、0 歧义。
        normalized_to_embedding: dict[str, int] = {}
        for embedding_index, row in enumerate(task_meta):
            normalized_text = row["task"].strip().lower()
            if normalized_text in normalized_to_embedding:
                raise ValueError(
                    f"normalized task text is ambiguous in task meta: {normalized_text!r}"
                )
            normalized_to_embedding[normalized_text] = embedding_index

        dataset_tasks = [json.loads(line) for line in dataset_task_path.open()]
        dataset_to_embedding = []
        for dataset_index, row in enumerate(dataset_tasks):
            if row["task_index"] != dataset_index:
                raise ValueError(
                    f"dataset task row {dataset_index} has task_index={row['task_index']}"
                )
            normalized_text = row["task"].strip().lower()
            if normalized_text not in normalized_to_embedding:
                raise ValueError(
                    f"dataset task has no embedding match at task_index={dataset_index}: "
                    f"{row['task']!r}"
                )
            dataset_to_embedding.append(normalized_to_embedding[normalized_text])

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        # 大型离线特征不写进 checkpoint；恢复训练时从配置路径重新读取。
        self.register_buffer("text_embeddings", embeddings, persistent=False)
        self.register_buffer("task_ids", task_ids, persistent=False)
        self.register_buffer(
            "dataset_to_embedding_index",
            torch.tensor(dataset_to_embedding, dtype=torch.long),
            persistent=False,
        )
        self.global_step = 0
        self.max_steps = 0

    def reset_alignment_parameters(self) -> None:
        """官方 base checkpoint 不含 alignment 权重时，显式恢复原模型初始化。"""
        self.alignment_model.apply(self.alignment_model._init_weights)
        for module in self.alignment_model.modules():
            if hasattr(module, "query") and isinstance(module.query, nn.Parameter):
                nn.init.normal_(module.query, mean=0.0, std=0.02)
        if self.alignment_model.log_temperature is not None:
            self.alignment_model.log_temperature.data.fill_(np.log(self.temperature))
        if self.alignment_model._temperature_buffer is not None:
            self.alignment_model._temperature_buffer.data.fill_(self.temperature)
        self.alignment_model.sigmoid_bias.data.fill_(self.sigmoid_bias)
        self.alignment_model._keep_scalar_trainables_fp32()
        for name, parameter in self.alignment_model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"alignment parameter is non-finite after reset: {name}")

    def set_training_progress(self, global_step: int, max_steps: int) -> None:
        self.global_step = int(global_step)
        self.max_steps = int(max_steps)

    def _should_detach_backbone(self) -> bool:
        detach_until = int(self.max_steps * self.backbone_update_start_pct + 0.5)
        return self.global_step < detach_until

    def _select_action_features(
        self,
        hidden_states: list[torch.Tensor],
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_steps = action_mask.bool().any(dim=-1)
        valid_counts = valid_steps.sum(dim=1)
        if not torch.all(valid_counts == valid_counts[0]):
            raise ValueError("all samples in one batch must have the same action horizon")

        valid_horizon = int(valid_counts[0].item())
        expected_mask = (
            torch.arange(action_mask.shape[1], device=action_mask.device)[None, :]
            < valid_counts[:, None]
        )
        if not torch.equal(valid_steps, expected_mask):
            raise ValueError("valid action timesteps must form a contiguous prefix")

        selected_layers = []
        for layer_index in self.layer_indices:
            if not 0 <= layer_index < len(hidden_states):
                raise ValueError(
                    f"alignment layer index {layer_index} is outside "
                    f"[0, {len(hidden_states) - 1}]"
                )
            layer_hidden = hidden_states[layer_index]
            action_hidden = layer_hidden[:, -action_mask.shape[1] :, :]
            selected_layers.append(action_hidden[:, :valid_horizon, :])

        action_features = torch.stack(selected_layers, dim=1)
        expected_dim = self.alignment_model.openvla_dim
        if action_features.shape[-1] != expected_dim:
            raise ValueError(
                f"GR00T hidden dim {action_features.shape[-1]} does not match "
                f"alignment openvla_dim {expected_dim}"
            )
        return action_features

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        action_mask: torch.Tensor,
        task_index: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        task_index = task_index.long()
        if task_index.ndim != 1:
            raise ValueError(f"task_index must have shape [B], got {task_index.shape}")
        if task_index.min() < 0 or task_index.max() >= self.dataset_to_embedding_index.shape[0]:
            raise IndexError("task_index is outside the LeRobot task table")
        embedding_index = self.dataset_to_embedding_index.index_select(0, task_index)

        action_features = self._select_action_features(hidden_states, action_mask)
        backbone_detached = self._should_detach_backbone()
        if backbone_detached:
            action_features = action_features.detach()

        model_dtype = next(self.alignment_model.parameters()).dtype
        action_features = action_features.to(dtype=model_dtype)
        text_features = self.text_embeddings.index_select(0, embedding_index).to(
            device=action_features.device, dtype=model_dtype
        )
        task_ids = self.task_ids.index_select(0, embedding_index).to(action_features.device)
        valid_samples = task_ids != -1
        if not valid_samples.any():
            # 保持 action hidden 和 alignment 参数都在计算图中，DDP/ZeRO 可得到显式零梯度。
            zero_loss = action_features.sum() * 0.0
            for parameter in self.alignment_model.parameters():
                zero_loss = zero_loss + parameter.sum() * 0.0
            return {
                "raw_loss": zero_loss,
                "weighted_loss": zero_loss,
                "loss_dict": {"total_loss": 0.0},
                "backbone_detached": torch.tensor(
                    float(backbone_detached), device=zero_loss.device
                ),
            }

        # 不在 adapter 外层删除无效样本：AlignmentModel 会用 task_id=-1 的 valid mask
        # 排除它们。保留固定 batch shape 可避免各 rank 无效样本数不同时 all_gather 卡死。

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        dist_args = SimpleNamespace(rank=rank, world_size=world_size)

        raw_loss, loss_dict, _ = self.alignment_model(
            openvla_features=action_features,
            egovlpv2_features=text_features,
            task_index=embedding_index,
            task_ids=task_ids,
            allgather=_AllGather.apply,
            n_gpu=world_size,
            args=dist_args,
        )
        weighted_loss = raw_loss * self.loss_weight
        return {
            "raw_loss": raw_loss,
            "weighted_loss": weighted_loss,
            "loss_dict": loss_dict,
            "backbone_detached": torch.tensor(
                float(backbone_detached), device=raw_loss.device
            ),
        }

    def reset_feature_bank(self) -> None:
        if hasattr(self.alignment_model, "reset_feature_bank"):
            self.alignment_model.reset_feature_bank()
