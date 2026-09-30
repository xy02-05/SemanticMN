from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import packaging.version
import torch
from typing import Any, Callable

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import Bridgev2RldsDataset, DroidRldsDataset, LiberoRldsDataset, RobotwinRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def load_task_mapping(task_mapping_path: str | None = None) -> dict[int, int]:
    """
    从显式指定的路径加载 task_index -> task_id 映射
    
    Args:
        task_mapping_path: tasks_with_id.jsonl 的显式路径，必须设置才能加载
    
    Returns:
        task_index_to_task_id: {task_index: task_id} 映射字典
    """
    task_index_to_task_id = {}
    
    if not task_mapping_path:
        logging.info("[TaskMapping] task_mapping_path未设置，task_id将回退为task_index")
        return {}
    
    tasks_path = Path(task_mapping_path)
    if not tasks_path.exists():
        logging.warning(f"[TaskMapping] 文件不存在: {task_mapping_path}")
        return {}
    
    with open(tasks_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            task_index = item.get("task_index", -1)
            if "task_id" in item and task_index >= 0:
                task_index_to_task_id[task_index] = item["task_id"]
    
    logging.info(f"[TaskMapping] Loaded {len(task_index_to_task_id)} task_id mappings from {tasks_path}")
    return task_index_to_task_id


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _column_to_numpy(values: Any) -> np.ndarray:
    """把 HuggingFace datasets 的 Column 统一转成 numpy，兼容新版返回类型。"""
    if hasattr(values, "to_pylist"):
        values = values.to_pylist()
    else:
        values = list(values)

    normalized = [v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v for v in values]
    return np.asarray(normalized)


def _column_to_tensor(values: Any) -> torch.Tensor:
    """把 HuggingFace datasets 的 Column 统一转成 torch.Tensor，兼容 select 后的返回类型。"""
    if isinstance(values, torch.Tensor):
        return values

    if hasattr(values, "to_pylist"):
        values = values.to_pylist()
    else:
        values = list(values)

    if not values:
        return torch.empty(0)

    tensors = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, np.ndarray):
            tensors.append(torch.from_numpy(value))
        else:
            tensors.append(torch.as_tensor(value))
    return torch.stack(tensors)


class CompatLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """兼容新版 datasets.Column 返回值，避免 LeRobot 初始化时 torch.stack 报错。"""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root) if root else lerobot_dataset.HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else lerobot_dataset.CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else lerobot_dataset.get_safe_default_codec()
        self.delta_indices = None

        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        print(f"[LeRobot] 加载元信息: {self.repo_id}")
        self.meta = lerobot_dataset.LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = lerobot_dataset.aggregate_stats(episodes_stats)

        print("[LeRobot] 加载 parquet 数据")
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = lerobot_dataset.get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = lerobot_dataset.get_episode_data_index(self.meta.episodes, self.episodes)

        print("[LeRobot] 检查 timestamp 同步")
        timestamps = _column_to_numpy(self.hf_dataset["timestamp"])
        episode_indices = _column_to_numpy(self.hf_dataset["episode_index"])
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        lerobot_dataset.check_timestamps_sync(
            timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s
        )

        if self.delta_timestamps is not None:
            lerobot_dataset.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = lerobot_dataset.get_delta_indices(self.delta_timestamps, self.fps)

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = _column_to_tensor(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        return {
            key: _column_to_tensor(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def __getitem__(self, idx) -> dict:
        # LeRobot 内部读取阶段仍然沿用 torch，等样本组装完成后再统一转回 numpy，
        # 这样可以保持 openpi 原有 transform 默认处理 numpy 的假设不变。
        item = super().__getitem__(idx)
        return jax.tree.map(
            lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x,
            item,
        )


def load_atomic_chunk_map(path: str | None) -> dict | None:
    """加载原子级对齐的 chunk→atomic_label_idx 映射（None=不启用）"""
    if not path:
        return None
    data = np.load(path)
    chunk_map = {
        "episode_offsets": data["episode_offsets"],
        "chunk_atomic_idx": data["chunk_atomic_idx"],
        "chunk_size": int(data["chunk_size"]),
    }
    n_ep = len(data["episode_offsets"]) - 1
    n_chunks = len(data["chunk_atomic_idx"])
    logging.info(f"[AtomicChunkMap] Loaded: {n_ep} episodes, {n_chunks} chunks from {path}")
    return chunk_map


def load_chunk_video_meta(path: str | None) -> tuple[object | None, int | None]:
    """从 chunk_video_features npz 读取 episode_offsets + chunk_size，
    用于 PromptFromLeRobotTask 计算 chunk_video_idx。
    """
    if not path:
        return None, None
    data = np.load(path)
    offsets = data["episode_offsets"]
    chunk_size = int(data["chunk_size"])
    n_ep = len(offsets) - 1
    n_chunks = int(offsets[-1])
    logging.info(f"[ChunkVideoMeta] Loaded offsets: {n_ep} episodes, {n_chunks} chunks, chunk_size={chunk_size} from {path}")
    return offsets, chunk_size


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = CompatLeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        # 加载 task_id 映射
        task_index_to_task_id = load_task_mapping(data_config.task_mapping_path)
        # 加载原子级对齐映射（可选，None=不启用）
        atomic_chunk_map = load_atomic_chunk_map(data_config.atomic_chunk_map_path)
        chunk_video_offsets, chunk_video_size = load_chunk_video_meta(data_config.chunk_video_features_path)
        dataset = TransformedDataset(dataset, [
            _transforms.PromptFromLeRobotTask(dataset_meta.tasks, task_index_to_task_id,
                                              atomic_chunk_map=atomic_chunk_map,
                                              chunk_video_offsets=chunk_video_offsets,
                                              chunk_video_size=chunk_video_size)
        ])

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training.
    使用 CompatLeRobotDataset 兼容 v2.0 格式（标准 LeRobotDataset 在
    v2.0 + 新版 datasets 库下会因 Column 类型不兼容而崩溃）。
    建议将数据集转换为 v2.1 格式后可改回标准 LeRobotDataset。
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = CompatLeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        task_index_to_task_id = load_task_mapping(data_config.task_mapping_path)
        atomic_chunk_map = load_atomic_chunk_map(data_config.atomic_chunk_map_path)
        chunk_video_offsets, chunk_video_size = load_chunk_video_meta(data_config.chunk_video_features_path)
        dataset = TransformedDataset(dataset, [
            _transforms.PromptFromLeRobotTask(dataset_meta.tasks, task_index_to_task_id,
                                              atomic_chunk_map=atomic_chunk_map,
                                              chunk_video_offsets=chunk_video_offsets,
                                              chunk_video_size=chunk_video_size)
        ])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # 检查是否为robotwin数据集
    data_dir = data_config.rlds_data_dir or ""
    repo_id = getattr(data_config, "repo_id", "") or ""
    is_robotwin = (
        getattr(data_config, 'is_robotwin_dataset', False) or 
        "robotwin" in data_dir.lower() or
        "agilex" in data_dir.lower() or
        repo_id.lower().startswith('robotwin') or
        repo_id.lower().startswith('agilex')
    )
    is_bridgev2 = (
        getattr(data_config, 'is_bridgev2_dataset', False) or 
        "bridgev2" in data_dir.lower() or
        repo_id.lower().startswith('bridgev2')
    )
    dataset_name = getattr(data_config, "dataset_name", None) or data_config.repo_id
    dataset_name_lower = (dataset_name or "").lower()
    is_libero = (
        getattr(data_config, "is_libero_dataset", False)
        or "libero" in data_dir.lower()
        or repo_id.lower().startswith("libero")
        or dataset_name_lower.startswith("libero")
    )
    if is_bridgev2:
        # 使用专门的bridgev2数据集加载器
        logging.info("使用Bridgev2RldsDataset加载bridgev2数据")
        return Bridgev2RldsDataset(
            data_dir=data_config.rlds_data_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            dataset_name=dataset_name,
            filter_dict_path=data_config.filter_dict_path,
            task_mapping_path=getattr(data_config, "task_mapping_path", None),
        )
    elif is_libero:
        logging.info("使用LiberoRldsDataset加载 LIBERO RLDS 数据")
        return LiberoRldsDataset(
            data_dir=data_config.rlds_data_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            shuffle_buffer_size=getattr(data_config, "shuffle_buffer_size", None) or 100_000,
            dataset_name=dataset_name,
            task_mapping_path=getattr(data_config, "task_mapping_path", None),
        )
    elif is_robotwin:
        # 使用专门的robotwin数据集加载器
        logging.info("使用RobotwinRldsDataset加载robotwin数据")
        return RobotwinRldsDataset(
            data_dir=data_config.rlds_data_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            dataset_name=dataset_name,
            filter_dict_path=data_config.filter_dict_path,
            task_mapping_path=getattr(data_config, "task_mapping_path", None),
        )
    elif hasattr(data_config, 'action_space') and data_config.action_space is not None:
        # 使用原有的DROID数据集加载器
        return DroidRldsDataset(
            data_dir=data_config.rlds_data_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            action_space=data_config.action_space,
            filter_dict_path=data_config.filter_dict_path,
        )
    else:
        raise NotImplementedError("Unknown dataset type.")


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        # RLDS datasets are instantiated independently by every PyTorch rank,
        # so their batch size must already be the per-rank batch size.  Unlike
        # create_torch_data_loader(), this path is not wrapped in a
        # DistributedSampler that divides the global batch by world size.
        rlds_batch_size = config.batch_size
        if framework == "pytorch" and getattr(config, "per_device_batch_size", None) is not None:
            rlds_batch_size = config.per_device_batch_size
        logging.info(
            "RLDS batch size: framework=%s global=%s per_rank=%s",
            framework,
            config.batch_size,
            rlds_batch_size,
        )
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=rlds_batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader, framework=framework)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        framework: The framework to use ("jax" or "pytorch").
    """
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    if framework == "pytorch":
        data_loader = PyTorchRLDSDataLoader(
            dataset,
            num_batches=num_batches,
        )
    else:
        data_loader = RLDSDataLoader(
            dataset,
            sharding=sharding,
            num_batches=num_batches,
        )

    return DataLoaderImpl(data_config, data_loader, framework=framework)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    batch = _filter_non_numeric(batch)
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(_maybe_to_torch, batch)


def _filter_non_numeric(batch):
    """Remove non-numeric (e.g. string) leaves from a pytree before JAX device_put."""
    if isinstance(batch, dict):
        return {k: _filter_non_numeric(v) for k, v in batch.items()
                if not (isinstance(v, np.ndarray) and not (np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.bool_)))}
    if isinstance(batch, (list, tuple)):
        filtered = [_filter_non_numeric(v) for v in batch]
        return type(batch)(filtered)
    return batch


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _maybe_to_torch(x):
    """仅把数值叶子转成 torch.Tensor，字符串字段保持原样。"""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        if np.issubdtype(x.dtype, np.str_) or np.issubdtype(x.dtype, np.bytes_) or x.dtype == object:
            return x
        return torch.as_tensor(x)
    if isinstance(x, np.generic):
        if np.issubdtype(x.dtype, np.str_) or np.issubdtype(x.dtype, np.bytes_):
            return x
        return torch.as_tensor(x)
    return x


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class PyTorchRLDSDataLoader:
    """PyTorch版本的RLDS数据加载器，用于替代JAX版本的RLDSDataLoader。
    
    该类专门为PyTorch框架设计，返回PyTorch张量而非JAX数组。
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

    @staticmethod
    def _tree_map(fn: Callable, *trees: Any) -> Any:
        """PyTorch专用的 tree_map 函数"""
        if len(trees) == 1:
            tree = trees[0]
            if isinstance(tree, dict):
                return {k: PyTorchRLDSDataLoader._tree_map(fn, v) for k, v in tree.items()}
            elif isinstance(tree, (list, tuple)):
                result = [PyTorchRLDSDataLoader._tree_map(fn, item) for item in tree]
                return type(tree)(result)
            else:
                return fn(tree)
        else:
            # 多个树的情况
            if all(isinstance(tree, dict) for tree in trees):
                if not all(set(tree.keys()) == set(trees[0].keys()) for tree in trees):
                    raise ValueError("All dictionaries must have the same keys")
                return {k: PyTorchRLDSDataLoader._tree_map(fn, *(tree[k] for tree in trees)) for k in trees[0].keys()}
            elif all(isinstance(tree, (list, tuple)) for tree in trees):
                if not all(len(tree) == len(trees[0]) for tree in trees):
                    raise ValueError("All sequences must have the same length")
                result = [PyTorchRLDSDataLoader._tree_map(fn, *(tree[i] for tree in trees)) for i in range(len(trees[0]))]
                return type(trees[0])(result)
            else:
                return fn(*trees)

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # 将numpy数组转换为PyTorch张量
                # yield self._tree_map(lambda x: torch.from_numpy(np.asarray(x)) if isinstance(x, (np.ndarray, jnp.ndarray)) else x, batch)
                yield self._tree_map(lambda x: torch.from_numpy(np.asarray(x)) if isinstance(x, (np.ndarray, jnp.ndarray)) and not np.issubdtype(x.dtype, np.str_) else x, batch)


class DataLoaderImpl(DataLoader):
    """JAX 训练返回官方 (observation, actions) 2-tuple；
    PyTorch alignment 训练返回扩展 tuple（含 task_index/task_id 等）。"""

    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader | PyTorchRLDSDataLoader, framework: str = "jax"):
        self._data_config = data_config
        self._data_loader = data_loader
        self._framework = framework

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            actions = batch["actions"]
            # JAX 训练走官方 2-tuple 格式
            if self._framework == "jax":
                yield observation, actions
                continue
            # PyTorch alignment 训练走扩展 tuple
            atomic_label_idx = batch.get("atomic_label_idx", None)
            chunk_video_idx = batch.get("chunk_video_idx", None)
            if "task_id" in batch and "task_index" in batch:
                yield (observation, actions, batch["prompt"],
                       batch["task_index"], batch["task_id"], atomic_label_idx, chunk_video_idx)
            elif "task_index" in batch:
                yield (observation, actions, batch["prompt"],
                       batch["task_index"], batch["task_index"], atomic_label_idx, chunk_video_idx)
            elif "task_id" in batch:
                yield (observation, actions, batch["prompt"],
                       batch["task_id"], batch["task_id"], atomic_label_idx, chunk_video_idx)
            else:
                yield (observation, actions, batch["prompt"],
                       None, None, atomic_label_idx, chunk_video_idx)
