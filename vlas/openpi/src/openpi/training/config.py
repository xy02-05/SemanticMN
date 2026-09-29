"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # RLDS 数据集名称（用于 tfds.builder 的 dataset_name）
    dataset_name: str | None = None
    # RLDS shuffle buffer 大小。仅 RLDS 路径使用。
    shuffle_buffer_size: int | None = None
    # 任务映射文件路径（可选，优先级高于自动搜索）
    task_mapping_path: str | None = None
    # 原子级对齐: chunk→atomic_label_idx 映射文件路径（None=不启用）
    atomic_chunk_map_path: str | None = None
    # chunk-level video 对齐: chunk video features npz 路径（None=不启用）
    # 文件由 data_process/video_feature/chunk_video/chunk_video_extract.py 生成；
    # transforms 仅读取其 episode_offsets/chunk_size，真正 embedding 由
    # EmbeddingModel.load_chunk_video_features 加载（路径由 alignment config 给出）。
    chunk_video_features_path: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        # RepackTransform 语义：dict 的 KEY 是输出 key，VALUE 是输入(flat_item)的查找 key
        # 例如 "observation/image": "image" 表示 output["observation/image"] = flat_item["image"]
        # 这将数据集的 "image" 映射为推理格式 "observation/image"（LiberoInputs 需要）
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "task_index": "task_index",
                        "task_id": "task_id",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                        # 透传 alignment 用的两个 chunk-level 索引（PromptFromLeRobotTask
                        # 始终输出 -1 sentinel，下游 alignment 按 -1 跳过）
                        "atomic_label_idx": "atomic_label_idx",
                        "chunk_video_idx": "chunk_video_idx",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSLiberoDataConfig(DataConfigFactory):
    """
    LIBERO 的 RLDS 版数据配置。

    设计原则：
    1. 尽量复用 LeRobotLiberoDataConfig 的训练语义。
    2. 只把数据入口改成 RLDS。
    3. repack 后的字段名继续对齐当前 LIBERO LeRobot 路径，
       这样可以直接复用 libero_policy.LiberoInputs/Outputs。
    """

    rlds_data_dir: str | None = None
    dataset_name: str = "libero_mix_no_noops"
    shuffle_buffer_size: int = 100_000
    filter_dict_path: str | None = None
    task_mapping_path: str | None = None
    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/wrist_image": "observation/wrist_image",
                        "observation/state": "observation/state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # 对齐训练依赖这两个字段，必须继续透传。
                        "task_index": "task_index",
                        "task_id": "task_id",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "需要设置 rlds_data_dir 用于 LIBERO RLDS 数据加载器。"
        assert self.task_mapping_path is not None, "需要设置 task_mapping_path 用于 LIBERO task 映射。"

        base_config = self.create_base_config(assets_dirs, model_config)
        base_config = dataclasses.replace(
            base_config,
            prompt_from_task=False,
            task_mapping_path=self.task_mapping_path,
        )

        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            dataset_name=self.dataset_name,
            shuffle_buffer_size=self.shuffle_buffer_size,
            action_space=None,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class RobotwinRLDSDataConfig(DataConfigFactory):
    """
    专门为robotwin RLDS数据集设计的配置类。
    与LeRobot Aloha格式完全对齐。
    """
    
    # robotwin RLDS数据目录路径
    rlds_data_dir: str | None = None
    
    # robotwin特定参数
    action_chunk_size: int = 50  # robotwin推荐的动作序列长度
    dataset_name: str = "robotwin_full_hard"  # robotwin数据集名称
    
    # 数据过滤文件路径（可选）
    filter_dict_path: str | None = None
    
    # 任务映射文件路径（可选，用于生成task_index和task_id）
    task_mapping_path: str | None = None
    
    # 默认提示词
    default_prompt: str | None = None
    
    # 是否从任务索引生成提示词
    prompt_from_task: bool = True
    
    # 与LeRobot对齐：使用delta关节动作，保持夹爪绝对值
    use_delta_joint_actions: bool = True
    
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
    
    # 是否只使用cam_high摄像头，忽略left_wrist和right_wrist摄像头
    use_only_cam_high: bool = False
    # robotwin数据重新打包变换，将RobotWin字段映射到LeRobot标准格式
    repack_transform: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
       default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",  # RobotWin的主摄像头
                        "cam_left_wrist": "observation/wrist_image",  # RobotWin的左手腕摄像头
                        "cam_right_wrist": "observation/wrist_image_right",  # RobotWin的右手腕摄像头
                    },
                    "state": "observation/state",  # 将joint_position映射为state
                    "actions": "actions",  # 动作字段保持不变
                    "prompt": "prompt",  # 提示词字段保持不变
                    "task_index": "task_index",  # 任务索引
                    "task_id": "task_id",  # 任务ID（若不存在由上游填充）
                })
            ]
        )
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        
        # robotwin数据变换 - 完全参照LeRobot Aloha格式
        # 根据use_only_cam_high配置决定是否只使用cam_high
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi, use_only_cam_high=self.use_only_cam_high)], 
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)]
        )
        
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        
        # 模型特定变换
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        
        assert self.rlds_data_dir is not None, "需要设置rlds_data_dir用于robotwin RLDS数据加载器。"
        
        # 创建基础配置
        base_config = self.create_base_config(assets_dirs, model_config)
        base_config = dataclasses.replace(
            base_config,
            prompt_from_task=self.prompt_from_task,
            task_mapping_path=self.task_mapping_path,  # 修复：传递task_mapping_path，否则task_id全为-1
        )
        
        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            dataset_name=self.dataset_name,
            action_space=None,  # robotwin不需要action_space
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class RLDSBridgev2DataConfig(DataConfigFactory):
    """
    专门为Bridgev2 RLDS数据集设计的配置类。
    与LeRobot Aloha格式完全对齐。
    """
    
    # Bridgev2 RLDS数据目录路径
    rlds_data_dir: str | None = None
    
    # Bridgev2特定参数
    action_chunk_size: int = 5  # Bridgev2推荐的动作序列长度
    dataset_name: str = "bridgev2"  # Bridgev2数据集名称
    
    # 数据过滤文件路径（可选）
    filter_dict_path: str | None = None
    
    # 任务映射文件路径（可选，优先级高于自动搜索）
    task_mapping_path: str | None = None
    
    # 默认提示词
    default_prompt: str | None = None
    
    # 是否从任务索引生成提示词
    prompt_from_task: bool = False
    
    # 与LeRobot对齐：使用delta关节动作，保持夹爪绝对值
    use_delta_joint_actions: bool = True
    
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
    
    use_only_cam_high: bool = True
    # Bridgev2数据重新打包变换，将Bridgev2字段映射到LeRobot标准格式
    repack_transform: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
       default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image_0",  # Bridgev2的主摄像头
                    },
                    "state": "observation/state", 
                    "actions": "action",  
                    "prompt": "language_instruction",  
                    "task_index": "task_index",  # 任务索引
                    "task_id": "task_id",  # 任务ID（若不存在由上游填充）
                })
            ]
        )
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        
        # Bridgev2数据变换 - 完全参照LeRobot Aloha格式
        # 根据use_only_cam_high配置决定是否只使用cam_high
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi, use_only_cam_high=self.use_only_cam_high)], 
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)]
        )
        
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        
        # 模型特定变换
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        
        assert self.rlds_data_dir is not None, "需要设置rlds_data_dir用于Bridgev2 RLDS数据加载器。"
        
        # 创建基础配置
        base_config = self.create_base_config(assets_dirs, model_config)
        base_config = dataclasses.replace(
            base_config,
            prompt_from_task=self.prompt_from_task,
            task_mapping_path=self.task_mapping_path,
        )
        
        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            dataset_name=self.dataset_name,
            action_space=None,  # Bridgev2不需要action_space
            filter_dict_path=self.filter_dict_path,
        )



@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    deepspeed_config_file: str | None = None
    
    # ================== 内存优化参数 ==================
    # 是否启用梯度检查点（gradient checkpointing）来节省显存，会牺牲一定训练速度
    enable_gradient_checkpointing: bool = True

    # ================== LoRA相关参数 ==================
    lora_config: dict[str, Any] | None = None  # LoRA配置字典

    # ================== 对齐训练相关参数 ==================
    # 是否启用EgoVLPv2训练（用于多模态对齐）
    use_egovlpv2: bool = False
    # 是否启用Alignment模型训练（用于VLA和VLM的特征对齐）
    use_alignment: bool = False
    # EgoVLPv2配置文件路径（当use_egovlpv2或use_alignment为True时必需）
    egovlpv2_config_path: str | None = None
    # VLM损失权重（EgoVLPv2 loss的权重系数）
    vlm_loss_weight: float = 1.0
    # 对齐损失权重（Alignment loss的权重系数）
    alignment_loss_weight: float = 1.0
    # 动态alignment loss权重调度：warmup期间从init_weight线性增加到alignment_loss_weight
    # 用于避免训练初期PI0特征不稳定时alignment梯度的干扰
    alignment_loss_warmup: bool = False  # 是否启用alignment loss warmup
    alignment_loss_warmup_init: float = 0.0  # warmup起始权重
    alignment_loss_warmup_steps: int | None = None  # warmup步数（None则复用LR warmup_steps）
    # 是否冻结VLM模型参数（冻结时不更新VLM参数，可减少显存计算）
    freeze_vlm: bool = False
    # 是否使用learnable token（镜像神经元对齐，用于VLA和VLM的特征对齐）
    use_learnable_token: bool = False
    # 用于对齐的层索引（默认选择后几层，例如[20, 25]）
    use_new_optimizer: bool = True

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size（期望的总batch size，会根据GPU数量和梯度累积自动调整）.
    # batch_size = per_device_batch_size * GPU数量 * gradient_accumulation_steps
    batch_size: int = 32
    # 每个GPU的实际batch size（如果不指定，会自动计算）
    per_device_batch_size: int | None = None
    # 梯度累积步数（累积多少步后才更新一次权重）
    gradient_accumulation_steps: int = 1
    # 是否在每次optimizer step后重置feature bank
    # False（默认）: 跨step保留bank，利用FIFO循环缓冲区自然淘汰旧特征，所有micro-batch负样本数一致
    # True: 每次step后清空bank，每个梯度累积周期第1个micro-batch没有bank负样本（不对称）
    reset_bank_every_step: bool = True
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False
    # Explicit recovery mode for checkpoints whose optimizer.pt was intentionally removed.
    # Model/alignment/metadata are still loaded strictly; AdamW moments restart empty.
    allow_missing_optimizer_state: bool = False
    # Optional LR ramp applied after an optimizer-state reset at a nonzero global step.
    optimizer_restart_step: int | None = None
    optimizer_restart_warmup_steps: int = 0
    optimizer_restart_warmup_init_ratio: float = 0.1

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def _libero_qwen_dsn_30k_config(name: str, alignment_config_name: str) -> TrainConfig:
    """构造一组只改变 DSN loss 权重的 LIBERO 30k 受控消融配置。"""
    work_root = "/mnt/bn/2d-videos/xy/work/mirror_neuron"
    return TrainConfig(
        name=name,
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id=f"{work_root}/data/physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir=(
                    f"{work_root}/mirror_neuron/vlas/openpi/"
                    "checkpoints/pi0_libero_full/raw/30000/assets"
                ),
                asset_id="libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path=(
                    f"{work_root}/data/physical-intelligence/libero/meta/tasks_with_id.jsonl"
                ),
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path=f"{work_root}/weights/openpi/pi0_base_pytorch",
        batch_size=32,
        per_device_batch_size=None,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=2,
        log_interval=50,
        save_interval=5_000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path=(
            f"{work_root}/mirror_neuron/egovlpv2/egovlpv2/configs/ft/"
            f"{alignment_config_name}"
        ),
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
        alignment_loss_warmup=False,
        alignment_loss_warmup_init=0.0,
        alignment_loss_warmup_steps=None,
        reset_bank_every_step=True,
        wandb_enabled=True,
    )


def _libero_qwen_recon001_50k_config(name: str, alignment_config_name: str) -> TrainConfig:
    """构造 50k LIBERO DSN 对照，学习率仍按原 30k 时间尺度衰减。"""
    work_root = "/mnt/bn/2d-videos/xy/work/mirror_neuron"
    return TrainConfig(
        name=name,
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id=f"{work_root}/data/physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir=(
                    f"{work_root}/mirror_neuron/vlas/openpi/"
                    "checkpoints/pi0_libero_full/raw/30000/assets"
                ),
                asset_id="libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path=(
                    f"{work_root}/data/physical-intelligence/libero/meta/tasks_with_id.jsonl"
                ),
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path=f"{work_root}/weights/openpi/pi0_base_pytorch",
        batch_size=32,
        per_device_batch_size=None,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=50_000,
        num_workers=2,
        log_interval=50,
        save_interval=5_000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path=(
            f"{work_root}/mirror_neuron/egovlpv2/egovlpv2/configs/ft/"
            f"{alignment_config_name}"
        ),
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
        alignment_loss_warmup=False,
        alignment_loss_warmup_init=0.0,
        alignment_loss_warmup_steps=None,
        reset_bank_every_step=True,
        wandb_enabled=True,
    )


def _bridge_egohod_infonce_8gpu_config(
    name: str, alignment_config_name: str
) -> TrainConfig:
    """Bridge full fine-tuning with global batch 128 on 8 GPUs."""
    work_root = os.environ.get(
        "MIRROR_NEURON_WORK_ROOT",
        "/mnt/bn/2d-videos/xy/work/mirror_neuron",
    )
    code_root = os.environ.get(
        "MIRROR_NEURON_CODE_ROOT",
        "/mnt/bn/2d-videos/xy/work/mirror_neuron_final",
    )
    return TrainConfig(
        name=name,
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
            max_token_len=64,
            vlm_mode="embedding",
        ),
        # freeze_vlm only freezes the external semantic encoder. The pi0
        # backbone remains fully trainable because no freeze_filter is set.
        freeze_vlm=True,
        data=RLDSBridgev2DataConfig(
            repo_id="bridgev2",
            # Reuse the exact Bridge normalization statistics from the
            # action-only baseline so pre/post-pool runs differ only in DSN.
            assets=AssetsConfig(
                assets_dir=(
                    f"{work_root}/mirror_neuron/vlas/openpi/"
                    "assets/bridgev2_rlds_train"
                ),
                asset_id="bridgev2",
            ),
            adapt_to_pi=False,
            rlds_data_dir=f"{work_root}/data/bridge-rlds",
            action_chunk_size=5,
            dataset_name="bridge_orig",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=False,
            task_mapping_path=(
                f"{work_root}/data/bridge_orig/bridge_orig_lerobot/"
                "meta/tasks_with_id.jsonl"
            ),
            use_only_cam_high=True,
            repack_transform=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {"cam_high": "observation/image"},
                            "state": "observation/state",
                            "actions": "actions",
                            "prompt": "prompt",
                            "task_index": "task_index",
                            "task_id": "task_id",
                        }
                    )
                ]
            ),
        ),
        pytorch_weight_path=f"{work_root}/weights/openpi/pi0_base_pytorch",
        batch_size=128,
        per_device_batch_size=None,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=True,
        num_train_steps=50_000,
        num_workers=0,
        log_interval=50,
        save_interval=5_000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=50_000,
            decay_lr=5e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path=(
            f"{code_root}/egovlpv2/egovlpv2/configs/ft/{alignment_config_name}"
        ),
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
        wandb_enabled=False,
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    _bridge_egohod_infonce_8gpu_config(
        "bridgev2_egohod_infonce_dsn_prepool_h512_8gpu",
        "pi0_bridge_egohod_infonce_dsn_prepool_h512.json",
    ),
    _bridge_egohod_infonce_8gpu_config(
        "bridgev2_egohod_infonce_dsn_postpool_h1024_8gpu",
        "pi0_bridge_egohod_infonce_dsn_postpool_h1024.json",
    ),
    # ===== 与官方 openpi 完全对齐的 pi0 LIBERO 配置 =====
    # 用于 JAX 全参数训练，或作为 serve_policy 加载 PyTorch checkpoint 的配置
    TrainConfig(
        name="pi0_libero",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            # JAX 训练直接复用仓库内现成的 LIBERO norm stats，
            # 避免默认去找不存在的 assets/pi0_libero/libero。
            assets=AssetsConfig(assets_dir="./assets", asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/data/xuyuan1/dataset/pi0_base_official/params"
        ),
        num_train_steps=30_000,
    ),
    # ===== 官方 pi0_libero checkpoint 专用评测配置 =====
    # 只用于加载 /root/data/xuyuan1/dataset/pi0_libero_official 这类官方 JAX checkpoint。
    # 这里保持官方 repo_id，目的是让 asset_id 与官方 checkpoint 里的
    # assets/physical-intelligence/libero/norm_stats.json 完全对齐。
    TrainConfig(
        name="pi0_libero_official_eval",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/data/xuyuan1/dataset/pi0_base_official/params"
        ),
        num_train_steps=30_000,
    ),
    # ===== Pi0.5 LIBERO 官方 checkpoint 评测配置 =====
    # Pi0.5 checkpoint 的 norm_stats 是原始 delta 动作统计量（非 double delta），
    # 不需要 extra_delta_transform。
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/data/xuyuan1/dataset/pi05_libero_official/params"
        ),
        num_train_steps=30_000,
    ),
    # ===== Pi0.5 LIBERO PyTorch checkpoint 评测配置 =====
    TrainConfig(
        name="pi05_libero_pytorch",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=False,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi05_libero_pytorch",
        num_train_steps=30_000,
    ),
    # ===== Pi0-FAST LIBERO 官方微调 checkpoint 评测配置 =====
    TrainConfig(
        name="pi0_fast_libero",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/data/xuyuan1/dataset/pi0_fast_libero_official/params"
        ),
        num_train_steps=30_000,
    ),
    # ===== Pi0-FAST LIBERO 从 base 权重全量微调（JAX 训练） =====
    # assets 指向 libero 已有的 norm_stats，避免重复计算
    TrainConfig(
        name="pi0_fast_libero_train",
        exp_name="pi0_fast_libero_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="/root/data/xuyuan1/Codes/mirror_neuron/vlas/openpi/assets",
                asset_id="libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/data/xuyuan1/dataset/pi0_fast_base/params"
        ),
        # 对齐官方 lerobot/pi0fast-libero 训练设置
        num_train_steps=20_000,
        fsdp_devices=2,
        batch_size=32,
        gradient_accumulation_steps=2,
        seed=1000,
        enable_gradient_checkpointing=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=4_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=0.01,
            clip_gradient_norm=1.0,
        ),
        log_interval=10,
        save_interval=2500,
        wandb_enabled=False,
        overwrite=True,
    ),
    # ===== LIBERO LeRobot 配置（兼容当前 PyTorch + Alignment 训练链路） =====
    TrainConfig(
        name="pi0_libero_lora",
        resume=False,
        # PyTorch 训练链路里的 LoRA 统一由 train_pytorch_*.py 里的 PEFT 注入。
        # 这里保持基础 Pi0Config，避免和官方 JAX 的 *_lora 变体语义混用。
        model=pi0_config.Pi0Config(),
        freeze_vlm=True,
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
            ),
            # 与官方 pi0_libero 保持一致，兼容旧版 pi0 checkpoint 的动作定义。
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=32,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=50_000,
        num_workers=2,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=50000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0,
        lora_config={
            "lora_rank_paligemma": 16,
            "lora_alpha_paligemma": 16,
            "lora_rank_gemma_expert": 32,
            "lora_alpha_gemma_expert": 32,
        },
    ),
    TrainConfig(
        name="pi0_libero_full",
        resume=False,
        freeze_vlm=True,
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
            ),
            # 与官方 pi0_libero 保持一致，兼容旧版 pi0 checkpoint 的动作定义。
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=32,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=2,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0,
    ),
    TrainConfig(
        name="pi0_libero_rlds_lora",
        resume=False,
        freeze_vlm=True,
        data=RLDSLiberoDataConfig(
            repo_id="libero_mix_no_noops",
            assets=AssetsConfig(asset_id="libero_mix_no_noops"),
            rlds_data_dir="/root/data/xuyuan1/dataset/libero_rlds",
            dataset_name="libero_mix_no_noops",
            task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=32,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=0,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0,
        lora_config={
            "lora_rank_paligemma": 16,
            "lora_alpha_paligemma": 16,
            "lora_rank_gemma_expert": 32,
            "lora_alpha_gemma_expert": 32,
        },
    ),
    TrainConfig(
        name="pi0_libero_align_lora",
        resume=False,
        # Alignment 版本只在配置层复用现有类和函数，不改训练主逻辑。
        model=pi0_config.Pi0Config(vlm_mode="egohod"),
        freeze_vlm=True,
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
            ),
            # 与官方 pi0_libero 保持一致，兼容旧版 pi0 checkpoint 的动作定义。
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=32,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=0,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        # 直接复用当前 alignment 版本已经稳定使用的 EgoHOD + Alignment 配置。
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_fixed_lora.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
        lora_config={
            "lora_rank_paligemma": 16,
            "lora_alpha_paligemma": 16,
            "lora_rank_gemma_expert": 32,
            "lora_alpha_gemma_expert": 32,
        },
    ),
    TrainConfig(
        name="pi0_libero_full_qwen",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
            ),
            # 与官方 pi0_libero 保持一致，兼容旧版 pi0 checkpoint 的动作定义。
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=16,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=2,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        # Qwen3 Embedding 离线查表配置：egovlpv2_dim=4096, token_egovlpv2_dim=4096
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3_embedding_libero.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1
    ),
    # ===== 0 -> 50k / 2-GPU / global batch 32 stability comparison =====
    # 两个配置除 alignment 相关字段外保持一致。batch_size 是全局 batch；
    # 两卡、无梯度累积时每卡 batch 自动计算为 16。
    TrainConfig(
        name="pi0_libero_full_50k_gbs32",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id=(
                "/mnt/bn/2d-videos/xy/work/mirror_neuron/data/physical-intelligence/libero"
            ),
            assets=AssetsConfig(
                assets_dir=(
                    "/mnt/bn/2d-videos/xy/work/mirror_neuron/mirror_neuron/vlas/openpi/"
                    "checkpoints/pi0_libero_full/raw/30000/assets"
                ),
                asset_id="libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path=(
                    "/mnt/bn/2d-videos/xy/work/mirror_neuron/data/physical-intelligence/"
                    "libero/meta/tasks_with_id.jsonl"
                ),
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path=(
            "/mnt/bn/2d-videos/xy/work/mirror_neuron/weights/openpi/pi0_base_pytorch"
        ),
        batch_size=32,
        per_device_batch_size=None,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=50_000,
        num_workers=2,
        log_interval=50,
        save_interval=5_000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0,
        alignment_loss_warmup=False,
        alignment_loss_warmup_init=0.0,
        alignment_loss_warmup_steps=None,
        reset_bank_every_step=True,
        wandb_enabled=True,
    ),
    TrainConfig(
        name="pi0_libero_full_qwen_50k_gbs32",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id=(
                "/mnt/bn/2d-videos/xy/work/mirror_neuron/data/physical-intelligence/libero"
            ),
            assets=AssetsConfig(
                assets_dir=(
                    "/mnt/bn/2d-videos/xy/work/mirror_neuron/mirror_neuron/vlas/openpi/"
                    "checkpoints/pi0_libero_full/raw/30000/assets"
                ),
                asset_id="libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path=(
                    "/mnt/bn/2d-videos/xy/work/mirror_neuron/data/physical-intelligence/"
                    "libero/meta/tasks_with_id.jsonl"
                ),
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path=(
            "/mnt/bn/2d-videos/xy/work/mirror_neuron/weights/openpi/pi0_base_pytorch"
        ),
        batch_size=32,
        per_device_batch_size=None,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=50_000,
        num_workers=2,
        log_interval=50,
        save_interval=5_000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path=(
            "/mnt/bn/2d-videos/xy/work/mirror_neuron/mirror_neuron/egovlpv2/"
            "egovlpv2/configs/ft/pi0_align_qwen3_embedding_libero_50k_gbs32.json"
        ),
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
        alignment_loss_warmup=False,
        alignment_loss_warmup_init=0.0,
        alignment_loss_warmup_steps=None,
        reset_bank_every_step=True,
        wandb_enabled=True,
    ),
    _libero_qwen_recon001_50k_config(
        "pi0_libero_full_qwen_recon001_prepool_50k_gbs32",
        "pi0_align_qwen3_embedding_libero_recon001_prepool_50k_gbs32.json",
    ),
    _libero_qwen_recon001_50k_config(
        "pi0_libero_full_qwen_recon001_postpool_50k_gbs32",
        "pi0_align_qwen3_embedding_libero_recon001_postpool_50k_gbs32.json",
    ),
    _libero_qwen_dsn_30k_config(
        "pi0_libero_full_qwen_dsn_full_30k_gbs32",
        "pi0_align_qwen3_embedding_libero_dsn_full_30k_gbs32.json",
    ),
    _libero_qwen_dsn_30k_config(
        "pi0_libero_full_qwen_dsn_no_recon_30k_gbs32",
        "pi0_align_qwen3_embedding_libero_dsn_no_recon_30k_gbs32.json",
    ),
    _libero_qwen_dsn_30k_config(
        "pi0_libero_full_qwen_dsn_no_diff_30k_gbs32",
        "pi0_align_qwen3_embedding_libero_dsn_no_diff_30k_gbs32.json",
    ),
    # ===== 30k checkpoint -> 50k / 2-GPU / global batch 32 comparison =====
    TrainConfig(
        name="pi0_libero_full_continue_30k_50k_gbs32",
        resume=True,
        allow_missing_optimizer_state=True,
        optimizer_restart_step=30_000,
        optimizer_restart_warmup_steps=200,
        optimizer_restart_warmup_init_ratio=0.1,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id=(
                "/mnt/bn/2d-videos/xy/work/mirror_neuron/data/physical-intelligence/libero"
            ),
            assets=AssetsConfig(
                assets_dir=(
                    "/mnt/bn/2d-videos/xy/work/mirror_neuron/mirror_neuron/vlas/openpi/"
                    "checkpoints/pi0_libero_full/raw/30000/assets"
                ),
                asset_id="libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path=(
                    "/mnt/bn/2d-videos/xy/work/mirror_neuron/data/physical-intelligence/"
                    "libero/meta/tasks_with_id.jsonl"
                ),
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path=(
            "/mnt/bn/2d-videos/xy/work/mirror_neuron/weights/openpi/pi0_base_pytorch"
        ),
        batch_size=32,
        per_device_batch_size=None,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=50_000,
        num_workers=2,
        log_interval=50,
        save_interval=5_000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0,
        alignment_loss_warmup=False,
        alignment_loss_warmup_init=0.0,
        alignment_loss_warmup_steps=None,
        reset_bank_every_step=True,
        wandb_enabled=True,
    ),
    TrainConfig(
        name="pi0_libero_full_qwen_continue_30k_50k_gbs32",
        resume=True,
        allow_missing_optimizer_state=True,
        optimizer_restart_step=30_000,
        optimizer_restart_warmup_steps=200,
        optimizer_restart_warmup_init_ratio=0.1,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id=(
                "/mnt/bn/2d-videos/xy/work/mirror_neuron/data/physical-intelligence/libero"
            ),
            assets=AssetsConfig(
                assets_dir=(
                    "/mnt/bn/2d-videos/xy/work/mirror_neuron/mirror_neuron/vlas/openpi/"
                    "checkpoints/pi0_libero_full/raw/30000/assets"
                ),
                asset_id="libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path=(
                    "/mnt/bn/2d-videos/xy/work/mirror_neuron/data/physical-intelligence/"
                    "libero/meta/tasks_with_id.jsonl"
                ),
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path=(
            "/mnt/bn/2d-videos/xy/work/mirror_neuron/weights/openpi/pi0_base_pytorch"
        ),
        batch_size=32,
        per_device_batch_size=None,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=50_000,
        num_workers=2,
        log_interval=50,
        save_interval=5_000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path=(
            "/mnt/bn/2d-videos/xy/work/mirror_neuron/mirror_neuron/egovlpv2/"
            "egovlpv2/configs/ft/"
            "pi0_align_qwen3_embedding_libero_new_dsn_new_continue.json"
        ),
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
        alignment_loss_warmup=False,
        alignment_loss_warmup_init=0.0,
        alignment_loss_warmup_steps=None,
        reset_bank_every_step=True,
        wandb_enabled=True,
    ),
    TrainConfig(
        name="pi0_libero_full_qwen_video",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
            ),
            # 与官方 pi0_libero 保持一致，兼容旧版 pi0 checkpoint 的动作定义。
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=32,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=2,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        # Qwen3 Embedding 离线查表 + video anchor 对齐 (as2ts + as2vs)
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3_embedding_libero_video.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.05
    ),
    # ===== Video strict（去 ek100 + strong=0.9, neg=0.75 + 不 detach + 双卡）=====
    # 关键差异 vs pi0_libero_full_qwen_video:
    #   - 用 video_pool_merged_no_ek100 / video_topk_no_ek100（去掉 ek100，仅留 vitra/bridge/droid/fractal/libero）
    #   - sigmoid_weighted strong=0.9, weak=0.9, neg=0.75，bias=0（hard binary + ignore 中间区）
    #   - as2vs weight=0.3，alignment_loss_weight=0.1
    #   - DSN encoder cosine_hinge（与 align_new_hinge 对齐）
    #   - fg_alignment_model 中 _forward_as2vs 不 detach（在主流程修改）
    #   - 双卡 batch_size=32 (16/卡)
    TrainConfig(
        name="pi0_libero_full_qwen_video_strict",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=32,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=2,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3_embedding_libero_video_strict.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1
    ),
    TrainConfig(
        # ===== 基于 align_new 的设定 + at2tt_soft seq 细粒度对齐 + Qwen3-VL embedding =====
        # 关键开关（在对齐 config 中）：
        #   mode_config.at2tt_soft.enabled = true（aggregation = 'wti'）
        #   disentangle_config.align_target = 'both'，pooled_align_weight = 0.3
        #   use_distributed_negatives = true（多卡 allgather 合并负样本）
        # 文本 embedding 来自 Qwen3-VL-Embedding-8B（离线生成的 npz）
        # 单卡场景 batch_size 16；如果跑 2 卡可以恢复到 32
        name="pi0_libero_full_qwen_seq",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=16,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=2,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3vl_libero_seq.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1
    ),
    # 双粒度对齐：task-level(as2ts) + chunk-level(as2atomic) 并行
    # 与 pi0_libero_full_qwen 的区别：
    #   1. as2ts 保留（task_id模式，全局锚点），as2atomic 新增（diagonal模式，chunk级细粒度）
    #   2. 两套独立投影头，避免梯度冲突
    #   3. atomic_embeddings_path 指向174个原子标签的Qwen3 embedding
    #   4. atomic_chunk_map_path 提供 (episode_idx, chunk_idx) → atomic_label_idx 映射
    # SMOKE 版本：使用真实 chunk video features，跑 20 step 验证链路
    TrainConfig(
        name="pi0_libero_chunk_video_smoke",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
                chunk_video_features_path="/root/data/xuyuan1/Codes/mirror_neuron/data_process/video_feature/output/robot_videos/libero/chunk_video/libero_chunk_video_features.npz",
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=4,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=20,
        num_workers=0,
        log_interval=1,
        save_interval=1000,
        keep_period=10000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5, peak_lr=1e-5, decay_steps=20, decay_lr=1e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3_chunk_video_libero.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
    ),
    # 双粒度 chunk video 对齐：as2ts + as2cv（trajectory + chunk 双 head）
    # 与 pi0_libero_full_qwen_atomic 区别：
    #   1) anchor 来自 video 而非 atomic text；
    #   2) 同一 sim 矩阵跑 task_id（trajectory）+ diagonal（chunk）两个 mask；
    #   3) chunk_video_features_path 在 base_config 给出；
    #      embedding 表本身由 alignment config 中 EmbeddingModel 加载。
    TrainConfig(
        name="pi0_libero_chunk_video",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
                chunk_video_features_path="/root/data/xuyuan1/Codes/mirror_neuron/data_process/video_feature/output/robot_videos/libero/chunk_video/libero_chunk_video_features.npz",
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=16,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=2,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3_chunk_video_libero.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
    ),
    TrainConfig(
        name="pi0_libero_full_qwen_atomic",
        resume=False,
        freeze_vlm=True,
        model=pi0_config.Pi0Config(vlm_mode="embedding"),
        data=LeRobotLiberoDataConfig(
            repo_id="/root/data/xuyuan1/dataset/physical-intelligence/libero",
            assets=AssetsConfig(asset_id="libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                task_mapping_path="/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl",
                atomic_chunk_map_path="/root/data/xuyuan1/Codes/mirror_neuron/data_process/atomic_label/output/atomic_embeds/atomic_chunk_map.npz",
            ),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=32,
        gradient_accumulation_steps=1,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=2,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=2.5e-5,
            decay_steps=30000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3_atomic_libero.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1
    ),
    
    TrainConfig(
        name="bridgev2_rlds_train_lora",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
        ),
        freeze_vlm=True,
        data=RLDSBridgev2DataConfig(
            repo_id="bridgev2",
            adapt_to_pi=False,
            rlds_data_dir="/mnt/bn/2d-videos/xy/work/mirror_neuron/data/bridge-rlds",
            action_chunk_size=5,
            dataset_name="bridge_orig",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=False,
            task_mapping_path="/mnt/bn/2d-videos/xy/work/mirror_neuron/data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl",
            use_only_cam_high=True,
            repack_transform = _transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                    },
                    "state": "observation/state", 
                    "actions": "actions", 
                    "prompt": "prompt", 
                    "task_index": "task_index",  # 任务索引
                    "task_id": "task_id",  # 任务ID
                }),
            ])
        ),
        pytorch_weight_path="/mnt/bn/2d-videos/xy/work/mirror_neuron/weights/openpi/pi0_base_pytorch",
        batch_size=128, 
        gradient_accumulation_steps=4, 
        enable_gradient_checkpointing=False, 
        num_train_steps=50_000,
        num_workers=0,       # RLDS数据加载器要求num_workers=0
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=50000,
            decay_lr=1e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_fixed.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0,
        lora_config={
            "lora_rank_paligemma": 16,        # PaliGemma视觉编码器的LoRA秩
            "lora_alpha_paligemma": 32,       # PaliGemma的alpha缩放因子
            "lora_rank_gemma_expert": 16,     # Gemma Expert动作生成器的LoRA秩
            "lora_alpha_gemma_expert": 32,    # Gemma Expert的alpha缩放因子
        },
    ),
    TrainConfig(
        name="bridgev2_rlds_train",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
            max_token_len=64,
        ),
        freeze_vlm=True,
        data=RLDSBridgev2DataConfig(
            repo_id="bridgev2",
            adapt_to_pi=False,
            rlds_data_dir="/root/data/xuyuan1/Codes/mirror_neuron/data",
            action_chunk_size=5,
            dataset_name="bridge_orig",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=False,
            task_mapping_path="/root/data/xuyuan1/Codes/mirror_neuron/data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl",
            use_only_cam_high=True,
            repack_transform = _transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                    },
                    "state": "observation/state", 
                    "actions": "actions", 
                    "prompt": "prompt", 
                    "task_index": "task_index",  # 任务索引
                    "task_id": "task_id",  # 任务ID
                }),
            ])
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=128, 
        gradient_accumulation_steps=4, 
        enable_gradient_checkpointing=False, 
        num_train_steps=50_000,
        num_workers=0,       # RLDS数据加载器要求num_workers=0
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=50000,
            decay_lr=5e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_fixed.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0
    ),
    # Bridge + Qwen3VL Embedding 离线查表对齐：
    #   - 训练参数（batch/lr/steps/optimizer）完全沿用 bridgev2_rlds_train baseline；
    #   - vlm_mode="embedding"、use_alignment=True、alignment_loss_weight=0.1（与 pi0_libero_full_qwen 同口径）；
    #   - egovlpv2_config_path 指向新的 bridge alignment json，内部加载预计算的 bridge_qwen3vl_text_features.npz。
    #   - bridge npz 仅含 sentence_embeddings（无 token_embeddings），故 alignment json 关闭 at2tt_soft。
    TrainConfig(
        name="bridgev2_rlds_train_qwen",
        resume=False,
        # 与 pi0_libero_full_qwen 一致：vlm_mode="embedding" 触发离线查表路径
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
            max_token_len=64,
            vlm_mode="embedding",
        ),
        freeze_vlm=True,
        data=RLDSBridgev2DataConfig(
            repo_id="bridgev2",
            adapt_to_pi=False,
            rlds_data_dir="/mnt/bn/2d-videos/xy/work/mirror_neuron/data/bridge-rlds",
            action_chunk_size=5,
            dataset_name="bridge_orig",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=False,
            task_mapping_path="/mnt/bn/2d-videos/xy/work/mirror_neuron/data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl",
            use_only_cam_high=True,
            repack_transform=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                    },
                    "state": "observation/state",
                    "actions": "actions",
                    "prompt": "prompt",
                    "task_index": "task_index",
                    "task_id": "task_id",
                }),
            ])
        ),
        pytorch_weight_path="/mnt/bn/2d-videos/xy/work/mirror_neuron/weights/openpi/pi0_base_pytorch",
        # 与 bridgev2_rlds_train baseline 完全一致的 batch 设置（保持不变）。
        # data_loader 每次 yield local_batch = batch_size//world_size = 128/2 = 64/GPU/forward。
        # 64-sample forward 在 80GB A800 上必须开 gradient_checkpointing 才装得下
        # （PaliGemma 各 transformer 层 activation 重算，省 ~2-3x activation memory）。
        batch_size=128,
        gradient_accumulation_steps=4,
        enable_gradient_checkpointing=True,
        num_train_steps=50_000,
        num_workers=0,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=50000,
            decay_lr=5e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        # Qwen3VL 离线查表 alignment 配置：embeddings_path 指向 bridge_qwen3vl_text_features.npz
        egovlpv2_config_path="/mnt/bn/2d-videos/xy/work/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_qwen3vl_embedding_bridge.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
    ),
    TrainConfig(
        name="bridgev2_rlds_train_egohod",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
            max_token_len=64,
            vlm_mode="embedding",
        ),
        freeze_vlm=True,
        data=RLDSBridgev2DataConfig(
            repo_id="bridgev2",
            adapt_to_pi=False,
            rlds_data_dir="/mnt/bn/2d-videos/xy/work/mirror_neuron/data/bridge-rlds",
            action_chunk_size=5,
            dataset_name="bridge_orig",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=False,
            task_mapping_path="/mnt/bn/2d-videos/xy/work/mirror_neuron/data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl",
            use_only_cam_high=True,
            repack_transform=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                    },
                    "state": "observation/state",
                    "actions": "actions",
                    "prompt": "prompt",
                    "task_index": "task_index",
                    "task_id": "task_id",
                }),
            ])
        ),
        pytorch_weight_path="/mnt/bn/2d-videos/xy/work/mirror_neuron/weights/openpi/pi0_base_pytorch",
        batch_size=128,
        gradient_accumulation_steps=8,
        enable_gradient_checkpointing=False,
        num_train_steps=50_000,
        num_workers=0,
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=50000,
            decay_lr=5e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/mnt/bn/2d-videos/xy/work/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_egohod_embedding_bridge.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
    ),
    TrainConfig(
        name="bridgev2_rlds_train_align",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
            max_token_len=64,
        ),
        freeze_vlm=True,
        data=RLDSBridgev2DataConfig(
            repo_id="bridgev2",
            adapt_to_pi=False,
            rlds_data_dir="/root/data/xuyuan1/Codes/mirror_neuron/data",
            action_chunk_size=5,
            dataset_name="bridge_orig",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=False,
            task_mapping_path="/root/data/xuyuan1/Codes/mirror_neuron/data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl",
            use_only_cam_high=True,
            repack_transform = _transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                    },
                    "state": "observation/state", 
                    "actions": "actions", 
                    "prompt": "prompt", 
                    "task_index": "task_index",  # 任务索引
                    "task_id": "task_id",  # 任务ID
                }),
            ])
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=128, 
        gradient_accumulation_steps=4, 
        enable_gradient_checkpointing=False, 
        num_train_steps=50_000,
        num_workers=0,       # RLDS数据加载器要求num_workers=0
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=50000,
            decay_lr=5e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_fixed_full.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=1.0
    ),
    TrainConfig(
        name="bridgev2_rlds_train_align_cotrain",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
            max_token_len=64,
        ),
        freeze_vlm=False,
        data=RLDSBridgev2DataConfig(
            repo_id="bridgev2",
            adapt_to_pi=False,
            rlds_data_dir="/root/data/xuyuan1/Codes/mirror_neuron/data",
            action_chunk_size=5,
            dataset_name="bridge_orig",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=False,
            task_mapping_path="/root/data/xuyuan1/Codes/mirror_neuron/data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl",
            use_only_cam_high=True,
            repack_transform = _transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                    },
                    "state": "observation/state", 
                    "actions": "actions", 
                    "prompt": "prompt", 
                    "task_index": "task_index",  # 任务索引
                    "task_id": "task_id",  # 任务ID
                }),
            ])
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=128, 
        gradient_accumulation_steps=4, 
        enable_gradient_checkpointing=False, 
        num_train_steps=50_000,
        num_workers=0,       # RLDS数据加载器要求num_workers=0
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=50000,
            decay_lr=5e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=True,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_cotrain_full.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0
    ),


    # ===== Agilex 真机数据集（从HDF5转换的RLDS格式） =====
    TrainConfig(
        name="agilex_rlds_train",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,       # 内部 action_dim（padding到32）
            action_horizon=50,   # 动作序列长度
        ),
        freeze_vlm=True,
        data=RobotwinRLDSDataConfig(
            repo_id="agilex_rlds_fruit",
            adapt_to_pi=False,   # Agilex 数据不需要 Aloha 空间转换
            rlds_data_dir="/root/data/xuyuan1/dataset/agilex/agilex_rlds",
            action_chunk_size=50,
            dataset_name="agilex_dataset_fruit",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=True,
            use_delta_joint_actions=False,  # 先不使用 delta actions
            use_only_cam_high=False,        # 使用全部3个摄像头
            repack_transform=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                        "cam_left_wrist": "observation/wrist_image",
                        "cam_right_wrist": "observation/wrist_image_right",
                    },
                    "state": "observation/state",
                    "actions": "actions",
                    "prompt": "prompt",
                    "task_index": "task_index",  # 任务索引（RobotwinRldsDataset.__iter__填充）
                    "task_id": "task_id",         # 任务ID（RobotwinRldsDataset.__iter__填充）
                }),
            ]),
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=64,
        gradient_accumulation_steps=4,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=0,        # RLDS数据加载器要求num_workers=0
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=30000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0,
    ),
    TrainConfig(
        name="agilex_rlds_train_align",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,       # 内部 action_dim（padding到32）
            action_horizon=50,   # 动作序列长度
            vlm_mode="egohod",   # EgoHODModel需要使用egohod模式
        ),
        freeze_vlm=True,
        data=RobotwinRLDSDataConfig(
            repo_id="agilex_rlds_fruit",
            adapt_to_pi=False,   # Agilex 数据不需要 Aloha 空间转换
            rlds_data_dir="/root/data/xuyuan1/dataset/agilex/agilex_rlds",
            action_chunk_size=50,
            dataset_name="agilex_dataset_fruit",
            filter_dict_path=None,
            task_mapping_path="/root/data/xuyuan1/dataset/agilex/agilex_rlds/meta/tasks_with_id.jsonl",
            default_prompt="",
            prompt_from_task=True,
            use_delta_joint_actions=False,  # 先不使用 delta actions
            use_only_cam_high=False,        # 使用全部3个摄像头
            repack_transform=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                        "cam_left_wrist": "observation/wrist_image",
                        "cam_right_wrist": "observation/wrist_image_right",
                    },
                    "state": "observation/state",
                    "actions": "actions",
                    "prompt": "prompt",
                    "task_index": "task_index",  # 任务索引（RobotwinRldsDataset.__iter__填充）
                    "task_id": "task_id",         # 任务ID（RobotwinRldsDataset.__iter__填充）
                }),
            ]),
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=64,
        gradient_accumulation_steps=4,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=0,        # RLDS数据加载器要求num_workers=0
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=30000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_fixed_lora_real.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.005,
    ),
    TrainConfig(
        name="agilex_rlds_train_blocks",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,       # 内部 action_dim（padding到32）
            action_horizon=50,   # 动作序列长度
        ),
        freeze_vlm=True,
        data=RobotwinRLDSDataConfig(
            repo_id="agilex_rlds_blocks",
            adapt_to_pi=False,   # Agilex 数据不需要 Aloha 空间转换
            rlds_data_dir="/root/data/xuyuan1/dataset/agilex/agilex_rlds",
            action_chunk_size=50,
            dataset_name="agilex_dataset_blocks",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=True,
            use_delta_joint_actions=False,  # 先不使用 delta actions
            use_only_cam_high=False,        # 使用全部3个摄像头
            repack_transform=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                        "cam_left_wrist": "observation/wrist_image",
                        "cam_right_wrist": "observation/wrist_image_right",
                    },
                    "state": "observation/state",
                    "actions": "actions",
                    "prompt": "prompt",
                    "task_index": "task_index",  # 任务索引（RobotwinRldsDataset.__iter__填充）
                    "task_id": "task_id",         # 任务ID（RobotwinRldsDataset.__iter__填充）
                }),
            ]),
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=64,
        gradient_accumulation_steps=4,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=0,        # RLDS数据加载器要求num_workers=0
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=30000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=False,
        egovlpv2_config_path="",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.0,
    ),
    TrainConfig(
        name="agilex_rlds_train_align_blocks",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,       # 内部 action_dim（padding到32）
            action_horizon=50,   # 动作序列长度
        ),
        freeze_vlm=True,
        data=RobotwinRLDSDataConfig(
            repo_id="agilex_rlds_blocks",
            adapt_to_pi=False,   # Agilex 数据不需要 Aloha 空间转换
            rlds_data_dir="/root/data/xuyuan1/dataset/agilex/agilex_rlds",
            action_chunk_size=50,
            dataset_name="agilex_dataset_blocks",
            filter_dict_path=None,
            default_prompt="",
            prompt_from_task=True,
            use_delta_joint_actions=False,  # 先不使用 delta actions
            use_only_cam_high=False,        # 使用全部3个摄像头
            repack_transform=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation/image",
                        "cam_left_wrist": "observation/wrist_image",
                        "cam_right_wrist": "observation/wrist_image_right",
                    },
                    "state": "observation/state",
                    "actions": "actions",
                    "prompt": "prompt",
                    "task_index": "task_index",  # 任务索引（RobotwinRldsDataset.__iter__填充）
                    "task_id": "task_id",         # 任务ID（RobotwinRldsDataset.__iter__填充）
                }),
            ]),
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=64,
        gradient_accumulation_steps=4,
        enable_gradient_checkpointing=False,
        num_train_steps=30_000,
        num_workers=0,        # RLDS数据加载器要求num_workers=0
        log_interval=50,
        save_interval=2000,
        keep_period=5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=30000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_fixed_lora.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
    ),
    
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
