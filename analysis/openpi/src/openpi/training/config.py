"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

from sympy import false

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
    # 任务映射文件路径（可选，优先级高于自动搜索）
    task_mapping_path: str | None = None
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
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
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


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="bridgev2_rlds_train_align_lora",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
            use_learnable_token=False,  # 推理时需要这个来创建learnable_token参数
            vlm_mode="egohod",
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
            peak_lr=1e-4,
            decay_steps=50000,
            decay_lr=1e-7,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        use_egovlpv2=False,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_fixed_lora.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
        lora_config={
            "lora_rank_paligemma": 16,        # PaliGemma视觉编码器的LoRA秩
            "lora_alpha_paligemma": 32,       # PaliGemma的alpha缩放因子
            "lora_rank_gemma_expert": 16,     # Gemma Expert动作生成器的LoRA秩
            "lora_alpha_gemma_expert": 32,    # Gemma Expert的alpha缩放因子
        },
    ),
    TrainConfig(
        name="bridgev2_rlds_train_align_lora_egohod_cotrain",
        resume=False,
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=5,
            use_learnable_token=False,
            vlm_mode="egohod",  # 使用EgoHOD作为VLM
        ),
        freeze_vlm=False,  # 关键区别：不冻结EgoHOD，使其参与co-training
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
                    "task_index": "task_index",
                    "task_id": "task_id",
                }),
            ])
        ),
        pytorch_weight_path="/root/data/xuyuan1/dataset/pi0_base_pytorch",
        batch_size=128,                        # EgoHOD co-training显存开销更大，适当减小batch
        gradient_accumulation_steps=4,        
        enable_gradient_checkpointing=False, 
        num_train_steps=50_000,
        num_workers=0,
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
        use_egovlpv2=True,
        use_alignment=True,
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_cotrain_lora.json",
        vlm_loss_weight=1.0,                  # EgoHOD ClipLoss权重
        alignment_loss_weight=0.1,            # Alignment对比学习损失权重（与SpatialVLA对齐）
        alignment_loss_warmup=False,           # 启用alignment loss warmup
        alignment_loss_warmup_init=0.01,       # warmup从0开始，避免早期干扰PI0训练
        # alignment_loss_warmup_steps 默认None，复用LR warmup_steps=500
        lora_config={
            "lora_rank_paligemma": 16,
            "lora_alpha_paligemma": 32,
            "lora_rank_gemma_expert": 16,
            "lora_alpha_gemma_expert": 32,
        },
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
        egovlpv2_config_path="/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2/egovlpv2/configs/ft/pi0_align_fixed_lora.json",
        vlm_loss_weight=0.0,
        alignment_loss_weight=0.1,
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
