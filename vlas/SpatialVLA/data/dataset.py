import os
import json
import torch
import itertools
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import IterableDataset

IGNORE_INDEX = -100


def load_task_mapping(data_root_dir: str, data_mix: str, task_filename: str = None) -> tuple:
    """
    统一加载任务映射函数
    
    加载指定的任务文件，自动检测是否包含task_id字段：
    - 如果文件包含task_id字段：同时构建lang_to_task_index和task_index_to_task_id
    - 如果文件不包含task_id字段：仅构建lang_to_task_index，task_index_to_task_id为空
    
    Args:
        data_root_dir: 数据根目录
        data_mix: 数据集名称  
        task_filename: 任务文件名（如"tasks.jsonl"或"tasks_with_id.jsonl"），默认"tasks.jsonl"
    
    Returns:
        tuple: (lang_to_task_index, task_index_to_task_id)
    """
    # 默认文件名
    if task_filename is None:
        task_filename = "tasks.jsonl"
    
    # 查找文件路径（优先级从高到低）
    possible_paths = [
        Path(data_root_dir) / data_mix / f"{data_mix}_lerobot" / "meta" / task_filename,
        Path(data_root_dir) / f"{data_mix}_lerobot" / "meta" / task_filename,
        Path(data_root_dir) / data_mix / "meta" / task_filename,
    ]
    
    tasks_path = None
    for p in possible_paths:
        if p.exists():
            tasks_path = p
            break
    
    if tasks_path is None:
        print(f"[Warning] {task_filename} not found, task_index and task_id will be -1")
        return {}, {}
    
    # 读取文件，构建映射（自动检测是否有task_id字段）
    lang_to_task_index = {}
    task_index_to_task_id = {}
    has_task_id = False  # 标记文件是否包含task_id字段
    
    with open(tasks_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            task_index = item["task_index"]
            # 构建 lang -> task_index 映射（统一转小写）
            task_text = item.get("task", "")
            if task_text:
                lang_to_task_index[task_text.lower()] = task_index
            # 构建 task_index -> task_id 映射
            # 如果有task_id字段则使用，否则用task_index作为task_id
            if "task_id" in item:
                task_index_to_task_id[task_index] = item["task_id"]
                has_task_id = True
            else:
                task_index_to_task_id[task_index] = task_index  # 无task_id时用task_index
    
    # 打印加载信息
    if has_task_id:
        print(f"[Dataset] Loaded {len(lang_to_task_index)} tasks with {len(task_index_to_task_id)} task_ids from {tasks_path}")
    else:
        print(f"[Dataset] Loaded {len(lang_to_task_index)} tasks (task_id=task_index) from {tasks_path}")
    
    return lang_to_task_index, task_index_to_task_id


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from .utils.data_utils import NormalizationType, save_dataset_statistics
from .rlds import dataset_statistics, build_interleaved_dataset

from egovlpv2.data_loader.transforms import init_video_transform_dict

class OpenXIterableDataset(IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_root_dir,
        output_dir,
        data_mix,
        image_size=224,
        max_length=1024,
        is_train=True,
        shuffle_buffer_size=1000_000,
        tsfm_thread_muti=1,
        read_thread_muti=1,
        obs_backward_steps=0,
        obs_backward_delta=1,
        action_forward_steps=0,
        use_raw_dataloader=False,
        fix_raw_length=None,
        vla_processor=None,
        load_future_images=False,
        task_filename=None,  # 任务文件名（如"tasks.jsonl"或"tasks_with_id.jsonl"），默认"tasks.jsonl"
    ):
        super(OpenXIterableDataset, self).__init__()
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.use_raw_dataloader = use_raw_dataloader
        self.vla_processor = vla_processor
        self.image_size = image_size
        self.max_length = max_length
        self.is_train = is_train
        # 保存load_future_images参数，用于控制是否在轨迹变换中生成未来图像
        self.load_future_images = load_future_images

        self.total_ranks = torch.distributed.get_world_size()
        self.current_rank = torch.distributed.get_rank()

        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(os.path.join(self.data_mix, "1.0.0"), 1.0)]
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        self.dataset_num = len(weights)
        self.rlds_config = dict(
            traj_transform_kwargs=dict(
                backward_windows_size=obs_backward_steps,  # If we wanted to feed / predict more than one step
                backward_delta=obs_backward_delta,
                forward_window_size=action_forward_steps,  # For action chunking
                skip_unlabeled=True,  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",  # Goals are currently unused
                # 将load_future_images参数传递给chunk_act_obs函数，控制是否生成未来图像序列
                future_images=self.load_future_images,
            ),
            frame_transform_kwargs=dict(
                resize_size=(self.image_size, self.image_size),
                num_parallel_calls=16,  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec) * tsfm_thread_muti,
            traj_read_threads=len(mixture_spec) * read_thread_muti,
            train=self.is_train,
            shuffle_seed=3407 * self.current_rank,
        )        
        self.rlds_config["frame_transform_kwargs"].update(
            {
                "image_augment_kwargs": dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )
            }
        )
        self.rlds_dataset = None
        expected_length, self.ds_stats, self.sample_weights = dataset_statistics(**self.rlds_config)
        self.raw_length = expected_length * self.dataset_num

        # NOTE: in staget 2 ptraining, we use much less data, thus the resume'll stop immediately
        # set a fixed dataset length avoids the unexceptable traing interrupt
        if fix_raw_length:
            self.raw_length = fix_raw_length
            print(f"[Dataset] set a fixed dataset length {fix_raw_length} avoids the unexceptable traing interrupt!")

        
        self.ds_stats_pc = save_dataset_statistics(self.ds_stats, Path(output_dir) / "ds_stats.json")

        # --- ADDITION START: Initialize EgoVLPv2's video transform ---
        self.egovlp_video_transform = init_video_transform_dict(
            input_res=self.image_size # 使用你数据集中配置的image_size
        )['val']
        
        # 统一加载任务映射：从指定文件加载，自动检测是否有task_id字段
        self.lang_to_task_index, self.task_index_to_task_id = load_task_mapping(
            self.data_root_dir, self.data_mix, task_filename
        )

    def _compute_task_progress(self, data_item, num_actions):
        """
        从data_item计算任务进度，归一化到[0, 1]
        
        核心逻辑：
        1. 从observation["timestep"]获取当前timestep（chunked后的最后一个）
        2. 从data_item["traj_len"]获取轨迹总长度（真实值，不估算）
        3. 计算action chunk的起止timestep
        4. 归一化到[0, 1]范围
        
        Args:
            data_item: 从RLDS iterator获取的数据项，包含：
                - observation["timestep"]: [T] chunked timesteps
                - traj_len: [T] 轨迹总长度（每个元素相同）
            num_actions: action chunk的长度
            
        Returns:
            task_progress: torch.Tensor [2] 包含[start_progress, end_progress]
            
        示例：
            如果当前timestep=50, traj_len=100, num_actions=8:
            - action_start_t = 50
            - action_end_t = 57
            - progress_start = 50/99 ≈ 0.505
            - progress_end = 57/99 ≈ 0.576
        """
        # 1. 从observation中提取当前timestep
        # observation["timestep"]是chunked的，形状为[T]，T=backward_windows_size+1
        # 当前timestep是最后一个（最新的观测）
        obs_timesteps = data_item["observation"]["timestep"]
        current_timestep = int(obs_timesteps[-1])
        
        # 2. 从data_item中获取轨迹总长度
        # traj_len是在rlds.py中添加的
        # 经过numpy iterator后可能是标量、0维数组或1维数组，需要兼容处理
        traj_len_raw = data_item["traj_len"]
        if isinstance(traj_len_raw, (int, np.integer)):
            # 情况1: 已经是标量（经过某些transform后）
            traj_len = int(traj_len_raw)
        elif isinstance(traj_len_raw, np.ndarray):
            if traj_len_raw.ndim == 0:
                # 情况2: 0维数组，直接用item()取值
                traj_len = int(traj_len_raw.item())
            else:
                # 情况3: 1维或更高维数组，取第一个元素
                traj_len = int(traj_len_raw.flat[0])
        else:
            # 情况4: 其他情况，直接转换
            traj_len = int(traj_len_raw)
        
        # 3. 计算action chunk的timestep范围
        # action chunk从当前timestep开始，持续num_actions步
        action_start_t = current_timestep
        action_end_t = min(current_timestep + num_actions - 1, traj_len - 1)
        
        # 4. 归一化到[0, 1]
        # 除以(traj_len - 1)使得最后一个timestep归一化为1.0
        # 使用max(..., 1)防止除零错误（单步轨迹）
        denominator = max(traj_len - 1, 1)
        progress_start = float(action_start_t) / denominator
        progress_end = float(action_end_t) / denominator
        
        # 5. 返回torch tensor [2]
        task_progress = torch.tensor([progress_start, progress_end], dtype=torch.float32)
        
        return task_progress
    
    def _process_future_images(self, future_images_np):
        """
        将RLDS输出的numpy数组 (T, H, W, C) 转换为EgoVLPv2 vision encoder所需的PyTorch张量 (T, C, 224, 224)
        """
        # 1. 将numpy数组直接转换为PyTorch Tensor (避免逐帧循环，提高效率)
        #    输入形状: (T, H, W, C), dtype: uint8, 范围: [0, 255]
        video_tensor = torch.from_numpy(future_images_np)

        # 2. 维度变换 (HWC -> CHW) 并归一化值域至 [0.0, 1.0]。
        #    这一步完美复刻了EgoVLPv2中read_frames_cv2等函数的核心操作。
        #    输入形状: (T, H, W, C) -> 输出形状: (T, C, H, W)
        video_tensor = video_tensor.permute(0, 3, 1, 2).float() / 255.0
        
        # 3. 为视频变换准备数据，调整维度顺序。
        #    torchvision的视频变换 (如NormalizeVideo) 要求输入形状为 (C, T, H, W)。
        #    输入形状: (T, C, H, W) -> 输出形状: (C, T, H, W)
        video_tensor_for_transform = video_tensor.transpose(0, 1)
        
        # 4. 应用在 __init__ 中缓存的官方EgoVLPv2变换。
        #    这一步完成了所有必要的缩放、裁剪和ImageNet归一化。
        transformed_video = self.egovlp_video_transform(video_tensor_for_transform)
        
        # 5. 将维度恢复为更通用的 (T, C, H, W) 格式，方便DataLoader进行批处理。
        #    输入形状: (C, T, 224, 224) -> 输出形状: (T, C, 224, 224)
        final_video_tensor = transformed_video.transpose(0, 1)

        return final_video_tensor

    def __len__(self):
        if self.use_raw_dataloader:
            return self.raw_length // self.total_ranks
        else:
            return self.raw_length

    def multi_modal_get_item(self, data_item):
        pixel_values_seq = []
        
        # TODO: add mutiple image inputs support (processor, model)
        for image_primary in data_item["observation"]["image_primary"]:  # (t h w c)
            image = Image.fromarray(image_primary)
            pixel_values_seq += [image] # [c h w]

        actions = torch.from_numpy(data_item["action"])  # (t e)
        lang = data_item["task"]["language_instruction"].lower()
        if isinstance(lang, bytes): lang = lang.decode()

        # ============ 关键 fix：从 data_item 取 dataset_name 作为 unnorm_key ============
        # rlds.py:177 在每条轨迹的每 step 注入 dataset_name = tf.repeat(name, traj_len)。
        # 经 dlimp/numpy 转换后，data_item['dataset_name'] 是 **bytes scalar**，
        # 形如 b'libero_mix_no_noops/1.0.0'（不是 array）。注意 bytes 索引会返回 int
        # （bytes[0] == ord('l') == 108），所以**绝对不能 _dn[0]**——这是上一版的 bug 来源。
        _dn = data_item.get("dataset_name", None)
        # 1) numpy 数组：取首项 / .item()；之后还会落到 bytes 分支去 decode
        if _dn is not None and not isinstance(_dn, (bytes, str)):
            import numpy as _np  # 局部 import 避免顶部加 import 改动其它代码
            if isinstance(_dn, _np.ndarray):
                _dn = _dn.item() if _dn.ndim == 0 else _dn[0]
        # 2) bytes → str
        if isinstance(_dn, bytes):
            _dn = _dn.decode()
        unnorm_key = str(_dn) if _dn else None
        # 首次：dump 完整 data_item 字段类型 + 值，便于排查 dataset_name 未被识别的情况。
        if not getattr(self, "_unnorm_key_logged", False):
            _raw = data_item.get("dataset_name", None)
            print(f"[DATASET_FIX_V3_MARKER] data_item keys = {list(data_item.keys())}", flush=True)
            print(f"[DATASET_FIX_V3_MARKER] dataset_name raw type={type(_raw).__name__} value={_raw!r}", flush=True)
            print(f"[DATASET_FIX_V3_MARKER] _dn type={type(_dn).__name__} value={_dn!r}", flush=True)
            print(f"[DATASET_FIX_V3_MARKER] resolved unnorm_key='{unnorm_key}' (type={type(unnorm_key).__name__})", flush=True)
            self._unnorm_key_logged = True

        # TODO: move to processor
        ret = self.vla_processor(
            text=lang,
            images=pixel_values_seq,
            unnorm_key=unnorm_key,
            suffix_actions=actions,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length,
            truncation=True,
            do_normalize=False, # do not normalize the image for zoe
        )

        future_pixel_values = None
        if "future_images" in data_item:
            # data_item["future_images"] 是一个 (T_future, H, W, C) 的 uint8 NumPy 数组
            future_pixel_values = self._process_future_images(
                data_item["future_images"]
            )

        # ============ 新增：计算任务进度（Task Progress Encoding） ============
        # 从RLDS数据中提取当前timestep信息
        task_progress = self._compute_task_progress(data_item, actions.shape[0])
        
        # ============ 获取task_index和task_ids（O(1)哈希查找） ============
        # task_index: 原始任务索引（与tasks.jsonl对应）
        # task_ids: 聚类后的任务类别ID（无task_id时等于task_index）
        # 注意：lang_to_task_index的key是lower()存储的，查找时也需要lower()
        task_index = self.lang_to_task_index.get(lang.lower(), -1)
        task_ids = self.task_index_to_task_id.get(task_index, task_index)  # 找不到时用task_index
        
        # ============ 提取轨迹标识符（用于特征提取） ============
        # traj_index: 轨迹在数据集中的索引
        # timestep: 当前帧在轨迹内的时间步
        traj_index_raw = data_item.get("traj_index", -1)
        if isinstance(traj_index_raw, np.ndarray):
            traj_index_val = int(traj_index_raw.flat[0]) if traj_index_raw.size > 0 else -1
        else:
            traj_index_val = int(traj_index_raw) if traj_index_raw is not None else -1
        
        timestep_val = int(data_item["observation"]["timestep"][-1])  # 取最后一个（当前）timestep
        
        model_inputs = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            token_type_ids=ret["token_type_ids"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=ret["pixel_values"],
            intrinsic=ret["intrinsic"],
            actions=actions,
            lang=lang,
            future_pixel_values=future_pixel_values,
            task_progress=task_progress,  # 任务进度信息
            task_index=task_index,  # 原始任务索引
            task_ids=task_ids,  # 聚类后的任务类别ID（与model forward参数对齐）
            traj_index=traj_index_val,  # 轨迹索引（用于特征提取唯一标识）
            timestep=timestep_val,  # 轨迹内时间步（用于特征提取唯一标识）
        )
        return model_inputs

    def __iter__(self):
        if self.rlds_dataset is None:
            self.rlds_dataset = build_interleaved_dataset(weights=self.sample_weights, dataset_statistics=self.ds_stats, **self.rlds_config).as_numpy_iterator()
            if torch.utils.data.get_worker_info() is not None:
                worker_total_num = torch.utils.data.get_worker_info().num_workers
                worker_id = torch.utils.data.get_worker_info().id
            else:
                worker_id = 0
                worker_total_num = 1
            self.rlds_dataset = itertools.islice(iter(self.rlds_dataset), worker_id, None, worker_total_num)

        for i, data_item in enumerate(self.rlds_dataset):
            ret = self.multi_modal_get_item(data_item)
            if i < len(self):
                yield ret
            else:
                break


class OpenXIterableDatasetWithPairData(IterableDataset):
    """
    Dataset for supervised fine-tuning with Pair Data support for Video Alignment.
    
    继承自IterableDataset，在OpenXIterableDataset基础上增加了Pair Data功能，
    用于加载预提取的Ego4D视频特征，实现human video到robot action的对齐。
    
    新增参数:
        pair_data_config: dict, 包含以下字段:
            - enabled: bool, 是否启用pair data
            - feature_dir: str, npz特征文件目录
            - index_path: str, query_to_clip_index.json路径
            - similarity_threshold: float, 正样本相似度阈值
    """

    def __init__(
        self,
        data_root_dir,
        output_dir,
        data_mix,
        image_size=224,
        max_length=1024,
        is_train=True,
        shuffle_buffer_size=1000_000,
        tsfm_thread_muti=1,
        read_thread_muti=1,
        obs_backward_steps=0,
        obs_backward_delta=1,
        action_forward_steps=0,
        use_raw_dataloader=False,
        fix_raw_length=None,
        vla_processor=None,
        load_future_images=False,
        task_filename=None,  # 任务文件名（如"tasks.jsonl"或"tasks_with_id.jsonl"），默认"tasks.jsonl"
        # ============ 新增: Pair数据支持 (Video Alignment) ============
        pair_data_config=None,  # dict: {feature_dir, index_path, similarity_threshold, enabled}
    ):
        super(OpenXIterableDatasetWithPairData, self).__init__()
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.use_raw_dataloader = use_raw_dataloader
        self.vla_processor = vla_processor
        self.image_size = image_size
        self.max_length = max_length
        self.is_train = is_train
        # 保存load_future_images参数，用于控制是否在轨迹变换中生成未来图像
        self.load_future_images = load_future_images

        self.total_ranks = torch.distributed.get_world_size()
        self.current_rank = torch.distributed.get_rank()

        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(os.path.join(self.data_mix, "1.0.0"), 1.0)]
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        self.dataset_num = len(weights)
        self.rlds_config = dict(
            traj_transform_kwargs=dict(
                backward_windows_size=obs_backward_steps,  # If we wanted to feed / predict more than one step
                backward_delta=obs_backward_delta,
                forward_window_size=action_forward_steps,  # For action chunking
                skip_unlabeled=True,  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",  # Goals are currently unused
                # 将load_future_images参数传递给chunk_act_obs函数，控制是否生成未来图像序列
                future_images=self.load_future_images,
            ),
            frame_transform_kwargs=dict(
                resize_size=(self.image_size, self.image_size),
                num_parallel_calls=16,  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec) * tsfm_thread_muti,
            traj_read_threads=len(mixture_spec) * read_thread_muti,
            train=self.is_train,
            shuffle_seed=3407 * self.current_rank,
        )        
        self.rlds_config["frame_transform_kwargs"].update(
            {
                "image_augment_kwargs": dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )
            }
        )
        self.rlds_dataset = None
        expected_length, self.ds_stats, self.sample_weights = dataset_statistics(**self.rlds_config)
        self.raw_length = expected_length * self.dataset_num

        # NOTE: in staget 2 ptraining, we use much less data, thus the resume'll stop immediately
        # set a fixed dataset length avoids the unexceptable traing interrupt
        if fix_raw_length:
            self.raw_length = fix_raw_length
            print(f"[Dataset] set a fixed dataset length {fix_raw_length} avoids the unexceptable traing interrupt!")

        
        self.ds_stats_pc = save_dataset_statistics(self.ds_stats, Path(output_dir) / "ds_stats.json")

        # --- ADDITION START: Initialize EgoVLPv2's video transform ---
        self.egovlp_video_transform = init_video_transform_dict(
            input_res=self.image_size # 使用你数据集中配置的image_size
        )['val']
        
        # 统一加载任务映射：从指定文件加载，自动检测是否有task_id字段
        self.lang_to_task_index, self.task_index_to_task_id = load_task_mapping(
            self.data_root_dir, self.data_mix, task_filename
        )
        
        # ============ 新增: 初始化Pair数据索引 (Video Alignment) ============
        self.pair_data_config = pair_data_config or {}
        self.use_pair_data = self.pair_data_config.get('enabled', False)
        self.pair_query_to_clip_index = {}  # query_hash -> [{clip_hash, similarity}, ...]
        
        if self.use_pair_data:
            self._init_pair_data_index()

    def _init_pair_data_index(self):
        """
        初始化Pair数据索引
        
        加载query_to_clip_index.json，建立query_hash -> [clip_info] 的映射
        """
        import hashlib
        import json
        
        feature_dir = self.pair_data_config.get('feature_dir')
        index_path = self.pair_data_config.get('index_path')
        
        if not feature_dir or not index_path:
            print("[Dataset] Pair data config incomplete, disabling pair data")
            self.use_pair_data = False
            return
        
        if not os.path.exists(index_path):
            print(f"[Dataset] Pair index file not found: {index_path}, disabling pair data")
            self.use_pair_data = False
            return
        
        # 加载索引
        with open(index_path, 'r') as f:
            self.pair_query_to_clip_index = json.load(f)
        
        self.pair_feature_dir = feature_dir
        self.pair_similarity_threshold = self.pair_data_config.get('similarity_threshold', 0.7)
        
        print(f"[Dataset] Loaded pair data index with {len(self.pair_query_to_clip_index)} queries")
        print(f"[Dataset] Feature dir: {feature_dir}")
        print(f"[Dataset] Similarity threshold: {self.pair_similarity_threshold}")
    
    def _load_pair_video_features(self, lang: str):
        """
        根据lang加载对应的pair video特征
        
        Args:
            lang: 任务指令文本
        
        Returns:
            dict或None: {
                'video_cls': [D] np.array,
                'video_frames': [T, D] np.array,
                'ego4d_text': str,
                'similarity': float,
                'is_positive': bool,
            }
        """
        import hashlib
        import random
        
        if not self.use_pair_data:
            return None
        
        # 计算lang的hash
        query_hash = hashlib.md5(lang.encode()).hexdigest()
        
        # 查找匹配的clips
        clips = self.pair_query_to_clip_index.get(query_hash, [])
        
        if not clips:
            return None
        
        # 过滤高相似度的clips作为正样本
        valid_clips = [c for c in clips if c.get('similarity', 0) >= self.pair_similarity_threshold]
        
        if not valid_clips:
            # 没有高相似度的clip，随机选一个作为负样本
            clip_info = random.choice(clips)
            is_positive = False
        else:
            # 随机选择一个高相似度的clip
            clip_info = random.choice(valid_clips)
            is_positive = True
        
        # 加载npz文件
        clip_hash = clip_info['clip_hash']
        npz_path = os.path.join(self.pair_feature_dir, f"{clip_hash}.npz")
        
        if not os.path.exists(npz_path):
            return None
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            return {
                'video_cls': data['video_cls'],  # [D]
                'video_frames': data['video_frames'],  # [T, D]
                'ego4d_text': str(data['text']),
                'similarity': float(clip_info['similarity']),
                'is_positive': is_positive,
            }
        except Exception as e:
            print(f"[Dataset] Error loading pair npz {npz_path}: {e}")
            return None

    def _compute_task_progress(self, data_item, num_actions):
        """
        从data_item计算任务进度，归一化到[0, 1]
        
        核心逻辑：
        1. 从observation["timestep"]获取当前timestep（chunked后的最后一个）
        2. 从data_item["traj_len"]获取轨迹总长度（真实值，不估算）
        3. 计算action chunk的起止timestep
        4. 归一化到[0, 1]范围
        
        Args:
            data_item: 从RLDS iterator获取的数据项，包含：
                - observation["timestep"]: [T] chunked timesteps
                - traj_len: [T] 轨迹总长度（每个元素相同）
            num_actions: action chunk的长度
            
        Returns:
            task_progress: torch.Tensor [2] 包含[start_progress, end_progress]
            
        示例：
            如果当前timestep=50, traj_len=100, num_actions=8:
            - action_start_t = 50
            - action_end_t = 57
            - progress_start = 50/99 ≈ 0.505
            - progress_end = 57/99 ≈ 0.576
        """
        # 1. 从observation中提取当前timestep
        # observation["timestep"]是chunked的，形状为[T]，T=backward_windows_size+1
        # 当前timestep是最后一个（最新的观测）
        obs_timesteps = data_item["observation"]["timestep"]
        current_timestep = int(obs_timesteps[-1])
        
        # 2. 从data_item中获取轨迹总长度
        # traj_len是在rlds.py中添加的
        # 经过numpy iterator后可能是标量、0维数组或1维数组，需要兼容处理
        traj_len_raw = data_item["traj_len"]
        if isinstance(traj_len_raw, (int, np.integer)):
            # 情况1: 已经是标量（经过某些transform后）
            traj_len = int(traj_len_raw)
        elif isinstance(traj_len_raw, np.ndarray):
            if traj_len_raw.ndim == 0:
                # 情况2: 0维数组，直接用item()取值
                traj_len = int(traj_len_raw.item())
            else:
                # 情况3: 1维或更高维数组，取第一个元素
                traj_len = int(traj_len_raw.flat[0])
        else:
            # 情况4: 其他情况，直接转换
            traj_len = int(traj_len_raw)
        
        # 3. 计算action chunk的timestep范围
        # action chunk从当前timestep开始，持续num_actions步
        action_start_t = current_timestep
        action_end_t = min(current_timestep + num_actions - 1, traj_len - 1)
        
        # 4. 归一化到[0, 1]
        # 除以(traj_len - 1)使得最后一个timestep归一化为1.0
        # 使用max(..., 1)防止除零错误（单步轨迹）
        denominator = max(traj_len - 1, 1)
        progress_start = float(action_start_t) / denominator
        progress_end = float(action_end_t) / denominator
        
        # 5. 返回torch tensor [2]
        task_progress = torch.tensor([progress_start, progress_end], dtype=torch.float32)
        
        return task_progress
    
    def _process_future_images(self, future_images_np):
        """
        将RLDS输出的numpy数组 (T, H, W, C) 转换为EgoVLPv2 vision encoder所需的PyTorch张量 (T, C, 224, 224)
        """
        # 1. 将numpy数组直接转换为PyTorch Tensor (避免逐帧循环，提高效率)
        #    输入形状: (T, H, W, C), dtype: uint8, 范围: [0, 255]
        video_tensor = torch.from_numpy(future_images_np)

        # 2. 维度变换 (HWC -> CHW) 并归一化值域至 [0.0, 1.0]。
        #    这一步完美复刻了EgoVLPv2中read_frames_cv2等函数的核心操作。
        #    输入形状: (T, H, W, C) -> 输出形状: (T, C, H, W)
        video_tensor = video_tensor.permute(0, 3, 1, 2).float() / 255.0
        
        # 3. 为视频变换准备数据，调整维度顺序。
        #    torchvision的视频变换 (如NormalizeVideo) 要求输入形状为 (C, T, H, W)。
        #    输入形状: (T, C, H, W) -> 输出形状: (C, T, H, W)
        video_tensor_for_transform = video_tensor.transpose(0, 1)
        
        # 4. 应用在 __init__ 中缓存的官方EgoVLPv2变换。
        #    这一步完成了所有必要的缩放、裁剪和ImageNet归一化。
        transformed_video = self.egovlp_video_transform(video_tensor_for_transform)
        
        # 5. 将维度恢复为更通用的 (T, C, H, W) 格式，方便DataLoader进行批处理。
        #    输入形状: (C, T, 224, 224) -> 输出形状: (T, C, 224, 224)
        final_video_tensor = transformed_video.transpose(0, 1)

        return final_video_tensor

    def __len__(self):
        if self.use_raw_dataloader:
            return self.raw_length // self.total_ranks
        else:
            return self.raw_length

    def multi_modal_get_item(self, data_item):
        pixel_values_seq = []
        
        # TODO: add mutiple image inputs support (processor, model)
        for image_primary in data_item["observation"]["image_primary"]:  # (t h w c)
            image = Image.fromarray(image_primary)
            pixel_values_seq += [image] # [c h w]

        actions = torch.from_numpy(data_item["action"])  # (t e)
        lang = data_item["task"]["language_instruction"].lower()
        if isinstance(lang, bytes): lang = lang.decode()

        # ============ 关键 fix：从 data_item 取 dataset_name 作为 unnorm_key ============
        # rlds.py:177 在每条轨迹的每 step 注入 dataset_name = tf.repeat(name, traj_len)。
        # 经 dlimp/numpy 转换后，data_item['dataset_name'] 是 **bytes scalar**，
        # 形如 b'libero_mix_no_noops/1.0.0'（不是 array）。注意 bytes 索引会返回 int
        # （bytes[0] == ord('l') == 108），所以**绝对不能 _dn[0]**——这是上一版的 bug 来源。
        _dn = data_item.get("dataset_name", None)
        # 1) numpy 数组：取首项 / .item()；之后还会落到 bytes 分支去 decode
        if _dn is not None and not isinstance(_dn, (bytes, str)):
            import numpy as _np  # 局部 import 避免顶部加 import 改动其它代码
            if isinstance(_dn, _np.ndarray):
                _dn = _dn.item() if _dn.ndim == 0 else _dn[0]
        # 2) bytes → str
        if isinstance(_dn, bytes):
            _dn = _dn.decode()
        unnorm_key = str(_dn) if _dn else None
        # 首次：dump 完整 data_item 字段类型 + 值，便于排查 dataset_name 未被识别的情况。
        if not getattr(self, "_unnorm_key_logged", False):
            _raw = data_item.get("dataset_name", None)
            print(f"[DATASET_FIX_V3_MARKER] data_item keys = {list(data_item.keys())}", flush=True)
            print(f"[DATASET_FIX_V3_MARKER] dataset_name raw type={type(_raw).__name__} value={_raw!r}", flush=True)
            print(f"[DATASET_FIX_V3_MARKER] _dn type={type(_dn).__name__} value={_dn!r}", flush=True)
            print(f"[DATASET_FIX_V3_MARKER] resolved unnorm_key='{unnorm_key}' (type={type(unnorm_key).__name__})", flush=True)
            self._unnorm_key_logged = True

        # TODO: move to processor
        ret = self.vla_processor(
            text=lang,
            images=pixel_values_seq,
            unnorm_key=unnorm_key,
            suffix_actions=actions,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length,
            truncation=True,
            do_normalize=False, # do not normalize the image for zoe
        )

        future_pixel_values = None
        if "future_images" in data_item:
            # data_item["future_images"] 是一个 (T_future, H, W, C) 的 uint8 NumPy 数组
            future_pixel_values = self._process_future_images(
                data_item["future_images"]
            )

        # ============ 新增：计算任务进度（Task Progress Encoding） ============
        # 从RLDS数据中提取当前timestep信息
        task_progress = self._compute_task_progress(data_item, actions.shape[0])
        
        # ============ 获取task_index和task_ids（O(1)哈希查找） ============
        # task_index: 原始任务索引（与tasks.jsonl对应）
        # task_ids: 聚类后的任务类别ID（无task_id时等于task_index）
        # 注意：lang_to_task_index的key是lower()存储的，查找时也需要lower()
        task_index = self.lang_to_task_index.get(lang.lower(), -1)
        task_ids = self.task_index_to_task_id.get(task_index, task_index)  # 找不到时用task_index
        
        # ============ 新增: 加载Pair Video Features (Video Alignment) ============
        pair_video_cls = None
        pair_video_frames = None
        pair_similarity = None
        pair_is_positive = None
        
        if self.use_pair_data:
            pair_data = self._load_pair_video_features(lang)
            if pair_data is not None:
                pair_video_cls = torch.from_numpy(pair_data['video_cls']).float()  # [D]
                pair_video_frames = torch.from_numpy(pair_data['video_frames']).float()  # [T, D]
                pair_similarity = pair_data['similarity']
                pair_is_positive = pair_data['is_positive']
        
        model_inputs = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            token_type_ids=ret["token_type_ids"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=ret["pixel_values"],
            intrinsic=ret["intrinsic"],
            actions=actions,
            lang=lang,
            future_pixel_values=future_pixel_values,
            task_progress=task_progress,  # 任务进度信息
            task_index=task_index,  # 原始任务索引
            task_ids=task_ids,  # 聚类后的任务类别ID（与model forward参数对齐）
            # ============ 新增: Pair Video Features ============
            pair_video_cls=pair_video_cls,  # [D] Ego4D视频CLS特征
            pair_video_frames=pair_video_frames,  # [T, D] Ego4D视频帧级特征
            pair_similarity=pair_similarity,  # float 匹配相似度
            pair_is_positive=pair_is_positive,  # bool 是否为正样本
        )
        return model_inputs

    def __iter__(self):
        if self.rlds_dataset is None:
            self.rlds_dataset = build_interleaved_dataset(weights=self.sample_weights, dataset_statistics=self.ds_stats, **self.rlds_config).as_numpy_iterator()
            if torch.utils.data.get_worker_info() is not None:
                worker_total_num = torch.utils.data.get_worker_info().num_workers
                worker_id = torch.utils.data.get_worker_info().id
            else:
                worker_id = 0
                worker_total_num = 1
            self.rlds_dataset = itertools.islice(iter(self.rlds_dataset), worker_id, None, worker_total_num)

        for i, data_item in enumerate(self.rlds_dataset):
            ret = self.multi_modal_get_item(data_item)
            if i < len(self):
                yield ret
            else:
                break

def build_datasets(
    data_args,
    output_dir,  # NOTE: from training_args.output_dir
    vla_processor=None,
) -> IterableDataset:
    train_dataset = OpenXIterableDataset(
        data_args.data_root_dir,
        output_dir,
        data_args.data_mix,
        is_train=True,
        max_length=data_args.max_seq_length,
        shuffle_buffer_size=data_args.shuffle_buffer_size,
        tsfm_thread_muti=data_args.tsfm_thread_muti,
        read_thread_muti=data_args.read_thread_muti,
        obs_backward_steps=data_args.obs_backward_steps,
        obs_backward_delta=data_args.obs_backward_delta,
        action_forward_steps=data_args.action_forward_steps,
        use_raw_dataloader=data_args.use_raw_dataloader,
        fix_raw_length=data_args.fix_raw_length,
        vla_processor=vla_processor,
        # 从data_args中传递load_future_images参数，控制是否加载未来图像序列
        load_future_images=data_args.load_future_images,
        # 从data_args中传递task_filename参数，指定任务文件（默认"tasks.jsonl"）
        task_filename=getattr(data_args, 'task_filename', None),
    )
    eval_dataset = None
    # 可选：构建验证集（使用RLDS的val split）
    if getattr(data_args, 'use_eval_split', False):
        eval_dataset = OpenXIterableDataset(
            data_args.data_root_dir,
            output_dir,
            data_args.data_mix,
            is_train=False,
            max_length=data_args.max_seq_length,
            shuffle_buffer_size=data_args.shuffle_buffer_size,
            tsfm_thread_muti=data_args.tsfm_thread_muti,
            read_thread_muti=data_args.read_thread_muti,
            obs_backward_steps=data_args.obs_backward_steps,
            obs_backward_delta=data_args.obs_backward_delta,
            action_forward_steps=data_args.action_forward_steps,
            use_raw_dataloader=data_args.use_raw_dataloader,
            fix_raw_length=data_args.fix_raw_length,
            vla_processor=vla_processor,
            load_future_images=False,
            task_filename=getattr(data_args, 'task_filename', None),
        )
    return train_dataset, eval_dataset
