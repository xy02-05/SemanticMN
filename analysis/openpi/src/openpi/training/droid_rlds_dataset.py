"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path

import tqdm

import openpi.shared.download as download


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        filter_dict_path=None,  # Path to json file with indices to sample during training
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )

        # # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        # Load the filter dictionary if provided.
        # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
        # (e.g.,
        # {
        #     "<episode key>": [[0, 100], [200, 300]]
        # }
        # means keep frames 0-99 and 200-299).
        if filter_dict_path is not None:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    (
                        traj["action_dict"]["joint_position"]
                        if action_space == DroidActionSpace.JOINT_POSITION
                        else traj["action_dict"]["joint_velocity"]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]

            traj_len = tf.shape(traj["action"])[0]
            indices = tf.as_string(tf.range(traj_len))

            # Data filtering:
            # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
            # and each step's time step index. This will index into the filter hash table, and if it returns true,
            # then the frame passes the filter.
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "step_id": step_id,
                "passes_filter": passes_filter,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Filter data that doesn't pass the filter
        def filter_from_dict(frame):
            return frame["passes_filter"]

        dataset = dataset.filter(filter_from_dict)

        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return traj

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000



class RobotwinRldsDataset:
    """
    专门为robotwin RLDS数据集设计的数据加载器。
    基于检测到的robotwin数据结构进行优化。
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 50,  # robotwin推荐的动作序列长度
        shuffle_buffer_size: int = 1_000,  # 防止爆内存
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        filter_dict_path=None,
        dataset_name: str = "robotwin_full_hard",  # robotwin默认数据集名称
        dataset_version: str = "1.0.0",
        task_mapping_path: str | None = None,
    ):
        """
        专门为robotwin数据集设计的RLDS数据加载器
        
        Args:
            data_dir: robotwin RLDS数据目录路径
            batch_size: 批次大小
            action_chunk_size: 动作序列长度，robotwin推荐50
            dataset_name: robotwin数据集名称，默认为robotwin_full_hard
            task_mapping_path: tasks_with_id.jsonl 的路径，用于生成task_index和task_id
        """
        # Import tensorflow here to not make it mandatory
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        
        # Configure Tensorflow with *no GPU devices*
        tf.config.set_visible_devices([], "GPU")
        
        # 加载任务映射（用于 task_index 和 task_id 查找）
        self.lang_to_task_index, self.task_index_to_task_id = load_task_mapping_for_rlds(task_mapping_path)
        
        logging.info(f"Loading RobotWin RLDS dataset: {dataset_name} from {data_dir}")
        
        try:
            print(f"Loading RobotWin RLDS dataset: {dataset_name} from {data_dir}")
            builder = tfds.builder(dataset_name, data_dir=data_dir, version=dataset_version)
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)
            print(f"Finish loading RobotWin RLDS dataset: {dataset_name} from {data_dir}")
            # 尝试从 metadata 获取 total_steps，如果不可用则从 splits 信息估算
            try:
                self._length = builder.info.metadata["total_steps"]
            except Exception:
                # 没有 metadata 时，用 splits 的 num_examples 估算
                total_examples = sum(s.num_examples for s in builder.info.splits.values())
                self._length = total_examples * 400  # 粗略估计：每个 episode 平均约100步
                logging.info(f"No metadata found, estimating total_steps from {total_examples} episodes")
            print(f"total_steps = {self._length}")
        except Exception as e:
            logging.error(f"Failed to load robotwin dataset {dataset_name}: {e}")
            raise RuntimeError(f"无法加载robotwin RLDS数据集从 {data_dir}。请检查数据路径和格式。") from e
        
        # robotwin数据集通常不需要成功轨迹过滤，因为数据已经预处理过
        dataset = dataset.repeat()
        
        # robotwin数据通常质量较高，只在需要时启用过滤
        self.use_filter = filter_dict_path is not None
        if self.use_filter:
            # 简化的过滤逻辑
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)
            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")
            # 可以在这里实现具体的过滤逻辑
        
        def restructure_robotwin(traj):
            """重构robotwin轨迹数据，完全保持原始数据结构"""
            
            # robotwin原始数据结构（基于测试结果）：
            # action: [T, 14] - 机器人动作（14维）
            # observation/state: [T, 14] - 机器人状态（14维，与动作维度相同）
            # observation/cam_high: [T, H, W, 3] - 高位摄像头
            # observation/cam_left_wrist: [T, H, W, 3] - 左手腕摄像头
            # observation/cam_right_wrist: [T, H, W, 3] - 右手腕摄像头
            # language_instruction: Text - 任务指令
            
            # 直接提取动作数据（保持原始14维）
            actions = traj["action"]
            
            # 提取观测数据
            observation = traj["observation"]
            
            # 提取图像数据（保持原始格式）
            cam_high = observation["cam_high"]
            cam_left_wrist = observation["cam_left_wrist"]
            cam_right_wrist = observation["cam_right_wrist"]
            
            # 提取机器人状态（保持原始14维）
            robot_state = observation["state"]
            
            # 提取语言指令
            language_instruction = traj["language_instruction"]

            # robotwin数据质量较高，默认通过所有数据
            traj_len = tf.shape(actions)[0]
            passes_filter = tf.ones([traj_len], dtype=tf.bool)
            
            # 返回与OpenPI期望格式兼容的数据结构 
            restructure_data = {
                "actions": actions,  # [T, 14] - 保持原始动作维度
                "observation": {
                    # 图像数据 - 直接映射到OpenPI期望的键名
                    "image": cam_high,                      # 主要图像：cam_high -> image
                    "wrist_image": cam_left_wrist,          # 手腕图像：cam_left_wrist -> wrist_image
                    "wrist_image_right": cam_right_wrist,   # 右手腕图像：cam_right_wrist -> wrist_image_right
                    "state": robot_state,                   # [T, 14] - 完整的原始状态
                },
                "prompt": language_instruction,
                "passes_filter": passes_filter,
            }
            if "task_id" in traj:
                restructure_data["task_id"] = traj["task_id"]
            #     print("task_id")

            # if "episode_id" in traj:
            #     restructure_data["episode_id"] = traj["episode_id"]
            #     print("episode_id")

            # if "frame_id" in traj:
            #     restructure_data["frame_id"] = traj["frame_id"]
            #     print("prame_id")

            return restructure_data
        
        dataset = dataset.traj_map(restructure_robotwin, num_parallel_calls)
        
        def chunk_actions_robotwin(traj):
            """为robotwin数据创建动作块"""
            traj_len = tf.shape(traj["actions"])[0]
            
            # robotwin使用较长的动作序列（50步）
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )
            
            # 限制在轨迹长度内，重复最后一个动作
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            
            # 收集动作块
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            
            # # 收集每个step后续50帧的obs["image"]
            # # future_images需要获取后续的50帧（t+1到t+50）
            # # 所以将索引加1，然后限制在轨迹长度内
            # future_image_indices = action_chunk_indices + 1
            # future_image_indices = tf.minimum(future_image_indices, traj_len - 1)
            
            # obs = traj["observation"]
            # future_images = tf.gather(obs["image"], future_image_indices)  # [traj_len, action_chunk_size, H, W, 3]
            # obs["future_images"] = future_images
            
            return traj
        
        dataset = dataset.traj_map(chunk_actions_robotwin, num_parallel_calls)
        
        # 展平：从轨迹数据集转换为单个动作块数据集
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        
        # 应用数据过滤
        dataset = dataset.filter(lambda frame: frame["passes_filter"])
        dataset = dataset.map(lambda frame: {k: v for k, v in frame.items() if k != "passes_filter"})
        
        # 图像解码：robotwin图像处理
        def decode_robotwin_images(traj):
            """解码robotwin图像数据"""
            obs = traj["observation"]
            
            obs["image"] = tf.io.decode_image(obs["image"], expand_animations=False, dtype=tf.uint8)
            obs["wrist_image"] = tf.io.decode_image(obs["wrist_image"], expand_animations=False, dtype=tf.uint8)
            obs["wrist_image_right"] = tf.io.decode_image(obs["wrist_image_right"], expand_animations=False, dtype=tf.uint8)
            
            # # 解码future_images序列中的每一帧图像
            # # future_images形状: [action_chunk_size, H, W, 3] (在展平后)
            # if "future_images" in obs:
            #     def decode_single_image(img):
            #         return tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
            #     obs["future_images"] = tf.map_fn(
            #         decode_single_image,
            #         obs["future_images"],
            #         fn_output_signature=tf.TensorSpec(shape=[None, None, None], dtype=tf.uint8)
            #     )
            
            return traj
        
        dataset = dataset.frame_map(decode_robotwin_images, num_parallel_calls)

        # 批处理和随机化
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.with_ram_budget(1)  # 内存优化
        
        self.dataset = dataset
        
        logging.info(f"RobotWin RLDS dataset initialized: batch_size={batch_size}, action_chunk_size={action_chunk_size}")
        
    def __iter__(self):
        """迭代数据集，在 numpy 阶段添加 task_index 和 task_id"""
        import numpy as np
        
        for batch in self.dataset.as_numpy_iterator():
            # 从 prompt 获取 task_index 和 task_id
            prompts = batch["prompt"]
            batch_size = len(prompts)
            task_indices = np.zeros(batch_size, dtype=np.int64)
            task_ids = np.zeros(batch_size, dtype=np.int64)
            
            for i, prompt in enumerate(prompts):
                # 解码 bytes 为 string，并做最小规范化（对齐 tasks_with_id.jsonl 的 lower key）
                prompt_str = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
                prompt_norm = prompt_str.strip().lower()
                
                # 查找 task_index
                task_index = self.lang_to_task_index.get(prompt_norm, -1)
                task_indices[i] = task_index
                
                # 查找 task_id：如果有映射则使用，否则用 task_index 代替
                if task_index in self.task_index_to_task_id:
                    task_ids[i] = self.task_index_to_task_id[task_index]
                else:
                    task_ids[i] = task_index  # 没有 task_id 时使用 task_index
            
            batch["task_index"] = task_indices
            batch["task_id"] = task_ids
            yield batch
        
    def __len__(self):
        # robotwin数据集的大概样本数量
        return self._length * 10 


def load_task_mapping_for_rlds(task_mapping_path: str | None = None) -> tuple[dict, dict]:
    """
    从显式指定的路径加载任务映射
    
    Args:
        task_mapping_path: tasks_with_id.jsonl 的显式路径，必须设置才能加载
        
    Returns:
        (lang_to_task_index, task_index_to_task_id)
    """
    lang_to_task_index = {}
    task_index_to_task_id = {}
    
    if not task_mapping_path:
        logging.info("[TaskMapping] task_mapping_path未设置，task_id将回退为task_index")
        return {}, {}
    
    tasks_path = Path(task_mapping_path)
    assert tasks_path.exists() or not task_mapping_path, f"[TaskMapping] 文件不存在: {task_mapping_path}"
    
    # 读取文件，构建映射
    with open(tasks_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            task_text = item.get("task", "")
            task_index = item.get("task_index", -1)
            if task_text and task_index >= 0:
                lang_to_task_index[task_text.lower()] = task_index
            if "task_id" in item and task_index >= 0:
                task_index_to_task_id[task_index] = item["task_id"]
    
    logging.info(f"[TaskMapping] Loaded {len(lang_to_task_index)} tasks from {tasks_path}")
    if task_index_to_task_id:
        logging.info(f"[TaskMapping] Loaded {len(task_index_to_task_id)} task_id mappings")
    
    return lang_to_task_index, task_index_to_task_id


class Bridgev2RldsDataset:
    """
    专门为Bridgev2 RLDS数据集设计的数据加载器。
    基于检测到的Bridgev2数据结构进行优化。
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 5,  # Bridgev2推荐的动作序列长度
        shuffle_buffer_size: int = 1_000,  # 防止爆内存
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        filter_dict_path=None,
        dataset_name: str = "bridgev2",  # Bridgev2默认数据集名称
        dataset_version: str = "1.0.0",
        task_mapping_path: str | None = None,
    ):
        """
        专门为Bridgev2数据集设计的RLDS数据加载器
        
        Args:
            data_dir: Bridgev2 RLDS数据目录路径
            batch_size: 批次大小
            action_chunk_size: 动作序列长度，Bridgev2推荐5
            dataset_name: Bridgev2数据集名称，默认为bridgev2
        """
        # Import tensorflow here to not make it mandatory
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        
        # Configure Tensorflow with *no GPU devices*
        tf.config.set_visible_devices([], "GPU")
        
        # 加载任务映射（用于 task_index 和 task_id 查找）
        self.lang_to_task_index, self.task_index_to_task_id = load_task_mapping_for_rlds(task_mapping_path)
        
        logging.info(f"Loading Bridgev2 RLDS dataset: {dataset_name} from {data_dir}")
        
        try:
            print(f"Loading Bridgev2 RLDS dataset: {dataset_name} from {data_dir}")
            builder = tfds.builder(dataset_name, data_dir=data_dir, version=dataset_version)
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)
            print(f"Finish loading Bridgev2 RLDS dataset: {dataset_name} from {data_dir}")
        except Exception as e:
            logging.error(f"Failed to load Bridgev2 dataset {dataset_name}: {e}")
            raise RuntimeError(f"无法加载Bridgev2 RLDS数据集从 {data_dir}。请检查数据路径和格式。") from e
        
        # robotwin数据集通常不需要成功轨迹过滤，因为数据已经预处理过
        dataset = dataset.repeat()
        
        # robotwin数据通常质量较高，只在需要时启用过滤
        self.use_filter = filter_dict_path is not None
        if self.use_filter:
            # 简化的过滤逻辑
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)
            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")
            # 可以在这里实现具体的过滤逻辑
        
        def restructure_bridgev2(traj):
            """重构robotwin轨迹数据，完全保持原始数据结构"""
            
            # Bridgev2原始数据结构（基于测试结果）：
            # action: [T, 7] - 机器人动作（7维）
            # observation/image_0: [T, H, W, 3] - 主摄像头
            # language_instruction: Text - 任务指令
            
            # 直接提取动作数据（保持原始1维）
            actions = traj["action"]
            
            # 提取观测数据
            observation = traj["observation"]
            
            # 提取图像数据（保持原始格式）
            image_0 = observation["image_0"]
            
            # 提取机器人状态（保持原始7维）
            robot_state = observation["state"]
            
            # 提取语言指令
            language_instruction = traj["language_instruction"]
            
            # Bridgev2数据质量较高，默认通过所有数据
            traj_len = tf.shape(actions)[0]
            passes_filter = tf.ones([traj_len], dtype=tf.bool)
            
            # 返回与OpenPI期望格式兼容的数据结构
            return {
                "actions": actions,  # [T, 7] - 保持原始动作维度
                "observation": {
                    # 图像数据 - 直接映射到OpenPI期望的键名
                    "image": image_0,                      # 主要图像：image_0 -> image
                    "state": robot_state,                   # [T, 14] - 完整的原始状态
                },
                "prompt": language_instruction,
                "passes_filter": passes_filter,
            }
        
        dataset = dataset.traj_map(restructure_bridgev2, num_parallel_calls)
        
        # 不过滤空指令轨迹，空prompt样本仍参与VLA训练，alignment计算时会被mask掉
        dataset = dataset.filter(lambda traj: tf.reduce_any(traj["prompt"] != b""))
        
        def chunk_actions_bridgev2(traj):
            """为Bridgev2数据创建动作块"""
            traj_len = tf.shape(traj["actions"])[0]
            
            # Bridgev2使用较长的动作序列（5步）
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )
            
            # 限制在轨迹长度内，重复最后一个动作
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            
            # 收集动作块
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj
        
        dataset = dataset.traj_map(chunk_actions_bridgev2, num_parallel_calls)
        
        # 展平：从轨迹数据集转换为单个动作块数据集
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        
        # 应用数据过滤
        dataset = dataset.filter(lambda frame: frame["passes_filter"])
        dataset = dataset.map(lambda frame: {k: v for k, v in frame.items() if k != "passes_filter"})
        
        # 图像解码：Bridgev2图像处理
        def decode_bridgev2_images(traj):
            """解码Bridgev2图像数据"""
            obs = traj["observation"]
            
            obs["image"] = tf.io.decode_image(obs["image"], expand_animations=False, dtype=tf.uint8)
            return traj
        
        dataset = dataset.frame_map(decode_bridgev2_images, num_parallel_calls)
        
        # 批处理和随机化
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.with_ram_budget(1)  # 内存优化
        
        self.dataset = dataset
        
        logging.info(f"Bridgev2 RLDS dataset initialized: batch_size={batch_size}, action_chunk_size={action_chunk_size}")
        
    def __iter__(self):
        """迭代数据集，在 numpy 阶段添加 task_index 和 task_id"""
        import numpy as np
        
        for batch in self.dataset.as_numpy_iterator():
            # 从 prompt 获取 task_index 和 task_id
            prompts = batch["prompt"]
            batch_size = len(prompts)
            task_indices = np.zeros(batch_size, dtype=np.int64)
            task_ids = np.zeros(batch_size, dtype=np.int64)
            
            for i, prompt in enumerate(prompts):
                # 解码 bytes 为 string，并做最小规范化（对齐 tasks_with_id.jsonl 的 lower key）
                prompt_str = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
                prompt_norm = prompt_str.strip().lower()
                
                # 查找 task_index
                task_index = self.lang_to_task_index.get(prompt_norm, -1)
                task_indices[i] = task_index
                
                # 查找 task_id：如果有映射则使用，否则用 task_index 代替
                if task_index in self.task_index_to_task_id:
                    task_ids[i] = self.task_index_to_task_id[task_index]
                else:
                    task_ids[i] = task_index  # 没有 task_id 时使用 task_index
            
            batch["task_index"] = task_indices
            batch["task_id"] = task_ids
            yield batch
        
    def __len__(self):
        # Bridgev2数据集的大概样本数量
        return 1_000_000  # 根据实际Bridgev2数据集大小调整
