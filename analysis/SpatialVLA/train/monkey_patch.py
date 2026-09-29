import os
import hashlib
import torch
import numpy as np
import torch.nn as nn
import datasets
from torch.utils.data import DataLoader
import transformers
from transformers import logging, TrainerCallback, Trainer
from transformers.trainer import LengthGroupedSampler, RandomSampler, has_length, is_datasets_available, seed_worker, _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_pt_utils import logger
from typing import List, Optional
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100

# data patch
def concat_pad_data_collator(features, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        # handel temp_token_type_ids for gemma
        temp_token_type_ids = torch.LongTensor([0] * max_item_length) # pad with 0 to indicate first scentence
        temp_token_type_ids[:feat['token_type_ids'].shape[0]] = feat['token_type_ids']
        feat['token_type_ids'] = temp_token_type_ids

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k == 'lang' and v is not None:
            # Special handling for lang field - collect as list of strings
            batch[k] = [f[k] for f in features]
        elif k == 'future_pixel_values' and v is not None:
            # 最终形状将是 (B, T_future, C, H, W)，这正是我们想要的。
            batch[k] = torch.stack([f[k] for f in features])
        elif k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch

# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    将索引列表按长度均匀分割成指定数量的chunks，用于长度分组采样
    
    算法：
    1. 如果索引数量无法被chunk数整除，使用轮询分配
    2. 否则计算每个chunk的目标大小，动态平衡各chunk的总长度
    3. 优先分配到当前总长度最短的chunk
    4. 当chunk达到目标大小时，将其长度设为无穷大避免继续分配
    
    Args:
        indices (List[int]): 要分割的索引列表
        lengths (List[int]): 每个索引对应的序列长度
        num_chunks (int): 目标chunk数量
        
    Returns:
        List[List[int]]: 分割后的chunks列表
    """
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks

# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    生成长度分组的采样索引，优化训练效率减少填充浪费
    
    算法流程：
    1. 随机打乱所有索引
    2. 按megabatch_size (world_size * batch_size) 分组
    3. 在每个megabatch内按序列长度降序排列
    4. 将每个megabatch均匀分割成world_size个子batch
    5. 展平所有子batch得到最终索引序列
    
    作用：将相似长度的样本分组到同一batch，减少padding开销
    
    Args:
        lengths (List[int]): 每个样本的序列长度
        batch_size (int): 单个设备的批次大小
        world_size (int): 分布式训练的进程数
        generator (torch.Generator, optional): 随机数生成器
        merge (bool): 兼容参数，暂未使用
        
    Returns:
        List[int]: 长度分组后的索引序列
    """
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L99
class LengthGroupedSampler(Sampler):
    """
    长度分组采样器：将数据集按序列长度分组采样，提高训练效率
    
    设计目标：
    - 将相似长度的样本分组到同一batch，减少padding浪费
    - 保持一定随机性，避免过度规律化
    - 支持分布式训练，考虑world_size参数
    - 自动从数据集推断序列长度或接受预计算的长度列表
    
    适用场景：
    - 序列长度差异较大的数据集
    - 需要优化训练速度和内存使用的场景
    - SpatialVLA等多模态模型的序列数据
    """
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

# patch trainer
def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    """
    替换Trainer的训练采样器获取方法，支持SpatialVLA的长度分组采样
    
    功能改进：
    1. 支持多数据集的长度提取（self.train_dataset.datasets）
    2. 使用自定义LengthGroupedSampler替代原生实现
    3. 考虑gradient_accumulation_steps调整world_size
    4. 兼容原有的group_by_length配置选项
    
    替换原因：
    - 原生LengthGroupedSampler不支持多数据集场景
    - 需要自定义长度提取逻辑适配SpatialVLA数据格式
    - 优化分布式训练的采样策略
    
    Args:
        self: Trainer实例
        
    Returns:
        Optional[torch.utils.data.Sampler]: 训练采样器实例或None
    """
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    # Build the sampler.
    if self.args.group_by_length:
        lengths = []
        for dataset in self.train_dataset.datasets:
            lengths = lengths + dataset.length
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            # self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        return RandomSampler(self.train_dataset)

def replace_train_sampler():
    transformers.Trainer._get_train_sampler = _get_train_sampler
    print('Replace train sampler!!')

def get_train_dataloader(self) -> DataLoader:
    """
    替换Trainer的训练数据加载器创建方法，支持SpatialVLA特有功能
    
    关键改进：
    1. 支持自定义数据整理器（concat_pad_data_collator）
    2. 集成自定义长度分组采样器
    3. 支持SpatialVLA数据集的use_raw_dataloader属性
    4. 保持与HuggingFace Trainer的完全兼容性
    
    use_raw_dataloader功能：
    - 当数据集设置use_raw_dataloader=True时，直接返回DataLoader
    - 否则使用accelerator.prepare()进行分布式和混合精度包装
    - 适配SpatialVLA特殊的数据加载需求
    
    Args:
        self: Trainer实例
        
    Returns:
        DataLoader: 配置好的训练数据加载器
    """
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    if train_dataset.use_raw_dataloader:
        return DataLoader(train_dataset, **dataloader_params)
    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

def replace_train_dataloader():
    transformers.Trainer.get_train_dataloader = get_train_dataloader
    print("Replace train dataloader!!")

# 拆分函数 - 预处理输入和标签
def _prepare_inputs_and_labels(trainer, inputs, num_items_in_batch=None):
    """
    预处理输入数据和标签，提取labels并设置loss_kwargs
    
    Args:
        trainer: Trainer实例
        inputs: 输入数据字典
        num_items_in_batch: 批次中的样本数量
        
    Returns:
        tuple: (processed_inputs, labels)
    """
    if (trainer.label_smoother is not None or trainer.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    
    if trainer.model_accepts_loss_kwargs:
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
    
    return inputs, labels

# 拆分函数 - 计算标准损失
def _compute_standard_loss(trainer, model, outputs, labels, inputs, num_items_in_batch=None):
    """
    使用HuggingFace标准方法计算损失
    
    Args:
        trainer: Trainer实例
        model: 模型实例
        outputs: 模型输出
        labels: 标签数据
        inputs: 输入数据
        num_items_in_batch: 批次中的样本数量
        
    Returns:
        torch.Tensor: 计算得到的损失值
    """
    # Save past state if it exists
    if trainer.args.past_index >= 0:
        trainer._past = outputs[trainer.args.past_index]

    if labels is not None:
        unwrapped_model = trainer.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        if trainer.compute_loss_func is not None:
            loss = trainer.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = trainer.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = trainer.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    if trainer.args.average_tokens_across_devices and trainer.model_accepts_loss_kwargs:
        loss *= trainer.accelerator.num_processes
    
    return loss

# 拆分函数 - 计算SpatialVLA特有指标
def _compute_spatialvla_metrics(trainer, model, outputs, inputs):
    """
    计算SpatialVLA特有的动作预测指标
    
    Args:
        trainer: Trainer实例  
        model: SpatialVLA模型实例
        outputs: 模型输出
        inputs: 输入数据
    """
    with torch.no_grad():
        logits = outputs["logits"]  # (bs, seq, voc)
        labels = inputs["labels"]  # (bs, seq)
        shift_logits = logits[..., :-1, :].argmax(-1).contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 创建动作token掩码（检测所有动作token）
        mask = (shift_labels >= model.action_tokenizer.translation_tokenizer.token_start_idx) & (
            shift_labels <= model.action_tokenizer.gripper_tokenizer.token_end_idx
        )
        gt_action_ids, pred_action_ids = shift_labels[mask], shift_logits[mask]
        correct_preds = gt_action_ids == pred_action_ids
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # 计算分类动作准确率：平移、旋转、抓取
        token_start_idx, token_end_idx = (
            model.action_tokenizer.translation_tokenizer.token_start_idx,
            model.action_tokenizer.translation_tokenizer.token_end_idx,
        )
        translation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        token_start_idx, token_end_idx = (
            model.action_tokenizer.rotation_tokenizer.token_start_idx,
            model.action_tokenizer.rotation_tokenizer.token_end_idx,
        )
        rotation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        token_start_idx, token_end_idx = (
            model.action_tokenizer.gripper_tokenizer.token_start_idx,
            model.action_tokenizer.gripper_tokenizer.token_end_idx,
        )
        gripper_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        translation_gt_action_ids, translation_pred_action_ids = gt_action_ids[translation_mask], pred_action_ids[translation_mask]
        rotation_gt_action_ids, rotation_pred_action_ids = gt_action_ids[rotation_mask], pred_action_ids[rotation_mask]
        gripper_gt_action_ids, gripper_pred_action_ids = gt_action_ids[gripper_mask], pred_action_ids[gripper_mask]

        translation_correct_preds = translation_gt_action_ids == translation_pred_action_ids
        rotation_correct_preds = rotation_gt_action_ids == rotation_pred_action_ids
        gripper_correct_preds = gripper_gt_action_ids == gripper_pred_action_ids

        translation_action_accuracy = translation_correct_preds.sum().float() / translation_mask.sum().float()
        rotation_action_accuracy = rotation_correct_preds.sum().float() / rotation_mask.sum().float()
        gripper_action_accuracy = gripper_correct_preds.sum().float() / gripper_mask.sum().float()

        # 计算连续动作L1损失
        gt_actions = inputs["actions"].reshape(-1, 7).to(device="cpu", dtype=torch.float32)
        pred_actions = model.action_tokenizer.decode_token_ids_to_actions(pred_action_ids.cpu().numpy().reshape(-1, 3))
        l1_loss = nn.functional.l1_loss(torch.tensor(pred_actions), torch.tensor(gt_actions))

        # 只在符合logging_steps间隔时记录详细指标
        if trainer.state.global_step % trainer.args.logging_steps == 0:
            trainer.log(
                {
                    "accuracy": action_accuracy.item(),
                    "translation_accuracy": translation_action_accuracy.item(),
                    "rotation_accuracy": rotation_action_accuracy.item(),
                    "gripper_accuracy": gripper_action_accuracy.item(),
                    "l1_loss": l1_loss.item(),
                }
            )

def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    替换Trainer的损失计算方法，专为SpatialVLA的动作预测任务设计
    
    核心功能：
    1. 计算标准的因果语言模型损失（VLA损失）
    2. 提取并分析动作token的预测准确率
    3. 计算细分的动作准确率（平移、旋转、抓取）
    4. 将离散动作token解码为连续动作并计算L1损失
    5. 记录详细的训练指标用于监控
    
    SpatialVLA特有功能：
    - 动作token范围检测：从translation_tokenizer到gripper_tokenizer
    - 三类动作准确率：translation_accuracy, rotation_accuracy, gripper_accuracy
    - 连续动作解码：使用action_tokenizer.decode_token_ids_to_actions
    - L1损失计算：预测动作与真实动作的L1距离
    
    Args:
        self: Trainer实例
        model: SpatialVLA模型
        inputs: 批次输入数据（包含input_ids, labels, actions等）
        return_outputs: 是否返回模型输出
        num_items_in_batch: 批次中的样本数量
        
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, ModelOutput]]: 损失值或(损失值, 模型输出)
    """
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    # 1. 预处理输入和标签
    inputs, labels = _prepare_inputs_and_labels(self, inputs, num_items_in_batch)
    
    # 2. 获取模型输出
    outputs = model(**inputs, output_hidden_states=True)
    
    # 3. 计算标准损失
    loss = _compute_standard_loss(self, model, outputs, labels, inputs, num_items_in_batch)

    # 计算SpatialVLA特有的动作预测指标
    _compute_spatialvla_metrics(self, model, outputs, inputs)

    return (loss, outputs) if return_outputs else loss

def replace_compute_loss():
    transformers.Trainer.compute_loss = compute_loss
    print("Replace compute_loss!!")

class SaveProcessorCallback(TrainerCallback):
    """
    自定义回调：在模型保存时同时保存SpatialVLA处理器
    
    功能：
    1. 监听Trainer的保存事件
    2. 在主进程中保存processor到相应目录
    3. 保持与模型检查点的目录结构一致
    4. 确保processor与模型版本同步
    
    重要性：
    - SpatialVLA的processor包含action_tokenizer等关键组件
    - 推理时需要完整的processor进行数据预处理
    - 支持检查点恢复时的完整状态还原
    """
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            output_dir = args.output_dir
            if state.global_step > 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.processor.save_pretrained(output_dir)
        return control

class ProfilerTrainer(Trainer):
    """
    性能分析版本的Trainer：集成PyTorch Profiler进行性能监控
    
    功能：
    1. 在训练过程中自动收集性能数据
    2. 生成TensorBoard可视化的性能报告
    3. 分析GPU利用率、内存使用、算子耗时等
    4. 调试和优化训练性能的重要工具
    
    配置：
    - wait=2: 跳过前2个step的数据收集
    - warmup=2: 2个step的预热阶段
    - active=4: 收集4个step的详细数据
    - 输出到./profiler_output目录
    
    使用场景：
    - 训练性能调优
    - 内存瓶颈分析
    - 模型效率评估
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=4),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output")
        )
        self.profiler.__enter__()

    def training_step(self, model, inputs):
        output = super().training_step(model, inputs)
        self.profiler.step()
        return output

    def __del__(self):
        self.profiler.__exit__(None, None, None)