# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import logging
import pandas as pd
import torch
import numpy as np
import decord
from decord import cpu

logger = logging.getLogger(__name__)

from egovlpv2.base.base_dataset import TextVideoDataset


class FHO_Dataset_EgoHOD(TextVideoDataset):
    """
    FHO Dataset for EgoHOD - 完全对齐EgoHOD的数据处理流程
    
    关键对齐点：
    1. 使用decord库加载视频（与EgoHOD的data_utils.py中的video_loader一致）
    2. 采用get_frame_ids进行帧采样（jitter模式：训练时随机，测试时中心采样）
    3. 视频输出格式: [T, H, W, C] float32范围[0,1]，后续需要permute到[T, C, H, W]
    4. Transform应用顺序：先除以255归一化，再permute，最后应用transforms
    """
    
    def _load_metadata(self):
        """加载CSV元数据文件"""
        split_files = {
            'train': 'fho_processed_shuffled_normalized.csv',
            'val': 'fho_processed_shuffled_normalized.csv',
            'test': 'fho_processed_shuffled_normalized.csv'
        }
        target_split_fp = split_files[self.split]
        csv_path = os.path.join(self.meta_dir, target_split_fp)
        
        self.metadata = pd.read_csv(csv_path)
        
        if self.cut:
            self.metadata = self.metadata[:self.cut]
        if self.subsample > 1:
            self.metadata = self.metadata[::self.subsample]
            
        self.metadata = self.metadata.dropna(subset=['processed_text', 'clip_uid'])
        self.metadata = self.metadata[self.metadata['duration_sec'] > 0]
        
        print(f"Loaded {len(self.metadata)} EgoHOD samples for split '{self.split}'")
    
    def _get_frame_ids(self, start_frame, end_frame, num_segments, jitter):
        """
        EgoHOD风格的帧采样函数
        对齐 EgoHOD/dataset/data_utils.py 中的 get_frame_ids
        
        参数:
            start_frame: 起始帧索引
            end_frame: 结束帧索引
            num_segments: 采样帧数
            jitter: True=随机采样(训练), False=中心采样(测试)
        """
        seg_size = float(end_frame - start_frame - 1) / num_segments
        seq = []
        for i in range(num_segments):
            start = int(np.round(seg_size * i) + start_frame)
            end = int(np.round(seg_size * (i + 1)) + start_frame)
            start = min(start, end_frame - 1)
            end = min(end, end_frame)
            
            if jitter:
                frame_id = np.random.randint(low=start, high=(end + 1))
            else:
                frame_id = (start + end) // 2
            seq.append(frame_id)
        return seq
    
    def _load_video_frames(self, video_path, start_sec, end_sec, num_frames, jitter):
        """
        使用decord加载视频帧 - 对齐EgoHOD的video_loader_by_timestamp
        
        返回: torch.Tensor [T, H, W, C] float32, 范围[0, 1]
        """
        # 使用decord读取视频
        vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        
        # 计算帧索引范围
        start_frame = int(np.round(fps * start_sec)) if start_sec else 0
        end_frame = int(np.ceil(fps * end_sec)) if end_sec else len(vr) - 1
        end_frame = min(end_frame, len(vr) - 1)
        
        # 采样帧索引
        frame_ids = self._get_frame_ids(start_frame, end_frame, num_frames, jitter)
        
        # 读取帧（兼容不同版本的decord）
        batch = vr.get_batch(frame_ids)
        if hasattr(batch, 'asnumpy'):
            # decord NDArray
            frames = batch.asnumpy()
        elif isinstance(batch, torch.Tensor):
            # PyTorch Tensor
            frames = batch.numpy()
        else:
            # 其他类型，尝试转为numpy
            frames = np.array(batch)
        
        frames = torch.tensor(frames, dtype=torch.float32)  # [T, H, W, C] float32
        
        return frames
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, item, max_retries=10):
        """
        获取单个样本 - 完全按照EgoHOD的处理流程
        
        返回格式与egovlpv2兼容:
            'video': [T, C, H, W] float32
            'text': str
            'meta': dict
            'target': tensor (兼容字段)
        """
        for retry in range(max_retries):
            try:
                item = item % len(self.metadata)
                sample = self.metadata.iloc[item]
                
                # 视频路径: 优先使用video_path列(统一CSV), 否则用默认规则
                clip_uid = sample['clip_uid']
                if 'video_path' in sample.index and pd.notna(sample['video_path']):
                    video_path = sample['video_path']
                else:
                    video_path = os.path.join(self.data_dir, f'{clip_uid}.mp4')
                
                # 时间范围
                start_sec = float(sample['clip_start_sec'])
                end_sec = float(sample['clip_end_sec'])
                
                # 采样模式：训练时jitter=True，测试时jitter=False
                jitter = (self.split == 'train')
                
                # 加载视频帧：[T, H, W, C] float32
                frames = self._load_video_frames(
                    video_path, 
                    start_sec, 
                    end_sec, 
                    self.video_params['num_frames'],
                    jitter
                )
                
                # EgoHOD风格的预处理：
                # 1. 先归一化到[0,1]
                frames = frames / 255.0
                
                # 2. Permute到[T, C, H, W]
                frames = frames.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
                
                # 3. 应用transforms (RandomResizedCrop + Normalize)
                if self.transforms is not None:
                    # 对于多帧视频，需要转换为[C, T, H, W]格式
                    frames = frames.transpose(0, 1)  # [T, C, H, W] -> [C, T, H, W]
                    frames = self.transforms(frames)
                    frames = frames.transpose(0, 1)  # [C, T, H, W] -> [T, C, H, W]
                
                # 文本
                caption = sample['processed_text']
                
                # 元数据
                meta_arr = {
                    'raw_captions': caption,
                    'paths': f'{clip_uid}.mp4',
                    'dataset': self.dataset_name
                }
                
                # 兼容字段
                target = torch.IntTensor(157).zero_()
                
                return {
                    'video': frames,     # [T, C, H, W]
                    'text': caption,     # str
                    'meta': meta_arr,    # dict
                    'target': target     # tensor
                }
            except Exception as e:
                logger.warning(
                    f"[FHO_Dataset_EgoHOD] Failed to load sample {item} "
                    f"(retry {retry + 1}/{max_retries}): {e}"
                )
                # 随机选择一个新的样本索引进行重试
                item = random.randint(0, len(self.metadata) - 1)
        
        # 所有重试都失败，抛出异常
        raise RuntimeError(
            f"[FHO_Dataset_EgoHOD] Failed to load any valid sample after {max_retries} retries"
        )


if __name__ == "__main__":
    """简单测试代码"""
    from egovlpv2.data_loader.transforms import init_video_transform_dict
    
    kwargs = dict(
        dataset_name="FHO_Dataset_EgoHOD",
        text_params={"input": "text"},
        video_params={"input_res": 224, "num_frames": 4, "loading": "lax"},
        data_dir="/data/xuyuan/UniVLA_env/mirror_neuron/data/Ego4D/v2",
        meta_dir="/data/xuyuan/UniVLA_env/mirror_neuron/data/Ego4D/v2/annotations",
        tsfms=init_video_transform_dict()['test'],
        reader='decord',  # 这里不使用reader，直接用decord
        split='train',
        cut=5
    )
    
    dataset = FHO_Dataset_EgoHOD(**kwargs)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Video shape: {sample['video'].shape}")
    print(f"Text: {sample['text'][:100]}")
    print(f"Meta: {sample['meta']}") 