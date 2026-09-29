# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from egovlpv2.base.base_dataset import TextVideoDataset
from egovlpv2.data_loader.transforms import init_transform_dict, init_video_transform_dict


class FHO_Dataset(TextVideoDataset):
    """
    FHO (Forecasting Human Object Interaction) Dataset
    
    This dataset loads preprocessed FHO data from a CSV file and provides
    video clips with corresponding action descriptions, following the same
    architecture as other EgoVLPv2 datasets.
    """
    
    def _load_metadata(self):
        """Load metadata from preprocessed CSV file"""
        split_files = {
            'train': 'fho_processed_shuffled_normalized.csv',
            'val': 'fho_processed_shuffled_normalized.csv',     # 暂时都用同一个文件，可以后续按需分割
            'test': 'fho_processed_shuffled_normalized.csv'
        }
        target_split_fp = split_files[self.split]
        
        # 读取预处理后的CSV文件
        csv_path = os.path.join(self.meta_dir, target_split_fp)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"FHO CSV file not found: {csv_path}")
        
        print(f"Loading FHO metadata from: {csv_path}")
        self.metadata = pd.read_csv(csv_path)
        
        # 数据过滤和预处理
        if self.cut:
            self.metadata = self.metadata[:self.cut]
        
        if self.subsample > 1:
            self.metadata = self.metadata[::self.subsample]
            
        # 移除无效数据
        self.metadata = self.metadata.dropna(subset=['processed_text', 'clip_uid'])
        self.metadata = self.metadata[self.metadata['duration_sec'] > 0]
        
        print(f"Loaded {len(self.metadata)} samples for split '{self.split}'")
        
        # 设置帧采样方式 - 和其他数据集保持一致
        self.frame_sample = 'rand' if self.split == 'train' else 'uniform'
    
    def _get_video_path(self, sample):
        """Get video file path for a sample"""
        clip_uid = sample['clip_uid']
        rel_video_fp = f'{clip_uid}.mp4'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp
    
    def _get_caption(self, sample):
        """Get text caption for a sample"""
        return sample['processed_text']
    
    def _get_video_frames(self, video_fp, sample):
        """Load video frames for a sample with precise time alignment"""
        video_loading = self.video_params.get('loading', 'strict')
        
        # 获取精确的时间信息 - 这是关键！
        start_sec = max(float(sample['clip_start_sec']), 0)  # 动作在视频clip中的开始时间
        end_sec = max(float(sample['clip_end_sec']), 0)      # 动作在视频clip中的结束时间
        
        try:
            if os.path.isfile(video_fp):
                # 使用cv2_charades reader，支持start_sec和end_sec参数
                # 这确保了视频时间段和文本的精确对应
                imgs, idxs = self.video_reader(
                    video_path=video_fp, 
                    num_frames=self.video_params['num_frames'], 
                    sample=self.frame_sample,
                    start_sec=start_sec, 
                    end_sec=end_sec
                )
            else:
                print(f"Warning: missing video file {video_fp}")
                if video_loading == 'strict':
                    raise FileNotFoundError(f"Video file not found: {video_fp}")
                else:
                    # 创建黑色占位帧
                    imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                    imgs = transforms.ToTensor()(imgs).unsqueeze(0)
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(f'Video loading failed for {video_fp}') from e
            else:
                print(f"Warning: Video loading failed for {video_fp}, using black frames")
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)
        
        # 应用transforms - 和其他数据集完全一致的逻辑
        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover [T, C, H, W]
            else:
                imgs = self.transforms(imgs)
        
        # 确保输出shape一致
        final = torch.zeros([self.video_params['num_frames'], 3, 
                           self.video_params['input_res'], 
                           self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        
        return final
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, item):
        """Get a single sample - 输出格式严格按照要求"""
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        
        # 获取视频路径和加载帧
        video_fp, rel_fp = self._get_video_path(sample)
        video_frames = self._get_video_frames(video_fp, sample)
        
        # 获取文本标题
        caption = self._get_caption(sample)
        
        # 创建元数据 - 格式和其他数据集保持一致
        meta_arr = {
            'raw_captions': caption,
            'paths': rel_fp,  # 保持和CharadesEgo一致的格式（字符串而非列表）
            'dataset': self.dataset_name
        }
        
        # 创建target字段 - 兼容CharadesEgo的trainer
        # FHO数据集主要用于视频-文本对齐，不是多标签分类任务
        # 这里创建一个虚拟的全零target，维度157与CharadesEgo保持一致
        target = torch.IntTensor(157).zero_()
        
        # 返回标准格式 - 严格按照要求，保持和CharadesEgo完全一致
        return {
            'video': video_frames,  # torch.Tensor [num_frames, 3, input_res, input_res]
            'text': caption,        # str - 原始文本描述
            'meta': meta_arr,       # dict - 元数据信息
            'target': target        # torch.Tensor [157] - 多标签分类target（FHO数据集用全零）
        }


if __name__ == "__main__":
    # 测试代码 - 按照其他数据集的格式
    print("Testing FHO Dataset...")
    
    kwargs = dict(
        dataset_name="FHO_Dataset",
        text_params={
            "input": "text"
        },
        video_params={
            "input_res": 224,
            "num_frames": 4,
            "loading": "lax"  # 使用lax模式以便测试
        },
        data_dir="/data/xuyuan/UniVLA_env/mirror_neuron/data/Ego4D/v2",
        meta_dir="/data/xuyuan/UniVLA_env/mirror_neuron/data/Ego4D/v2/annotations",
        tsfms=init_video_transform_dict()['test'],  # 使用和其他数据集相同的transforms
        reader='cv2_charades',  # 使用支持时间段的reader
        split='train',
        cut=10  # 只测试前10个样本
    )
    
    try:
        dataset = FHO_Dataset(**kwargs)
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # 测试加载样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample keys: {sample.keys()}")
            print(f"✓ Video shape: {sample['video'].shape}")
            print(f"✓ Text: {sample['text'][:100]}...")  # 只显示前100个字符
            print(f"✓ Meta keys: {list(sample['meta'].keys())}")
            
            # 验证输出格式
            assert isinstance(sample['video'], torch.Tensor), "Video should be torch.Tensor"
            assert len(sample['video'].shape) == 4, "Video should be 4D tensor"
            assert sample['video'].shape[1] == 3, "Video should have 3 channels"
            assert isinstance(sample['text'], str), "Text should be string"
            assert isinstance(sample['meta'], dict), "Meta should be dict"
            assert 'raw_captions' in sample['meta'], "Meta should contain raw_captions"
            assert 'paths' in sample['meta'], "Meta should contain paths"
            assert 'dataset' in sample['meta'], "Meta should contain dataset"
            
            print("✓ All format validations passed!")
            print("✓ FHO Dataset test completed successfully!")
        else:
            print("✗ No samples found in dataset")
            
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc() 