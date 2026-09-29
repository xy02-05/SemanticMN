#!/usr/bin/env python3
"""
RLDS图像预处理脚本（HDF5版本）：
将指定数据集的所有图像帧，以 numpy 数组的形式，提取并存储到一个HDF5数据库中。
"""

import os
import argparse
from tqdm import tqdm
import tensorflow as tf
import h5py
import numpy as np

# 从你现有的代码库中导入核心的数据加载函数
from .rlds import make_dataset_from_rlds

# 配置TensorFlow不使用GPU
tf.config.set_visible_devices([], "GPU")

def main():
    parser = argparse.ArgumentParser(description='Preprocess RLDS images to an HDF5 database.')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of RLDS datasets.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset folder to process, e.g., "bridge_orig".')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output the HDF5 file.')
    
    args = parser.parse_args()
    
    # 处理输出路径：如果是目录，则在其中创建默认文件名；如果是文件，则创建父目录
    if os.path.isdir(args.output_path):
        # 如果输出路径是目录，在其中创建默认的HDF5文件
        args.output_path = os.path.join(args.output_path, "images.h5")
    
    # 创建输出文件的父目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir:  # 如果父目录不为空（即不是当前目录）
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 初始化HDF5文件
    print(f"Opening HDF5 file at: {args.output_path}")
    
    # 使用 'w' 模式，如果文件存在则覆盖
    with h5py.File(args.output_path, 'w') as hf:
        
        # 2. 使用rlds.py中的函数来加载数据集
        dataset_full_name = f"{args.dataset_name}/1.0.0"
        dataset_kwargs = {
            'name': dataset_full_name,
            'data_dir': args.data_root,
            'image_obs_keys': {"primary": "image_0"},
            'state_obs_keys': [],
        }
        dataset, _ = make_dataset_from_rlds(**dataset_kwargs, train=True, shuffle=False, shuffle_seed=0)

        # 3. 开始写入数据
        total_images = 0
        
        # 遍历每个轨迹
        for traj_data in tqdm(dataset.as_numpy_iterator(), desc=f"Processing {args.dataset_name}"):
            

            # 获取轨迹ID
            traj_index = traj_data['traj_index']
            traj_group = hf.create_group(f'traj_{traj_index}')
            
            image_bytes_list = traj_data['observation']['image_primary']

            # 遍历轨迹中的每一帧
            for step_index, image_bytes in enumerate(image_bytes_list):
                # 1. 解码JPEG字节串为numpy数组
                image_np = tf.io.decode_jpeg(image_bytes).numpy()
                
                # 3. 将处理后的图像作为数据集写入HDF5
                # dataset_name_in_hdf5 = f"image_{step_index}" # 或者可以简单命名
                traj_group.create_dataset(f"image_{step_index}", data=image_np)
                
                total_images += 1

    print(f"All done! Total images processed and saved to HDF5: {total_images}")

if __name__ == '__main__':
    main()