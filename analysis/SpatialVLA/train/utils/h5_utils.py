"""
HDF5 特征存储工具
用于存储和读取 VLA/VLM 特征，支持分片写入和高效随机访问

核心类：
- H5FeatureWriter: 分块写入 HDF5 文件
- H5FeatureDataset: PyTorch Dataset，支持随机访问
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional


class H5FeatureWriter:
    """
    HDF5 特征写入器（仅 VLA action 特征，不含文本特征）
    
    存储结构：
    - identifiers/: task_ids, task_index, traj_index, timestep, progress
    - text/: instructions(变长字符串) - 只存指令文本，不存特征
    - action/: features [N, L, A, D_action]
    - metadata/: 配置信息
    
    使用方法：
        writer = H5FeatureWriter(path, layer_indices, action_dim)
        for batch in dataloader:
            writer.append(action_feats, identifiers, texts)
        writer.close()  # 重要：必须调用 close() 刷新缓冲区
    
    性能优化：
    - 内存缓冲：累积多个 batch 后一次性写入，减少 I/O 次数
    - 批量 resize：按 chunk_size 的倍数预分配空间，避免频繁 resize
    - 默认 lzf 压缩：比 gzip 快 5-10 倍，压缩比略低
    """
    
    def __init__(
        self,
        path: str,
        layer_indices: List[int],
        action_dim: int,
        action_chunk_size: int = 1,
        chunk_size: int = 10000,  # HDF5 分块大小
        compression: Optional[str] = None,  # None(最快)/lzf(快)/gzip(小)
        buffer_size: int = 10000,  # 内存缓冲大小（越大越快，但占用更多内存）
    ):
        """
        Args:
            path: HDF5 文件路径
            layer_indices: VLA 层索引列表，如 [2,4,6,8,10,12,14,16,18]
            action_dim: action 特征维度（通常是 hidden_size，如 2048）
            action_chunk_size: action chunk 长度
            chunk_size: HDF5 写入分块大小
            compression: 压缩方式 (None 最快不压缩；lzf 快；gzip 压缩比高但慢)
            buffer_size: 内存缓冲大小，累积多少样本后写入磁盘
        """
        self.path = path
        self.layer_indices = layer_indices
        self.num_layers = len(layer_indices)
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.chunk_size = chunk_size
        # 处理命令行传入的字符串 "none" 或 Python None
        self.compression = None if (compression is None or compression.lower() == "none") else compression
        self.buffer_size = buffer_size
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 创建 HDF5 文件
        self.h5f = h5py.File(path, 'w')
        self._create_datasets()
        self.current_idx = 0  # 已写入磁盘的样本数
        self.allocated_size = 0  # 已分配的空间大小
        
        # 内存缓冲区
        self._buffer = {
            'action_features': [],
            'task_ids': [],
            'task_index': [],
            'traj_index': [],
            'timestep': [],
            'progress': [],
            'instructions': [],
        }
        self._buffer_count = 0
        
    def _create_datasets(self):
        """创建可扩展的 HDF5 datasets"""
        # 使用 maxshape=(None, ...) 实现可扩展
        # 标识符 (int64) - 不压缩小数据，压缩反而慢
        self.h5f.create_dataset(
            'identifiers/task_ids', shape=(0,), maxshape=(None,),
            dtype='i8', chunks=(self.chunk_size,)
        )
        self.h5f.create_dataset(
            'identifiers/task_index', shape=(0,), maxshape=(None,),
            dtype='i8', chunks=(self.chunk_size,)
        )
        self.h5f.create_dataset(
            'identifiers/traj_index', shape=(0,), maxshape=(None,),
            dtype='i8', chunks=(self.chunk_size,)
        )
        self.h5f.create_dataset(
            'identifiers/timestep', shape=(0,), maxshape=(None,),
            dtype='i8', chunks=(self.chunk_size,)
        )
        self.h5f.create_dataset(
            'identifiers/progress', shape=(0, 2), maxshape=(None, 2),
            dtype='f4', chunks=(self.chunk_size, 2)
        )
        
        # 文本指令 (变长字符串) - 只存指令，不存特征
        dt = h5py.special_dtype(vlen=str)
        self.h5f.create_dataset(
            'text/instructions', shape=(0,), maxshape=(None,),
            dtype=dt, chunks=(self.chunk_size,)
        )
        
        # Action 特征 [N, L, A, D] - 这是主要数据，压缩有意义
        # L=层数, A=action_chunk_size, D=hidden_dim
        # chunk 大小优化：每个 chunk 约 5-10MB，便于高效随机访问
        bytes_per_sample = self.num_layers * self.action_chunk_size * self.action_dim * 2  # float16
        samples_per_chunk = max(1, min(100, 5 * 1024 * 1024 // bytes_per_sample))
        self.h5f.create_dataset(
            'action/features',
            shape=(0, self.num_layers, self.action_chunk_size, self.action_dim),
            maxshape=(None, self.num_layers, self.action_chunk_size, self.action_dim),
            dtype='f2',
            chunks=(samples_per_chunk, self.num_layers, self.action_chunk_size, self.action_dim),
            compression=self.compression  # 只对大数据压缩
        )
        
        # 元数据
        meta = self.h5f.create_group('metadata')
        meta.attrs['layer_indices'] = self.layer_indices
        meta.attrs['action_dim'] = self.action_dim
        meta.attrs['action_chunk_size'] = self.action_chunk_size
        meta.attrs['num_layers'] = self.num_layers
        
    def append(
        self,
        action_features: torch.Tensor,  # [B, L, A, D]
        task_ids: torch.Tensor,         # [B]
        task_index: torch.Tensor,       # [B]
        traj_index: torch.Tensor,       # [B]
        timestep: torch.Tensor,         # [B]
        progress: torch.Tensor,         # [B, 2]
        instructions: List[str],        # [B]
    ):
        """追加一个 batch 的数据到缓冲区（不含文本特征）"""
        # 添加到缓冲区（在 CPU 上，避免 GPU 内存占用）
        self._buffer['action_features'].append(action_features.cpu().half().numpy())
        self._buffer['task_ids'].append(task_ids.cpu().numpy())
        self._buffer['task_index'].append(task_index.cpu().numpy())
        self._buffer['traj_index'].append(traj_index.cpu().numpy())
        self._buffer['timestep'].append(timestep.cpu().numpy())
        self._buffer['progress'].append(progress.cpu().numpy())
        self._buffer['instructions'].extend(instructions)
        
        self._buffer_count += action_features.shape[0]
        
        # 缓冲区满时刷新到磁盘
        if self._buffer_count >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """将缓冲区数据刷新到磁盘"""
        if self._buffer_count == 0:
            return
        
        # 合并缓冲区数据
        action_np = np.concatenate(self._buffer['action_features'], axis=0)
        task_ids_np = np.concatenate(self._buffer['task_ids'], axis=0)
        task_index_np = np.concatenate(self._buffer['task_index'], axis=0)
        traj_index_np = np.concatenate(self._buffer['traj_index'], axis=0)
        timestep_np = np.concatenate(self._buffer['timestep'], axis=0)
        progress_np = np.concatenate(self._buffer['progress'], axis=0)
        instructions_np = np.array(self._buffer['instructions'], dtype=object)
        
        batch_size = action_np.shape[0]
        new_idx = self.current_idx + batch_size
        
        # 批量扩展空间（按 chunk_size 的倍数预分配，减少 resize 次数）
        if new_idx > self.allocated_size:
            # 预分配额外空间：至少 chunk_size，或当前需要的 2 倍
            new_allocated = max(
                self.allocated_size + self.chunk_size,
                new_idx + self.chunk_size,
                int(new_idx * 1.5)
            )
            self._resize_datasets(new_allocated)
            self.allocated_size = new_allocated
        
        # 写入数据（直接赋值，不需要 resize）
        self.h5f['identifiers/task_ids'][self.current_idx:new_idx] = task_ids_np
        self.h5f['identifiers/task_index'][self.current_idx:new_idx] = task_index_np
        self.h5f['identifiers/traj_index'][self.current_idx:new_idx] = traj_index_np
        self.h5f['identifiers/timestep'][self.current_idx:new_idx] = timestep_np
        self.h5f['identifiers/progress'][self.current_idx:new_idx] = progress_np
        self.h5f['text/instructions'][self.current_idx:new_idx] = instructions_np
        self.h5f['action/features'][self.current_idx:new_idx] = action_np
        
        self.current_idx = new_idx
        
        # 清空缓冲区
        self._buffer = {k: [] for k in self._buffer}
        self._buffer_count = 0
    
    def _resize_datasets(self, new_size: int):
        """批量调整所有 dataset 的大小"""
        for key in ['identifiers/task_ids', 'identifiers/task_index', 
                    'identifiers/traj_index', 'identifiers/timestep']:
            self.h5f[key].resize((new_size,))
        self.h5f['identifiers/progress'].resize((new_size, 2))
        self.h5f['text/instructions'].resize((new_size,))
        self.h5f['action/features'].resize(
            (new_size, self.num_layers, self.action_chunk_size, self.action_dim)
        )
        
    def close(self):
        """刷新缓冲区、截断多余空间并关闭文件"""
        # 刷新剩余缓冲区
        self._flush_buffer()
        
        # 截断到实际大小（移除预分配的多余空间）
        if self.current_idx < self.allocated_size:
            self._resize_datasets(self.current_idx)
        
        # 更新元数据
        self.h5f['metadata'].attrs['num_samples'] = self.current_idx
        self.h5f.close()
        print(f"✅ H5 特征已保存: {self.path} ({self.current_idx} samples)")


class H5FeatureDataset(Dataset):
    """
    HDF5 特征读取 Dataset
    
    支持：
    - 随机访问（高效）
    - 多分片读取（合并多个 rank 的文件）
    
    使用方法：
        dataset = H5FeatureDataset(['/path/rank0.h5', '/path/rank1.h5'])
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
    """
    
    def __init__(self, h5_paths: List[str]):
        """
        Args:
            h5_paths: HDF5 文件路径列表（支持多分片）
        """
        self.h5_paths = h5_paths
        self.h5_files = []
        self.cumulative_sizes = [0]
        
        # 打开所有文件，计算累计大小
        for path in h5_paths:
            h5f = h5py.File(path, 'r')
            self.h5_files.append(h5f)
            size = h5f['metadata'].attrs['num_samples']
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
        
        self.total_size = self.cumulative_sizes[-1]
        
        # 读取元数据（从第一个文件）
        meta = self.h5_files[0]['metadata']
        self.layer_indices = list(meta.attrs['layer_indices'])
        self.action_dim = meta.attrs['action_dim']
        
    def __len__(self):
        return self.total_size
    
    def _find_file_and_idx(self, idx: int):
        """找到 idx 对应的文件和文件内索引"""
        for i, (start, end) in enumerate(zip(self.cumulative_sizes[:-1], self.cumulative_sizes[1:])):
            if start <= idx < end:
                return self.h5_files[i], idx - start
        raise IndexError(f"Index {idx} out of range")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个样本（不含文本特征，需通过 task_index 从外部查表获取）"""
        h5f, local_idx = self._find_file_and_idx(idx)
        
        return {
            'action_features': torch.from_numpy(h5f['action/features'][local_idx].astype(np.float32)),
            'task_ids': torch.tensor(h5f['identifiers/task_ids'][local_idx], dtype=torch.long),
            'task_index': torch.tensor(h5f['identifiers/task_index'][local_idx], dtype=torch.long),
            'traj_index': torch.tensor(h5f['identifiers/traj_index'][local_idx], dtype=torch.long),
            'timestep': torch.tensor(h5f['identifiers/timestep'][local_idx], dtype=torch.long),
            'progress': torch.from_numpy(h5f['identifiers/progress'][local_idx]),
            'instruction': h5f['text/instructions'][local_idx],
        }
    
    def close(self):
        """关闭所有文件"""
        for h5f in self.h5_files:
            h5f.close()


def merge_h5_shards(shard_paths: List[str], output_path: str):
    """
    合并多个 HDF5 分片文件
    
    Args:
        shard_paths: 分片文件路径列表
        output_path: 输出文件路径
    """
    # 读取第一个分片获取元数据
    with h5py.File(shard_paths[0], 'r') as first:
        meta = dict(first['metadata'].attrs)
        layer_indices = list(meta['layer_indices'])
    
    # 计算总样本数
    total_samples = 0
    for path in shard_paths:
        with h5py.File(path, 'r') as f:
            total_samples += f['metadata'].attrs['num_samples']
    
    print(f"合并 {len(shard_paths)} 个分片，共 {total_samples} 样本")
    
    # 创建输出文件
    writer = H5FeatureWriter(
        output_path,
        layer_indices=layer_indices,
        action_dim=meta['action_dim'],
        action_chunk_size=meta['action_chunk_size'],
    )
    
    # 逐个复制
    for path in shard_paths:
        with h5py.File(path, 'r') as src:
            n = src['metadata'].attrs['num_samples']
            if n == 0:
                continue
            # 批量读取并写入
            batch_size = 10000
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                writer.append(
                    action_features=torch.from_numpy(src['action/features'][start:end]),
                    task_ids=torch.from_numpy(src['identifiers/task_ids'][start:end]),
                    task_index=torch.from_numpy(src['identifiers/task_index'][start:end]),
                    traj_index=torch.from_numpy(src['identifiers/traj_index'][start:end]),
                    timestep=torch.from_numpy(src['identifiers/timestep'][start:end]),
                    progress=torch.from_numpy(src['identifiers/progress'][start:end]),
                    instructions=list(src['text/instructions'][start:end]),
                )
    
    writer.close()
    print(f"✅ 合并完成: {output_path}")
