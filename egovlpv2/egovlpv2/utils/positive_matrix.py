# coding=utf-8
"""
正样本矩阵加载和采样

功能：
1. 加载预处理的 positive_matrix.json
2. 支持单个和批量正样本采样
3. 使用 NumPy 批量索引，避免循环
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union



class PositiveMatrix:
    """正样本矩阵加载和采样器（训练用）"""
    
    def __init__(self):
        """初始化"""
        # 核心数据
        self.positives_matrix = None   # int32, shape: (N_inst, N_pos)
        self.texts_vocab = None        # object array, shape: (N_unique,)
        
        # 索引
        self.instructions = None       # List[str] - 原始指令列表
        self.inst_to_idx = None        # Dict[str, int]
        self.task_ids = None           # np.array[int]
        self.episode_ids = None        # np.array[int]
        self.task_to_inst_indices = None
        
    @classmethod
    def load(cls, filepath: Path):
        """
        从文件加载正样本矩阵
        
        Args:
            filepath: positive_matrix.json 的路径
            
        Returns:
            PositiveMatrix 实例
        """
        instance = cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # --- 1. 构建 Vocab ---
        raw_positives = data["data"]["positives"]  # List[List[str]]
        all_texts = []
        for row in raw_positives:
            all_texts.extend(row)
        unique_texts = sorted(set(all_texts))
        
        instance.texts_vocab = np.array(unique_texts, dtype=object)
        text_to_int = {t: i for i, t in enumerate(unique_texts)}
        
        # --- 2. 转为整数矩阵 ---
        def to_int_matrix(raw_matrix):
            return np.array(
                [[text_to_int[t] for t in row] for row in raw_matrix],
                dtype=np.int32
            )
        
        instance.positives_matrix = to_int_matrix(raw_positives)
        
        # --- 3. 其他索引 ---
        instance.instructions = data["data"]["instructions"]
        instance.inst_to_idx = data["index"]["instruction_to_idx"]
        instance.task_ids = np.array(data["data"]["task_ids"], dtype=np.int32)
        instance.episode_ids = np.array(data["data"]["episode_ids"], dtype=np.int32)
        instance.task_to_inst_indices = {
            int(k): v for k, v in data["index"]["task_to_instruction_indices"].items()
        }

        # 内存统计
        matrix_mem = instance.positives_matrix.nbytes / 1024
        vocab_mem = sum(len(s.encode('utf-8')) for s in unique_texts) / 1024
        print(f"✅ PositiveMatrixV2 Loaded:")
        print(f"   Matrix: {instance.positives_matrix.shape}, {matrix_mem:.1f} KB")
        print(f"   Vocab: {len(unique_texts)} unique texts, ~{vocab_mem:.1f} KB")
        
        return instance
    
    def sample_positives(
        self,
        anchor_indices: Union[int, List[int], np.ndarray],
        num_samples: int,
        return_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        统一采样接口（全向量化）
        
        Args:
            anchor_indices: 锚点的行索引（单个 int 或 batch array）
            num_samples: 每个锚点采样数量
            return_indices: 是否返回采样的列索引
            
        Returns:
            texts: shape (B, num_samples) 或 (num_samples,) if 单个输入
            [可选] col_indices: 采样的列索引
        """
        anchors = np.atleast_1d(anchor_indices)
        B = len(anchors)
        num_positives = self.positives_matrix.shape[1]
        
        # 向量化随机采样（有放回，速度最快）
        col_indices = np.random.randint(0, num_positives, size=(B, num_samples))
        
        # 批量索引获取 token_ids
        token_ids = self.positives_matrix[anchors[:, None], col_indices]
        
        # Token ID -> 文本
        texts = self.texts_vocab[token_ids]
        
        # 处理单个输入的情况
        if np.isscalar(anchor_indices):
            texts = texts.squeeze(0)
            col_indices = col_indices.squeeze(0)
        
        if return_indices:
            return texts, col_indices
        return texts
    
    def sample_by_instructions(
        self,
        anchor_instructions: List[str],
        num_samples: int
    ) -> np.ndarray:
        """
        通过指令文本采样（兼容旧接口）
        """
        indices = np.array([self.inst_to_idx[inst] for inst in anchor_instructions])
        return self.sample_positives(indices, num_samples)

    def batch_sample_positives(
        self,
        anchor_instructions: List[str],
        num_samples: int
    ) -> List[List[str]]:
        """
        批量采样正样本（兼容旧版本接口）
        
        注意：统一使用有放回采样（符合用户要求："只要是pos里的就行"）
        
        Args:
            anchor_instructions: [B] 锚点指令列表
            num_samples: 每个锚点采样的正样本数量
            
        Returns:
            List[List[str]]: [B, num_samples] 正样本列表
        """
        # 获取指令索引
        indices = np.array([self.inst_to_idx[inst] for inst in anchor_instructions])
        
        # 调用统一采样接口
        texts_array = self.sample_positives(indices, num_samples)
        
        # 转换为list格式（保持旧接口兼容）
        return [row.tolist() for row in texts_array]
    
    def get_task_id(self, instruction: str) -> int:
        return int(self.task_ids[self.inst_to_idx[instruction]])
    
    def get_episode_id(self, instruction: str) -> int:
        return int(self.episode_ids[self.inst_to_idx[instruction]])