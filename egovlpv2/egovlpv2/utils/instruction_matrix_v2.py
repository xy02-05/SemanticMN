# coding=utf-8
"""
指令矩阵加载器 V2（训练时使用，按 task_id 和 episode_id 索引）

关键设计：
1. 第一维索引：task_id
2. 第二维索引：episode_id
3. 采样时：随机 task_id 和 episode_id，直接索引获取
"""

import json
from pathlib import Path
import numpy as np
from typing import Tuple, Union, List


def load_json(filepath):
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class InstructionMatrixV2:
    """指令矩阵加载器 V2（按 task_id 和 episode_id 索引）"""
    
    def __init__(self):
        self.data = None
        # 核心数据结构
        self.seen_array = None   # int32 matrix
        self.unseen_array = None # int32 matrix
        self.texts_vocab = None  # object array (strings)
        
        # 映射表
        self.task_id_to_idx_seen = None 
        self.task_id_to_idx_unseen = None
        self.idx_to_task_id_seen = None   # 用于反查 task_id
        self.idx_to_task_id_unseen = None
    
    @classmethod
    def load(cls, filepath: Path):
        instance = cls()
        data = load_json(filepath)
        
        # --- 1. 构建词汇表 (Vocab) ---
        seen_raw = data["seen"]["texts"]
        unseen_raw = data["unseen"]["texts"]
        all_texts = np.array(seen_raw).flatten().tolist() + \
                    np.array(unseen_raw).flatten().tolist()
        unique_texts = sorted(list(set(all_texts)))
        
        instance.texts_vocab = np.array(unique_texts, dtype=object)
        text_to_int = {t: i for i, t in enumerate(unique_texts)}
        
        # --- 2. 文本转整数矩阵 ---
        def to_int_matrix(raw_matrix):
            vec_lookup = np.vectorize(lambda x: text_to_int.get(x, 0))
            return vec_lookup(np.array(raw_matrix)).astype(np.int32)

        instance.seen_array = to_int_matrix(seen_raw)
        instance.unseen_array = to_int_matrix(unseen_raw)
        
        # --- 3. 构建索引映射 ---
        def build_maps(task_id_list):
            t_ids = np.array(task_id_list)
            # ID -> Matrix Index
            id_to_idx = {tid: i for i, tid in enumerate(t_ids)}
            # Matrix Index -> ID (用于 return_indices 时反查)
            idx_to_id = t_ids 
            return id_to_idx, idx_to_id
            
        instance.task_id_to_idx_seen, instance.idx_to_task_id_seen = \
            build_maps(data["seen"]["task_ids"])
        instance.task_id_to_idx_unseen, instance.idx_to_task_id_unseen = \
            build_maps(data["unseen"]["task_ids"])
            
        print(f"✅ V3 Loaded: Mem={instance.seen_array.nbytes/1024:.1f}KB, Vocab={len(instance.texts_vocab)}")
        return instance

    def get_instruction(self, task_id: int, episode_id: int, split: str = "seen") -> str:
        """获取单条指令"""
        if split == "seen":
            idx = self.task_id_to_idx_seen[task_id]
            token_id = self.seen_array[idx, episode_id]
        else:
            idx = self.task_id_to_idx_unseen[task_id]
            token_id = self.unseen_array[idx, episode_id]
        return self.texts_vocab[token_id]

    def sample_negatives(
        self,
        anchor_task_ids: Union[int, List[int], np.ndarray],
        num_samples: int,
        split: str = "seen",
        return_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        统一采样接口 (替代了原有的三个冗余函数)
        
        Args:
            anchor_task_ids: 单个 ID (int) 或 Batch IDs (list/array)
            num_samples: 每个 anchor 需要采样的负样本数
            split: "seen" / "unseen"
            return_indices: 是否返回对应的 task_id 和 episode_id (用于调试/分析)
            
        Returns:
            如果 return_indices=False: 
                negative_texts (np.ndarray)
            如果 return_indices=True: 
                (negative_texts, sampled_task_ids, sampled_episode_ids)
        """
        # 1. 统一输入格式为 Array
        anchors = np.atleast_1d(anchor_task_ids)
        B = len(anchors)
        
        # 2. 获取资源
        if split == "seen":
            matrix = self.seen_array
            map_to_idx = self.task_id_to_idx_seen
            map_to_id_arr = self.idx_to_task_id_seen
        else:
            matrix = self.unseen_array
            map_to_idx = self.task_id_to_idx_unseen
            map_to_id_arr = self.idx_to_task_id_unseen
            
        num_tasks, num_episodes = matrix.shape

        # 3. ID -> Row Index (列表推导比 np.vectorize 处理 dict 更快)
        anchor_row_indices = np.array([map_to_idx[tid] for tid in anchors])
        
        # 4. 极速采样算法 (Modulo Offset)
        # 偏移量 k ∈ [1, N-1]，保证不采到自己
        offsets = np.random.randint(1, num_tasks, size=(B, num_samples))
        sampled_row_indices = (anchor_row_indices[:, None] + offsets) % num_tasks
        
        # 随机 Episode
        sampled_col_indices = np.random.randint(0, num_episodes, size=(B, num_samples))
        
        # 5. 获取结果
        # 获取 Token ID -> 转回 String
        token_indices = matrix[sampled_row_indices, sampled_col_indices]
        texts = self.texts_vocab[token_indices]
        
        # 6. 处理返回格式
        # 如果输入是标量(单个int)，输出去掉 Batch 维度
        if np.isscalar(anchor_task_ids):
            texts = texts.squeeze(0) # [1, K] -> [K]
            sampled_row_indices = sampled_row_indices.squeeze(0)
            sampled_col_indices = sampled_col_indices.squeeze(0)

        if not return_indices:
            return texts
        
        # 如果需要 ID，反查回去
        sampled_task_ids = map_to_id_arr[sampled_row_indices]
        return texts, sampled_task_ids, sampled_col_indices