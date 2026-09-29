"""
model_embedding.py

预计算Embedding模型封装，用于fine-grained alignment训练
支持加载text-embedding-3-large等预计算的文本特征

核心功能：
1. 加载预计算的embedding特征（.npz格式）
2. 加载文本索引（.json格式）
3. 提供compute_text方法，通过文本查找对应的预计算特征
4. 与EgoVLPv2/EgoHOD/Qwen3保持一致的接口设计

使用示例：
    model = EmbeddingModel(
        embeddings_path="/path/to/robotwin_embeddings_1024.npz",
        index_path="/path/to/robotwin_index_1024.json",
        device="cuda:0"
    )
    text_embeds = model.compute_text({"text": ["sample text"]})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict

class EmbeddingModel(nn.Module):
    def __init__(
        self,
        embeddings_path: str,
        index_path: str,
        device: str = "cuda:0", # ⚠️注意：外部调用时一定要传对 device！
        dtype: torch.dtype = torch.float32,
        normalize_embeddings: bool = True,
        preload_to_gpu: bool = True,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        print(f"🔧 Loading Embeddings on {device}...")
        
        # 1. Load Numpy
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings_np = data['embeddings']
        
        # 2. Convert to Tensor (CPU first)
        embeddings_tensor = torch.from_numpy(embeddings_np.astype(np.float32))
        if normalize_embeddings:
            embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=-1)

        self.embed_layer = nn.Embedding.from_pretrained(
            embeddings_tensor, 
            freeze=True 
        )
        
        # 4. 显式移动到指定 GPU，防止挤占 GPU 0
        # ⚠️ 关键：.to() 不是 inplace 操作，需要赋值回去或使用 inplace 版本
        if preload_to_gpu:
            self.embed_layer = self.embed_layer.to(device=self.device, dtype=self.dtype)
        
        # 5. 显式禁用梯度（freeze=True 应该已经处理了，但这里双保险）
        for param in self.embed_layer.parameters():
            param.requires_grad = False
        self.eval()  # 设置为 eval 模式
        
        # 6. 设置 output_dim 属性（兼容旧代码）
        self.output_dim = self.embed_layer.embedding_dim
        
        # Load Index - 构建多种形式的文本索引，提高匹配成功率
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.text_to_idx = {}
        for item in index_data:
            original_text = item['text']
            idx = item['idx']
            
            # 存储多种形式的文本作为键，提高匹配率
            self.text_to_idx[original_text] = idx                    # 原始文本
            self.text_to_idx[original_text.strip()] = idx            # 去除首尾空格
            self.text_to_idx[original_text.lower()] = idx            # 小写
            self.text_to_idx[original_text.strip().lower()] = idx    # 去除空格+小写

    def compute_text(self, text_data: Dict) -> torch.Tensor:
        texts = text_data['text']
        
        # 1. 查找索引 (CPU 操作) - 字典已包含多种文本变体
        indices = []
        for text in texts:
            idx = self.text_to_idx.get(text)  # 先尝试精确匹配
            if idx is None:
                idx = self._find_text_index(text)  # 尝试其他变体
            indices.append(idx)
        
        current_device = self.embed_layer.weight.device
        
        # 2. 转 Tensor 并移到当前设备
        indices_tensor = torch.tensor(indices, device=current_device, dtype=torch.long)
        
        # 3. 查表 (nn.Embedding 内部处理极其高效)
        # .detach() 是双重保险，确保这里生成的 Tensor 是全新的叶子节点
        with torch.no_grad():
            return self.embed_layer(indices_tensor).detach()
    
    def _find_text_index(self, text: str) -> int:
        """
        查找文本对应的embedding索引
        
        由于初始化时已经存储了多种形式的文本（原始、小写、strip等），
        这里只需要尝试几种常见变体即可
        
        Args:
            text: 输入文本
        
        Returns:
            idx: embedding索引
        """
        # 尝试几种常见的文本变体（字典中已预存储）
        for variant in [text, text.strip(), text.lower(), text.strip().lower()]:
            idx = self.text_to_idx.get(variant)
            if idx is not None:
                return idx
        
        # 如果都找不到，返回索引0（避免训练中断）
        print(f"⚠️ Warning: Text not found '{text[:50]}...', using index 0")
        return 0
    
    def forward(self, text_data: Dict) -> torch.Tensor:
        """前向传播（调用compute_text）"""
        return self.compute_text(text_data)


def create_embedding_model(
    embeddings_path: str,
    index_path: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
    normalize_embeddings: bool = True,
    preload_to_gpu: bool = True,  # 新增：是否预加载到GPU（默认True）
) -> EmbeddingModel:
    """
    创建EmbeddingModel实例的工厂函数
    
    Args:
        embeddings_path: 嵌入特征文件路径
        index_path: 文本索引文件路径
        device: 设备
        dtype: 数据类型
        normalize_embeddings: 是否L2归一化
        preload_to_gpu: 是否预加载到GPU（推荐True，显存使用更稳定）
    
    Returns:
        EmbeddingModel实例
    """
    return EmbeddingModel(
        embeddings_path=embeddings_path,
        index_path=index_path,
        device=device,
        dtype=dtype,
        normalize_embeddings=normalize_embeddings,
        preload_to_gpu=preload_to_gpu
    )

