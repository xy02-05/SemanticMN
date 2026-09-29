"""
Alignment Discovery Module
用于验证VLA和VLM之间的天然对齐性

核心功能:
1. CKA分析 - 验证天然对齐程度
2. Alignment Probes - 训练简单对齐层验证跨域迁移
3. Evaluation - 检索准确率评测
"""

from .cka_utils import linear_cka, compute_cka_per_layer
from .action_pooler import ActionPooler
from .alignment_probes import AlignmentProbe, MultiLayerAlignmentProbe
from .evaluation import (
    compute_retrieval_topk,
    format_retrieval_results,
    save_retrieval_results,
    EvalFeatureCollector,
)

__all__ = [
    'linear_cka',
    'compute_cka_per_layer',
    'ActionPooler', 
    'AlignmentProbe',
    'MultiLayerAlignmentProbe',
    'compute_retrieval_topk',
    'format_retrieval_results',
    'save_retrieval_results',
    'EvalFeatureCollector',
]
