"""
Adaptive Action Ensembler — 复刻 CogACT (microsoft/CogACT) 风格的自适应动作集成。
来源：sim_cogact/adaptive_ensemble.py（MIT 协议）。SpatialVLA RSS 论文与社区实践
均参考此方案；与 Octo 风格 `exp(temp * k)` 时间衰减相比，CogACT 用 cosine 相似度
作为权重核——可以把"和最新预测明显不同（不同 mode）"的旧候选权重压低，
避免简单时序平均把决断（如 grasp / release 切换）平滑掉。

**算法**：
  1. 维护一个长度 H=action_chunk_size 的 deque，存放历史每次推理给出的 chunk。
  2. 每次新 chunk 进来后，把所有历史 chunk 中**对应当前 timestep**的那个 action
     抽取出来：最旧 chunk 已经走过 N-1 步，所以取 chunk[N-1]；
     最新 chunk 刚到，取 chunk[0]。这样 N 个候选都是"针对当前 t 的预测"。
  3. 以最新候选为 ref，算 N 个候选与 ref 的 cos similarity。
  4. weights = softmax(alpha * cos_sim)，alpha=0 等权，alpha 越大越偏向与 ref 相似的。
  5. 加权平均，得到当前 timestep 应执行的 action。

**接口语义**：
  - `ensemble_action(cur_chunk)`：每个 timestep 调用一次。
    输入：模型刚 predict 出的 chunk，shape [chunk_size, action_dim]
    输出：当前 timestep 的集成 action，shape [action_dim]

  - 当 ndim==1 时（即输入只是一个 action 而不是 chunk），退化为
    "用过去 N 个单 action 做 cos 加权平均"——CogACT 兼容，但 SpatialVLA 不会走这分支。
"""
from __future__ import annotations

from collections import deque

import numpy as np


class AdaptiveEnsembler:
    """CogACT 风格 Adaptive Action Ensembler (AAE)，与原仓库 sim_cogact/adaptive_ensemble.py 等价。"""

    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.1):
        """
        Args:
            pred_action_horizon: 模型一次推理输出的 future actions 数（SpatialVLA 训练默认 4）
            adaptive_ensemble_alpha: cos-similarity 权重温度。0=等权；正值越大越偏向与最新预测相似的候选。
                默认 0.1（CogACT 实战常用 0.05~0.5；可由 eval 入口 --action_ensemble_alpha 覆盖）。
        """
        self.pred_action_horizon = pred_action_horizon
        self.action_history: deque = deque(maxlen=pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self) -> None:
        """每个 episode 开始时调用，清空历史。"""
        self.action_history.clear()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        """
        集成当前 timestep 的 action。
        Args:
            cur_action: 模型刚 predict 出的 chunk [chunk_size, action_dim] 或单 action [action_dim]
        Returns:
            集成后的 action [action_dim]，可直接喂给 env.step()
        """
        # 入队最新 chunk；deque 自动 pop 最旧一项保证最大长度 = pred_action_horizon
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)

        if cur_action.ndim == 1:
            # 退化路径：history 里每项都是单 action，直接堆叠
            curr_act_preds = np.stack(self.action_history)
        else:
            # 主路径：每项是 chunk [H, action_dim]
            # 最旧 chunk 已走 N-1 步，对当前 t 的预测在 chunk 内位置 = N-1
            # 最新 chunk 刚加入，对当前 t 的预测在 chunk 内位置 = 0
            # 这里用 zip(range(N-1, -1, -1), history) 把每个 history 元素配对一个倒序索引
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # 以"最新预测"作为相似度比较的参考向量（curr_act_preds 中最后一行）
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds  # 含 ref 自己（cos=1，权重最大）

        # 计算每个候选与 ref 的余弦相似度
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # softmax over cos_sim：alpha 控制"偏向 ref"的强度
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()

        # 加权求和得到集成 action
        ensembled = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return ensembled
