"""
fg_alignment_model.py

Fine-Grained Alignment Model - Multi-Mode Alignment Framework

支持模式：
1. as2ts (Action Sequence to Text Sentence): 全局对齐
   - global_action_features + global_text_features
2. as2tt (Action Sequence to Text Token): 细粒度对齐（XClip风格）
   - global_action_features + local_text_features (tokens)
   - 用softmax加权text tokens得到最终相似度
3. at2tt (Action Token to Text Token): FILIP风格细粒度对齐
   - local_action_features + local_text_features (tokens)

Key Interfaces:
- Input: OpenVLA features [B, L, A, D1], EgoVLPv2 features [B, D2]
- Output: multi-mode losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from egovlpv2.base import BaseModel
from egovlpv2.model.feature_bank import DualFeatureBank, DualTokenFeatureBank, AT2TTTokenFeatureBank
from egovlpv2.model.utils import AlignmentUtilsMixin
from egovlpv2.model.loss import ContrastiveLossMixin, create_alignment_masks
from egovlpv2.model.action_pooling import ActionPooler, create_action_pooler
from egovlpv2.model.disentangle import DisentangleHead, compute_disentangle_losses
from egovlpv2.model.info_theory import log_information_plane_step
from egovlpv2.model.vicreg_loss import VICRegLoss


class AlignmentModel(AlignmentUtilsMixin, ContrastiveLossMixin, BaseModel):
    """
    Alignment model for OpenVLA-EgoVLPv2 feature alignment.
    
    架构设计：
    - forward: 只做特征提取（global/local），不做投影
    - as2ts/as2tt/at2tt/as2vs: 各自有独立的投影层，在内部完成投影和loss计算
    - 只支持simple模式：batch内对角线为正样本
    """
    
    def __init__(
        self,
        egovlpv2_dim: int = 4096,      # text sentence embedding维度
        openvla_dim: int = 4096,        # action token维度
        projection_dim: int = 512,      # 投影后的公共维度
        token_egovlpv2_dim: int = 768,  # text token维度
        video_dim: int = None,          # video embedding维度，默认None则回退到egovlpv2_dim（向后兼容）
        temperature: float = 0.07,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        num_selected_layers: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        mode_config: dict = None,
        use_feature_bank: bool = False,
        feature_bank_size: int = 1024,
        loss_type: str = 'infonce',
        sigmoid_bias: float = 0.0,
        learnable_temperature: bool = False,
        learnable_bias: bool = False,
        alignment_mode: str = 'diagonal',  # 对齐模式: 'diagonal', 'task_id', 'task_index'
        vla_mode: str = 'spatialvla',  # VLA类型: 'openpi'去掉第一个token, 'spatialvla'去掉最后一个token
        min_valid_ratio: float = 0.0,  # 最小有效样本比例阈值，低于此比例的anchor的loss被mask掉
        action_pool_mode: str = 'mean',  # action池化模式: 'mean', 'mean_mlp', 'learnable_query', 'target_attention'
        action_pool_config: dict = None,  # action池化超参数配置
        use_disentangle: bool = False,  # 是否启用共享/特有表征分离
        disentangle_config: dict = None,  # 分离模块配置 {hidden_dim, diff_weight, recon_weight}
        text_proj_config: dict = None,    # text投影层配置 {use_fc_projection, proj_num_layers, proj_hidden_dim}
        action_proj_config: dict = None,  # action投影层配置，字段同上
        use_distributed_negatives: bool = False,  # 多卡负样本 merge 开关：True 时所有对齐 loss 通过allgather跨rank合并特征
        anchor_bank_config: dict = None,  # AnchorBank (LIBERO 40 task × paraphrase + counterfactual) 配置
        vicreg_config: dict = None,  # VICReg配置 {sim_coeff, std_coeff, cov_coeff, variance_target, eps}
    ):
        super().__init__()
        # 多卡负样本合并开关；运行时依赖 forward 传入的 allgather_fn / n_gpu / args
        self.use_distributed_negatives = use_distributed_negatives
        # 占位：在 forward 入口被设置，用于 _compute_*_loss 中调用
        self._allgather_fn = None
        self._n_gpu = 1
        self._dist_args = None
        
        # 基础维度配置
        self.egovlpv2_dim = egovlpv2_dim
        self.openvla_dim = openvla_dim
        self.projection_dim = projection_dim
        self.token_egovlpv2_dim = token_egovlpv2_dim
        # video_dim: as2vs模式中video特征的输入维度，默认回退到egovlpv2_dim（向后兼容）
        self.video_dim = video_dim if video_dim is not None else egovlpv2_dim
        self.use_feature_bank = use_feature_bank
        self.num_selected_layers = num_selected_layers
        self.loss_type = loss_type
        self.learnable_temperature = learnable_temperature
        self.learnable_bias = learnable_bias
        self.vicreg_config = vicreg_config or {}
        
        # text/action 投影层独立配置（各自必须包含 use_fc_projection, proj_num_layers, proj_hidden_dim）
        _default_proj = {'use_fc_projection': False, 'proj_num_layers': 2, 'proj_hidden_dim': projection_dim}
        _text_cfg = {**_default_proj, **(text_proj_config or {})}
        _action_cfg = {**_default_proj, **(action_proj_config or {})}
        self._text_proj_kw = {
            'use_fc': _text_cfg['use_fc_projection'],
            'n_proj_layers': _text_cfg['proj_num_layers'],
            'hidden_dim': _text_cfg['proj_hidden_dim'],
        }
        self._action_proj_kw = {
            'use_fc': _action_cfg['use_fc_projection'],
            'n_proj_layers': _action_cfg['proj_num_layers'],
            'hidden_dim': _action_cfg['proj_hidden_dim'],
        }
        print(f"📐 Text投影层配置: {_text_cfg}")
        print(f"📐 Action投影层配置: {_action_cfg}")
        
        # 对齐模式: 'diagonal'=对角线正样本, 'task_id'=相同task_id正样本, 'task_index'=相同task_index正样本
        self.alignment_mode = alignment_mode
        # VLA类型: 'openpi'去掉第一个token, 'spatialvla'去掉最后一个token
        self.vla_mode = vla_mode
        # 最小有效样本比例阈值：如果某anchor的有效配对数/总样本数 < 此阈值，则该anchor的loss不计入
        # 但该样本仍可作为其他anchor的正/负样本
        self.min_valid_ratio = min_valid_ratio
        
        # ============ Action Token 池化配置 ============
        # 每个需要 global_action_features 的模式独立创建 pooler（与投影层设计一致）
        # at2tt 直接用 local_action_features，不需要池化
        self.action_pool_mode = action_pool_mode
        self.action_pool_config = action_pool_config
        
        # ============ 共享/特有表征分离配置（对齐DSN原始实现） ============
        # DSN: DiffLoss(private, shared) + MSE(recon, orig) + SIMSE(recon, orig)
        # position: "pre_pool"  — token级分解，shared tokens送入pooler（推荐）
        #           "post_pool" — pooler之后在全局向量上分解
        self.use_disentangle = use_disentangle
        self.disentangle_config = disentangle_config or {}
        self.disentangle_diff_weight = self.disentangle_config.get('diff_weight', 0.075)
        self.disentangle_recon_weight = self.disentangle_config.get('recon_weight', 0.01)
        self.disentangle_position = self.disentangle_config.get('position', 'post_pool')
        # diff_mode: "cosine"=cos²（推荐）, "cosine_hinge"=DLF风格hinge, "dsn"=原始DiffLoss
        self.disentangle_diff_mode = self.disentangle_config.get('diff_mode', 'cosine')
        self.detach_recon_target = self.disentangle_config.get('detach_recon_target', False)
        # 对齐路径固定走 'shared'：DSN 后的 z_shared 与 text 镜像；
        # token级对齐（at2tt/at2tt_soft）在 pre_pool 模式下用 shared tokens（来自 DSN 的 z_s_flat）。
        # （旧版的 'raw' / 'both' 配置项已删除）
        
        if use_disentangle:
            _dis_hidden = self.disentangle_config.get('hidden_dim', openvla_dim)
            _dis_dropout = self.disentangle_config.get('dropout', dropout)
            _dis_encoder = self.disentangle_config.get('encoder_type', 'mlp')
            self.as2ts_disentangle = DisentangleHead(
                input_dim=openvla_dim, hidden_dim=_dis_hidden,
                dropout=_dis_dropout, layer_norm_eps=layer_norm_eps,
                encoder_type=_dis_encoder,
            )
            print(f"🔀 表征分离已启用: position={self.disentangle_position}, "
                  f"diff_mode={self.disentangle_diff_mode}, encoder={_dis_encoder}, "
                  f"hidden={_dis_hidden}, diff_w={self.disentangle_diff_weight}, "
                  f"recon_w={self.disentangle_recon_weight}, "
                  f"detach_recon_target={self.detach_recon_target}")
        else:
            self.as2ts_disentangle = None
        
        # 温度参数（支持可学习）
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
            self.register_buffer('_temperature_buffer', None)  # placeholder
        else:
            self.log_temperature = None
            # 使用register_buffer预创建，避免每次调用_get_temperature都创建新Tensor
            self.register_buffer('_temperature_buffer', torch.tensor(temperature, dtype=torch.float32))

        # sigmoid_bias 需要参与 model.state_dict / to(device)；
        # 当 learnable_bias=False 时保持原来的“固定常数”行为，当为 True 时交给优化器学习。
        if learnable_bias:
            self.sigmoid_bias = nn.Parameter(torch.tensor(sigmoid_bias, dtype=torch.float32))
        else:
            self.register_buffer('sigmoid_bias', torch.tensor(sigmoid_bias, dtype=torch.float32))
        
        # 模式配置
        if mode_config is None:
            mode_config = {
                'as2ts': {'enabled': True, 'weight': 1.0},
                'as2tt': {'enabled': False, 'weight': 0.0, 'temperature': 0.01},
                'at2tt': {'enabled': False, 'weight': 0.0},  # FILIP风格细粒度对齐
                # at2tt_soft = at2tt 的 softmax 软化版（DRL-WTI / 类 X-CLIP frame-word AOSM）
                # attn_temperature: token importance 权重的softmax温度（与as2tt的'temperature'语义对齐）
                'at2tt_soft': {'enabled': False, 'weight': 0.0, 'attn_temperature': 0.01},
                'as2vs': {'enabled': False, 'weight': 1.0, 'soft_label_temp': 0.1,
                          'use_softmax_label': False,  # True=softmax归一化, False=线性归一化（推荐）
                          'sim_threshold': 0.0,        # 相似度阈值，低于此值的视频作为负样本
                          'sim_base': 0.0},            # 归一化基准：w=(sim-base)/(1-base)
                # as2cv: Action Sequence to Chunk Video（双粒度，trajectory+chunk 共用 sim 矩阵）
                'as2cv': {'enabled': False, 'weight': 1.0,
                          'traj_weight': 1.0, 'chunk_weight': 1.0},
            }
        self.mode_config = mode_config
        self.active_modes = [m for m, cfg in self.mode_config.items() if cfg.get('enabled', False)]

        if self.loss_type == 'vicreg':
            if self.active_modes != ['as2ts']:
                raise ValueError("VICReg目前只支持单独启用as2ts句子级对齐")
            if self.use_feature_bank:
                raise ValueError("VICReg不使用负样本，必须关闭feature bank")
            if self.use_distributed_negatives:
                raise ValueError(
                    "VICReg不使用分布式负样本，请关闭use_distributed_negatives；"
                    "跨卡统计由vicreg_config.gather_distributed控制"
                )
            if (anchor_bank_config or {}).get('enabled', False):
                raise ValueError("VICReg不使用正负anchor，必须关闭anchor bank")
            if self.learnable_temperature or self.learnable_bias:
                raise ValueError("VICReg不使用temperature或sigmoid bias")
            self.vicreg_loss = VICRegLoss(
                sim_coeff=self.vicreg_config.get('sim_coeff', 25.0),
                std_coeff=self.vicreg_config.get('std_coeff', 25.0),
                cov_coeff=self.vicreg_config.get('cov_coeff', 1.0),
                variance_target=self.vicreg_config.get('variance_target', 1.0),
                eps=self.vicreg_config.get('eps', 1e-4),
            )
            self.vicreg_gather_distributed = self.vicreg_config.get(
                'gather_distributed', True
            )
        else:
            self.vicreg_loss = None
            self.vicreg_gather_distributed = False
        
        print(f"🎯 多模式对齐配置: {self.active_modes}")
        
        # ============ as2ts 投影层 + 池化 ============
        # global_action [openvla_dim] -> [projection_dim]
        # global_text [egovlpv2_dim] -> [projection_dim]
        self.as2ts_action_pooler = create_action_pooler(
            mode=action_pool_mode, action_dim=openvla_dim,
            target_dim=egovlpv2_dim, config=action_pool_config,
        )
        self.as2ts_action_proj_per_layer = self._create_proj_layers(
            openvla_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
            **self._action_proj_kw,
        )
        self.as2ts_text_proj_per_layer = self._create_proj_layers(
            egovlpv2_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
            **self._text_proj_kw,
        )
        
        # ============ as2tt 投影层 + 池化 ============
        if 'as2tt' in self.active_modes:
            self.as2tt_action_pooler = create_action_pooler(
                mode=action_pool_mode, action_dim=openvla_dim,
                target_dim=egovlpv2_dim, config=action_pool_config,
            )
            # global_action [openvla_dim] -> [projection_dim]
            self.as2tt_action_proj_per_layer = self._create_proj_layers(
                openvla_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._action_proj_kw,
            )
            # local_text tokens [token_egovlpv2_dim] -> [projection_dim]
            self.as2tt_text_proj_per_layer = self._create_proj_layers(
                token_egovlpv2_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._text_proj_kw,
            )
        else:
            self.as2tt_action_pooler = None
            self.as2tt_action_proj_per_layer = None
            self.as2tt_text_proj_per_layer = None
        
        
        # ============ as2vs 投影层（Action Seq to Video Sampled） ============
        # Mirror Neuron 设计：复用 as2ts 的 DSN + pooler 得到共享 pooled_action，
        # 仅投影头独立——video 和 text 是同一个 z_shared 的两面镜子。
        if 'as2vs' in self.active_modes:
            # action 投影：pooled_action [openvla_dim] → [projection_dim]
            self.as2vs_action_proj_per_layer = self._create_proj_layers(
                openvla_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._action_proj_kw,
            )
            # video 投影：video_dim 可能与 text 的 egovlpv2_dim 不同
            self.as2vs_video_proj_per_layer = self._create_proj_layers(
                self.video_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._text_proj_kw,
            )
            self.as2vs_soft_label_temp = mode_config['as2vs'].get('soft_label_temp', 0.1)
            self.as2vs_use_softmax_label = mode_config['as2vs'].get('use_softmax_label', False)
            self.as2vs_sim_threshold = mode_config['as2vs'].get('sim_threshold', 0.0)
            # 归一化基准：w_norm = (sim - sim_base) / (1 - sim_base)，sim_base < sim_threshold 时等效无归一化
            self.as2vs_sim_base = mode_config['as2vs'].get('sim_base', 0.0)
        else:
            self.as2vs_action_proj_per_layer = None
            self.as2vs_video_proj_per_layer = None
        
        # ============ as2atomic 投影层（Action Seq to Atomic Text，chunk级细粒度对齐） ============
        # 设计要点（与用户方案一致）：
        #   1) 共享 as2ts 的 DSN + pooler，得到同一个 pooled_action [B, D_action]
        #   2) 仅投影头独立：as2atomic 有自己的 action_proj 和 text_proj MLP
        # 这样 as2ts 学全局task锚点，as2atomic 学chunk级子动作锚点，但共用底层"shared表征"。
        if 'as2atomic' in self.active_modes:
            self.as2atomic_action_proj_per_layer = self._create_proj_layers(
                openvla_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._action_proj_kw,
            )
            self.as2atomic_text_proj_per_layer = self._create_proj_layers(
                egovlpv2_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._text_proj_kw,
            )
        else:
            self.as2atomic_action_proj_per_layer = None
            self.as2atomic_text_proj_per_layer = None

        # ============ as2cv 投影层（Action Sequence to Chunk Video） ============
        # 设计要点：
        #   1) 共享 as2ts 的 pooled_action（不再单独 pool/DSN，与 as2vs/as2atomic 一致）
        #   2) action / video 各有独立投影头
        #   3) 同一个 sim 矩阵跑两次 mask：trajectory(task_id) + chunk(diagonal)
        if 'as2cv' in self.active_modes:
            self.as2cv_action_proj_per_layer = self._create_proj_layers(
                openvla_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._action_proj_kw,
            )
            self.as2cv_video_proj_per_layer = self._create_proj_layers(
                self.video_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._text_proj_kw,
            )
            self.as2cv_traj_weight = mode_config['as2cv'].get('traj_weight', 1.0)
            self.as2cv_chunk_weight = mode_config['as2cv'].get('chunk_weight', 1.0)
        else:
            self.as2cv_action_proj_per_layer = None
            self.as2cv_video_proj_per_layer = None
        
        # ============ at2tt / at2tt_soft 投影层（FILIP / WTI 风格） ============
        # local_action tokens [openvla_dim] -> [projection_dim]
        # local_text tokens [token_egovlpv2_dim] -> [projection_dim]
        # at2tt 与 at2tt_soft 共享同一套投影头和feature bank，仅token聚合方式不同
        if 'at2tt' in self.active_modes or 'at2tt_soft' in self.active_modes:
            self.at2tt_action_proj_per_layer = self._create_proj_layers(
                openvla_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._action_proj_kw,
            )
            self.at2tt_text_proj_per_layer = self._create_proj_layers(
                token_egovlpv2_dim, projection_dim, num_selected_layers, dropout, layer_norm_eps,
                **self._text_proj_kw,
            )
        else:
            self.at2tt_action_proj_per_layer = None
            self.at2tt_text_proj_per_layer = None
        
        # ============ Feature Bank ============
        self.feature_bank_size = feature_bank_size
        
        if self.use_feature_bank:
            # 简化版Feature Bank：每个optimizer step后由训练脚本调用reset()
            self.feature_banks = nn.ModuleList([
                DualFeatureBank(
                    bank_size=feature_bank_size,
                    feature_dim1=projection_dim,
                    feature_dim2=projection_dim,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                for _ in range(num_selected_layers)
            ])
            self.token_feature_banks = None
            self.token_bank_max_tokens = None
            # at2tt专用的双token feature bank（延迟初始化）
            self.at2tt_feature_banks = None
            self.at2tt_bank_params = None  # (max_action_tokens, max_text_tokens)
        else:
            self.feature_banks = None
            self.token_feature_banks = None
            self.at2tt_feature_banks = None
        
        self.apply(self._init_weights)
        self.to(dtype=dtype)
        self._keep_scalar_trainables_fp32()

        # 保存模型的目标dtype，用于外部正确配置autocast
        self.model_dtype = dtype

        # ============ Anchor Bank: LIBERO 40 task × (paraphrase + counterfactual) ============
        # 离线编码 (Qwen3-VL-Embedding-8B) → libero_anchor_qwen3vl_features.npz
        # 训练时 lookup(task_index) → 给 a2t 方向 sim matrix 多接 1+K_p (positive) + K_neg (negative) 列,
        # 走同一份 baseline sigmoid/InfoNCE, 不破坏 loss_type 选择.
        self.anchor_bank_config = anchor_bank_config or {}
        self.use_anchor_bank = self.anchor_bank_config.get('enabled', False)
        if self.use_anchor_bank:
            from egovlpv2.utils.anchor_bank import AnchorBank
            bank_path = self.anchor_bank_config['path']
            prefer_ood = self.anchor_bank_config.get('prefer_ood', True)
            self.anchor_bank = AnchorBank(bank_path, prefer_ood=prefer_ood)
            self.anchor_K_neg_sample = int(self.anchor_bank_config.get('K_neg_sample', 10))
            print(f"⚓ AnchorBank enabled: {self.anchor_bank} | K_neg_sample={self.anchor_K_neg_sample}")
        else:
            self.anchor_bank = None

        print("=" * 50)
        print(f"AlignmentModel: proj_dim={projection_dim}, loss={loss_type}, pool={action_pool_mode}")
        print("=" * 50)

    def _keep_scalar_trainables_fp32(self):
        """
        将温度和bias这两个标量固定为float32。

        原因：
        1. 整个模型可能会统一 `.to(dtype=torch.bfloat16)`。
        2. 但这两个标量如果也被转成bf16/fp16，学习会不稳定，容易看起来像“不更新”。
        3. 这里只修正这两个标量，不影响其它模块原本的dtype行为。
        """
        if self.log_temperature is not None:
            self.log_temperature.data = self.log_temperature.data.to(dtype=torch.float32)
        if self._temperature_buffer is not None:
            self._temperature_buffer.data = self._temperature_buffer.data.to(dtype=torch.float32)
        self.sigmoid_bias.data = self.sigmoid_bias.data.to(dtype=torch.float32)
    
    def _create_proj_layers(self, in_dim, out_dim, num_layers, dropout, layer_norm_eps,
                            use_fc, n_proj_layers, hidden_dim):
        """
        创建投影层（复用，减少代码重复）
        
        - use_fc=True: 单层FC（忽略n_proj_layers）
        - n_proj_layers=1: in_dim -> out_dim（单层Linear）
        - n_proj_layers=2: in_dim -> hidden_dim -> out_dim（2层MLP）
        - n_proj_layers=3: in_dim -> hidden_dim -> hidden_dim -> out_dim（3层MLP）
        """
        if use_fc:
            # 单层FC模式，忽略 proj_num_layers
            return nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_layers)])
        else:
            def build_mlp():
                layers = []
                if n_proj_layers == 1:
                    # 单层：in -> out
                    layers.append(nn.Linear(in_dim, out_dim))
                elif n_proj_layers == 2:
                    # 两层：in -> hidden -> out
                    layers.extend([
                        nn.Linear(in_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, out_dim),
                    ])
                else:
                    # 多层：in -> hidden -> ... -> hidden -> out
                    # 第一层：in -> hidden
                    layers.extend([
                        nn.Linear(in_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ])
                    # 中间层：hidden -> hidden
                    for _ in range(n_proj_layers - 2):
                        layers.extend([
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        ])
                    # 最后一层：hidden -> out
                    layers.append(nn.Linear(hidden_dim, out_dim))
                return nn.Sequential(*layers)
            
            return nn.ModuleList([build_mlp() for _ in range(num_layers)])
    
    def _ensure_token_feature_banks(self, max_tokens: int):
        """确保TokenFeatureBank已创建"""
        if self.use_feature_bank and self.token_feature_banks is None:
            self.token_bank_max_tokens = max_tokens
            device = next(self.parameters()).device
            self.token_feature_banks = nn.ModuleList([
                DualTokenFeatureBank(
                    bank_size=self.feature_bank_size,
                    max_tokens=max_tokens,
                    feature_dim=self.projection_dim,
                    device=device
                )
                for _ in range(self.num_selected_layers)
            ])
            self.token_feature_banks.to(device)
    
    def _ensure_at2tt_feature_banks(self, max_action_tokens: int, max_text_tokens: int):
        """确保AT2TTTokenFeatureBank已创建（延迟初始化）"""
        if self.use_feature_bank and self.at2tt_feature_banks is None:
            self.at2tt_bank_params = (max_action_tokens, max_text_tokens)
            device = next(self.parameters()).device
            self.at2tt_feature_banks = nn.ModuleList([
                AT2TTTokenFeatureBank(
                    bank_size=self.feature_bank_size,
                    max_action_tokens=max_action_tokens,
                    max_text_tokens=max_text_tokens,
                    feature_dim=self.projection_dim,
                    device=device
                )
                for _ in range(self.num_selected_layers)
            ])
            self.at2tt_feature_banks.to(device)
    
    
    def forward(
        self, 
        data: Dict = None,
        allgather=None, 
        n_gpu: int = 1, 
        args=None, 
        config=None, 
        loss_fn=None, 
        gpu: int = 0, 
        return_embeds: bool = True, 
        task_names: str = 'Alignment',
        openvla_features: Optional[torch.Tensor] = None,  # [B, L, A, D_action]
        egovlpv2_features: Optional[torch.Tensor] = None,  # [B, D_text] global text
        egovlpv2_text_tokens: Optional[torch.Tensor] = None,  # [B, T, D_token] local text
        egovlpv2_text_mask: Optional[torch.Tensor] = None,  # [B, T]
        task_index: Optional[torch.Tensor] = None,  # [B] 原始任务索引
        task_ids: Optional[torch.Tensor] = None,  # [B] 聚类后的任务类别ID
        langs: list = None,  # List[str] 指令文本列表，用于检测空指令
        video_features: Optional[torch.Tensor] = None,     # [B*n, D] 采样的ego4d video features
        video_sim_weights: Optional[torch.Tensor] = None,  # [B, B*n] soft positive mask
        atomic_text_features: Optional[torch.Tensor] = None,  # [B, D_text] 原子级text embedding
        chunk_video_features: Optional[torch.Tensor] = None,  # [B, D_video] chunk-level video embedding（as2cv）
        chunk_video_task_ids: Optional[torch.Tensor] = None,  # [B] task_id 用于 as2cv trajectory 正样本判定
        chunk_video_idx: Optional[torch.Tensor] = None,        # [B] chunk 全局 id；同 id 的样本互为 chunk 正样本
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass - 支持3种对齐模式
        
        对齐模式（self.alignment_mode）：
        - 'diagonal': 对角线为正样本（默认）
        - 'task_id': 相同task_id为正样本
        - 'task_index': 相同task_index为正样本
        
        特征命名：
        - global_action_features: mean pooling后的action特征 [B, D_action]
        - local_action_features: 原始action token序列 [B, A, D_action]
        - global_text_features: text sentence embedding [B, D_text]
        - local_text_features: text token序列 [B, T, D_token]
        """
        # ============ 绑定多卡 allgather 上下文（供 _compute_*_loss 使用） ============
        # 仅在 use_distributed_negatives=True 时生效；单卡或未初始化分布式时自动 fallback 不合并
        self._allgather_fn = allgather
        self._n_gpu = n_gpu
        self._dist_args = args

        # ============ 处理输入 ============
        if data is not None:
            openvla_features = data.get('openvla_features', openvla_features)
            egovlpv2_features = data.get('egovlpv2_features', egovlpv2_features)
            egovlpv2_text_tokens = data.get('egovlpv2_text_tokens', egovlpv2_text_tokens)
            egovlpv2_text_mask = data.get('egovlpv2_text_mask', egovlpv2_text_mask)
        
        if openvla_features is None or egovlpv2_features is None:
            raise ValueError("openvla_features and egovlpv2_features must be provided")
        
        # ============ 根据vla_mode切片action tokens ============
        # openpi: 去掉第一个token（包括mean时也去掉）
        # spatialvla: 去掉最后一个token
        if self.vla_mode == 'openpi':
            openvla_features = openvla_features[:, :, 1:, :]  # [B, L, A-1, D]
        elif self.vla_mode == 'spatialvla':
            openvla_features = openvla_features[:, :, :-1, :]  # [B, L, A-1, D]
        
        batch_size, num_selected_layers, num_actions, _ = openvla_features.shape
        device = openvla_features.device
        
        # ============ 确定使用的ids（统一为一个变量） ============
        # 优先级：alignment_mode指定 > task_ids > task_index > 默认arange
        if self.alignment_mode == 'task_id':
            current_ids = task_ids
            alignment_mode_for_mask = 'task_id'
        elif self.alignment_mode == 'task_index':
            current_ids = task_index
            alignment_mode_for_mask = 'task_index'
        elif self.alignment_mode == 'diagonal':
            current_ids = torch.arange(batch_size, device=device)
            alignment_mode_for_mask = 'diagonal'
        else:
            assert False, f"Unsupported alignment mode: {self.alignment_mode}"
        
        # simple模式：直接使用传入的特征
        global_text_features = egovlpv2_features  # [B, D_text]
        local_text_features = egovlpv2_text_tokens  # [B, T, D_token]
        local_text_mask = egovlpv2_text_mask
        
        # ============ 逐层处理 ============
        mode_losses = {mode: [] for mode in self.active_modes}
        # 按模式分别收集stats，便于分开log
        mode_stats = {mode: [] for mode in self.active_modes}
        # 表征分离损失收集（DSN风格正交+重建）
        disentangle_losses = []
        disentangle_stats_list = []
        # IT 诊断用：每条 (z_s, z_p, x_recon, x_orig) detach 暂存，仅做 logging 不进 backward
        disentangle_pairs = []
        
        # target_attention 模式下的 target 特征（text embedding）
        _pool_target = global_text_features if self.action_pool_mode == 'target_attention' else None
        
        for layer_idx in range(num_selected_layers):
            # 提取当前层的action特征（不投影）
            local_action_features = openvla_features[:, layer_idx, :, :]  # [B, A, D_action]
            # token级"分解后表征"占位：pre_pool DSN 完成后会被赋值为 shared tokens [B, A, D]
            # token-level 对齐（at2tt/at2tt_soft）使用 shared tokens（DSN 后）
            local_action_shared = None

            # ============ 共享的 DSN + Pooling（as2ts / as2atomic / as2vs 三路共用） ============
            # Mirror Neuron 核心：text / atomic text / video 三面镜子共享同一个 z_shared。
            # DSN（如启用）→ pool → pooled_action，然后各路径用独立 MLP 投影。
            _need_pooled_action = ('as2ts' in self.active_modes) or (
                'as2atomic' in self.active_modes and atomic_text_features is not None
            ) or (
                'as2vs' in self.active_modes and video_features is not None
            ) or (
                'as2cv' in self.active_modes and chunk_video_features is not None
            )
            pooled_action = None
            if _need_pooled_action:
                _do_dis = self.use_disentangle and self.as2ts_disentangle is not None
                
                if _do_dis and self.disentangle_position == 'pre_pool':
                    # token级分解：对每个action token独立做shared/private拆分
                    # shared tokens → pooler → 对齐；private tokens不参与
                    B_a, A_a, D_a = local_action_features.shape
                    flat = local_action_features.reshape(B_a * A_a, D_a)
                    z_s_flat, z_p_flat, x_hat_flat = self.as2ts_disentangle(flat)
                    dis_loss, dis_stats = compute_disentangle_losses(
                        z_s_flat, z_p_flat, x_hat_flat, flat,
                        diff_weight=self.disentangle_diff_weight,
                        recon_weight=self.disentangle_recon_weight,
                        diff_mode=self.disentangle_diff_mode,
                        detach_recon_target=self.detach_recon_target,
                    )
                    disentangle_losses.append(dis_loss)
                    disentangle_stats_list.append(dis_stats)
                    # IT logging（pre_pool 走 token-level）
                    disentangle_pairs.append((z_s_flat.detach(), z_p_flat.detach(),
                                              x_hat_flat.detach(), flat.detach()))
                    # 保存 shared tokens 供 token级对齐使用（at2tt/at2tt_soft 在 shared 模式下用它）
                    local_action_shared = z_s_flat.view(B_a, A_a, D_a)
                    pooled_action = self.as2ts_action_pooler(local_action_shared, _pool_target)
                else:
                    pooled_action = self.as2ts_action_pooler(local_action_features, _pool_target)

                if _do_dis and self.disentangle_position == 'post_pool':
                    # 全局级分解：在pooled向量上做shared/private拆分
                    z_shared, z_private, x_recon = self.as2ts_disentangle(pooled_action)
                    dis_loss, dis_stats = compute_disentangle_losses(
                        z_shared, z_private, x_recon, pooled_action,
                        diff_weight=self.disentangle_diff_weight,
                        recon_weight=self.disentangle_recon_weight,
                        diff_mode=self.disentangle_diff_mode,
                        detach_recon_target=self.detach_recon_target,
                    )
                    disentangle_losses.append(dis_loss)
                    disentangle_stats_list.append(dis_stats)
                    # IT logging（post_pool 走 batch-level）
                    disentangle_pairs.append((z_shared.detach(), z_private.detach(),
                                              x_recon.detach(), pooled_action.detach()))
                    pooled_action = z_shared

            # ============ as2ts 模式（全局task级对齐） ============
            if 'as2ts' in self.active_modes:
                as2ts_loss, as2ts_stats = self._forward_as2ts(
                    global_action_features=pooled_action,
                    global_text_features=global_text_features,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    current_ids=current_ids,
                    alignment_mode_for_mask=alignment_mode_for_mask,
                    task_index=task_index,
                )
                mode_losses['as2ts'].append(as2ts_loss)
                mode_stats['as2ts'].append(as2ts_stats)
            
            # ============ as2tt 模式 ============
            if 'as2tt' in self.active_modes and local_text_features is not None:
                as2tt_action = self.as2tt_action_pooler(local_action_features, _pool_target)
                as2tt_loss, as2tt_stats = self._forward_as2tt(
                    global_action_features=as2tt_action,
                    local_text_features=local_text_features,
                    text_mask=local_text_mask,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    current_ids=current_ids,
                    alignment_mode_for_mask=alignment_mode_for_mask,
                )
                mode_losses['as2tt'].append(as2tt_loss)
                mode_stats['as2tt'].append(as2tt_stats)
            
            # ============ at2tt 模式（FILIP风格，直接用 local_action，无需池化） ============
            if 'at2tt' in self.active_modes and local_text_features is not None:
                at2tt_loss, at2tt_stats = self._forward_at2tt(
                    local_action_features=local_action_features,
                    local_text_features=local_text_features,
                    text_mask=local_text_mask,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    current_ids=current_ids,
                    alignment_mode_for_mask=alignment_mode_for_mask,
                    aggregation='filip',
                )
                mode_losses['at2tt'].append(at2tt_loss)
                mode_stats['at2tt'].append(at2tt_stats)

            # ============ at2tt_soft 模式（WTI/AOSM 软化版，as2tt 的 seq-to-seq 推广） ============
            # 与at2tt共享投影头和feature bank（避免双倍参数和显存开销），
            # token-wise 聚合方式由 mode_config['at2tt_soft']['aggregation'] 决定：
            #   'wti'  = softmax-weighted(max)（默认）
            #   'aosm' = X-CLIP AOSM 双层 softmax
            # 与disentangle协同：pre_pool 时使用 z_shared tokens 代替 raw tokens
            if 'at2tt_soft' in self.active_modes and local_text_features is not None:
                _soft_cfg = self.mode_config['at2tt_soft']
                _soft_aggregation = _soft_cfg.get('aggregation', 'wti')
                _soft_attn_temp = _soft_cfg.get('attn_temperature', 0.01)
                # 优先用分解后的 shared tokens；DSN 未启用或非 pre_pool 时退回 raw tokens
                _soft_action = (
                    local_action_shared
                    if local_action_shared is not None
                    else local_action_features
                )
                at2tt_soft_loss, at2tt_soft_stats = self._forward_at2tt(
                    local_action_features=_soft_action,
                    local_text_features=local_text_features,
                    text_mask=local_text_mask,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    current_ids=current_ids,
                    alignment_mode_for_mask=alignment_mode_for_mask,
                    aggregation=_soft_aggregation,
                    attn_temperature=_soft_attn_temp,
                )
                mode_losses['at2tt_soft'].append(at2tt_soft_loss)
                mode_stats['at2tt_soft'].append(at2tt_soft_stats)
            
            # ============ as2vs 模式（Video Mirror） ============
            # 不再 detach：让 video 对齐梯度反传到主干，video 信号真正注入 backbone。
            # 前期噪声风险靠收紧 video pool（仅保留 vitra ego 子集）+ 严格阈值
            # （strong=0.9, neg=0.75）+ sigmoid_weighted ignore 中间区域来控制。
            if 'as2vs' in self.active_modes and video_features is not None:
                as2vs_loss, as2vs_stats = self._forward_as2vs(
                    global_action_features=pooled_action,
                    video_features=video_features,
                    sim_weights=video_sim_weights,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    task_index=task_index,
                )
                mode_losses['as2vs'].append(as2vs_loss)
                mode_stats['as2vs'].append(as2vs_stats)
            
            # ============ as2cv 模式（chunk-level video，trajectory + chunk 双 head） ============
            if 'as2cv' in self.active_modes and chunk_video_features is not None:
                _cv_traj_ids = chunk_video_task_ids if chunk_video_task_ids is not None else current_ids
                # chunk head 的正样本判定：必须用 chunk_video_idx 作 id，
                # 这样 batch 内若有两个样本落在同一个 chunk（同 ep+chunk_idx），
                # 它们的 video feature 完全相同 → 应互为正样本，而不是 diagonal-only。
                # 缺失 chunk_video_idx 时退化为 diagonal（兼容老调用）。
                if chunk_video_idx is not None:
                    _cv_chunk_ids = chunk_video_idx.to(pooled_action.device).long()
                else:
                    _cv_chunk_ids = torch.arange(batch_size, device=pooled_action.device)
                as2cv_loss, as2cv_stats = self._forward_as2cv(
                    global_action_features=pooled_action,
                    chunk_video_features=chunk_video_features,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    traj_ids=_cv_traj_ids,
                    chunk_ids=_cv_chunk_ids,
                )
                mode_losses['as2cv'].append(as2cv_loss)
                mode_stats['as2cv'].append(as2cv_stats)

            # ============ as2atomic 模式（chunk级atomic对齐，复用 pooled_action） ============
            # 复用上面共享计算得到的 pooled_action（经过 DSN 和 pool）
            # 仅通过独立的 as2atomic_action_proj / as2atomic_text_proj MLP 投影到对齐空间
            if 'as2atomic' in self.active_modes and atomic_text_features is not None:
                as2atomic_loss, as2atomic_stats = self._forward_as2atomic(
                    global_action_features=pooled_action,
                    atomic_text_features=atomic_text_features,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                )
                mode_losses['as2atomic'].append(as2atomic_loss)
                mode_stats['as2atomic'].append(as2atomic_stats)
        
        # ============ 计算总损失 ============
        total_loss, loss_dict = self._compute_total_loss(mode_losses, num_selected_layers)
        
        # ============ 表征分离损失：加入总loss ============
        if disentangle_losses:
            dis_total = torch.stack(disentangle_losses).mean()
            total_loss = total_loss + dis_total
            # 汇总分离损失统计到loss_dict
            avg_dis_stats = {}
            for k in disentangle_stats_list[0]:
                avg_dis_stats[k] = sum(s[k] for s in disentangle_stats_list) / len(disentangle_stats_list)
            loss_dict.update(avg_dis_stats)
            loss_dict['total_loss'] = total_loss.item()

        # ============ IT 诊断：HSIC / InfoNCE-MI 下界 / recon NLL / 自由能 ============
        # 仅 logging，不进 backward；DSN 关闭时 disentangle_pairs 为空，函数 no-op
        log_information_plane_step(
            loss_dict, disentangle_pairs,
            # VICReg 不是 InfoNCE，不能套用 log(B)-L_NCE 的 MI 下界。
            # 仍保留 HSIC 与 reconstruction NLL 两项通用诊断。
            mode_losses=None if self.loss_type == 'vicreg' else mode_losses,
            batch_size=batch_size,
        )

        # ============ 按模式分别添加logit/样本数统计 ============
        if self.loss_type != 'vicreg':
            loss_dict['temperature'] = self._get_temperature().item()
        if self.loss_type == 'sigmoid':
            loss_dict['sigmoid_bias'] = self.sigmoid_bias.detach().float().item()
        
        # 所有模式的混合统计（兼容旧日志）
        all_stats = [s for stats_list in mode_stats.values() for s in stats_list]
        if all_stats:
            pos_logits = [s['pos_mean_logit'] for s in all_stats if isinstance(s.get('pos_mean_logit'), torch.Tensor)]
            neg_logits = [s['neg_mean_logit'] for s in all_stats if isinstance(s.get('neg_mean_logit'), torch.Tensor)]
            if pos_logits:
                loss_dict['pos_mean_logit'] = torch.stack(pos_logits).mean().item()
                loss_dict['neg_mean_logit'] = torch.stack(neg_logits).mean().item()
        
        # 每个模式独立的统计（分开log）
        for mode, stats_list in mode_stats.items():
            if not stats_list:
                continue
            pos_logits = [s['pos_mean_logit'] for s in stats_list if isinstance(s.get('pos_mean_logit'), torch.Tensor)]
            neg_logits = [s['neg_mean_logit'] for s in stats_list if isinstance(s.get('neg_mean_logit'), torch.Tensor)]
            if pos_logits:
                loss_dict[f'{mode}_pos_logit'] = torch.stack(pos_logits).mean().item()
                loss_dict[f'{mode}_neg_logit'] = torch.stack(neg_logits).mean().item()
            pos_samples = [s['avg_pos_samples'] for s in stats_list if isinstance(s.get('avg_pos_samples'), torch.Tensor)]
            neg_samples = [s['avg_neg_samples'] for s in stats_list if isinstance(s.get('avg_neg_samples'), torch.Tensor)]
            total_samples = [s['avg_total_samples'] for s in stats_list if isinstance(s.get('avg_total_samples'), torch.Tensor)]
            if pos_samples:
                loss_dict[f'{mode}_pos_samples'] = torch.stack(pos_samples).mean().item()
                loss_dict[f'{mode}_neg_samples'] = torch.stack(neg_samples).mean().item()
                loss_dict[f'{mode}_total_samples'] = torch.stack(total_samples).mean().item()
            vicreg_keys = {
                key
                for stats in stats_list
                for key in stats
                if key.startswith('vicreg_')
            }
            for key in vicreg_keys:
                values = [
                    stats[key]
                    for stats in stats_list
                    if isinstance(stats.get(key), torch.Tensor)
                ]
                if values:
                    loss_dict[key] = torch.stack(values).mean().item()

        # 合并 AnchorBank / PKT 等自定义 stats: 凡是 key 含 'anchor' 或 'pkt' 的, 跨层取均值
        custom_keys = set()
        for stats_list in mode_stats.values():
            for s in stats_list:
                for k in s.keys():
                    kl = k.lower()
                    if 'anchor' in kl or 'pkt' in kl:
                        custom_keys.add(k)
        for k in custom_keys:
            vals = []
            for stats_list in mode_stats.values():
                for s in stats_list:
                    v = s.get(k, None)
                    if isinstance(v, torch.Tensor):
                        vals.append(v.detach().float())
            if vals:
                loss_dict[k] = torch.stack(vals).mean().item()
        
        return total_loss, loss_dict, {'num_selected_layers': num_selected_layers}
    
    def _forward_as2atomic(
        self,
        global_action_features: torch.Tensor,  # [B, D_action] pooled action特征
        atomic_text_features: torch.Tensor,    # [B, D_text] 原子级text embedding
        layer_idx: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        as2atomic模式：Action Sequence to Atomic Text 细粒度chunk级对齐
        
        与as2ts结构相同，但使用独立投影头，且强制diagonal对齐模式——
        每个sample的action只与自身的atomic label配对为正样本。
        """
        action_proj = self.as2atomic_action_proj_per_layer[layer_idx](global_action_features)
        text_proj = self.as2atomic_text_proj_per_layer[layer_idx](atomic_text_features)
        # 强制 diagonal：每个sample独立，不按task_id聚合正样本
        diagonal_ids = torch.arange(batch_size, device=action_proj.device)
        loss, stats = self._compute_contrastive_loss(
            text_proj, action_proj, layer_idx, diagonal_ids, 'diagonal'
        )
        return loss, stats

    def _forward_as2cv(
        self,
        global_action_features: torch.Tensor,  # [B, D_action] pooled action
        chunk_video_features: torch.Tensor,    # [B, D_video] frozen chunk video embedding
        layer_idx: int,
        batch_size: int,
        traj_ids: torch.Tensor,                # [B] task_id（含 -1 表无效）
        chunk_ids: torch.Tensor,               # [B] chunk_video_idx（同 id 的样本 video feature 完全相同）
    ) -> Tuple[torch.Tensor, Dict]:
        """
        as2cv：Action Sequence to Chunk Video（双粒度）

        - trajectory head：alignment_mode='task_id'，同 task 互为正样本
        - chunk head：alignment_mode='task_id' on chunk_ids，同 chunk_video_idx 互为正样本
          （注意：不能用 diagonal-only，因为 batch 内若两个 frame 落在同一 chunk，
            它们查到的 video feature 是同一行，必须互为正才不会“自相矛盾”地被推开）
        - 两 head 共用同一个 sim 矩阵，仅 mask 不同；权重 traj_weight / chunk_weight 控制
        """
        # 投影 + L2 归一化
        a = self.as2cv_action_proj_per_layer[layer_idx](global_action_features)
        v = self.as2cv_video_proj_per_layer[layer_idx](chunk_video_features)
        a = F.normalize(a, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)

        # similarity 已除以 temperature；后续 sigmoid/infonce 由 _compute_bidirectional_contrastive_loss 决定
        similarity = torch.mm(a, v.t()) / self._get_temperature()  # [B, B]

        # —— trajectory：同 task_id 互为正 ——
        pos_mask_traj, valid_traj, _ = create_alignment_masks(traj_ids, 'task_id')
        loss_traj, stats_traj = self._compute_bidirectional_contrastive_loss(
            similarity, batch_size, similarity.device, pos_mask_traj, valid_traj
        )

        # —— chunk：同 chunk_video_idx 互为正（diagonal 是其特例） ——
        pos_mask_chunk, valid_chunk, _ = create_alignment_masks(chunk_ids, 'task_id')
        loss_chunk, stats_chunk = self._compute_bidirectional_contrastive_loss(
            similarity, batch_size, similarity.device, pos_mask_chunk, valid_chunk
        )

        loss = self.as2cv_traj_weight * loss_traj + self.as2cv_chunk_weight * loss_chunk

        # 复用 stats 字段名，前缀 traj_/chunk_ 区分；上层会自动 reduce
        stats = {
            'pos_mean_logit': stats_chunk.get('pos_mean_logit'),
            'neg_mean_logit': stats_chunk.get('neg_mean_logit'),
            'avg_pos_samples': stats_chunk.get('avg_pos_samples'),
            'avg_neg_samples': stats_chunk.get('avg_neg_samples'),
            'avg_total_samples': stats_chunk.get('avg_total_samples'),
            'traj_pos_mean_logit': stats_traj.get('pos_mean_logit'),
            'traj_neg_mean_logit': stats_traj.get('neg_mean_logit'),
            'traj_avg_pos_samples': stats_traj.get('avg_pos_samples'),
        }
        return loss, stats

    def _forward_as2ts(
        self,
        global_action_features: torch.Tensor,  # [B, D_action] 原始特征
        global_text_features: torch.Tensor,    # [B, D_text]
        layer_idx: int,
        batch_size: int,
        current_ids: torch.Tensor,  # [B] 当前batch的ids (用于 SupCon mask)
        alignment_mode_for_mask: str,  # 对齐模式
        task_index: Optional[torch.Tensor] = None,  # [B] 原始 task_index ∈ [0, 40), AnchorBank lookup 用
    ) -> Tuple[torch.Tensor, Dict]:
        """
        as2ts模式：Action Sequence to Text Sentence 全局对齐

        支持Feature Bank + 多正样本模式 + (可选) AnchorBank a2t-direction extension
        AnchorBank: 给 a2t 方向 sim matrix 多接 (1 orig + K_p paraphrase) 正样本 + (K_neg cfact) 负样本列,
                    走同一份 sigmoid/InfoNCE, 不破坏 loss_type 选择.
        """
        # 投影 action
        action_proj = self.as2ts_action_proj_per_layer[layer_idx](global_action_features)
        # 投影 text
        text_proj = self.as2ts_text_proj_per_layer[layer_idx](global_text_features)

        # ===== AnchorBank: 构造 per-sample 的 anchor extra text + target =====
        anchor_extra_text = None
        anchor_extra_target = None
        if self.use_anchor_bank and task_index is not None:
            ti = task_index.long().clamp(min=0)
            e_pos_raw, e_para_raw, e_cfact_raw = self.anchor_bank.lookup(
                ti, K_neg_sample=self.anchor_K_neg_sample
            )
            # cast 同 dtype, 同 proj head (与 baseline text 共享空间)
            e_pos_raw = e_pos_raw.to(text_proj.dtype)
            e_para_raw = e_para_raw.to(text_proj.dtype)
            e_cfact_raw = e_cfact_raw.to(text_proj.dtype)
            B = ti.shape[0]
            K_p = e_para_raw.shape[1]
            K_n = e_cfact_raw.shape[1]
            D_in = e_pos_raw.shape[-1]
            proj_layer = self.as2ts_text_proj_per_layer[layer_idx]
            e_pos_proj = proj_layer(e_pos_raw)                                              # [B, D]
            if K_p > 0:
                e_para_proj = proj_layer(e_para_raw.reshape(B * K_p, D_in)).reshape(B, K_p, -1)
            else:
                e_para_proj = e_pos_proj.new_zeros(B, 0, e_pos_proj.shape[-1])
            if K_n > 0:
                e_cfact_proj = proj_layer(e_cfact_raw.reshape(B * K_n, D_in)).reshape(B, K_n, -1)
            else:
                e_cfact_proj = e_pos_proj.new_zeros(B, 0, e_pos_proj.shape[-1])
            # 拼到 [B, S=1+K_p+K_n, D]
            anchor_extra_text = torch.cat([e_pos_proj.unsqueeze(1), e_para_proj, e_cfact_proj], dim=1)
            # target: orig + paraphrase = 1 (positive); counterfactual = 0 (negative)
            anchor_extra_target = torch.zeros(B, 1 + K_p + K_n, device=text_proj.device, dtype=text_proj.dtype)
            anchor_extra_target[:, : 1 + K_p] = 1.0

        if self.loss_type == 'vicreg':
            if current_ids is None:
                raise ValueError("VICReg需要task_ids或task_index来屏蔽无效样本")
            valid_mask = current_ids != -1
            loss, stats = self.vicreg_loss(
                action_features=action_proj,
                text_features=text_proj,
                valid_mask=valid_mask,
                # 与 InfoNCE 保持相同语义：task_id/task_index 下同 ID 样本互为正配对。
                sample_ids=(
                    current_ids if alignment_mode_for_mask != 'diagonal' else None
                ),
                gather_distributed=self.vicreg_gather_distributed,
            )
        else:
            # 对比学习基线继续使用原有正负样本、feature bank 和 anchor 逻辑。
            loss, stats = self._compute_contrastive_loss(
                text_proj, action_proj, layer_idx, current_ids, alignment_mode_for_mask,
                anchor_extra_text=anchor_extra_text,
                anchor_extra_target=anchor_extra_target,
            )

        return loss, stats
    
    def _forward_as2tt(
        self,
        global_action_features: torch.Tensor,  # [B, D_action] 原始特征
        local_text_features: torch.Tensor,     # [B, T, D_token]
        text_mask: Optional[torch.Tensor],     # [B, T]
        layer_idx: int,
        batch_size: int,
        current_ids: torch.Tensor,  # [B] 当前batch的ids
        alignment_mode_for_mask: str,  # 对齐模式
    ) -> Tuple[torch.Tensor, Dict]:
        """
        as2tt模式：Action Sequence to Text Token 细粒度对齐（XClip风格）
        
        支持Feature Bank + 多正样本模式
        """
        # 投影 action
        action_proj = self.as2tt_action_proj_per_layer[layer_idx](global_action_features)  # [B, proj_dim]
        
        # 投影 text tokens
        B_text, T, D_token = local_text_features.shape
        text_flat = local_text_features.view(B_text * T, D_token)
        text_proj_flat = self.as2tt_text_proj_per_layer[layer_idx](text_flat)
        text_proj = text_proj_flat.view(B_text, T, -1)  # [B, T, proj_dim]
        
        temperature = self.mode_config['as2tt']['temperature']
        
        # 计算loss，传递current_ids用于Feature Bank扩展
        loss, stats = self._compute_token_contrastive_loss(
            action_proj, text_proj, text_mask, temperature, layer_idx,
            current_ids, alignment_mode_for_mask
        )
        
        return loss, stats
    
    
    def _forward_at2tt(
        self,
        local_action_features: torch.Tensor,  # [B, A, D_action] 原始action token序列
        local_text_features: torch.Tensor,    # [B, T, D_token]
        text_mask: Optional[torch.Tensor],    # [B, T]
        layer_idx: int,
        batch_size: int,
        current_ids: torch.Tensor,  # [B] 当前batch的ids
        alignment_mode_for_mask: str,  # 对齐模式
        aggregation: str = 'filip',   # 'filip' = mean(max), 'wti' = softmax-weighted(max)
        attn_temperature: float = 0.01,  # WTI 中 token importance softmax 温度
    ) -> Tuple[torch.Tensor, Dict]:
        """
        at2tt模式：Action Token to Text Token 细粒度对齐

        支持两种 token-wise 聚合方式：
        - aggregation='filip': FILIP原文，双向 mean(max)，硬聚合
        - aggregation='wti'  : DRL-WTI 软化版，双向 Σ softmax(max/τ_attn)·max
                               即 as2tt（X-Clip Video-Word）的seq-to-seq推广

        注意：action tokens已在forward中根据vla_mode进行了切片处理
        共享 at2tt_action_proj / at2tt_text_proj / at2tt_feature_banks，
        以便两种模式可同时启用且不重复消耗显存。
        支持Feature Bank增大有效负样本数量；支持多正样本模式
        """
        from egovlpv2.model.loss import compute_at2tt_contrastive_loss
        
        # 投影 action tokens
        B, A, D_action = local_action_features.shape
        action_flat = local_action_features.contiguous().view(B * A, D_action)
        action_proj_flat = self.at2tt_action_proj_per_layer[layer_idx](action_flat)
        action_proj = action_proj_flat.view(B, A, -1)  # [B, A, proj_dim]

        # 投影 text tokens
        B_text, T, D_token = local_text_features.shape
        text_flat = local_text_features.contiguous().view(B_text * T, D_token)
        text_proj_flat = self.at2tt_text_proj_per_layer[layer_idx](text_flat)
        text_proj = text_proj_flat.view(B_text, T, -1)  # [B, T, proj_dim]

        # 多卡 merge：token级也跨rank合并（投影后再 gather，节省 allgather 流量）
        action_proj, text_proj, text_mask, current_ids = self._maybe_dist_gather(
            action_proj, text_proj, text_mask, current_ids
        )
        batch_size = action_proj.shape[0]

        # Feature Bank: 扩展负样本，同时获取bank中的ids
        if self.use_feature_bank:
            self._ensure_at2tt_feature_banks(A, T)
            bank_action, bank_text, bank_text_mask, bank_ids = self.at2tt_feature_banks[layer_idx].get_all()
            if len(self.at2tt_feature_banks[layer_idx]) > 0:
                action_all = torch.cat([action_proj, bank_action], dim=0)
                text_all = torch.cat([text_proj, bank_text], dim=0)
                ids_all = torch.cat([current_ids, bank_ids], dim=0)
                
                # Text Mask Concatenation（B 用 gathered 后的 batch_size，避免与 distributed merge 冲突）
                if text_mask is not None:
                    text_mask_all = torch.cat([text_mask, bank_text_mask], dim=0)
                else:
                    text_mask_all = torch.cat([
                        torch.ones(batch_size, T, device=text_proj.device, dtype=torch.long),
                        bank_text_mask
                    ], dim=0)
            else:
                action_all, text_all = action_proj, text_proj
                text_mask_all = text_mask
                ids_all = current_ids
        else:
            action_all, text_all = action_proj, text_proj
            text_mask_all = text_mask
            ids_all = current_ids
        
        # 使用扩展后的ids重新生成masks
        positive_mask, valid_mask, _ = create_alignment_masks(ids_all, alignment_mode_for_mask)
        
        # FILIP / WTI 风格对比学习loss（aggregation 决定聚合方式）
        loss, stats = compute_at2tt_contrastive_loss(
            action_tokens=action_all,
            text_tokens=text_all,
            batch_size=batch_size,
            text_mask=text_mask_all,
            temperature=self._get_temperature(),
            loss_type=self.loss_type,
            sigmoid_bias=self.sigmoid_bias,
            positive_mask=positive_mask,
            valid_mask=valid_mask,
            min_valid_ratio=self.min_valid_ratio,  # 传递min_valid_ratio配置
            aggregation=aggregation,
            attn_temperature=attn_temperature,
        )
        
        # 添加当前batch到feature bank（包含ids）
        if self.use_feature_bank and self.at2tt_feature_banks is not None:
            self.at2tt_feature_banks[layer_idx].add(action_proj, text_proj, text_mask, current_ids)
        
        return loss, stats
    
    def _forward_as2vs(
        self,
        global_action_features: torch.Tensor,  # [B, D_action] action mean pooling特征
        video_features: torch.Tensor,           # [M, D_video] 去重后的shared sampled anchors
        sim_weights: torch.Tensor,              # [B, M] dense query-to-anchor text 相似度矩阵
        layer_idx: int,
        batch_size: int,
        task_index: Optional[torch.Tensor] = None,  # [B] 原始任务索引，-1表示无效
    ) -> Tuple[torch.Tensor, Dict]:
        """
        as2vs模式：Action Sequence to Video Sampled

        当前版本的监督信号设计：
        1. sampler 先按旧逻辑采样，再在 batch 内去重得到 shared sampled anchors。
        2. 对每个 case 和每个 sampled anchor 都计算一个 text 相似度，形成 dense 矩阵 [B, M]。
        3. 只把相似度高于 threshold 的位置视为正样本。
        4. 不做行归一化；每个 case 的 loss 只除以该 case 的正样本个数。

        支持通过 as2vs config 中的 'loss_type' 字段选择 loss 模式：
        - 'infonce' (默认): 原有 Soft InfoNCE，负样本在 softmax 分母中
        - 'sigmoid_bce': Weighted Sigmoid BCE Loss（soft label，w∈[0,1]）
        - 'sigmoid_binary': Hard Binary Sigmoid（>threshold=+1，否则=-1，与as2ts一致）

        Args:
            global_action_features: [B, D_action] 经过 DSN+Pool 的 action 特征
            video_features: [M, D_video] 去重后的 sampled anchor 特征
            sim_weights: [B, M] dense text 相似度矩阵
            layer_idx: 当前层索引
            batch_size: B
            task_index: [B] 原始任务索引，-1=无效
        """
        B = batch_size
        M = video_features.shape[0]
        device = global_action_features.device

        # ============ 有效样本 mask ============
        if task_index is not None:
            valid_anchor_mask = (task_index != -1).float()  # [B]
        else:
            valid_anchor_mask = torch.ones(B, device=device)

        # ============ 投影 + 归一化 ============
        action_proj = self.as2vs_action_proj_per_layer[layer_idx](global_action_features)
        video_proj = self.as2vs_video_proj_per_layer[layer_idx](video_features)
        action_norm = F.normalize(action_proj, p=2, dim=-1)
        video_norm = F.normalize(video_proj, p=2, dim=-1)

        # 相似度矩阵 [B, M] / temperature
        temperature = self._get_temperature()
        logits = torch.mm(action_norm, video_norm.t()) / temperature

        # ============ Threshold 过滤 + 权重变换 ============
        # 这里只决定哪些位置是正样本，以及正样本的权重大小。
        threshold = self.as2vs_sim_threshold
        base = self.as2vs_sim_base
        raw_sim_weights = sim_weights.float()

        if threshold > 0:
            pos_mask = raw_sim_weights > threshold
        else:
            pos_mask = raw_sim_weights > 0

        positive_weights = torch.zeros_like(raw_sim_weights)
        positive_weights[pos_mask] = raw_sim_weights[pos_mask]
        if base > 0:
            positive_weights[pos_mask] = (
                (positive_weights[pos_mask] - base).clamp(min=0.0) /
                max(1.0 - base, 1e-8)
            )

        # ============ 根据 as2vs 配置选择 loss 模式 ============
        # as2vs_loss_type: 从 mode_config['as2vs'] 读取，默认 'infonce'
        as2vs_cfg = self.mode_config.get('as2vs', {})
        as2vs_loss_type = as2vs_cfg.get('loss_type', 'infonce')
        # as2vs 使用独立的 bias（默认0），不复用 as2ts 的 sigmoid_bias（通常=1.5）
        # 因为 as2vs 的正负比极端（~1.6%），高 bias 会让正样本更难被推过决策边界
        as2vs_bias = as2vs_cfg.get('sigmoid_bias', 0.0)
        stats_pos_mask = pos_mask
        stats_neg_mask = ~pos_mask

        if as2vs_loss_type == 'sigmoid_binary':
            # ============ Hard Binary Sigmoid Loss ============
            # sim > threshold → z=+1（正样本），否则 z=-1（负样本）
            # L = -log(σ(z · s))，其中 s = logits - as2vs_bias
            logits_with_bias = logits - as2vs_bias
            binary_labels = pos_mask.float() * 2 - 1  # +1 或 -1
            elem_loss = -F.logsigmoid(binary_labels * logits_with_bias)

            per_sample_loss = elem_loss.mean(dim=-1)
            per_sample_loss = per_sample_loss * valid_anchor_mask
            num_valid = valid_anchor_mask.sum().clamp(min=1)
            loss = per_sample_loss.sum() / num_valid

        elif as2vs_loss_type == 'sigmoid_bce':
            # ============ Weighted Sigmoid BCE Loss (soft label) ============
            # SigLIP 风格：每个 pair 独立做二元分类，w∈[0,1] 是 soft label
            # L = -[w·log(σ(s)) + (1-w)·log(σ(-s))]
            soft_labels = positive_weights.clamp(0.0, 1.0)
            logits_with_bias = logits - as2vs_bias

            bce_loss = -(
                soft_labels * F.logsigmoid(logits_with_bias)
                + (1 - soft_labels) * F.logsigmoid(-logits_with_bias)
            )

            per_sample_loss = bce_loss.mean(dim=-1)
            per_sample_loss = per_sample_loss * valid_anchor_mask
            num_valid = valid_anchor_mask.sum().clamp(min=1)
            loss = per_sample_loss.sum() / num_valid
        elif as2vs_loss_type == 'sigmoid_weighted':
            # ============ Weighted Sigmoid Loss（推荐） ============
            # 设计目标：
            # 1. >= strong_pos_threshold 的样本是强正样本，权重=1
            # 2. weak_pos_threshold ~ strong_pos_threshold 之间是弱正样本，线性增加权重
            # 3. <= neg_threshold 的样本才视为负样本
            # 4. 中间不确定区域直接忽略，避免把“部分相关样本”硬压成负样本
            strong_pos_threshold = as2vs_cfg.get('strong_pos_threshold', 0.85)
            weak_pos_threshold = as2vs_cfg.get('weak_pos_threshold', 0.70)
            neg_threshold = as2vs_cfg.get('neg_threshold', 0.55)

            strong_pos_mask = raw_sim_weights >= strong_pos_threshold
            weak_pos_mask = (
                (raw_sim_weights > weak_pos_threshold)
                & (raw_sim_weights < strong_pos_threshold)
            )
            neg_mask = raw_sim_weights <= neg_threshold

            logits_with_bias = logits - as2vs_bias
            targets = torch.zeros_like(raw_sim_weights)
            pair_weights = torch.zeros_like(raw_sim_weights)

            # 强正样本：确定的正样本，直接给满权重
            targets[strong_pos_mask] = 1.0
            pair_weights[strong_pos_mask] = 1.0

            # 弱正样本：按相似度线性映射到 [0, 1]
            # 这样只需要 3 个配置参数，不额外引入权重超参。
            if weak_pos_mask.any():
                weak_scores = raw_sim_weights[weak_pos_mask]
                weak_scale = (weak_scores - weak_pos_threshold) / max(
                    strong_pos_threshold - weak_pos_threshold, 1e-8
                )
                weak_scale = weak_scale.clamp(0.0, 1.0)
                targets[weak_pos_mask] = 1.0
                pair_weights[weak_pos_mask] = weak_scale

            # 只有低于负阈值的样本才作为负样本；中间区域忽略
            pair_weights[neg_mask] = 1.0

            elem_loss = F.binary_cross_entropy_with_logits(
                logits_with_bias, targets, reduction='none'
            )
            weighted_elem_loss = elem_loss * pair_weights

            # 每个样本按参与监督的总权重归一化，避免弱正样本数量变化带来尺度波动
            weight_sum_per_row = pair_weights.sum(dim=-1)
            has_supervision = weight_sum_per_row > 0
            per_sample_loss = weighted_elem_loss.sum(dim=-1) / weight_sum_per_row.clamp(min=1e-6)

            effective_mask = valid_anchor_mask * has_supervision.float()
            num_valid = effective_mask.sum().clamp(min=1)
            loss = (per_sample_loss * effective_mask).sum() / num_valid

            # 统计时只把真正参与监督的正/负样本算进去
            stats_pos_mask = strong_pos_mask | weak_pos_mask
            stats_neg_mask = neg_mask
        else:
            # ============ 原有 Soft InfoNCE Loss（默认） ============
            # softmax 分母包含所有 M 个 anchor（正+负），正样本加权求和后除以正样本数
            log_probs = F.log_softmax(logits, dim=-1)  # [B, M]
            num_pos = pos_mask.sum(dim=-1).clamp(min=1).to(log_probs.dtype)
            per_sample_loss = -(positive_weights * log_probs).sum(dim=-1) / num_pos

            # 有效 query：task_index != -1 且至少有一个正样本
            has_pos = pos_mask.any(dim=-1).float()
            effective_mask = valid_anchor_mask * has_pos

            # loss = mean over 有效 query
            per_sample_loss = per_sample_loss * effective_mask
            num_valid = effective_mask.sum().clamp(min=1)
            loss = per_sample_loss.sum() / num_valid

        # ============ 统计信息 ============
        total_anchors = logits.shape[1]
        with torch.no_grad():
            # sigmoid 系列模式下统计减去 as2vs 独立 bias 后的 logits
            use_bias_stat = as2vs_loss_type in ('sigmoid_binary', 'sigmoid_bce', 'sigmoid_weighted')
            stat_logits = (logits - as2vs_bias) if use_bias_stat else logits
            valid_rows = valid_anchor_mask.bool().unsqueeze(1).expand_as(logits)
            pos_locs = stats_pos_mask & valid_rows
            neg_locs = stats_neg_mask & valid_rows
            pos_logits_vals = stat_logits[pos_locs]
            neg_logits_vals = stat_logits[neg_locs]
            avg_pos_per_query = (
                stats_pos_mask[valid_anchor_mask.bool()].float().sum(dim=-1).mean()
                if valid_anchor_mask.sum() > 0
                else torch.tensor(0.0, device=device)
            )
            avg_neg_per_query = (
                stats_neg_mask[valid_anchor_mask.bool()].float().sum(dim=-1).mean()
                if valid_anchor_mask.sum() > 0
                else torch.tensor(0.0, device=device)
            )
            stats = {
                'pos_mean_logit': pos_logits_vals.mean() if pos_logits_vals.numel() > 0 else torch.tensor(0.0, device=device),
                'neg_mean_logit': neg_logits_vals.mean() if neg_logits_vals.numel() > 0 else torch.tensor(0.0, device=device),
                'avg_pos_samples': avg_pos_per_query if isinstance(avg_pos_per_query, torch.Tensor) else torch.tensor(avg_pos_per_query, device=device),
                'avg_neg_samples': avg_neg_per_query if isinstance(avg_neg_per_query, torch.Tensor) else torch.tensor(avg_neg_per_query, device=device),
                'avg_total_samples': (avg_pos_per_query + avg_neg_per_query) if isinstance(avg_pos_per_query, torch.Tensor) else torch.tensor(float(total_anchors), device=device),
            }

        return loss, stats
    
    def _compute_total_loss(self, mode_losses: Dict[str, list], num_layers: int) -> Tuple[torch.Tensor, Dict]:
        """计算加权总损失"""
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device, dtype=next(self.parameters()).dtype)
        loss_dict = {}
        
        for mode in self.active_modes:
            if mode_losses[mode]:
                # mode_losses已按层收集，这里只平均一次，避免重复除以num_layers
                mode_loss = torch.stack(mode_losses[mode]).mean()
                mode_weight = self.mode_config[mode]['weight']
                weighted_loss = mode_loss * mode_weight
                total_loss = total_loss + weighted_loss
                loss_dict[f'{mode}_loss'] = mode_loss.item()
        
        # 已对每个mode做层内平均，这里不再除以num_layers
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def _maybe_dist_gather(self, *tensors):
        """
        多卡负样本 merge 工具：用 AllGather_multi 在 batch 维拼接所有 rank 的张量。

        - 仅当 use_distributed_negatives=True 且分布式环境（n_gpu>1）真实可用时合并；
          否则直接返回原张量。
        - AllGather_multi 的 backward 只对当前 rank 的样本回梯度，因此安全。
        - None 元素直接透传，便于按位置 unpack。
        """
        if (not self.use_distributed_negatives
                or self._allgather_fn is None
                or self._n_gpu <= 1
                or self._dist_args is None):
            return tensors
        return tuple(
            None if t is None else self._allgather_fn(t.contiguous(), self._n_gpu, self._dist_args)
            for t in tensors
        )

    def _compute_contrastive_loss(
        self,
        text_features: torch.Tensor,  # [B, D]
        action_features: torch.Tensor,  # [B, D]
        layer_idx: int,
        current_ids: torch.Tensor,  # [B] 当前batch的ids
        alignment_mode_for_mask: str,  # 对齐模式
        anchor_extra_text: Optional[torch.Tensor] = None,  # [B, S, D] per-sample anchor 文本特征 (已投影同空间)
        anchor_extra_target: Optional[torch.Tensor] = None,  # [B, S] per-sample anchor 正负标签 (1=pos, 0=neg)
    ) -> Tuple[torch.Tensor, Dict]:
        """计算对比学习loss，支持 Feature Bank + 多正样本 + 多卡 allgather + AnchorBank a2t-direction extension."""
        # 多卡 merge：先把所有rank的特征 allgather 到一起再计算 loss（扩大有效负样本池）
        text_features, action_features, current_ids = self._maybe_dist_gather(
            text_features, action_features, current_ids
        )
        batch_size = text_features.shape[0]

        # Feature Bank: 扩展负样本，同时获取bank中的ids
        if self.use_feature_bank and self.feature_banks is not None:
            bank_text, bank_action, bank_ids = self.feature_banks[layer_idx].get_all()
            if len(self.feature_banks[layer_idx]) > 0:
                text_all = torch.cat([text_features, bank_text], dim=0)
                action_all = torch.cat([action_features, bank_action], dim=0)
                ids_all = torch.cat([current_ids, bank_ids], dim=0)
            else:
                text_all, action_all = text_features, action_features
                ids_all = current_ids
        else:
            text_all, action_all = text_features, action_features
            ids_all = current_ids

        # 使用扩展后的ids重新生成masks
        positive_mask, valid_mask, _ = create_alignment_masks(ids_all, alignment_mode_for_mask)
        # L2归一化 + 相似度（统一在此处应用温度，使用除法）
        text_norm = F.normalize(text_all, p=2, dim=1)
        action_norm = F.normalize(action_all, p=2, dim=1)

        # 统一使用除法应用温度，无论是 infonce 还是 sigmoid
        similarity = torch.mm(text_norm, action_norm.t()) / self._get_temperature()

        # ===== AnchorBank: a2t 方向 sim matrix 扩展 (per-sample) =====
        # baseline 走 _compute_bidirectional_contrastive_loss; 当 anchor 提供时, 手写两方向调 helper.
        if anchor_extra_text is None:
            loss, stats = self._compute_bidirectional_contrastive_loss(
                similarity, batch_size, similarity.device, positive_mask, valid_mask
            )
        else:
            from egovlpv2.model.loss import (
                compute_single_direction_sigmoid,
                compute_single_direction_infonce,
                compute_anchor_valid_mask_by_ratio,
            )
            S = anchor_extra_text.shape[1]
            # anchor 也 L2 normalize, 同 temperature (与 baseline 共享同一空间)
            anchor_norm = F.normalize(anchor_extra_text, p=2, dim=-1)             # [B, S, D]
            sim_extra_a2t = torch.einsum('bd,bsd->bs', action_norm[:batch_size], anchor_norm) / self._get_temperature()  # [B, S]

            # baseline 部分 mask 准备
            valid_mask_batch = valid_mask[:batch_size]
            # 全部列 valid: batch_all + anchor S
            anchor_valid_col = torch.ones(batch_size, S, device=similarity.device, dtype=valid_mask.dtype)
            min_valid_ratio = getattr(self, 'min_valid_ratio', 0.0)

            # 方向1: 1→2 (text→action, t2a) — 不动, 仍方阵 [B, N]
            #   note: 复刻 _compute_*_loss 的内部逻辑
            valid_mask_2d_t2a = valid_mask_batch.unsqueeze(1) * valid_mask.unsqueeze(0)  # [B, N]
            valid_anchor_t2a = compute_anchor_valid_mask_by_ratio(
                valid_mask_2d_t2a, valid_mask_batch, min_valid_ratio
            )

            # 方向2: 2→1 (action→text, a2t) — sim cat anchor 列, mask cat anchor target
            sim_a2t = similarity[:, :batch_size].t()                                         # [B, N]
            pos_a2t = positive_mask[:, :batch_size].t()                                      # [B, N]
            sim_a2t_full = torch.cat([sim_a2t, sim_extra_a2t], dim=1)                        # [B, N+S]
            pos_a2t_full = torch.cat([pos_a2t, anchor_extra_target.float()], dim=1)          # [B, N+S]
            valid_a2t_2d_full = torch.cat([valid_mask_2d_t2a, anchor_valid_col], dim=1)      # [B, N+S]
            valid_anchor_a2t = compute_anchor_valid_mask_by_ratio(
                valid_a2t_2d_full, valid_mask_batch, min_valid_ratio
            )

            if self.loss_type == 'sigmoid':
                logits_t2a = similarity[:batch_size, :] - self.sigmoid_bias.float()          # [B, N]
                loss_t2a = compute_single_direction_sigmoid(
                    logits_t2a, positive_mask[:batch_size, :], valid_mask_2d_t2a, valid_anchor_t2a
                )
                logits_a2t_full = sim_a2t_full - self.sigmoid_bias.float()                   # [B, N+S]
                loss_a2t = compute_single_direction_sigmoid(
                    logits_a2t_full, pos_a2t_full, valid_a2t_2d_full, valid_anchor_a2t
                )
            else:
                # InfoNCE: valid_col_mask 是 [1, N]
                valid_col_t2a = valid_mask.unsqueeze(0)                                      # [1, N]
                valid_col_a2t_full = torch.cat([valid_mask, torch.ones(S, device=valid_mask.device, dtype=valid_mask.dtype)], dim=0).unsqueeze(0)  # [1, N+S]
                loss_t2a = compute_single_direction_infonce(
                    similarity[:batch_size, :].clone(), positive_mask[:batch_size, :], valid_col_t2a, valid_anchor_t2a
                )
                loss_a2t = compute_single_direction_infonce(
                    sim_a2t_full.clone(), pos_a2t_full, valid_col_a2t_full, valid_anchor_a2t
                )

            loss = (loss_t2a + loss_a2t) / 2.0

            # 统计 (沿用 _compute_*_loss 中的 batch-only 统计)
            pos_mask_batch = positive_mask[:batch_size, :batch_size]
            valid_2d_batch = valid_mask_batch.unsqueeze(1) * valid_mask_batch.unsqueeze(0)
            valid_pos_mask = (pos_mask_batch > 0) & (valid_2d_batch > 0)
            valid_neg_mask = (pos_mask_batch == 0) & (valid_2d_batch > 0)
            sim_batch_only = similarity[:batch_size, :batch_size]
            pos_logits = sim_batch_only[valid_pos_mask] - (self.sigmoid_bias.float() if self.loss_type == 'sigmoid' else 0.0)
            neg_logits = sim_batch_only[valid_neg_mask] - (self.sigmoid_bias.float() if self.loss_type == 'sigmoid' else 0.0)
            pos_mean_logit = pos_logits.mean() if len(pos_logits) > 0 else torch.tensor(0.0, device=similarity.device)
            neg_mean_logit = neg_logits.mean() if len(neg_logits) > 0 else torch.tensor(0.0, device=similarity.device)
            num_pos_per_row = (pos_mask_batch * valid_2d_batch).sum(dim=1)
            num_neg_per_row = ((1 - pos_mask_batch) * valid_2d_batch).sum(dim=1)
            num_valid_rows = (valid_mask_batch > 0).sum().clamp(min=1)
            stats = {
                'pos_mean_logit': pos_mean_logit.detach(),
                'neg_mean_logit': neg_mean_logit.detach(),
                'avg_pos_samples': (num_pos_per_row.sum() / num_valid_rows).detach(),
                'avg_neg_samples': (num_neg_per_row.sum() / num_valid_rows).detach(),
                'avg_total_samples': ((num_pos_per_row + num_neg_per_row).sum() / num_valid_rows).detach(),
                # anchor 诊断
                'anchor_S': torch.tensor(float(S), device=similarity.device).detach(),
                'anchor_pos_count': anchor_extra_target.sum().detach() / batch_size,
                'anchor_neg_count': (S - anchor_extra_target.sum() / batch_size),
            }

        # 添加到bank（包含ids）
        if self.use_feature_bank and self.feature_banks is not None:
            self.feature_banks[layer_idx].add(text_features, action_features, current_ids)

        return loss, stats
    
    def _compute_token_contrastive_loss(
        self,
        action_features: torch.Tensor,  # [B, D]
        text_tokens: torch.Tensor,       # [B, T, D]
        text_mask: Optional[torch.Tensor],
        temperature: float,
        layer_idx: int,
        current_ids: torch.Tensor,  # [B] 当前batch的ids
        alignment_mode_for_mask: str,  # 对齐模式
    ) -> Tuple[torch.Tensor, Dict]:
        """计算token级别对比学习loss（XClip风格），支持 Feature Bank + 多正样本 + 多卡 allgather"""
        # 多卡 merge：global action + token-level text + mask + ids 一并 gather
        action_features, text_tokens, text_mask, current_ids = self._maybe_dist_gather(
            action_features, text_tokens, text_mask, current_ids
        )
        batch_size = action_features.shape[0]
        num_tokens = text_tokens.shape[1]
        
        # Feature Bank: 扩展负样本，同时获取bank中的ids
        if self.use_feature_bank:
            self._ensure_token_feature_banks(num_tokens)
            bank_tokens, bank_masks, bank_actions, bank_ids = self.token_feature_banks[layer_idx].get_all()
            if len(self.token_feature_banks[layer_idx]) > 0:
                text_all = torch.cat([text_tokens, bank_tokens], dim=0)
                action_all = torch.cat([action_features, bank_actions], dim=0)
                ids_all = torch.cat([current_ids, bank_ids], dim=0)
                if text_mask is not None:
                    mask_all = torch.cat([text_mask, bank_masks], dim=0)
                else:
                    mask_all = torch.cat([
                        torch.ones(batch_size, num_tokens, device=text_tokens.device, dtype=torch.long),
                        bank_masks
                    ], dim=0)
            else:
                text_all, action_all, mask_all = text_tokens, action_features, text_mask
                ids_all = current_ids
        else:
            text_all, action_all, mask_all = text_tokens, action_features, text_mask
            ids_all = current_ids
        
        # 使用扩展后的ids重新生成masks
        positive_mask, valid_mask, _ = create_alignment_masks(ids_all, alignment_mode_for_mask)
        
        # L2归一化
        action_norm = F.normalize(action_all, p=2, dim=-1)
        text_norm = F.normalize(text_all, p=2, dim=-1)
        
        # XClip风格相似度
        sim_matrix = torch.matmul(text_norm, action_norm.t()).permute(2, 0, 1)
        if mask_all is not None:
            mask_exp = mask_all.unsqueeze(0).expand(sim_matrix.shape[0], -1, -1)
            sim_matrix = sim_matrix.masked_fill(mask_exp == 0, -1e9)
        
        weights = torch.softmax(sim_matrix / temperature, dim=-1)
        logits = torch.sum(weights * sim_matrix, dim=-1)
        
        # 统一使用除法应用温度，无论是 infonce 还是 sigmoid
        similarity = logits / self._get_temperature()
        
        # 传递positive_mask和valid_mask到loss计算
        loss, stats = self._compute_bidirectional_contrastive_loss(
            similarity, batch_size, action_features.device, positive_mask, valid_mask
        )
        
        # 添加到bank（包含ids）
        if self.use_feature_bank and self.token_feature_banks is not None:
            self.token_feature_banks[layer_idx].add(text_tokens, action_features, text_mask, current_ids)
        
        return loss, stats
    
    def reset_feature_bank(self):
        """重置feature bank（在每个optimizer.step()后调用）"""
        if self.use_feature_bank:
            if self.feature_banks:
                for fb in self.feature_banks:
                    fb.reset()
            if self.token_feature_banks:
                for tfb in self.token_feature_banks:
                    tfb.reset()
            if self.at2tt_feature_banks:
                for atfb in self.at2tt_feature_banks:
                    atfb.reset()
    
    def infer(self, data: Dict, task_names: str = 'Alignment', ret: Dict = None) -> Dict:
        """Inference method"""
        if ret is None:
            ret = {}
        _, _, output = self.forward(data=data, task_names=task_names, return_embeds=True)
        ret.update(output)
        return ret


def create_alignment_model(config: Dict, dtype: torch.dtype) -> AlignmentModel:
    """Factory function to create alignment model from config."""
    return AlignmentModel(
        egovlpv2_dim=config['egovlpv2_dim'],
        openvla_dim=config['openvla_dim'],
        projection_dim=config['projection_dim'],
        token_egovlpv2_dim=config['token_egovlpv2_dim'],
        video_dim=config.get('video_dim', None),  # video维度，默认None回退到egovlpv2_dim
        temperature=config['temperature'],
        dropout=config['dropout'],
        layer_norm_eps=config['layer_norm_eps'],
        num_selected_layers=len(config['layer_indices']),
        dtype=dtype,
        mode_config=config['mode_config'],
        use_feature_bank=config['use_feature_bank'],
        feature_bank_size=config['feature_bank_size'],
        loss_type=config['loss_type'],
        sigmoid_bias=config['sigmoid_bias'],
        learnable_temperature=config['learnable_temperature'],
        # 兼容用户配置中的 learnbale_bias 拼写，同时允许后续使用标准 learnable_bias。
        learnable_bias=config.get('learnbale_bias', config.get('learnable_bias', config.get('learnbale bias', False))),
        alignment_mode=config.get('alignment_mode', 'diagonal'),
        vla_mode=config.get('vla_mode', 'spatialvla'),
        min_valid_ratio=config.get('min_valid_ratio', 0.0),
        # action token 池化配置（默认 'mean' 保持向后兼容）
        action_pool_mode=config.get('action_pool_mode', 'mean'),
        action_pool_config=config.get('action_pool_config', None),
        # 共享/特有表征分离配置（默认关闭，完全向后兼容）
        use_disentangle=config.get('use_disentangle', False),
        disentangle_config=config.get('disentangle_config', None),
        # text/action 投影层独立配置
        text_proj_config=config.get('text_proj_config', None),
        action_proj_config=config.get('action_proj_config', None),
        # 多卡负样本 merge（allgather）开关，运行时由 trainer 注入 allgather_fn
        use_distributed_negatives=config.get('use_distributed_negatives', False),
        # AnchorBank: LIBERO 40 task × (paraphrase + counterfactual) 软/硬正负样本支持
        anchor_bank_config=config.get('anchor_bank_config', None),
        # 无负样本的 VICReg 对齐配置
        vicreg_config=config.get('vicreg_config', None),
    )
