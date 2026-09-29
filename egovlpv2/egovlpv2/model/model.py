# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb

import timm
import torch
import yaml
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange, repeat

from egovlpv2.base import BaseModel
from egovlpv2.model import video_transformer
from egovlpv2.model.video_transformer import SpaceTimeTransformer
from egovlpv2.utils.util import state_dict_data_parallel_fix

from egovlpv2.model import roberta
from egovlpv2.model.roberta import RobertaModel, _prepare_decoder_attention_mask
from egovlpv2.model import heads
from egovlpv2.model.feature_bank import DualFeatureBank
from transformers import RobertaConfig
from functools import partial
import copy
import torch.distributed as dist

import os
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'EgoNCE_MLM_ITM_Config.yml')
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class FrozenInTime(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=4096,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='bilinear',
                 config = config,
                 task_names = 'EgoNCE_ITM_MLM',
                 norm_layer = None,
                 embed_dim=768,
                 use_feature_bank=False,
                 feature_bank_size=1024):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        self.config = config
        self.task_names = task_names
        
        # Feature Bank参数
        self.use_feature_bank = use_feature_bank
        self.feature_bank_size = feature_bank_size
        self.feature_bank = None
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        if self.text_params['model'].startswith('roberta'):
            # 使用相对路径加载本地roberta-base模型
            roberta_path = os.path.join(os.path.dirname(__file__), '..', '..', 'pretrain_weight', 'roberta-base')
            self.text_model = RobertaModel.from_pretrained(roberta_path)
        self.text_model.train()

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            self.num_frames = video_params['num_frames']
            time_init = 'zeros'
            attention_style = 'frozen-in-time'
            arch_config = 'base_patch16_224'
            vit_init = 'imagenet-21k'
            if arch_config == 'base_patch16_224':
                # 使用相对路径加载权重文件
                weight_path = os.path.join(os.path.dirname(__file__), '..', '..', 'pretrain_weight', 'jx_vit_base_p16_224-80ecf9dd.pth')
                vit_model = torch.load(weight_path, map_location="cpu")
                model = SpaceTimeTransformer(num_frames=self.num_frames,
                                            time_init=time_init,
                                            attention_style=attention_style)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
           
            if load_checkpoint in ["", None]:
                vit_checkpoint = vit_model
                new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
                model.load_state_dict(new_vit_dict, strict=False)
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':

            txt_proj = nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, projection_dim, bias=False),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
            )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim, bias=False),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
            )

        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if ('MLM' in self.task_names or 'ITM' in self.task_names):
            # for FIBER-like cross-attention

            bert_config = RobertaConfig(
                vocab_size=self.config["vocab_size"],
                hidden_size=self.config["hidden_size"],
                num_hidden_layers=self.config["num_layers"],
                num_attention_heads=self.config["num_heads"],
                intermediate_size=self.config["hidden_size"] * self.config["mlp_ratio"],
                #max_position_embeddings=maxlen, [was used in BTGOT script]
                hidden_dropout_prob=self.config["drop_rate"],
                attention_probs_dropout_prob=self.config["drop_rate"],
            )

            self.num_fuse_block=self.config["num_fuse_block"]
            self.num_text_layer=self.config["num_layers"]
            roberta.NUM_FUSE_BLOCK = self.video_model.NUM_FUSE_BLOCK=self.num_fuse_block
            roberta.DIM_IMG=self.config["input_image_embed_size"]
            self.video_model.DIM_TXT=self.config["input_text_embed_size"]

            self.cross_modal_text_transform = nn.Linear(self.config["input_text_embed_size"], self.config["hidden_size"])
            self.cross_modal_text_transform.apply(init_weights)
            self.cross_modal_video_transform = nn.Linear(self.config["input_image_embed_size"], self.config["hidden_size"])
            self.cross_modal_video_transform.apply(init_weights)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.num_patches = self.video_model.patch_embed.num_patches
            self.patches_per_frame = self.num_patches//self.num_frames
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(embed_dim)
            self.pre_logits = nn.Identity()


            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.cross_modal_video_pooler = heads.Pooler(self.config["hidden_size"])
            self.cross_modal_video_pooler.apply(init_weights)
            self.cross_modal_text_pooler = heads.Pooler(self.config["hidden_size"])
            self.cross_modal_text_pooler.apply(init_weights)

            ## einops transformations
            self.einops_from_space = 'b (f n) d'
            self.einops_to_space = '(b f) n d'
            self.einops_from_time = 'b (f n) d'
            self.einops_to_time = '(b n) f d'

        if 'MLM' in self.task_names:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(init_weights)

        if 'ITM' in self.task_names:
            self.itm_score = heads.ITMHead(self.config["hidden_size"] * 2)
            self.itm_score.apply(init_weights)

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint, map_location='cpu', weights_only=False)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=False)
        
        # ========== 初始化Feature Bank（如果启用） ==========
        if self.use_feature_bank:
            # EgoVLPv2的对比学习使用video_embeds和text_embeds
            # 两者都是projection_dim维度
            self.feature_bank = DualFeatureBank(
                bank_size=feature_bank_size,
                feature_dim1=projection_dim,
                feature_dim2=projection_dim
            )
            print(f"✅ EgoVLPv2 Feature Bank已启用:")
            print(f"   - Bank容量: {feature_bank_size}")
            print(f"   - 特征维度: {projection_dim}")
            print(f"   - 预期增大对比学习视野: {feature_bank_size}个额外负样本")

    def set_device(self, device):
        self.device = device

    def infer(self, data, video_only=False, return_embeds=True, task_names=None, ret={}):
        
        text_data = data['text']
        video_data = data['video']

        if task_names is not None:
            self.task_names = task_names


        if 'EgoNCE' in self.task_names: 

            text_embeddings = self.compute_text(text_data)
            video_embeddings = self.compute_video(video_data)


            if return_embeds:
                ret.update({'text_embeds':text_embeddings,
                'video_embeds':video_embeddings
                })

        if 'ITM' in self.task_names:

            b, curr_frames, channels, _, _ = video_data.shape
            video_data_itm = self.video_model.patch_embed(video_data)
            video_data_itm = video_data_itm.flatten(2).transpose(2, 1)
            video_data_itm = video_data_itm.reshape(b, -1, self.video_model.patch_embed.embed_dim)

            BF = video_data_itm.shape[0]
            cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            video_data_itm = torch.cat((cls_tokens, video_data_itm), dim=1)
            # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
            cls_embed = self.video_model.pos_embed[:, 0, :].unsqueeze(1)
            tile_pos_embed = self.video_model.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
            # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            tile_temporal_embed = self.video_model.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            total_pos_embed = tile_pos_embed + tile_temporal_embed
            total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

            n = self.patches_per_frame
            f = curr_frames

            curr_patches = video_data_itm.shape[1]
            video_data_itm = video_data_itm + total_pos_embed[:, :curr_patches]
            video_data_itm = self.video_model.pos_drop(video_data_itm)

            unfused_blocks = self.num_text_layer - self.num_fuse_block

            

            for blk_i, blk in enumerate(self.video_model.blocks[:unfused_blocks]):
                if self.config['use_checkpoint']:
                    video_data_itm = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                n, f)
                else:
                    video_data_itm = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                time_n=n, space_f=f)
                            
            
            text_embeds = self.text_model.embeddings(input_ids=text_data['input_ids']) # before it was input_ids=text_ids
            device = text_embeds.device
            text_masks = text_data['attention_mask']
            input_shape = text_masks.size()
            extend_text_masks = self.text_model.get_extended_attention_mask(text_masks, input_shape, device)
            for layer_i, layer in enumerate(self.text_model.encoder.layer[:unfused_blocks]):
                if self.config['use_checkpoint']:
                    text_embeds = torch.utils.checkpoint.checkpoint(layer, text_embeds, extend_text_masks)[0]
                else:
                    text_embeds = layer(text_embeds, extend_text_masks)[0]


            for blk_i, blk in enumerate(self.video_model.blocks[unfused_blocks:self.num_text_layer]):
                if self.config['use_checkpoint']:
                    fuse_video_data = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
                                          n, f, text_embeds, extend_text_masks)
                    text_embeds = torch.utils.checkpoint.checkpoint(self.text_model.encoder.layer[blk_i + unfused_blocks],
                                          text_embeds, extend_text_masks, None, (video_data_itm), None, None, False, True)[0]
                else:
                    fuse_video_data = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
                                          y=text_embeds, y_mask=extend_text_masks, time_n=n, space_f=f)
                    text_embeds = self.text_model.encoder.layer[blk_i + unfused_blocks](text_embeds, extend_text_masks, encoder_hidden_states=(video_data_itm), last_norm=True)[0]
                video_data_itm = fuse_video_data

            
            #print("Shape of model output", video_data.shape)
            video_data_itm = self.norm(video_data_itm)[:, 0]
            video_data_itm = self.pre_logits(video_data_itm)

            text_embeds = text_embeds[:, 0]
            text_embeds = self.cross_modal_text_transform(text_embeds)
            video_embeds = self.cross_modal_video_transform(video_data_itm)

            cls_feats_text = self.cross_modal_text_pooler(text_embeds)
            
            cls_feats_video = self.cross_modal_video_pooler(video_embeds)

            cls_feats = torch.cat([cls_feats_text, cls_feats_video], dim=-1)

            ret.update({
                "cross_attn_itm_logits": self.itm_score(cls_feats)
            })


        if 'MLM' in self.task_names:
            
            b, curr_frames, channels, _, _ = video_data.shape
            video_data_mlm = self.video_model.patch_embed(video_data)
            video_data_mlm = video_data_mlm.flatten(2).transpose(2, 1)
            video_data_mlm = video_data_mlm.reshape(b, -1, self.video_model.patch_embed.embed_dim)

            BF = video_data_mlm.shape[0]
            cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            video_data_mlm = torch.cat((cls_tokens, video_data_mlm), dim=1)
            # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
            cls_embed = self.video_model.pos_embed[:, 0, :].unsqueeze(1)
            tile_pos_embed = self.video_model.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
            # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            tile_temporal_embed = self.video_model.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            total_pos_embed = tile_pos_embed + tile_temporal_embed
            total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

            #print("total_pos_embed shape: ", total_pos_embed.shape)

            n = self.patches_per_frame
            f = curr_frames

            curr_patches = video_data_mlm.shape[1]
            video_data_mlm = video_data_mlm + total_pos_embed[:, :curr_patches]
            video_data_mlm = self.video_model.pos_drop(video_data_mlm)

            #print("video_data_mlm shape: ", video_data_mlm.shape)

            unfused_blocks = self.num_text_layer - self.num_fuse_block

            
            for blk_i, blk in enumerate(self.video_model.blocks[:unfused_blocks]):
                if self.config['use_checkpoint']:
                    video_data_mlm = torch.utils.checkpoint.checkpoint(blk, video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                n, f)
                else:
                    video_data_mlm = blk(video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                time_n=n, space_f=f)
                       
            
            text_embeds = self.text_model.embeddings(input_ids=data['text_mlm_ids']) # before it was input_ids=text_ids
            device = text_embeds.device
            text_masks = text_data['attention_mask']
            input_shape = text_masks.size()
            extend_text_masks = self.text_model.get_extended_attention_mask(text_masks, input_shape, device)

            for layer_i, layer in enumerate(self.text_model.encoder.layer[:unfused_blocks]):
                if self.config['use_checkpoint']:
                    text_embeds = torch.utils.checkpoint.checkpoint(layer, text_embeds, extend_text_masks)[0]
                else:
                    text_embeds = layer(text_embeds, extend_text_masks)[0]

            for blk_i, blk in enumerate(self.video_model.blocks[unfused_blocks:self.num_text_layer]):
                if self.config['use_checkpoint']:

                    fuse_video_data = torch.utils.checkpoint.checkpoint(blk, video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                            n, f, text_embeds, extend_text_masks)
                    text_embeds = torch.utils.checkpoint.checkpoint(self.text_model.encoder.layer[blk_i + unfused_blocks],
                                          text_embeds, extend_text_masks, None, (video_data_mlm), None, None, False, True)[0]
                else:
                    fuse_video_data = blk(video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                          y=text_embeds, y_mask=extend_text_masks, time_n=n, space_f=f)
                    text_embeds = self.text_model.encoder.layer[blk_i + unfused_blocks](text_embeds, extend_text_masks, encoder_hidden_states=(video_data_mlm), last_norm=True)[0]
                video_data_mlm = fuse_video_data


            text_embeds = text_embeds #[:, 0]
            text_embeds = self.cross_modal_text_transform(text_embeds)

            ret.update({
                "cross_attn_mlm_logits": self.mlm_score(text_embeds)
            })

        return ret

    
    def forward(self, data, n_embeds, v_embeds, allgather, n_gpu, args, config, loss_egonce, gpu, return_embeds=True, task_names='EgoNCE_ITM_MLM'):

        ret = {}
        loss_dict = {}

        if 'Feature_Extraction' in task_names:
            video_embeddings = self.compute_video(data['video'])
            return video_embeddings


        if 'EgoNCE' in task_names:

            ret = self.infer(data, task_names='EgoNCE')
            video_embeds = ret['video_embeds']  # [B, D]
            text_embeds = ret['text_embeds']    # [B, D]
            
            # AllGather跨GPU收集features
            video_embeds = allgather(video_embeds, n_gpu, args)
            text_embeds = allgather(text_embeds, n_gpu, args)
            n_embeds = allgather(n_embeds, n_gpu, args)
            v_embeds = allgather(v_embeds, n_gpu, args)
            
            # ========== Feature Bank逻辑：累积历史features增大对比学习视野 ==========
            if self.use_feature_bank and self.feature_bank is not None:
                # 1. 将当前batch的features（detached）添加到bank
                self.feature_bank.add(
                    text_embeds.detach(),
                    video_embeds.detach()
                )
                
                # 2. 获取bank中累积的历史features
                bank_text, bank_video = self.feature_bank.get_all()  # [K, D] each
                
                if bank_text is not None and bank_video is not None:
                    # 3. Concat当前features（带梯度）和bank features（无梯度）
                    text_embeds_all = torch.cat([text_embeds, bank_text], dim=0)  # [B+K, D]
                    video_embeds_all = torch.cat([video_embeds, bank_video], dim=0)  # [B+K, D]
                else:
                    # Bank还是空的，只使用当前features
                    text_embeds_all = text_embeds
                    video_embeds_all = video_embeds
            else:
                # 不使用feature bank，直接使用当前features
                text_embeds_all = text_embeds
                video_embeds_all = video_embeds
            
            # 4. 计算similarity matrix（增大了视野）
            output = sim_matrix(text_embeds_all, video_embeds_all)  # [B+K, B+K]
            
            # 5. 计算loss（只对当前batch的样本）
            batch_size = text_embeds.shape[0]
            
            # 问题：EgoNCE loss需要方阵输入（因为它有mask_diag和双向loss）
            # 解决方案：用自定义的对比学习loss，支持利用bank中的负样本
            
            if self.use_feature_bank and bank_text is not None and bank_video is not None:
                # 有feature bank时，使用增强的对比学习loss
                # Text->Video方向：利用所有video candidates (包括bank)
                sim_t2v = output[:batch_size, :]  # [B, B+K]
                # Video->Text方向：利用所有text candidates (包括bank)
                sim_v2t = output[:, :batch_size].t()  # [B, B+K]
                
                # 计算双向的cross-entropy loss
                # 对于当前batch，正样本在对角线（前B个位置）
                labels = torch.arange(batch_size, device=output.device)
                
                # Loss: text query找video target
                loss_t2v = F.cross_entropy(sim_t2v / 0.05, labels)
                # Loss: video query找text target
                loss_v2t = F.cross_entropy(sim_v2t / 0.05, labels)
                
                loss = (loss_t2v + loss_v2t) / 2.0
                
                # 为了兼容性，生成假的mask_bool和temp
                mask_bool = None
                temp = 0.05
            else:
                # 没有feature bank或bank为空时，使用原始的EgoNCE loss
                output_current = output[:batch_size, :batch_size]  # [B, B]
                
                if config['loss']['type'] == 'EgoNCE':
                    sim_v = sim_matrix(v_embeds, v_embeds)
                    sim_n = sim_matrix(n_embeds, n_embeds)
                    loss, mask_bool, temp = loss_egonce(output_current, sim_v, sim_n)
                else:
                    loss, mask_bool, temp = loss_egonce(output_current)

            # 为了兼容性，ret中返回前B×B的子矩阵
            ret.update({"sim_v2t": output[:batch_size, :batch_size], 
                       "sim_t2v": output[:batch_size, :batch_size].t(),})

            loss_dict.update({'EgoNCE': loss})

        
        # MLM
        if 'MLM' in task_names:

            ret = self.infer(data, task_names='MLM', ret=ret)

            mlm_logits = ret["cross_attn_mlm_logits"].view(-1, 50265)
            mlm_labels = data["text_mlm_labels"].view(-1)

            mlm_logits = allgather(mlm_logits, n_gpu, args)
            mlm_labels = allgather(mlm_labels, n_gpu, args)

            loss_mlm = torch.nn.functional.cross_entropy(
                                mlm_logits,
                                mlm_labels,
                                ignore_index=-100,
                                ).mean()

            loss = loss + loss_mlm

            loss_dict.update({"loss_mlm": loss_mlm})


        # ITM
        if 'ITM' in task_names:

            rank = dist.get_rank()

            all_video = allgather(data['video'], n_gpu, args)
            all_text_ids = allgather(data['text']['input_ids'], n_gpu, args)
            all_text_masks = allgather(data['text']['attention_mask'], n_gpu, args)

            pos_len = data['video'].size(0) // 2
            neg_len = data['video'].size(0) - pos_len
            itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).cuda(gpu, non_blocking=True)

            itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

            batch_size = len(itm_labels)

            with torch.no_grad():
                weights_v2t = F.softmax(ret['sim_v2t'][batch_size*rank : batch_size * (rank + 1), :]/temp, dim=1)
                weights_t2v = F.softmax(ret['sim_t2v'][batch_size*rank : batch_size * (rank + 1), :]/temp, dim=1)

                weights_v2t.masked_fill_(mask_bool[batch_size*rank : batch_size * (rank + 1), :], 0)
                weights_t2v.masked_fill_(mask_bool[batch_size*rank : batch_size * (rank + 1), :], 0)

            data_itm = copy.deepcopy(data)

            for idx in range(len(itm_labels)):
                if itm_labels[idx] == 1:
                    data_itm['video'][idx, :] = all_video[rank*batch_size + idx, :]
                    data_itm['text']['input_ids'][idx, :] = all_text_ids[rank*batch_size + idx, :]
                    data_itm['text']['attention_mask'][idx, :] = all_text_masks[rank*batch_size + idx, :]


                else:
                    if np.random.rand() > 0.5:
                        neg_idx = torch.multinomial(weights_t2v[idx] + 1e-9, 1).item()
                        data_itm['video'][idx, :] = all_video[neg_idx, :]
                        data_itm['text']['input_ids'][idx, :] = all_text_ids[rank*batch_size + idx, :]
                        data_itm['text']['attention_mask'][idx, :] = all_text_masks[rank*batch_size + idx, :]
                    else:
                        neg_idx = torch.multinomial(weights_v2t[idx] + 1e-9, 1).item()
                        data_itm['video'][idx, :] = all_video[rank*batch_size + idx, :]
                        data_itm['text']['input_ids'][idx, :] = all_text_ids[neg_idx, :]
                        data_itm['text']['attention_mask'][idx, :] = all_text_masks[neg_idx, :]


            ret = self.infer(data_itm, task_names='ITM', ret=ret)

            itm_logits = ret["cross_attn_itm_logits"]

            itm_logits = allgather(itm_logits, n_gpu, args)
            itm_labels = allgather(itm_labels, n_gpu, args)

            loss_itm = torch.nn.functional.cross_entropy(itm_logits, itm_labels.long()).mean()

            loss = loss + 2*loss_itm

            #print("ITM loss: ", loss_itm)
            loss_dict.update({"loss_itm": loss_itm})

        loss_dict.update({"loss_total": loss})

        return loss, loss_dict, ret



    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        if self.config['use_checkpoint']:
            text_embeddings = torch.utils.checkpoint.checkpoint(self.txt_proj, text_embeddings)
        else:
            text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']    # not implement for bert
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError

        if self.config['use_checkpoint']:
            text_embeddings = torch.utils.checkpoint.checkpoint(self.txt_proj, text_embeddings)
        else:
            text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        if self.config['use_checkpoint']:
            video_embeddings = torch.utils.checkpoint.checkpoint(self.vid_proj, video_embeddings)
        else:
            video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict
    
    def reset_feature_bank(self):
        """
        重置feature bank（在optimizer.step()后调用）
        
        这个方法应该在训练循环中，每次optimizer.step()后调用，
        清空累积的features，开始新的gradient accumulation周期。
        """
        if self.use_feature_bank and self.feature_bank is not None:
            self.feature_bank.reset()

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def sim_matrix_batch_val(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1).unsqueeze(-1), b.norm(dim=-1).unsqueeze(-1)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


if __name__ == "__main__":
    pass
