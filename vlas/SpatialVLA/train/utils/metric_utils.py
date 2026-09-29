"""
动作预测指标计算工具函数
专为SpatialVLA的动作预测任务设计，计算各类动作准确率和连续动作损失
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def compute_action_metrics(model, outputs, batch, action_tokenizer, accelerator, global_step):
    """
    计算SpatialVLA特有的动作预测指标
    
    功能：
    1. 计算总体动作预测准确率
    2. 分别计算平移、旋转、抓取动作的准确率
    3. 计算连续动作L1损失
    4. 通过accelerator记录指标
    
    Args:
        model: SpatialVLA模型实例（包含action_tokenizer）
        outputs: 模型输出（包含logits）
        batch: 输入批次数据（包含labels和actions）
        action_tokenizer: 动作分词器
        accelerator: accelerate实例，用于记录指标
        global_step: 当前训练步数
    """
    with torch.no_grad():
        # 支持两种访问方式：dict（旧）和 dataclass 属性（新）
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs["logits"]  # (bs, seq, voc)
        labels = batch["labels"]  # (bs, seq)
        shift_logits = logits[..., :-1, :].argmax(-1).contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 创建动作token掩码（检测所有动作token）
        mask = (shift_labels >= action_tokenizer.translation_tokenizer.token_start_idx) & (
            shift_labels <= action_tokenizer.gripper_tokenizer.token_end_idx
        )
        gt_action_ids, pred_action_ids = shift_labels[mask], shift_logits[mask]
        correct_preds = gt_action_ids == pred_action_ids
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # 计算分类动作准确率：平移、旋转、抓取
        token_start_idx, token_end_idx = (
            action_tokenizer.translation_tokenizer.token_start_idx,
            action_tokenizer.translation_tokenizer.token_end_idx,
        )
        translation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        token_start_idx, token_end_idx = (
            action_tokenizer.rotation_tokenizer.token_start_idx,
            action_tokenizer.rotation_tokenizer.token_end_idx,
        )
        rotation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        token_start_idx, token_end_idx = (
            action_tokenizer.gripper_tokenizer.token_start_idx,
            action_tokenizer.gripper_tokenizer.token_end_idx,
        )
        gripper_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        translation_gt_action_ids, translation_pred_action_ids = gt_action_ids[translation_mask], pred_action_ids[translation_mask]
        rotation_gt_action_ids, rotation_pred_action_ids = gt_action_ids[rotation_mask], pred_action_ids[rotation_mask]
        gripper_gt_action_ids, gripper_pred_action_ids = gt_action_ids[gripper_mask], pred_action_ids[gripper_mask]

        translation_correct_preds = translation_gt_action_ids == translation_pred_action_ids
        rotation_correct_preds = rotation_gt_action_ids == rotation_pred_action_ids
        gripper_correct_preds = gripper_gt_action_ids == gripper_pred_action_ids

        translation_action_accuracy = translation_correct_preds.sum().float() / translation_mask.sum().float()
        rotation_action_accuracy = rotation_correct_preds.sum().float() / rotation_mask.sum().float()
        gripper_action_accuracy = gripper_correct_preds.sum().float() / gripper_mask.sum().float()

        # 计算连续动作L1损失
        gt_actions = batch["actions"].reshape(-1, 7).to(device="cpu", dtype=torch.float32)
        pred_actions = action_tokenizer.decode_token_ids_to_actions(pred_action_ids.cpu().numpy().reshape(-1, 3))
        l1_loss = nn.functional.l1_loss(torch.tensor(pred_actions), torch.tensor(gt_actions))

        metric = {
            "train/accuracy": action_accuracy.item(),
            "train/translation_accuracy": translation_action_accuracy.item(),
            "train/rotation_accuracy": rotation_action_accuracy.item(), 
            "train/gripper_accuracy": gripper_action_accuracy.item(),
            "train/l1_loss": l1_loss.item(),
        }

        return metric


def evaluate_model(model, eval_dataloader, accelerator, egovlpv2_dataloader=None):
    """
    运行模型验证循环，计算验证损失和指标
    
    功能：
    1. 将模型设置为验证模式（model.eval()）
    2. 遍历验证数据集，计算各批次损失
    3. 支持MIMIC-VLA多模型验证（SpatialVLA + EgoVLPv2 + Alignment）
    4. 返回平均验证损失和详细指标
    5. 使用accelerator进行分布式验证统计
    
    Args:
        model: MIMIC-VLA模型实例
        eval_dataloader: 验证数据加载器
        accelerator: accelerate实例，用于分布式验证
        egovlpv2_dataloader: EgoVLPv2验证数据加载器（可选）
        
    Returns:
        dict: 包含验证损失和指标的字典
    """
    model.eval()
    total_eval_loss = 0.0
    spatial_vla_loss_total = 0.0
    egovlpv2_loss_total = 0.0
    alignment_loss_total = 0.0
    eval_steps = 0
    
    # 创建EgoVLPv2数据迭代器（如果提供）
    egovlpv2_iter = iter(egovlpv2_dataloader) if egovlpv2_dataloader else None
    
    accelerator.print(f"📊 开始验证，共 {len(eval_dataloader)} 个批次")
    
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            # 获取EgoVLPv2批次数据（如果可用）
            egovlpv2_inputs = None
            if egovlpv2_iter is not None:
                try:
                    egovlpv2_inputs = next(egovlpv2_iter)
                except StopIteration:
                    # 重新创建迭代器实现无限循环
                    egovlpv2_iter = iter(egovlpv2_dataloader)
                    egovlpv2_inputs = next(egovlpv2_iter)
            
            # 前向传播：集成三个模型的计算
            # 验证时也需要使用autocast确保混合精度一致性
            with accelerator.autocast():
                outputs = model(
                    egovlpv2_inputs=egovlpv2_inputs,
                    **batch
                )
            
            loss = outputs.loss
            
            # 统计损失
            total_eval_loss += loss.item()
            if hasattr(outputs, 'spatial_vla_loss'):
                spatial_vla_loss_total += outputs.spatial_vla_loss.item()
            if hasattr(outputs, 'egovlpv2_loss'):
                egovlpv2_loss_total += outputs.egovlpv2_loss.item()
            if hasattr(outputs, 'alignment_loss'):
                alignment_loss_total += outputs.alignment_loss.item()
            
            eval_steps += 1
    
    # 计算平均损失
    avg_eval_loss = total_eval_loss / eval_steps
    avg_spatial_loss = spatial_vla_loss_total / eval_steps
    avg_egovlpv2_loss = egovlpv2_loss_total / eval_steps
    avg_alignment_loss = alignment_loss_total / eval_steps
    
    # 使用accelerator进行跨进程平均
    avg_eval_loss = accelerator.gather_for_metrics(torch.tensor(avg_eval_loss, device=accelerator.device)).mean().item()
    avg_spatial_loss = accelerator.gather_for_metrics(torch.tensor(avg_spatial_loss, device=accelerator.device)).mean().item()
    avg_egovlpv2_loss = accelerator.gather_for_metrics(torch.tensor(avg_egovlpv2_loss, device=accelerator.device)).mean().item()
    avg_alignment_loss = accelerator.gather_for_metrics(torch.tensor(avg_alignment_loss, device=accelerator.device)).mean().item()
    
    # 输出验证结果
    accelerator.print(f"📊 验证结果:")
    accelerator.print(f"  总损失: {avg_eval_loss:.4f}")
    accelerator.print(f"  SpatialVLA损失: {avg_spatial_loss:.4f}")
    accelerator.print(f"  EgoVLPv2损失: {avg_egovlpv2_loss:.4f}")
    accelerator.print(f"  Alignment损失: {avg_alignment_loss:.4f}")
    
    # 恢复训练模式
    model.train()
    
    return {
        "eval_loss": avg_eval_loss,
        "spatial_vla_loss": avg_spatial_loss,
        "egovlpv2_loss": avg_egovlpv2_loss,
        "alignment_loss": avg_alignment_loss,
        "eval_steps": eval_steps,
    }