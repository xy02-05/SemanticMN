"""
Feature Bank for Contrastive Learning with Gradient Accumulation

该模块实现了Feature Bank机制，用于在gradient accumulation期间累积多个micro-batch的特征，
从而增大对比学习的有效batch size和负样本数量。

核心设计：
1. 使用circular buffer存储detached features
2. 支持动态添加和获取features
3. 训练时每个optimizer step后调用reset()清空
4. 采用FIFO循环队列策略

简化版本：移除了reset_mode和replacement_strategy参数，统一使用FIFO+每步reset模式
"""

import torch
import torch.nn as nn


class FeatureBank(nn.Module):
    """
    特征银行：循环队列方式存储detached features用于对比学习
    
    工作原理：
    1. 使用固定大小的buffer (circular buffer / ring buffer)
    2. 每次forward时添加当前batch的detached features
    3. 计算loss时，concat当前features（带梯度）和bank features（无梯度）
    4. 在optimizer.step()后由训练脚本调用reset()清空
    
    Args:
        bank_size: bank的最大容量（存储的feature数量）
        feature_dim: 特征维度
        device: 存储设备
    """
    
    def __init__(self, bank_size: int, feature_dim: int, device='cuda'):
        super().__init__()
        
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        
        # 使用register_buffer注册为模型buffer（会自动转移到正确device，但不计算梯度）
        self.register_buffer('features', torch.zeros(bank_size, feature_dim, device=device))
        self.register_buffer('ids', torch.full((bank_size,), -1, dtype=torch.long, device=device))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer('count', torch.zeros(1, dtype=torch.long, device=device))
        
    @torch.no_grad()
    def add(self, features: torch.Tensor, ids: torch.Tensor = None):
        """添加features到bank（FIFO循环队列）"""
        features = features.detach()
        batch_size = features.shape[0]
        
        if ids is None:
            ids = torch.arange(batch_size, device=features.device) - 1000000
        else:
            ids = ids.detach()
        
        ptr = int(self.ptr.item())
        count = int(self.count.item())
        
        # 批量写入（处理循环队列边界）
        if ptr + batch_size <= self.bank_size:
            self.features[ptr:ptr + batch_size] = features
            self.ids[ptr:ptr + batch_size] = ids
        else:
            first_part = self.bank_size - ptr
            second_part = batch_size - first_part
            self.features[ptr:] = features[:first_part]
            self.features[:second_part] = features[first_part:]
            self.ids[ptr:] = ids[:first_part]
            self.ids[:second_part] = ids[first_part:]
        
        self.ptr[0] = (ptr + batch_size) % self.bank_size
        self.count[0] = min(count + batch_size, self.bank_size)
    
    @torch.no_grad()
    def get_all(self):
        """获取bank中所有有效的features和ids"""
        count = int(self.count.item())
        if count == 0:
            return (
                torch.empty(0, self.feature_dim, device=self.features.device),
                torch.empty(0, dtype=torch.long, device=self.ids.device)
            )
        return self.features[:count].clone(), self.ids[:count].clone()
    
    @torch.no_grad()
    def reset(self):
        """重置bank（在optimizer.step()后调用）"""
        self.ptr.zero_()
        self.count.zero_()
        self.ids.fill_(-1)
    
    def __len__(self):
        return int(self.count.item())
    
    def is_empty(self):
        return self.count.item() == 0
    
    def is_full(self):
        return self.count.item() >= self.bank_size


class TokenFeatureBank(nn.Module):
    """
    Token特征银行：存储 [B, T, D] 形状的token序列，用于token级别对比学习
    
    Args:
        bank_size: bank的最大容量（存储的样本数量）
        max_tokens: token序列的最大长度（固定）
        feature_dim: 每个token的特征维度
        device: 存储设备
    """
    
    def __init__(self, bank_size: int, max_tokens: int, feature_dim: int, device='cuda'):
        super().__init__()
        
        self.bank_size = bank_size
        self.max_tokens = max_tokens
        self.feature_dim = feature_dim
        
        self.register_buffer('features', torch.zeros(bank_size, max_tokens, feature_dim, device=device))
        self.register_buffer('masks', torch.zeros(bank_size, max_tokens, device=device, dtype=torch.long))
        self.register_buffer('ids', torch.full((bank_size,), -1, dtype=torch.long, device=device))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer('count', torch.zeros(1, dtype=torch.long, device=device))
        
    @torch.no_grad()
    def add(self, features: torch.Tensor, masks: torch.Tensor = None, ids: torch.Tensor = None):
        """添加token序列到bank"""
        features = features.detach()
        batch_size, num_tokens, _ = features.shape
        
        assert num_tokens == self.max_tokens, f"Token长度不匹配: 输入{num_tokens}, 预期{self.max_tokens}"
        
        if masks is None:
            masks = torch.ones(batch_size, num_tokens, device=features.device, dtype=torch.long)
        else:
            masks = masks.detach()
        
        if ids is None:
            ids = torch.arange(batch_size, device=features.device) - 1000000
        else:
            ids = ids.detach()
        
        ptr = int(self.ptr.item())
        count = int(self.count.item())
        
        if ptr + batch_size <= self.bank_size:
            self.features[ptr:ptr + batch_size] = features
            self.masks[ptr:ptr + batch_size] = masks
            self.ids[ptr:ptr + batch_size] = ids
        else:
            first_part = self.bank_size - ptr
            second_part = batch_size - first_part
            self.features[ptr:] = features[:first_part]
            self.features[:second_part] = features[first_part:]
            self.masks[ptr:] = masks[:first_part]
            self.masks[:second_part] = masks[first_part:]
            self.ids[ptr:] = ids[:first_part]
            self.ids[:second_part] = ids[first_part:]
        
        self.ptr[0] = (ptr + batch_size) % self.bank_size
        self.count[0] = min(count + batch_size, self.bank_size)
    
    @torch.no_grad()
    def get_all(self):
        """获取bank中所有有效的token序列、mask和ids"""
        count = int(self.count.item())
        if count == 0:
            return (
                torch.empty(0, self.max_tokens, self.feature_dim, device=self.features.device),
                torch.empty(0, self.max_tokens, device=self.masks.device, dtype=torch.long),
                torch.empty(0, dtype=torch.long, device=self.ids.device)
            )
        return self.features[:count].clone(), self.masks[:count].clone(), self.ids[:count].clone()
    
    @torch.no_grad()
    def reset(self):
        self.ptr.zero_()
        self.count.zero_()
        self.ids.fill_(-1)
    
    def __len__(self):
        return int(self.count.item())
    
    def is_empty(self):
        return self.count.item() == 0
    
    def is_full(self):
        return self.count.item() >= self.bank_size


class DualFeatureBank(nn.Module):
    """
    双特征银行：为对比学习的两组features分别维护bank
    
    Args:
        bank_size: bank的最大容量
        feature_dim1: 第一组特征的维度
        feature_dim2: 第二组特征的维度（如果为None，则与feature_dim1相同）
        device: 存储设备
    """
    
    def __init__(self, bank_size: int, feature_dim1: int, feature_dim2: int = None, device='cuda'):
        super().__init__()
        
        if feature_dim2 is None:
            feature_dim2 = feature_dim1
        
        self.bank1 = FeatureBank(bank_size, feature_dim1, device)
        self.bank2 = FeatureBank(bank_size, feature_dim2, device)
        
    def add(self, features1: torch.Tensor, features2: torch.Tensor, ids: torch.Tensor = None):
        """同时添加两组features到各自的bank"""
        self.bank1.add(features1, ids)
        self.bank2.add(features2, ids)
    
    def get_all(self):
        """获取两个bank中的所有features和ids"""
        features1, ids = self.bank1.get_all()
        features2, _ = self.bank2.get_all()
        return features1, features2, ids
    
    def reset(self):
        self.bank1.reset()
        self.bank2.reset()
    
    def __len__(self):
        return len(self.bank1)
    
    def is_empty(self):
        return self.bank1.is_empty() and self.bank2.is_empty()


class DualTokenFeatureBank(nn.Module):
    """
    双Token特征银行：为token对比学习同时维护text tokens [B,T,D] 和 action [B,D]
    
    Args:
        bank_size: bank的最大容量
        max_tokens: text token序列的最大长度
        feature_dim: 特征维度
        device: 存储设备
    """
    
    def __init__(self, bank_size: int, max_tokens: int, feature_dim: int, device='cuda'):
        super().__init__()
        
        self.token_bank = TokenFeatureBank(bank_size, max_tokens, feature_dim, device)
        self.action_bank = FeatureBank(bank_size, feature_dim, device)
        
    def add(self, text_tokens: torch.Tensor, action_features: torch.Tensor, 
            text_masks: torch.Tensor = None, ids: torch.Tensor = None):
        """同时添加text tokens和action features到各自的bank"""
        self.token_bank.add(text_tokens, text_masks, ids)
        self.action_bank.add(action_features, ids)
    
    def get_all(self):
        """获取两个bank中的所有features和ids"""
        bank_text_tokens, bank_text_masks, ids = self.token_bank.get_all()
        bank_actions, _ = self.action_bank.get_all()
        return bank_text_tokens, bank_text_masks, bank_actions, ids
    
    def reset(self):
        self.token_bank.reset()
        self.action_bank.reset()
    
    def __len__(self):
        return len(self.token_bank)
    
    def is_empty(self):
        return self.token_bank.is_empty() and self.action_bank.is_empty()


class AT2TTTokenFeatureBank(nn.Module):
    """
    AT2TT双Token特征银行：存储action tokens和text tokens用于FILIP风格对比学习
    
    Args:
        bank_size: bank的最大容量
        max_action_tokens: action token序列最大长度
        max_text_tokens: text token序列最大长度
        feature_dim: 特征维度
        device: 存储设备
    """
    
    def __init__(self, bank_size: int, max_action_tokens: int, max_text_tokens: int, 
                 feature_dim: int, device='cuda'):
        super().__init__()
        
        self.bank_size = bank_size
        self.max_action_tokens = max_action_tokens
        self.max_text_tokens = max_text_tokens
        self.feature_dim = feature_dim
        
        self.register_buffer('action_tokens', torch.zeros(bank_size, max_action_tokens, feature_dim, device=device))
        self.register_buffer('text_tokens', torch.zeros(bank_size, max_text_tokens, feature_dim, device=device))
        self.register_buffer('text_masks', torch.zeros(bank_size, max_text_tokens, device=device, dtype=torch.long))
        self.register_buffer('ids', torch.full((bank_size,), -1, dtype=torch.long, device=device))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer('count', torch.zeros(1, dtype=torch.long, device=device))
        
    @torch.no_grad()
    def add(self, action_tokens: torch.Tensor, text_tokens: torch.Tensor,
            text_masks: torch.Tensor = None, ids: torch.Tensor = None):
        """添加action tokens和text tokens到bank"""
        action_tokens = action_tokens.detach()
        text_tokens = text_tokens.detach()
        batch_size = action_tokens.shape[0]
        
        if text_masks is None:
            text_masks = torch.ones(batch_size, self.max_text_tokens, 
                                    device=text_tokens.device, dtype=torch.long)
        else:
            text_masks = text_masks.detach()
        
        if ids is None:
            ids = torch.arange(batch_size, device=action_tokens.device) - 1000000
        else:
            ids = ids.detach()
        
        ptr = int(self.ptr.item())
        count = int(self.count.item())
        
        if ptr + batch_size <= self.bank_size:
            self.action_tokens[ptr:ptr + batch_size] = action_tokens
            self.text_tokens[ptr:ptr + batch_size] = text_tokens
            self.text_masks[ptr:ptr + batch_size] = text_masks
            self.ids[ptr:ptr + batch_size] = ids
        else:
            first_part = self.bank_size - ptr
            second_part = batch_size - first_part
            self.action_tokens[ptr:] = action_tokens[:first_part]
            self.action_tokens[:second_part] = action_tokens[first_part:]
            self.text_tokens[ptr:] = text_tokens[:first_part]
            self.text_tokens[:second_part] = text_tokens[first_part:]
            self.text_masks[ptr:] = text_masks[:first_part]
            self.text_masks[:second_part] = text_masks[first_part:]
            self.ids[ptr:] = ids[:first_part]
            self.ids[:second_part] = ids[first_part:]
        
        self.ptr[0] = (ptr + batch_size) % self.bank_size
        self.count[0] = min(count + batch_size, self.bank_size)
    
    @torch.no_grad()
    def get_all(self):
        """获取bank中所有features和ids"""
        count = int(self.count.item())
        if count == 0:
            return (
                torch.empty(0, self.max_action_tokens, self.feature_dim, device=self.action_tokens.device),
                torch.empty(0, self.max_text_tokens, self.feature_dim, device=self.text_tokens.device),
                torch.empty(0, self.max_text_tokens, device=self.text_masks.device, dtype=torch.long),
                torch.empty(0, dtype=torch.long, device=self.ids.device)
            )
        return (
            self.action_tokens[:count].clone(),
            self.text_tokens[:count].clone(),
            self.text_masks[:count].clone(),
            self.ids[:count].clone()
        )
    
    @torch.no_grad()
    def reset(self):
        self.ptr.zero_()
        self.count.zero_()
        self.ids.fill_(-1)
    
    def __len__(self):
        return int(self.count.item())
    
    def is_empty(self):
        return self.count.item() == 0
    
    def is_full(self):
        return self.count.item() >= self.bank_size
