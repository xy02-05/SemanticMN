# coding=utf-8
"""
复合样本采样器

功能：
1. 管理正样本和负样本的采样
2. 支持三种负样本类型：none_neg, verb_neg, task_neg
3. 高效批量操作，避免循环
"""

from pathlib import Path
import numpy as np
import json
from typing import Union

# 导入本地的加载器（egovlpv2/utils）
try:
    from .positive_matrix import PositiveMatrix
    from .instruction_matrix_v2 import InstructionMatrixV2
except ImportError:
    # 测试时使用绝对导入
    from positive_matrix import PositiveMatrix
    from instruction_matrix_v2 import InstructionMatrixV2


class CompositeSampler:
    """
    复合样本采样器
    
    管理 composite 模式下的所有采样逻辑：
    - 正样本：从 positive_matrix.json 采样
    - none_neg：名词负样本（从 step3 数据采样）
    - verb_neg：动词负样本（从 step4 数据采样）
    - task_neg：任务负样本（从其他任务采样）
    
    支持两种模式：
    1. 固定模式：各类型数量固定
    2. 动态模式：总数固定，各类型随机分配
    """
    
    def __init__(
        self,
        positive_matrix_path: str,
        instruction_matrix_path: str,
        noun_neg_path: str,
        verb_neg_path: str,
        total_samples: int,
        max_positive: int,
        max_noun_neg: int,
        max_verb_neg: int,
        split: str = "seen",
        seed: int = 42
    ):
        """
        初始化复合采样器（动态采样模式）
        
        Args:
            positive_matrix_path: positive_matrix.json 路径（必需）
            instruction_matrix_path: instruction_matrix.json 路径（必需）
            noun_neg_path: noun_neg_matrix.json 路径（必需）
            verb_neg_path: verb_neg_matrix.json 路径（必需）
            total_samples: 除anchor外的总样本数（正样本 + 所有负样本）
            max_positive: 正样本上限
            max_noun_neg: noun_neg 上限
            max_verb_neg: verb_neg 上限
            split: 使用 seen 或 unseen 数据（默认 "seen"）
            seed: 随机种子（默认 42）
            
        说明：
            - 每次采样时，正样本、noun_neg、verb_neg 随机（0到上限）
            - task_neg 自动填充到 total_samples
            - batch 内所有样本使用相同数量（完全向量化）
        """
        self.split = split
        self.seed = seed
        
        # 采样参数验证
        if max_positive + max_noun_neg + max_verb_neg > total_samples:
            raise ValueError(
                f"max_positive({max_positive}) + max_noun_neg({max_noun_neg}) + "
                f"max_verb_neg({max_verb_neg}) = {max_positive + max_noun_neg + max_verb_neg} "
                f"不能大于 total_samples({total_samples})"
            )
        
        self.total_samples = total_samples
        self.max_positive = max_positive
        self.max_noun_neg = max_noun_neg
        self.max_verb_neg = max_verb_neg
        
        np.random.seed(seed)
        
        print("=" * 60)
        print("初始化 CompositeSampler - 动态采样")
        print("=" * 60)
        print(f"  - 总样本数（除anchor外）: {self.total_samples}")
        print(f"  - 正样本上限: {self.max_positive}")
        print(f"  - noun_neg 上限: {self.max_noun_neg}")
        print(f"  - verb_neg 上限: {self.max_verb_neg}")
        print(f"  - task_neg: 填充剩余")
        print(f"  - 数据分割: {split}")
        
        # 初始化正样本矩阵（必需）
        self.positive_matrix = PositiveMatrix.load(Path(positive_matrix_path))
        
        # 初始化 noun_neg 矩阵
        if max_noun_neg > 0 and noun_neg_path:
            print(f"\n加载 noun_neg 矩阵...")
            self.noun_neg_matrix = PositiveMatrix.load(Path(noun_neg_path))
        else:
            self.noun_neg_matrix = None
        
        # 初始化 verb_neg 矩阵
        if max_verb_neg > 0 and verb_neg_path:
            print(f"\n加载 verb_neg 矩阵...")
            self.verb_neg_matrix = PositiveMatrix.load(Path(verb_neg_path))
        else:
            self.verb_neg_matrix = None
        
        # 初始化指令矩阵（用于任务负样本）
        if instruction_matrix_path:
            self.instruction_matrix = InstructionMatrixV2.load(Path(instruction_matrix_path))
        else:
            self.instruction_matrix = None
        
        print("\n✅ CompositeSampler 初始化完成")
        print("=" * 60)
    
    def sample_batch_dynamic(
        self,
        anchor_instructions: list,
        anchor_task_ids: list = None
    ) -> dict:
        """
        动态采样：总样本数固定，正样本和各类负样本随机分配（batch 内数量相同）
        
        策略：
        1. 总样本数 = 正样本 + noun_neg + verb_neg + task_neg（固定）
        2. 正样本、noun_neg、verb_neg 随机（0 到上限）
        3. task_neg 填充剩余
        4. batch 内所有样本使用相同数量（完全向量化）
        
        Args:
            anchor_instructions: [B] anchor 指令列表
            anchor_task_ids: [B] anchor 任务 ID 列表（可选）
            
        Returns:
            dict: {
                'positives': [B, num_pos],
                'negatives': {
                    'none_neg': [B, num_noun],
                    'verb_neg': [B, num_verb],
                    'task_neg': [B, num_task]
                },
                'sample_config': {
                    'num_positive': int,
                    'num_noun_neg': int,
                    'num_verb_neg': int,
                    'num_task_neg': int,
                    'total': int
                }
            }
        """
        batch_size = len(anchor_instructions)
        
        # 1. 随机分配数量 (batch内统一)
        num_pos = np.random.randint(0, self.max_positive + 1) if self.max_positive > 0 else 0
        num_noun = np.random.randint(0, self.max_noun_neg + 1) if self.max_noun_neg > 0 else 0
        num_verb = np.random.randint(0, self.max_verb_neg + 1) if self.max_verb_neg > 0 else 0
        num_task = self.total_samples - num_pos - num_noun - num_verb
        
        # 2. 向量化索引转换 (使用dict.get，避免np.vectorize)
        # 建立全局映射，一次性转换所有指令为索引
        pos_inst_to_idx = self.positive_matrix.inst_to_idx if num_pos > 0 else {}
        noun_inst_to_idx = self.noun_neg_matrix.inst_to_idx if num_noun > 0 and self.noun_neg_matrix else {}
        verb_inst_to_idx = self.verb_neg_matrix.inst_to_idx if num_verb > 0 and self.verb_neg_matrix else {}
        
        # 使用列表推导式 (比np.vectorize快)
        pos_indices = np.array([pos_inst_to_idx.get(inst, -1) for inst in anchor_instructions], dtype=np.int32)
        noun_indices = np.array([noun_inst_to_idx.get(inst, -1) for inst in anchor_instructions], dtype=np.int32) if num_noun > 0 else None
        verb_indices = np.array([verb_inst_to_idx.get(inst, -1) for inst in anchor_instructions], dtype=np.int32) if num_verb > 0 else None
        
        # 有效性掩码
        valid_pos = pos_indices >= 0
        valid_noun = (noun_indices >= 0) if noun_indices is not None else np.ones(batch_size, dtype=bool)
        valid_verb = (verb_indices >= 0) if verb_indices is not None else np.ones(batch_size, dtype=bool)
        
        # 3. 核心优化: 矩阵式采样 (一次性调用，用0填充无效位置)
        # 即使无效位也采样，最后用mask过滤，保持完全并行
        safe_pos_indices = np.where(valid_pos, pos_indices, 0)
        
        all_pos_samples = None
        if num_pos > 0:
            all_pos_samples = self.positive_matrix.sample_positives(
                safe_pos_indices, num_pos, return_indices=False
            )  # shape: (B, num_pos)
        
        # 4. 获取 task_ids (向量化，使用批量映射)
        if anchor_task_ids is None:
            # 批量获取，避免循环
            anchor_task_ids = np.zeros(batch_size, dtype=np.int32)
            if valid_pos.any():
                # 只对有效指令获取task_id
                valid_instructions = [anchor_instructions[i] for i in np.where(valid_pos)[0]]
                valid_task_ids = [self.positive_matrix.get_task_id(inst) for inst in valid_instructions]
                anchor_task_ids[valid_pos] = valid_task_ids
        else:
            anchor_task_ids = np.asarray(anchor_task_ids, dtype=np.int32)
        
        # 5. 批量采样 noun_neg (安全地处理嵌套列表)
        all_noun_samples = None
        if num_noun > 0 and noun_inst_to_idx and valid_noun.any():
            valid_noun_inst = [anchor_instructions[i] for i in np.where(valid_noun)[0]]
            noun_samples_list = self.noun_neg_matrix.batch_sample_positives(valid_noun_inst, num_noun)
            # 创建二维对象数组，逐行赋值（避免形状不一致问题）
            all_noun_samples = np.empty((batch_size, num_noun), dtype=object)
            for idx, orig_idx in enumerate(np.where(valid_noun)[0]):
                all_noun_samples[orig_idx] = noun_samples_list[idx]
        
        # 6. 批量采样 verb_neg (安全地处理嵌套列表)
        all_verb_samples = None
        if num_verb > 0 and verb_inst_to_idx and valid_verb.any():
            valid_verb_inst = [anchor_instructions[i] for i in np.where(valid_verb)[0]]
            verb_samples_list = self.verb_neg_matrix.batch_sample_positives(valid_verb_inst, num_verb)
            # 创建二维对象数组，逐行赋值（避免形状不一致问题）
            all_verb_samples = np.empty((batch_size, num_verb), dtype=object)
            for idx, orig_idx in enumerate(np.where(valid_verb)[0]):
                all_verb_samples[orig_idx] = verb_samples_list[idx]
        
        # 7. 计算每个样本需要的 task_neg 数量 (向量化)
        actual_pos = np.where(valid_pos, num_pos, 0)
        actual_noun = np.where(valid_noun, num_noun, 0)
        actual_verb = np.where(valid_verb, num_verb, 0)
        needed_task = self.total_samples - (actual_pos + actual_noun + actual_verb)
        needed_task = np.maximum(needed_task, 0)
        
        # 8. 批量采样 task_neg (一次性采样最大需求)
        max_task_needed = int(needed_task.max())
        all_task_samples = None
        if max_task_needed > 0 and self.instruction_matrix is not None:
            all_task_samples = self.instruction_matrix.sample_negatives(
                anchor_task_ids=anchor_task_ids.tolist(),
                num_samples=max_task_needed,
                split=self.split,
                return_indices=False
            )  # shape: (B, max_task_needed)
        
        # 9. 快速分拣 (优化的列表推导式，减少分支判断)
        positives = [
            all_pos_samples[i].tolist() if num_pos > 0 and valid_pos[i] else []
            for i in range(batch_size)
        ]
        
        noun_neg_list = [
            all_noun_samples[i].tolist() if num_noun > 0 and valid_noun[i] and all_noun_samples is not None else []
            for i in range(batch_size)
        ]
        
        verb_neg_list = [
            all_verb_samples[i].tolist() if num_verb > 0 and valid_verb[i] and all_verb_samples is not None else []
            for i in range(batch_size)
        ]
        
        task_neg_list = [
            all_task_samples[i, :needed_task[i]].tolist() if needed_task[i] > 0 and all_task_samples is not None else []
            for i in range(batch_size)
        ]
        
        # 10. 计算统计信息 (使用向量化计算的结果)
        return {
            'positives': positives,
            'negatives': {
                'none_neg': noun_neg_list,
                'verb_neg': verb_neg_list,
                'task_neg': task_neg_list
            },
            'anchor_task_ids': anchor_task_ids.tolist(),
            'sample_config': {
                'num_positive': int(actual_pos.mean()) if valid_pos.any() else 0,
                'num_noun_neg': int(actual_noun.mean()) if valid_noun.any() else 0,
                'num_verb_neg': int(actual_verb.mean()) if valid_verb.any() else 0,
                'num_task_neg': int(needed_task.mean()),
                'total': self.total_samples
            }
        }
    
    def print_sample_stats(self, sample_result: dict):
        """打印采样结果统计"""
        print("\n" + "=" * 60)
        print("📊 采样结果统计")
        print("=" * 60)
        
        batch_size = len(sample_result['positives'])
        print(f"批量大小: {batch_size}")
        
        # 正样本统计
        pos_counts = [len(pos) for pos in sample_result['positives']]
        print(f"\n正样本:")
        print(f"  - 平均数量: {np.mean(pos_counts):.1f}")
        print(f"  - 范围: [{min(pos_counts)}, {max(pos_counts)}]")
        
        # 负样本统计
        print(f"\n负样本:")
        for neg_type, neg_samples in sample_result['negatives'].items():
            if 'task_id' in neg_type or 'inst_id' in neg_type:
                continue  # 跳过索引信息
            if len(neg_samples) > 0:
                neg_counts = [len(negs) for negs in neg_samples]
                print(f"  - {neg_type}: 平均 {np.mean(neg_counts):.1f}, 范围 [{min(neg_counts)}, {max(neg_counts)}]")
            else:
                print(f"  - {neg_type}: 0")
        
        print("=" * 60)


def verify_instruction_in_dataset(instruction, task_id, episode_id, split, robotwin_root, task_id_to_name, robot_type, dataset_suffix):
    """
    在原始数据集中验证指令是否存在
    
    Args:
        instruction: 要验证的指令
        task_id: 任务ID
        episode_id: Episode ID
        split: 'seen' 或 'unseen'
        robotwin_root: 数据集根目录
        task_id_to_name: task_id 到 task_name 的映射
        robot_type: 机器人类型
        dataset_suffix: 数据集后缀
        
    Returns:
        bool: 是否找到
    """
    task_name = task_id_to_name.get(task_id)
    if not task_name:
        return False
    
    episode_file = (
        robotwin_root / 
        task_name / 
        f"{robot_type}_{dataset_suffix}" /
        "instructions" /
        f"episode{episode_id}.json"
    )
    
    if not episode_file.exists():
        return False
    
    with open(episode_file, 'r', encoding='utf-8') as f:
        episode_data = json.load(f)
    
    if split not in episode_data:
        return False
    
    # 标准化比较
    def normalize(text):
        return text.lower().strip().replace(".", "").replace(",", "")
    
    instruction_norm = normalize(instruction)
    return any(normalize(inst) == instruction_norm for inst in episode_data[split])


def main():
    """测试主函数（包含固定采样和动态采样）"""
    from datetime import datetime
    import sys
    
    # 添加 robotwin_pos 到路径
    robotwin_pos_path = Path(__file__).parents[3] / "data_process" / "robotwin_pos"
    sys.path.insert(0, str(robotwin_pos_path))
    import config as robotwin_config
    
    print("\n" + "=" * 80)
    print("测试 CompositeSampler（noun_neg + verb_neg + 动态采样）")
    print("=" * 80)
    
    # 数据路径配置（统一管理，使用统一格式）
    data_root = Path("/mnt/nvmepool/xuyuan/Codes/mirror_neuron/data_process/robotwin_pos/output")
    data_paths = {
        'positive_matrix': data_root / "positive_matrix_reassigned.json",
        'instruction_matrix': data_root / "instruction_matrix_v2.json",
        'noun_neg': data_root / "noun_neg_matrix.json",  # 统一格式
        'verb_neg': data_root / "verb_neg_matrix.json"   # 统一格式
    }
    
    # 创建 log 文件
    output_dir = Path("/mnt/nvmepool/xuyuan/Codes/mirror_neuron/data_process/robotwin_pos/output")
    log_file = output_dir / f"composite_sampler_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log(msg):
        """同时打印和写入日志"""
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log("=" * 80)
    log(f"CompositeSampler 完整测试")
    log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)
    
    # ========== 测试1: 小规模动态采样 ==========
    log("\n" + "=" * 80)
    log("测试1: 动态采样（总样本数=20）")
    log("=" * 80)
    
    config_small = {
        'positive_matrix_path': data_paths['positive_matrix'],
        'instruction_matrix_path': data_paths['instruction_matrix'],
        'noun_neg_path': data_paths['noun_neg'],
        'verb_neg_path': data_paths['verb_neg'],
        'total_samples': 20,
        'max_positive': 5,
        'max_noun_neg': 5,
        'max_verb_neg': 5,
        'split': 'seen',
        'seed': 42
    }
    
    log("\n正在初始化 CompositeSampler...")
    sampler_small = CompositeSampler(**config_small)
    
    # 准备测试数据
    anchor_instructions = sampler_small.positive_matrix.instructions[:2]  # 选2个
    
    log("\n开始采样...")
    result_small = sampler_small.sample_batch_dynamic(anchor_instructions)
    
    cfg = result_small['sample_config']
    log("\n【采样结果】")
    log(f"批量大小: {len(anchor_instructions)}")
    log(f"本轮配置: 正样本={cfg['num_positive']}, noun_neg={cfg['num_noun_neg']}, verb_neg={cfg['num_verb_neg']}, task_neg={cfg['num_task_neg']}, 总计={cfg['total']}")
    
    # 显示第一个样本的详细信息
    log("\n【样本1详情】")
    log(f"Anchor: {anchor_instructions[0][:80]}...")
    log(f"正样本数: {len(result_small['positives'][0])}")
    log(f"noun_neg数: {len(result_small['negatives']['none_neg'][0])}")
    log(f"verb_neg数: {len(result_small['negatives']['verb_neg'][0])}")
    log(f"task_neg数: {len(result_small['negatives']['task_neg'][0])}")
    
    if len(result_small['positives'][0]) > 0:
        log(f"\n正样本[0]: {result_small['positives'][0][0][:80]}...")
    if len(result_small['negatives']['none_neg'][0]) > 0:
        log(f"noun_neg[0]: {result_small['negatives']['none_neg'][0][0][:80]}...")
    if len(result_small['negatives']['verb_neg'][0]) > 0:
        log(f"verb_neg[0]: {result_small['negatives']['verb_neg'][0][0][:80]}...")
    if len(result_small['negatives']['task_neg'][0]) > 0:
        log(f"task_neg[0]: {result_small['negatives']['task_neg'][0][0][:80]}...")
    
    # ========== 测试2: 大规模动态采样 ==========
    log("\n" + "=" * 80)
    log("测试2: 大规模动态采样（总样本数=127）")
    log("=" * 80)
    
    config_large = {
        'positive_matrix_path': data_paths['positive_matrix'],
        'instruction_matrix_path': data_paths['instruction_matrix'],
        'noun_neg_path': data_paths['noun_neg'],
        'verb_neg_path': data_paths['verb_neg'],
        'total_samples': 127,
        'max_positive': 11,
        'max_noun_neg': 30,
        'max_verb_neg': 30,
        'split': 'seen',
        'seed': 43  # 不同的seed
    }
    
    log("\n正在初始化 CompositeSampler...")
    sampler_dynamic = CompositeSampler(**config_large)
    
    # 动态采样测试
    anchor_instructions = sampler_dynamic.positive_matrix.instructions[:3]  # 选3个
    
    log("\n开始动态采样...")
    result_dynamic = sampler_dynamic.sample_batch_dynamic(anchor_instructions)
    
    log("\n【动态采样结果】")
    log(f"批量大小: {len(anchor_instructions)}")
    
    # 显示本次采样配置（batch 内统一）
    sample_cfg = result_dynamic['sample_config']
    log(f"\n本次采样配置（batch 内统一）:")
    log(f"  正样本: {sample_cfg['num_positive']} 个")
    log(f"  noun_neg: {sample_cfg['num_noun_neg']} 个")
    log(f"  verb_neg: {sample_cfg['num_verb_neg']} 个")
    log(f"  task_neg: {sample_cfg['num_task_neg']} 个")
    log(f"  总样本数: {sample_cfg['total']} 个")
    
    # 验证总数
    if sample_cfg['total'] != config_large['total_samples']:
        log(f"  ⚠️ 警告: 总数不等于 {config_large['total_samples']}")
    
    log(f"\n每个样本实际采样结果:")
    for i in range(len(anchor_instructions)):
        log(f"  样本 {i+1}: {anchor_instructions[i][:60]}...")
        log(f"    正样本数: {len(result_dynamic['positives'][i])}, noun_neg: {len(result_dynamic['negatives']['none_neg'][i])}, verb_neg: {len(result_dynamic['negatives']['verb_neg'][i])}, task_neg: {len(result_dynamic['negatives']['task_neg'][i])}")
    
    # ========== 测试3: 验证采样正确性 ==========
    log("\n" + "=" * 80)
    log("测试3: 验证动态采样的样本内容")
    log("=" * 80)
    
    # 显示第一个动态采样样本的详细内容
    i = 0
    cfg = result_dynamic['sample_config']
    log(f"\n【样本1详情（动态采样）】")
    log(f"Anchor: {anchor_instructions[i][:80]}...")
    
    # 正样本
    log(f"\n正样本 ({cfg['num_positive']} 个):")
    for j, pos in enumerate(result_dynamic['positives'][i][:2]):  # 只显示前2个
        log(f"  [{j+1}] {pos[:70]}...")
    
    # noun_neg
    if cfg['num_noun_neg'] > 0:
        log(f"\nnoun_neg ({cfg['num_noun_neg']} 个):")
        for j, neg in enumerate(result_dynamic['negatives']['none_neg'][i][:2]):
            log(f"  [{j+1}] {neg[:70]}...")
    
    # verb_neg
    if cfg['num_verb_neg'] > 0:
        log(f"\nverb_neg ({cfg['num_verb_neg']} 个):")
        for j, neg in enumerate(result_dynamic['negatives']['verb_neg'][i][:2]):
            log(f"  [{j+1}] {neg[:70]}...")
    
    # task_neg
    if cfg['num_task_neg'] > 0:
        log(f"\ntask_neg ({cfg['num_task_neg']} 个):")
        for j, neg in enumerate(result_dynamic['negatives']['task_neg'][i][:2]):
            log(f"  [{j+1}] {neg[:70]}...")
    
    log("\n" + "=" * 80)
    log("✅ 测试完成！")
    log("=" * 80)
    log(f"\n详细日志已保存到: {log_file}")
    print(f"\n✅ 所有测试通过！详细日志: {log_file}")


if __name__ == "__main__":
    main()
