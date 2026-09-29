"""
表征分析配置文件
定义模型路径、数据路径、选定的task列表、提取参数
"""
import os

# ===================== 代码路径 =====================
# 使用analysis下的SpatialVLA副本（可自由修改，已添加task过滤）
SPATIALVLA_DIR = "/root/data/xuyuan1/Codes/analysis/SpatialVLA"

# ===================== 数据路径 =====================
DATA_ROOT = "/root/data/xuyuan1/dataset"
# RLDS data_root_dir: os.path.join(RLDS_DATA_ROOT, data_mix) → 指向 bridge_orig/1.0.0/
RLDS_DATA_ROOT = "/root/data/xuyuan1/Codes/mirror_neuron/data"
BRIDGE_DATA_DIR = os.path.join(DATA_ROOT, "bridge_orig")
META_DIR = os.path.join(BRIDGE_DATA_DIR, "bridge_orig_lerobot/meta")

# Text embeddings（预计算好的）
EGOHOD_EMB_PATH = os.path.join(DATA_ROOT, "embedding/bridge_text_egohod_proj.npz")
QWEN_EMB_PATH = os.path.join(DATA_ROOT, "embedding/bridge_text_embeddings.npz")

# Task metadata
TASK_RLDS_PATH = os.path.join(META_DIR, "task_rlds.jsonl")
EPISODES_PATH = os.path.join(META_DIR, "episodes.jsonl")
TASKS_WITH_ID_PATH = os.path.join(META_DIR, "tasks_with_id.jsonl")

# ===================== 模型路径 =====================
PRETRAINED_MODEL_PATH = os.path.join(DATA_ROOT, "spatialvla-4b-224-pt/spatialvla-4b-224-pt")

# Raw FT (LoRA adapter)
RAW_FT_ADAPTER_PATH = (
    "/root/data/xuyuan1/Codes/mirror_neuron/vlas/SpatialVLA/outputs/"
    "spatialvla_4b_mimic_align_v2/2026-01-27/"
    "raw_128_19-58-13_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_lr1e-4_bs12_node1_gpu2/"
    "checkpoint-10000"
)

# Cotrain FG (LoRA adapter) — best model
COTRAIN_FG_ADAPTER_PATH = (
    "/root/data/xuyuan1/Codes/mirror_neuron/vlas/SpatialVLA/outputs/"
    "spatialvla_4b_mimic_align_v2/2026-02-10/"
    "cotrain_more_fg_13-59-17_bridge_orig_spatialvla-4b-224-pt_mimic_align_v2_lr1e-4_bs12_node1_gpu2/"
    "checkpoint-16000"
)

MODEL_CONFIGS = {
    "pretrained": {
        "base_model": PRETRAINED_MODEL_PATH,
        "adapter": None,
        "label": "Pretrained",
    },
    "raw_ft": {
        "base_model": PRETRAINED_MODEL_PATH,
        "adapter": RAW_FT_ADAPTER_PATH,
        "label": "Raw FT",
    },
    "cotrain_fg": {
        "base_model": PRETRAINED_MODEL_PATH,
        "adapter": COTRAIN_FG_ADAPTER_PATH,
        "label": "Cotrain+FG",
    },
}

# ===================== 选定的67个分析task =====================
# 原始35个: EgoHOD embedding贪心选择, 最大化inter-task语义距离 (easy mode)
# 新增32个: 语义相近的任务变体, 用于hard mode区分度测试
# 候选条件：episodes >= 50
SELECTED_TASKS = [
    # ===== 原始 35 个 (easy mode — 最大语义距离) =====
    {"task_index": 18,   "text": "sweep into pile",                                      "episodes": 963},
    {"task_index": 79,   "text": "turn faucet front to left",                             "episodes": 97},
    {"task_index": 153,  "text": "unfold the cloth from left to right",                   "episodes": 125},
    {"task_index": 212,  "text": "open fridge",                                           "episodes": 220},
    {"task_index": 241,  "text": "put lid on pot or pan",                                 "episodes": 176},
    {"task_index": 40,   "text": "end effector transition from object to object",         "episodes": 89},
    {"task_index": 555,  "text": "put banana in pot cardboard fence",                     "episodes": 88},
    {"task_index": 653,  "text": "put the ball in the cup",                               "episodes": 57},
    {"task_index": 172,  "text": "put carrot on plate",                                   "episodes": 332},
    {"task_index": 35,   "text": "open oven",                                             "episodes": 130},
    {"task_index": 173,  "text": "take clothes out of laundry machine",                   "episodes": 87},
    {"task_index": 222,  "text": "put knife on cutting board cardboard fence",            "episodes": 88},
    {"task_index": 377,  "text": "topple metal pot cardboard fence",                      "episodes": 88},
    {"task_index": 101,  "text": "put the blue figure on the top edge of the cloth",      "episodes": 129},
    {"task_index": 951,  "text": "pick up glass cup",                                     "episodes": 50},
    {"task_index": 102,  "text": "close the drawer",                                      "episodes": 375},
    {"task_index": 763,  "text": "put fork from basket to tray",                          "episodes": 51},
    {"task_index": 1936, "text": "put pepper in pot or pan",                              "episodes": 88},
    {"task_index": 388,  "text": "upright basil bottle cardboard fence",                  "episodes": 87},
    {"task_index": 304,  "text": "take the eggplant and put it between the two right burners", "episodes": 81},
    {"task_index": 499,  "text": "pour almonds in pot",                                   "episodes": 65},
    {"task_index": 107,  "text": "turn lever vertical to front",                          "episodes": 510},
    {"task_index": 258,  "text": "take sushi out of pot cardboard fence",                 "episodes": 88},
    {"task_index": 161,  "text": "put detergent in sink",                                 "episodes": 88},
    {"task_index": 19,   "text": "put broccoli in pot",                                   "episodes": 179},
    {"task_index": 364,  "text": "put sweet potato in pot which is in sink",              "episodes": 86},
    {"task_index": 39,   "text": "put cup from anywhere into sink",                       "episodes": 134},
    {"task_index": 31,   "text": "close microwave",                                       "episodes": 291},
    {"task_index": 748,  "text": "pick up pan from stove",                                "episodes": 65},
    {"task_index": 528,  "text": "take carrot out of pot cardboard fence",                "episodes": 88},
    {"task_index": 126,  "text": "flip pot upright in sink",                              "episodes": 267},
    {"task_index": 436,  "text": "put potato on plate",                                   "episodes": 87},
    {"task_index": 361,  "text": "put corn in pan which is on stove",                     "episodes": 134},
    {"task_index": 414,  "text": "flip cup upright",                                      "episodes": 140},
    {"task_index": 53,   "text": "put pear in bowl",                                      "episodes": 176},
    # ===== 新增 32 个 (hard mode — 语义相近任务) =====
    # -- fold cloth (4 directions, 极度相近)
    {"task_index": 15,   "text": "fold the cloth from bottom left to top right",          "episodes": 175},
    {"task_index": 483,  "text": "fold the cloth from bottom to top",                     "episodes": 56},
    {"task_index": 205,  "text": "fold the cloth from right to left",                     "episodes": 100},
    {"task_index": 133,  "text": "fold the cloth from top right to bottom left",          "episodes": 100},
    # -- unfold cloth (补充2个方向)
    {"task_index": 997,  "text": "unfold the cloth from bottom left to top right",        "episodes": 86},
    {"task_index": 45,   "text": "unfold the cloth from right to left",                   "episodes": 107},
    # -- put X on plate
    {"task_index": 95,   "text": "put eggplant on plate",                                 "episodes": 66},
    {"task_index": 57,   "text": "put sushi on plate",                                    "episodes": 131},
    # -- put X in pot/pan
    {"task_index": 761,  "text": "put broccoli in pot or pan",                            "episodes": 66},
    {"task_index": 373,  "text": "put eggplant in pot or pan",                            "episodes": 86},
    {"task_index": 823,  "text": "put corn in pot which is in sink",                      "episodes": 86},
    # -- put X in pot cardboard fence
    {"task_index": 284,  "text": "put carrot in pot cardboard fence",                     "episodes": 88},
    {"task_index": 68,   "text": "put knife in pot cardboard fence",                      "episodes": 88},
    {"task_index": 196,  "text": "put potato in pot cardboard fence",                     "episodes": 88},
    {"task_index": 579,  "text": "put sushi in pot cardboard fence",                      "episodes": 88},
    # -- open/close (补充对称操作)
    {"task_index": 288,  "text": "close fridge",                                          "episodes": 176},
    {"task_index": 41,   "text": "open microwave",                                        "episodes": 290},
    {"task_index": 29,   "text": "close oven",                                            "episodes": 131},
    # -- take out / off (更多变体)
    {"task_index": 8,    "text": "take broccoli out of pan",                              "episodes": 220},
    {"task_index": 264,  "text": "take carrot off plate",                                 "episodes": 219},
    {"task_index": 33,   "text": "take lid off pot",                                      "episodes": 88},
    {"task_index": 184,  "text": "take lid off pot or pan",                               "episodes": 88},
    # -- lid (put on vs take off)
    {"task_index": 575,  "text": "put lid on pot",                                        "episodes": 88},
    # -- topple/upright (补充)
    {"task_index": 338,  "text": "topple basil bottle cardboard fence",                   "episodes": 88},
    {"task_index": 283,  "text": "upright metal pot cardboard fence",                     "episodes": 85},
    # -- cup/pot sink
    {"task_index": 183,  "text": "put cup from counter to sink",                          "episodes": 90},
    {"task_index": 77,   "text": "pick up pot from sink",                                 "episodes": 88},
    # -- lever (同义不同表述)
    {"task_index": 175,  "text": "lever vertical to front",                               "episodes": 152},
    # -- end effector
    {"task_index": 11,   "text": "end effector reaching pot or pan",                      "episodes": 81},
    # -- put pot / clothes / knife (方向对操作)
    {"task_index": 607,  "text": "put pot on stove which is near stove",                  "episodes": 81},
    {"task_index": 24,   "text": "put clothes in laundry machine",                        "episodes": 88},
    {"task_index": 224,  "text": "put knife on cutting board",                            "episodes": 100},
]

# Hard mode 语义组定义（用于 hard mode 分析）
HARD_GROUPS = {
    'fold_cloth': [
        'fold the cloth from bottom left to top right',
        'fold the cloth from bottom to top',
        'fold the cloth from right to left',
        'fold the cloth from top right to bottom left',
    ],
    'unfold_cloth': [
        'unfold the cloth from left to right',
        'unfold the cloth from bottom left to top right',
        'unfold the cloth from right to left',
    ],
    'fold_vs_unfold': [
        'fold the cloth from bottom left to top right',
        'fold the cloth from bottom to top',
        'fold the cloth from right to left',
        'fold the cloth from top right to bottom left',
        'unfold the cloth from left to right',
        'unfold the cloth from bottom left to top right',
        'unfold the cloth from right to left',
    ],
    'put_on_plate': [
        'put carrot on plate', 'put potato on plate',
        'put eggplant on plate', 'put sushi on plate',
    ],
    'put_in_pot_pan': [
        'put broccoli in pot', 'put broccoli in pot or pan',
        'put pepper in pot or pan', 'put eggplant in pot or pan',
        'put corn in pan which is on stove', 'put corn in pot which is in sink',
        'put sweet potato in pot which is in sink', 'pour almonds in pot',
    ],
    'put_X_cardboard_fence': [
        'put banana in pot cardboard fence', 'put carrot in pot cardboard fence',
        'put knife in pot cardboard fence', 'put potato in pot cardboard fence',
        'put sushi in pot cardboard fence', 'put knife on cutting board cardboard fence',
    ],
    'open_close': [
        'open fridge', 'close fridge', 'open microwave', 'close microwave',
        'open oven', 'close oven', 'close the drawer',
    ],
    'take_out': [
        'take broccoli out of pan', 'take carrot off plate',
        'take carrot out of pot cardboard fence', 'take sushi out of pot cardboard fence',
        'take clothes out of laundry machine',
        'take lid off pot', 'take lid off pot or pan',
    ],
    'lid_on_off': [
        'put lid on pot', 'put lid on pot or pan',
        'take lid off pot', 'take lid off pot or pan',
    ],
    'topple_vs_upright': [
        'topple basil bottle cardboard fence', 'topple metal pot cardboard fence',
        'upright basil bottle cardboard fence', 'upright metal pot cardboard fence',
    ],
    'cup_pot_sink': [
        'put cup from anywhere into sink', 'put cup from counter to sink',
        'pick up pot from sink', 'flip pot upright in sink',
        'put detergent in sink',
    ],
    'clothes_laundry': [
        'put clothes in laundry machine', 'take clothes out of laundry machine',
    ],
    'knife_ops': [
        'put knife on cutting board', 'put knife on cutting board cardboard fence',
        'put knife in pot cardboard fence',
    ],
}

SELECTED_TASK_INDICES = set(t["task_index"] for t in SELECTED_TASKS)
TASK_INDEX_TO_TEXT = {t["task_index"]: t["text"] for t in SELECTED_TASKS}

# ===================== 构建完整的task过滤集合 =====================
# 一个task_id可能对应多种text写法（如"open drawer"有18种变体）
# 需要用tasks_with_id.jsonl找到每个选定task的task_id，再收集该task_id下所有文本变体
def _build_target_task_langs():
    """根据选定task的text，找到其task_id，再收集同task_id下的所有文本变体"""
    import json
    from collections import defaultdict
    
    # 1. 从 tasks_with_id.jsonl 构建 text->task_id 和 task_id->texts 映射
    #    ★ 全部 lowercase 以消除大小写不一致
    text_to_task_id = {}
    task_id_to_texts = defaultdict(set)
    try:
        with open(TASKS_WITH_ID_PATH) as f:
            for line in f:
                d = json.loads(line)
                text_lower = d["task"].lower()
                text_to_task_id[text_lower] = d["task_id"]
                task_id_to_texts[d["task_id"]].add(text_lower)
    except FileNotFoundError:
        # fallback: 只用精确text
        return set(t["text"] for t in SELECTED_TASKS)
    
    # 2. 对每个选定task，找到其task_id，收集同task_id下的所有text变体
    all_langs = set()
    selected_task_id_map = {}  # task_id -> representative text (用于后续task label)
    for t in SELECTED_TASKS:
        text = t["text"].lower()  # ★ lowercase
        task_id = text_to_task_id.get(text, None)
        if task_id is not None and task_id != -1:
            variants = task_id_to_texts[task_id]
            all_langs.update(variants)
            selected_task_id_map[task_id] = text
        else:
            # 没找到task_id，直接用精确text
            all_langs.add(text)
    
    return all_langs, text_to_task_id, task_id_to_texts

_result = _build_target_task_langs()
if isinstance(_result, tuple):
    TARGET_TASK_LANGS, _TEXT_TO_TASK_ID, _TASK_ID_TO_TEXTS = _result
else:
    TARGET_TASK_LANGS = _result
    _TEXT_TO_TASK_ID = {}
    _TASK_ID_TO_TEXTS = {}

def get_canonical_task_label(lang_text: str) -> str:
    """将任意text变体映射回选定task的代表性文本（用于聚类label）"""
    lang_lower = lang_text.lower().strip()
    task_id = _TEXT_TO_TASK_ID.get(lang_lower, None)
    if task_id is None:
        return lang_text
    # 找到该task_id对应的选定task的代表性文本
    for t in SELECTED_TASKS:
        rep_task_id = _TEXT_TO_TASK_ID.get(t["text"].lower(), None)
        if rep_task_id == task_id:
            return t["text"]
    return lang_text

# ===================== 提取参数 =====================
# VLA层索引 (与alignment config一致)
LAYER_INDICES = [0, 4, 8, 10, 12, 15, 16, 20, 24, 26]

# 每个task采样的轨迹数
TRAJECTORIES_PER_TASK = 50

# ===================== 输出路径 (Train) =====================
OUTPUT_DIR = "/root/data/xuyuan1/Codes/analysis/bridge_representation/outputs"
FEATURE_DIR = os.path.join(OUTPUT_DIR, "features")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# SimplerEnv结果CSV
SIMPLERENV_CSV_DIR = "/root/data/xuyuan1/Codes/INT-ACT/scripts/eval/data_csv"

# 数据子集目录（build_data_subset.py 生成，所有模型共用）
SUBSET_DIR = os.path.join(OUTPUT_DIR, "data_subset")


# ===================== Val 集分析配置 =====================
# Val 集 task 列表 — 5 个语义 Hard Group, 共 84 个 task
# HG1: 叠布/展布方向 (最难: 仅方向不同)
# HG2: 放不同物体到盘子上
# HG3: 放不同物体到锅里
# HG4: 开/关不同物体
# HG5: cardboard fence 场景不同操作
VAL_SELECTED_TASKS = [
    # === HG1: Fold/Unfold Cloth Directions (16 tasks, ~186 traj) ===
    "unfold the cloth from top right to bottom left",
    "fold the cloth from bottom left to top right",
    "fold the cloth from left to right",
    "fold the cloth from bottom right to top left",
    "fold the cloth from top left to bottom right",
    "unfold the cloth from top left to bottom right",
    "unfold the cloth from bottom right to top left",
    "fold the cloth from right to left",
    "unfold the cloth from right to left",
    "fold the cloth from top right to bottom left",
    "unfold the cloth from left to right",
    "fold the cloth from bottom to top",
    "unfold the cloth from bottom left to top right",
    "fold cloth in half",
    "fold the cloth from the bottom to the top",
    "unfold the cloth from bottom to top",
    # === HG2: Put X on Plate (13 tasks, ~143 traj) ===
    "put carrot on plate",
    "put sushi on plate",
    "put potato on plate",
    "put eggplant on plate",
    "put corn on plate",
    "put lemon on plate",
    "put banana on plate",
    "put bowl on plate cardboard fence",
    "put pear on plate",
    "put blueberries on plate sink",
    "put cup on plate",
    "put bowl on plate",
    "put spatula on plate sink",
    # === HG3: Put X in Pot/Pan (top 15, ~169 traj) ===
    "put broccoli in pot",
    "put corn in pan which is on stove",
    "put sweet potato in pot which is in sink",
    "put corn in pot which is in sink",
    "put eggplant in pot or pan",
    "put pepper in pot or pan",
    "put broccoli in pot or pan",
    "put carrot in pot or pan",
    "put green squash in pot or pan",
    "put can in pot",
    "put eggplant into pan",
    "put eggplant into pot or pan",
    "put spatula in pan",
    "put sweet potato in pot",
    "put green squash into pot or pan",
    # === HG4: Open vs Close (核心 8 对, ~398 traj) ===
    # 合并大小写变体: Open/open/opened → 同一 canonical
    "open the drawer",
    "open microwave",
    "open fridge",
    "open oven",
    "open brown1fbox flap",
    "open small4fbox flaps",
    "open drawer of box",
    "open white1fbox flap",
    "open large4fbox flaps",
    "opened the drawer",
    "open low fridge",
    "open cabinet",
    "open book",
    "close the drawer",
    "close microwave",
    "close fridge",
    "close oven",
    "closed the drawer",
    "close large4fbox flaps",
    "close brown1fbox flap",
    "close cabinet",
    "close white1fbox flap",
    "close low fridge",
    "close small4fbox flaps",
    "close drawer of box",
    # === HG5: Cardboard Fence Operations (13 tasks, ~148 traj) ===
    "upright metal pot cardboard fence",
    "put banana in pot cardboard fence",
    "put sushi in pot cardboard fence",
    "put potato in pot cardboard fence",
    "put knife in pot cardboard fence",
    "take carrot out of pot cardboard fence",
    "take sushi out of pot cardboard fence",
    "put carrot in pot cardboard fence",
    "topple metal pot cardboard fence",
    "upright basil bottle cardboard fence",
    "topple basil bottle cardboard fence",
    "topple hot sauce bottle cardboard fence",
    "upright hot sauce bottle cardboard fence",
]

# Val target langs: 所有 task text 的 lowercase (用于 RLDS 过滤)
# 额外加入大写变体, 确保能匹配到 "Open the drawer" / "Close the drawer" 等
VAL_TARGET_TASK_LANGS = set(t.lower() for t in VAL_SELECTED_TASKS)

# Val 的 canonical label 映射: 全部 lowercase, 合并大小写变体
def get_val_canonical_task_label(lang_text: str) -> str:
    """Val 集: canonical label = lowercase text"""
    return lang_text.lower().strip()

# Val hard groups (用于分析, key 和 value 都是 lowercase)
VAL_HARD_GROUPS = {
    'fold_cloth_directions': [t.lower() for t in VAL_SELECTED_TASKS[:16]],
    'put_X_on_plate':        [t.lower() for t in VAL_SELECTED_TASKS[16:29]],
    'put_X_in_pot_pan':      [t.lower() for t in VAL_SELECTED_TASKS[29:44]],
    'open_vs_close':         sorted(set(t.lower() for t in VAL_SELECTED_TASKS[44:69])),
    'cardboard_fence_ops':   [t.lower() for t in VAL_SELECTED_TASKS[69:]],
}

# Val 输出路径 (独立于 train, 不会覆盖已有数据)
VAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "val")
VAL_SUBSET_DIR = os.path.join(VAL_OUTPUT_DIR, "data_subset")
VAL_FEATURE_DIR = os.path.join(VAL_OUTPUT_DIR, "features")
VAL_FIGURE_DIR = os.path.join(VAL_OUTPUT_DIR, "figures")

# Val 每个 task 最多收集多少条轨迹 (val 中部分 task 只有 5-10 条)
VAL_TRAJECTORIES_PER_TASK = 999  # 全部收集
