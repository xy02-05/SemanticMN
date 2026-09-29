"""
Step 0: 基于EgoHOD/Qwen text embedding分析task间语义距离，选定分析用的task。
选择策略: 选语义差异大的task（EgoHOD embedding空间中cosine distance大），
这样alignment后的结构效果更明显。
"""
import numpy as np
import json
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import os

# ========== 路径配置 ==========
EMBEDDING_DIR = "/root/data/xuyuan1/dataset/embedding"
META_DIR = "/root/data/xuyuan1/dataset/bridge_orig/bridge_orig_lerobot/meta"

EGOHOD_PATH = os.path.join(EMBEDDING_DIR, "bridge_text_egohod_proj.npz")
QWEN_PATH = os.path.join(EMBEDDING_DIR, "bridge_text_embeddings.npz")
TASK_RLDS_PATH = os.path.join(META_DIR, "task_rlds.jsonl")
EPISODES_PATH = os.path.join(META_DIR, "episodes.jsonl")

# ========== 加载数据 ==========
print("Loading embeddings...")
egohod_data = np.load(EGOHOD_PATH)
qwen_data = np.load(QWEN_PATH)

# npz文件的key
egohod_keys = list(egohod_data.keys())
qwen_keys = list(qwen_data.keys())
print(f"EgoHOD keys: {egohod_keys}, shape: {egohod_data[egohod_keys[0]].shape}")
print(f"Qwen keys: {qwen_keys}, shape: {qwen_data[qwen_keys[0]].shape}")

egohod_emb = egohod_data[egohod_keys[0]]  # (21938, 512)
qwen_emb = qwen_data[qwen_keys[0]]        # (21938, 4096)

print(f"EgoHOD embedding: {egohod_emb.shape}, dtype={egohod_emb.dtype}")
print(f"Qwen embedding: {qwen_emb.shape}, dtype={qwen_emb.dtype}")

# ========== 加载task元数据 ==========
print("\nLoading task metadata...")
task_index_to_text = {}
with open(TASK_RLDS_PATH) as f:
    for line in f:
        d = json.loads(line)
        task_index_to_text[d['task_index']] = d['task']

print(f"Total task_indices: {len(task_index_to_text)}")

# 统计每个task text对应多少episode
text_to_ep_count = Counter()
text_to_task_index = {}  # text -> first task_index
with open(EPISODES_PATH) as f:
    for line in f:
        ep = json.loads(line)
        for t in ep['tasks']:
            if t:
                text_to_ep_count[t] += 1

# 建立text -> task_index映射
for ti, txt in task_index_to_text.items():
    if txt and txt not in text_to_task_index:
        text_to_task_index[txt] = ti

# ========== 筛选候选task ==========
# 条件: episode数 >= 100
MIN_EPISODES = 100
candidates = {}
for text, count in text_to_ep_count.items():
    if count >= MIN_EPISODES and text in text_to_task_index:
        ti = text_to_task_index[text]
        candidates[text] = {
            'task_index': ti,
            'episode_count': count,
        }

print(f"\nCandidate tasks (>={MIN_EPISODES} episodes): {len(candidates)}")

# ========== 计算候选task间的EgoHOD cosine similarity ==========
candidate_texts = sorted(candidates.keys(), key=lambda x: -candidates[x]['episode_count'])
candidate_indices = [candidates[t]['task_index'] for t in candidate_texts]

# 提取对应的embedding
egohod_subset = egohod_emb[candidate_indices]  # (N_cand, 512)
qwen_subset = qwen_emb[candidate_indices]      # (N_cand, 4096)

# 计算EgoHOD cosine similarity
egohod_sim = cosine_similarity(egohod_subset)  # (N_cand, N_cand)
qwen_sim = cosine_similarity(qwen_subset)

print(f"\n{'='*80}")
print(f"EgoHOD cosine similarity matrix: shape={egohod_sim.shape}")
print(f"  Overall mean: {egohod_sim.mean():.4f}")
print(f"  Off-diagonal mean: {(egohod_sim.sum() - np.trace(egohod_sim)) / (len(egohod_sim)**2 - len(egohod_sim)):.4f}")
print(f"  Min off-diag: {(egohod_sim + np.eye(len(egohod_sim))*10).min():.4f}")
print(f"  Max off-diag: {(egohod_sim - np.eye(len(egohod_sim))*10).max():.4f}")

print(f"\nQwen cosine similarity matrix: shape={qwen_sim.shape}")
print(f"  Overall mean: {qwen_sim.mean():.4f}")
print(f"  Off-diagonal mean: {(qwen_sim.sum() - np.trace(qwen_sim)) / (len(qwen_sim)**2 - len(qwen_sim)):.4f}")

# ========== 打印每个候选task及其EgoHOD特征的平均距离 ==========
print(f"\n{'='*80}")
print("Candidate tasks with EgoHOD avg inter-task distance:")
print(f"{'Task':<60} {'Episodes':>8} {'AvgDist':>8} {'MinSim':>8} {'TaskIdx':>8}")
print("-" * 100)

for i, text in enumerate(candidate_texts):
    # 该task与其他所有候选task的平均cosine similarity
    sims = egohod_sim[i].copy()
    sims[i] = np.nan  # exclude self
    avg_sim = np.nanmean(sims)
    min_sim = np.nanmin(sims)
    avg_dist = 1 - avg_sim  # cosine distance
    
    info = candidates[text]
    print(f"{text:<60} {info['episode_count']:>8d} {avg_dist:>8.4f} {min_sim:>8.4f} {info['task_index']:>8d}")

# ========== 选择task的策略: 最大化inter-task distance ==========
# 贪心选择: 每次选与已选task集合平均距离最大的task
print(f"\n{'='*80}")
print("Greedy task selection (maximize inter-task EgoHOD distance):")
print("=" * 80)

N_SELECT = 15
selected_indices = []
selected_texts = []

# 从episode数最多的task开始
first_idx = 0  # sweep into pile
selected_indices.append(first_idx)
selected_texts.append(candidate_texts[first_idx])

for step in range(N_SELECT - 1):
    best_score = -1
    best_idx = -1
    
    for i in range(len(candidate_texts)):
        if i in selected_indices:
            continue
        # 计算与所有已选task的平均距离
        dists = [1 - egohod_sim[i][j] for j in selected_indices]
        avg_dist = np.mean(dists)
        min_dist = np.min(dists)
        # 用min_dist来确保每对task都有足够距离
        score = min_dist * 0.7 + avg_dist * 0.3
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    selected_indices.append(best_idx)
    selected_texts.append(candidate_texts[best_idx])

print(f"\nSelected {N_SELECT} tasks:")
print(f"{'#':>3} {'Task':<60} {'Episodes':>8} {'TaskIdx':>8}")
print("-" * 85)
for rank, (idx, text) in enumerate(zip(selected_indices, selected_texts)):
    info = candidates[text]
    print(f"{rank+1:>3} {text:<60} {info['episode_count']:>8d} {info['task_index']:>8d}")

# 打印选定task间的EgoHOD相似度矩阵
print(f"\n{'='*80}")
print("EgoHOD cosine similarity between selected tasks:")
sel_sim = egohod_sim[np.ix_(selected_indices, selected_indices)]
# Print as table
print(f"{'':>4}", end="")
for i in range(len(selected_texts)):
    print(f" {i+1:>6}", end="")
print()
for i in range(len(selected_texts)):
    print(f"{i+1:>3}:", end="")
    for j in range(len(selected_texts)):
        print(f" {sel_sim[i,j]:>6.3f}", end="")
    short_name = selected_texts[i][:25]
    print(f"  {short_name}")

# 也做Qwen的
print(f"\nQwen cosine similarity between selected tasks:")
sel_sim_qwen = qwen_sim[np.ix_(selected_indices, selected_indices)]
print(f"{'':>4}", end="")
for i in range(len(selected_texts)):
    print(f" {i+1:>6}", end="")
print()
for i in range(len(selected_texts)):
    print(f"{i+1:>3}:", end="")
    for j in range(len(selected_texts)):
        print(f" {sel_sim_qwen[i,j]:>6.3f}", end="")
    short_name = selected_texts[i][:25]
    print(f"  {short_name}")

# ========== 输出最终选定task信息 ==========
print(f"\n{'='*80}")
print("FINAL SELECTED TASKS (copy to config):")
print("=" * 80)
output = []
for idx, text in zip(selected_indices, selected_texts):
    info = candidates[text]
    entry = {
        'task_index': info['task_index'],
        'task_text': text,
        'episode_count': info['episode_count'],
    }
    output.append(entry)
    print(f"  task_index={info['task_index']:>5d}, episodes={info['episode_count']:>5d}, text=\"{text}\"")

# Save to JSON
output_path = os.path.join(os.path.dirname(__file__), "selected_tasks.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {output_path}")
