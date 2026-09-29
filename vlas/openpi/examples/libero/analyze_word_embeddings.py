"""
分析 LIBERO 任务文本中「相同词」在不同句子中的 Qwen3-VL-Embedding token 表征是否相似。

核心思路：
1. 从 npz 的 input_ids 中定位用户文本 token 的位置（跳过 chat template 的 system prompt 等特殊 token）
2. 利用 tokenizer 将每个 token id 解码回词，并建立 token→word 的映射
3. 对于出现在 ≥2 个句子中的同一个词，提取其 token embedding
4. 计算同词跨句子的 cosine similarity，并给出统计和可视化

输出：
- 同词跨句子相似度统计表
- 相似度热力图（高频词）
- t-SNE 可视化（相同词是否聚在一起）
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. 数据加载
# ============================================================

BASE_DIR = Path("/root/data/xuyuan1")
NPZ_PATH = BASE_DIR / "dataset/embedding/libero_qwen3_text_features.npz"
INDEX_PATH = BASE_DIR / "dataset/embedding/libero_qwen3_text_index.json"
TOKENIZER_PATH = str(BASE_DIR / "dataset/Qwen3-VL-Embedding-8B")
OUTPUT_DIR = BASE_DIR / "Codes/mirror_neuron/vlas/openpi/results/libero/word_embedding_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(NPZ_PATH)
token_embeddings = data["token_embeddings"]    # [N, T_max, D]
attention_mask   = data["attention_mask"]       # [N, T_max]
input_ids        = data["input_ids"]            # [N, T_max]
token_lengths    = data["token_lengths"]        # [N]

with open(INDEX_PATH) as f:
    index_info = json.load(f)

N, T_max, D = token_embeddings.shape
print(f"数据: {N} 个句子, 最大 token 长度 {T_max}, 隐层维度 {D}")


# ============================================================
# 2. 加载 tokenizer，定位用户文本 token
# ============================================================

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Qwen3 chat template 的特殊 token id
IM_START = tokenizer.convert_tokens_to_ids("<|im_start|>")   # 151644
IM_END   = tokenizer.convert_tokens_to_ids("<|im_end|>")     # 151645

def find_user_text_range(ids: np.ndarray, valid_len: int):
    """
    在 input_ids 中定位用户文本 token 的起止位置。
    chat template 结构：
      <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n  TEXT  <|im_end|>\n<|im_start|>assistant\n<eos>
    用户文本就是第二个 <|im_start|> 之后（跳过 "user\n"）到对应 <|im_end|> 之前的部分。
    """
    ids_list = ids[:valid_len].tolist()
    # 找到所有 im_start 的位置
    im_start_positions = [i for i, x in enumerate(ids_list) if x == IM_START]
    # 第二个 im_start 对应 user turn
    user_im_start = im_start_positions[1]
    # user_im_start 后面是 "user" token 和 "\n" token，之后才是真正的文本
    # 具体来说：<|im_start|> + "user"(872) + "\n"(198) + TEXT
    text_start = user_im_start + 3  # 跳过 im_start, "user", "\n"

    # 找到 user turn 对应的 im_end
    user_im_end = ids_list.index(IM_END, user_im_start + 1)
    text_end = user_im_end  # 不包含 im_end

    return text_start, text_end


# ============================================================
# 3. 提取每个句子中每个词的 token 位置和 embedding
# ============================================================

# word_occurrences[word] = [(sent_idx, token_pos, embedding_vec), ...]
word_occurrences = defaultdict(list)

for sent_idx in range(N):
    valid_len = int(token_lengths[sent_idx])
    ids = input_ids[sent_idx]
    start, end = find_user_text_range(ids, valid_len)

    for pos in range(start, end):
        tid = int(ids[pos])
        # 解码单个 token，去除可能的前导空格，得到"词"
        word = tokenizer.decode([tid]).strip().lower()
        if not word or word in ("", " "):
            continue
        emb = token_embeddings[sent_idx, pos]
        word_occurrences[word].append((sent_idx, pos, emb))

# 只保留出现在 >=2 个不同句子中的词
shared_words = {}
for word, occurrences in word_occurrences.items():
    sent_set = set(o[0] for o in occurrences)
    if len(sent_set) >= 2:
        shared_words[word] = occurrences

print(f"\n共 {len(word_occurrences)} 个不同的 token-word")
print(f"出现在 ≥2 个不同句子中的共享词: {len(shared_words)} 个")
print(f"共享词列表: {sorted(shared_words.keys())}")


# ============================================================
# 4. 计算同词跨句子的 cosine similarity
# ============================================================

word_stats = {}
for word, occurrences in sorted(shared_words.items()):
    embs = np.stack([o[2] for o in occurrences])  # [K, D]
    # 计算所有 pair 的 cosine similarity
    sim_matrix = cosine_similarity(embs)
    # 取上三角（排除对角线自身）
    triu_indices = np.triu_indices(len(embs), k=1)
    pairwise_sims = sim_matrix[triu_indices]

    n_sents = len(set(o[0] for o in occurrences))
    word_stats[word] = {
        "n_occurrences": len(occurrences),
        "n_sentences": n_sents,
        "mean_sim": float(pairwise_sims.mean()),
        "std_sim": float(pairwise_sims.std()),
        "min_sim": float(pairwise_sims.min()),
        "max_sim": float(pairwise_sims.max()),
        "embs": embs,
        "sim_matrix": sim_matrix,
        "sent_indices": [o[0] for o in occurrences],
    }

# 按平均相似度排序打印
print("\n" + "=" * 80)
print(f"{'词':<15} {'出现次数':>8} {'所在句数':>8} {'平均sim':>10} {'std':>8} {'min':>8} {'max':>8}")
print("=" * 80)
for word, s in sorted(word_stats.items(), key=lambda x: -x[1]["mean_sim"]):
    print(f"{word:<15} {s['n_occurrences']:>8} {s['n_sentences']:>8} "
          f"{s['mean_sim']:>10.4f} {s['std_sim']:>8.4f} {s['min_sim']:>8.4f} {s['max_sim']:>8.4f}")


# ============================================================
# 5. 可视化 1：高频共享词的平均相似度柱状图
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))
words_sorted = sorted(word_stats.keys(), key=lambda w: -word_stats[w]["mean_sim"])
means = [word_stats[w]["mean_sim"] for w in words_sorted]
stds  = [word_stats[w]["std_sim"] for w in words_sorted]

colors = plt.cm.RdYlGn(np.array(means))
bars = ax.bar(range(len(words_sorted)), means, yerr=stds, capsize=3, color=colors, edgecolor="gray")
ax.set_xticks(range(len(words_sorted)))
ax.set_xticklabels(words_sorted, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Cosine Similarity")
ax.set_title("Same Word Across Different Sentences: Cosine Similarity of Token Embeddings")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "word_similarity_bar.png", dpi=150)
print(f"\n柱状图已保存: {OUTPUT_DIR / 'word_similarity_bar.png'}")


# ============================================================
# 6. 可视化 2：t-SNE 降维，观察相同词是否聚类
# ============================================================

# 选取出现次数最多的 top-K 个共享词做 t-SNE
TOP_K = min(15, len(shared_words))
top_words = sorted(shared_words.keys(), key=lambda w: -len(shared_words[w]))[:TOP_K]

all_embs = []
all_labels = []
for word in top_words:
    for occ in shared_words[word]:
        all_embs.append(occ[2])
        all_labels.append(word)

all_embs = np.stack(all_embs)
print(f"\nt-SNE: {len(all_embs)} 个 token embedding, {TOP_K} 个词类")

tsne = TSNE(n_components=2, perplexity=min(15, len(all_embs) - 1), random_state=42, max_iter=1000)
coords = tsne.fit_transform(all_embs)

fig, ax = plt.subplots(figsize=(12, 10))
unique_words = list(dict.fromkeys(all_labels))
cmap = plt.cm.get_cmap("tab20", len(unique_words))

for i, word in enumerate(unique_words):
    mask = [j for j, l in enumerate(all_labels) if l == word]
    ax.scatter(coords[mask, 0], coords[mask, 1],
               color=cmap(i), label=word, s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
ax.set_title("t-SNE of Token Embeddings (Same Word = Same Color)")
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "word_tsne.png", dpi=150)
print(f"t-SNE 图已保存: {OUTPUT_DIR / 'word_tsne.png'}")


# ============================================================
# 7. 可视化 3：选几个典型词画跨句子相似度热力图
# ============================================================

# 选取一些典型词（出现次数适中、语义上有趣的）
interesting_words = [w for w in ["the", "put", "pick", "plate", "bowl", "on", "and", "in",
                                  "basket", "stove", "cabinet", "black", "white", "up", "mug"]
                     if w in word_stats]

n_interest = len(interesting_words)
if n_interest > 0:
    cols = min(4, n_interest)
    rows = (n_interest + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, word in enumerate(interesting_words):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        s = word_stats[word]
        sim = s["sim_matrix"]
        sent_ids = s["sent_indices"]
        im = ax.imshow(sim, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="equal")
        ax.set_title(f'"{word}" (n={s["n_occurrences"]}, sim={s["mean_sim"]:.3f})', fontsize=9)
        # 标注句子编号
        labels = [f"S{si}" for si in sent_ids]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=45)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隐藏多余的子图
    for idx in range(n_interest, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    plt.suptitle("Cross-Sentence Similarity Heatmaps for Shared Words", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "word_heatmaps.png", dpi=150)
    print(f"热力图已保存: {OUTPUT_DIR / 'word_heatmaps.png'}")


# ============================================================
# 8. 额外分析：同一个词在 libero_spatial 10个句子中的表现
#    （这10个句子结构几乎一样，只有空间描述不同）
# ============================================================

# libero_spatial 对应 index 30-39（从 libero_qwen3_text_index.json 看）
spatial_indices = list(range(30, 40))
spatial_texts = [index_info[i]["text"] for i in spatial_indices]

print("\n\n=== libero_spatial 句子（结构高度相似，仅空间描述不同）===")
for i, idx in enumerate(spatial_indices):
    print(f"  S{idx}: {spatial_texts[i]}")

# 找到在所有 spatial 句子中都出现的共享词
spatial_shared = {}
for word, occurrences in shared_words.items():
    spatial_occ = [o for o in occurrences if o[0] in spatial_indices]
    spatial_sents = set(o[0] for o in spatial_occ)
    if len(spatial_sents) >= 5:
        embs = np.stack([o[2] for o in spatial_occ])
        sim_matrix = cosine_similarity(embs)
        triu = np.triu_indices(len(embs), k=1)
        pairwise = sim_matrix[triu]
        spatial_shared[word] = {
            "n_occ": len(spatial_occ),
            "mean_sim": float(pairwise.mean()),
            "std_sim": float(pairwise.std()),
            "min_sim": float(pairwise.min()),
            "max_sim": float(pairwise.max()),
        }

print(f"\n在 ≥5 个 spatial 句子中出现的词:")
print(f"{'词':<15} {'次数':>6} {'mean_sim':>10} {'std':>8} {'min':>8} {'max':>8}")
for w, s in sorted(spatial_shared.items(), key=lambda x: -x[1]["mean_sim"]):
    print(f"{w:<15} {s['n_occ']:>6} {s['mean_sim']:>10.4f} {s['std_sim']:>8.4f} "
          f"{s['min_sim']:>8.4f} {s['max_sim']:>8.4f}")


# ============================================================
# 9. 总体结论统计
# ============================================================

all_means = [s["mean_sim"] for s in word_stats.values()]
print(f"\n\n=== 总体统计 ===")
print(f"共享词总数: {len(word_stats)}")
print(f"所有共享词平均 cosine similarity: {np.mean(all_means):.4f}")
print(f"标准差: {np.std(all_means):.4f}")
print(f"最高: {max(all_means):.4f}")
print(f"最低: {min(all_means):.4f}")

# 判断：相似度 > 0.9 的词比例
high_sim = sum(1 for m in all_means if m > 0.9)
med_sim  = sum(1 for m in all_means if 0.7 <= m <= 0.9)
low_sim  = sum(1 for m in all_means if m < 0.7)
print(f"\n相似度分布:")
print(f"  高 (>0.9): {high_sim}/{len(all_means)} ({100*high_sim/len(all_means):.1f}%)")
print(f"  中 (0.7-0.9): {med_sim}/{len(all_means)} ({100*med_sim/len(all_means):.1f}%)")
print(f"  低 (<0.7): {low_sim}/{len(all_means)} ({100*low_sim/len(all_means):.1f}%)")

print(f"\n所有图表已保存到: {OUTPUT_DIR}")
