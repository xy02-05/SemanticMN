"""
分析 LIBERO task 文本中【名词/动词/方位词组】在不同句子中的 Qwen3-VL-Embedding 表征相似度。

核心思路：
1. 手工定义三类短语：物体名词、动词短语、方位短语
2. 在每个句子的 token 序列中定位这些短语，用平均池化得到短语级 embedding
3. 同一短语跨句子计算 cosine similarity，分类别统计和可视化
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


# ============================================================
# 配置
# ============================================================

BASE_DIR = Path("/root/data/xuyuan1")
NPZ_PATH = BASE_DIR / "dataset/embedding/libero_qwen3_text_features.npz"
INDEX_PATH = BASE_DIR / "dataset/embedding/libero_qwen3_text_index.json"
TOKENIZER_PATH = str(BASE_DIR / "dataset/Qwen3-VL-Embedding-8B")
OUTPUT_DIR = BASE_DIR / "Codes/mirror_neuron/vlas/openpi/results/libero/word_embedding_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 定义三类关注的短语（按语义角色分类）
# ============================================================

# 物体名词（含组合名词）
OBJECT_NOUNS = [
    "black bowl", "bowl", "plate", "basket", "stove", "cabinet",
    "drawer", "mug", "white mug", "cream cheese", "alphabet soup",
    "tomato sauce", "bbq sauce", "chocolate pudding", "orange juice",
    "ketchup", "milk", "butter", "salad dressing", "wine bottle",
    "cookie box", "moka pot", "book", "caddy", "microwave", "rack",
    "ramekin", "wooden cabinet",
]

# 动词 / 动词短语
VERB_PHRASES = [
    "pick up", "put", "place", "open", "close", "turn on", "push",
]

# 方位 / 空间短语
SPATIAL_PHRASES = [
    "on the plate", "in the basket", "on the stove",
    "next to", "between", "on top of",
    "in the top drawer", "on the ramekin", "on the cookie box",
    "in the bottom drawer", "in the microwave",
    "on the left plate", "on the right plate",
    "to the front of", "from table center",
    "on the wooden cabinet", "on the rack",
]


# ============================================================
# 1. 加载数据
# ============================================================

data = np.load(NPZ_PATH)
token_embeddings = data["token_embeddings"]  # [N, T_max, D]
attention_mask = data["attention_mask"]
input_ids = data["input_ids"]
token_lengths = data["token_lengths"]

with open(INDEX_PATH) as f:
    index_info = json.load(f)

N, T_max, D = token_embeddings.shape
print(f"数据: {N} 个句子, 最大 token 长度 {T_max}, 隐层维度 {D}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

IM_START = tokenizer.convert_tokens_to_ids("<|im_start|>")
IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")


# ============================================================
# 2. 解码每个句子的用户文本 token
# ============================================================

def get_user_tokens(sent_idx):
    """返回 (token_texts, token_positions, embeddings) — 仅用户文本部分"""
    valid_len = int(token_lengths[sent_idx])
    ids = input_ids[sent_idx, :valid_len].tolist()

    # 定位第二个 im_start（user turn）后的文本
    im_starts = [i for i, x in enumerate(ids) if x == IM_START]
    text_start = im_starts[1] + 3  # 跳过 <|im_start|> user \n
    text_end = ids.index(IM_END, im_starts[1] + 1)

    texts = []
    positions = []
    embs = []
    for pos in range(text_start, text_end):
        # 解码时保留前导空格（用于后续拼接判断词边界）
        tok_text = tokenizer.decode([ids[pos]])
        texts.append(tok_text)
        positions.append(pos)
        embs.append(token_embeddings[sent_idx, pos])

    return texts, positions, embs


def reconstruct_text(token_texts):
    """将 token 文本列表拼回原文"""
    return "".join(token_texts).strip()


# ============================================================
# 3. 在 token 序列中查找短语并提取 embedding
# ============================================================

def find_phrase_in_tokens(phrase, token_texts, positions, embs):
    """
    在 token 文本序列中查找短语的所有出现位置。
    返回每次出现的平均池化 embedding 列表。

    策略：将 token 拼接成文本，用字符级定位找到短语位置，
    再反查对应的 token 范围，取平均 embedding。
    """
    # 拼接所有 token 文本（保留原始空格）
    full_text = "".join(token_texts)
    phrase_lower = phrase.lower()
    full_lower = full_text.lower()

    # 建立字符偏移到 token index 的映射
    char_to_tok = []
    for tok_idx, t in enumerate(token_texts):
        char_to_tok.extend([tok_idx] * len(t))

    results = []
    search_start = 0
    while True:
        pos = full_lower.find(phrase_lower, search_start)
        if pos == -1:
            break

        # 检查词边界：短语前面是空格或句首，后面是空格或句末
        if pos > 0 and full_lower[pos - 1] not in (" ", "\t"):
            search_start = pos + 1
            continue
        end_pos = pos + len(phrase_lower)
        if end_pos < len(full_lower) and full_lower[end_pos] not in (" ", "\t", ""):
            search_start = pos + 1
            continue

        # 找到对应的 token 范围
        tok_start = char_to_tok[pos]
        tok_end = char_to_tok[min(end_pos - 1, len(char_to_tok) - 1)]

        phrase_embs = [embs[i] for i in range(tok_start, tok_end + 1)]
        # 平均池化作为短语 embedding
        avg_emb = np.mean(phrase_embs, axis=0)
        # 记录涉及的 token 文本（调试用）
        matched_tokens = token_texts[tok_start:tok_end + 1]
        results.append({
            "embedding": avg_emb,
            "tok_range": (tok_start, tok_end),
            "matched_text": "".join(matched_tokens).strip(),
        })

        search_start = end_pos

    return results


# ============================================================
# 4. 对所有句子搜索所有短语，收集 embedding
# ============================================================

# phrase_data[phrase] = {"category": ..., "occurrences": [(sent_idx, emb, matched_text), ...]}
phrase_data = {}

all_phrases = (
    [(p, "物体名词") for p in OBJECT_NOUNS] +
    [(p, "动词短语") for p in VERB_PHRASES] +
    [(p, "方位短语") for p in SPATIAL_PHRASES]
)

for phrase, category in all_phrases:
    occurrences = []
    for sent_idx in range(N):
        tok_texts, tok_positions, tok_embs = get_user_tokens(sent_idx)
        matches = find_phrase_in_tokens(phrase, tok_texts, tok_positions, tok_embs)
        for m in matches:
            occurrences.append((sent_idx, m["embedding"], m["matched_text"]))

    if len(occurrences) >= 1:
        phrase_data[phrase] = {
            "category": category,
            "occurrences": occurrences,
        }

# 过滤：只保留在 ≥2 个不同句子中出现的短语
shared_phrases = {}
for phrase, info in phrase_data.items():
    sent_set = set(o[0] for o in info["occurrences"])
    if len(sent_set) >= 2:
        shared_phrases[phrase] = info

print(f"\n匹配到的短语总数: {len(phrase_data)}")
print(f"出现在 ≥2 个句子中的共享短语: {len(shared_phrases)}")


# ============================================================
# 5. 计算跨句子 cosine similarity
# ============================================================

phrase_stats = {}
for phrase, info in shared_phrases.items():
    embs = np.stack([o[1] for o in info["occurrences"]])
    sim_matrix = cosine_similarity(embs)
    triu = np.triu_indices(len(embs), k=1)
    pairwise = sim_matrix[triu]

    n_sents = len(set(o[0] for o in info["occurrences"]))
    phrase_stats[phrase] = {
        "category": info["category"],
        "n_occ": len(info["occurrences"]),
        "n_sents": n_sents,
        "mean_sim": float(pairwise.mean()),
        "std_sim": float(pairwise.std()),
        "min_sim": float(pairwise.min()),
        "max_sim": float(pairwise.max()),
        "embs": embs,
        "sim_matrix": sim_matrix,
        "sent_indices": [o[0] for o in info["occurrences"]],
        "matched_texts": [o[2] for o in info["occurrences"]],
    }


# ============================================================
# 6. 分类打印结果
# ============================================================

for cat_name in ["物体名词", "动词短语", "方位短语"]:
    cat_items = [(p, s) for p, s in phrase_stats.items() if s["category"] == cat_name]
    cat_items.sort(key=lambda x: -x[1]["mean_sim"])

    print(f"\n{'=' * 90}")
    print(f"【{cat_name}】")
    print(f"{'=' * 90}")
    print(f"{'短语':<25} {'出现数':>6} {'句子数':>6} {'mean_sim':>10} {'std':>8} {'min':>8} {'max':>8}")
    print("-" * 90)
    for phrase, s in cat_items:
        print(f"{phrase:<25} {s['n_occ']:>6} {s['n_sents']:>6} "
              f"{s['mean_sim']:>10.4f} {s['std_sim']:>8.4f} {s['min_sim']:>8.4f} {s['max_sim']:>8.4f}")


# ============================================================
# 7. 可视化 1: 分类别柱状图
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
cat_en = {"物体名词": "Object Nouns", "动词短语": "Verb Phrases", "方位短语": "Spatial Phrases"}

for ax_idx, cat_name in enumerate(["物体名词", "动词短语", "方位短语"]):
    ax = axes[ax_idx]
    cat_items = [(p, s) for p, s in phrase_stats.items() if s["category"] == cat_name]
    cat_items.sort(key=lambda x: -x[1]["mean_sim"])

    if not cat_items:
        ax.set_visible(False)
        continue

    names = [p for p, _ in cat_items]
    means = [s["mean_sim"] for _, s in cat_items]
    stds = [s["std_sim"] for _, s in cat_items]
    counts = [s["n_occ"] for _, s in cat_items]

    colors = plt.cm.RdYlGn(np.array(means))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=3,
                  color=colors, edgecolor="gray", linewidth=0.5)

    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"[{cat_en[cat_name]}] Cross-Sentence Similarity", fontsize=12, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 1.15)

plt.suptitle("LIBERO Task: Same Phrase in Different Sentences — Cosine Similarity", fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "phrase_similarity_bar.png", dpi=150)
print(f"\n柱状图已保存: {OUTPUT_DIR / 'phrase_similarity_bar.png'}")


# ============================================================
# 8. 可视化 2: 三类短语混合 t-SNE
# ============================================================

all_embs_list = []
all_labels = []
all_cats = []

for phrase, info in shared_phrases.items():
    cat = info["category"]
    for occ in info["occurrences"]:
        all_embs_list.append(occ[1])
        all_labels.append(phrase)
        all_cats.append(cat)

all_embs_arr = np.stack(all_embs_list)
print(f"\nt-SNE: {len(all_embs_arr)} 个短语 embedding, {len(shared_phrases)} 类")

perp = min(30, len(all_embs_arr) - 1)
tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
coords = tsne.fit_transform(all_embs_arr)

# 按类别分面绘制
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
cat_order = ["物体名词", "动词短语", "方位短语"]
markers = {"物体名词": "o", "动词短语": "s", "方位短语": "^"}

for ax_idx, cat_name in enumerate(cat_order):
    ax = axes[ax_idx]

    # 先画灰色背景（其他类别的点）
    other_mask = [i for i, c in enumerate(all_cats) if c != cat_name]
    if other_mask:
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                   c="lightgray", s=15, alpha=0.3, zorder=1)

    # 再画当前类别的点，按短语着色
    cat_phrases = sorted(set(p for p, info in shared_phrases.items() if info["category"] == cat_name))
    n_phrases = len(cat_phrases)
    cmap = plt.colormaps.get_cmap("tab20")

    for p_idx, phrase in enumerate(cat_phrases):
        mask = [i for i, (l, c) in enumerate(zip(all_labels, all_cats))
                if l == phrase and c == cat_name]
        color = cmap(p_idx / max(n_phrases, 1))
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], label=phrase, s=50, alpha=0.85,
                   marker=markers[cat_name], edgecolors="white", linewidth=0.5, zorder=2)

    ax.set_title(f"[{cat_en[cat_name]}]", fontsize=12, fontweight="bold")
    ax.legend(fontsize=6, loc="best", ncol=2 if n_phrases > 8 else 1)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

plt.suptitle("t-SNE: Same Phrase = Same Color (across different sentences)", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "phrase_tsne.png", dpi=150)
print(f"t-SNE 图已保存: {OUTPUT_DIR / 'phrase_tsne.png'}")


# ============================================================
# 9. 可视化 3: 选典型短语画热力图
# ============================================================

# 每类选几个典型的
heatmap_phrases = []
for cat_name in ["物体名词", "动词短语", "方位短语"]:
    cat_items = [(p, s) for p, s in phrase_stats.items() if s["category"] == cat_name]
    cat_items.sort(key=lambda x: -x[1]["n_occ"])
    heatmap_phrases.extend([p for p, _ in cat_items[:4]])

n_hm = len(heatmap_phrases)
cols = min(4, n_hm)
rows = (n_hm + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = axes[np.newaxis, :]
elif cols == 1:
    axes = axes[:, np.newaxis]

for idx, phrase in enumerate(heatmap_phrases):
    r, c = idx // cols, idx % cols
    ax = axes[r, c]
    s = phrase_stats[phrase]
    sim = s["sim_matrix"]
    cat = s["category"]

    im = ax.imshow(sim, vmin=0.3, vmax=1.0, cmap="RdYlGn", aspect="equal")
    labels = [f"S{si}" for si in s["sent_indices"]]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    cat_label = cat_en.get(cat, cat)
    ax.set_title(f'"{phrase}" [{cat_label}]\n(n={s["n_occ"]}, sim={s["mean_sim"]:.3f})', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for idx in range(n_hm, rows * cols):
    r, c = idx // cols, idx % cols
    axes[r, c].set_visible(False)

plt.suptitle("Cross-Sentence Similarity Heatmaps for Key Phrases", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "phrase_heatmaps.png", dpi=150)
print(f"热力图已保存: {OUTPUT_DIR / 'phrase_heatmaps.png'}")


# ============================================================
# 10. 对比分析：同一物体在不同句子结构中的表征差异
# ============================================================

print("\n\n" + "=" * 90)
print("【深入分析】同一物体/短语在不同句式中的表征变化")
print("=" * 90)

# 对几个关键物体，展示它在哪些句子中出现以及 pairwise similarity
key_phrases = ["bowl", "plate", "stove", "cabinet", "cream cheese", "black bowl",
               "pick up", "place", "on the plate", "in the basket"]

for phrase in key_phrases:
    if phrase not in phrase_stats:
        continue
    s = phrase_stats[phrase]
    print(f'\n--- "{phrase}" [{s["category"]}] ---')
    print(f"  出现在 {s['n_sents']} 个句子中，共 {s['n_occ']} 次")
    print(f"  平均相似度: {s['mean_sim']:.4f}, std: {s['std_sim']:.4f}")
    print(f"  范围: [{s['min_sim']:.4f}, {s['max_sim']:.4f}]")
    # 展示每次出现对应的句子
    for i, (si, _, mt) in enumerate(shared_phrases[phrase]["occurrences"]):
        sent_text = index_info[si]["text"]
        print(f"  [{i}] S{si}: \"{sent_text}\"  (matched: \"{mt}\")")


# ============================================================
# 11. 总体统计
# ============================================================

print("\n\n" + "=" * 90)
print("【总体统计】")
print("=" * 90)

for cat_name in ["物体名词", "动词短语", "方位短语"]:
    cat_means = [s["mean_sim"] for s in phrase_stats.values() if s["category"] == cat_name]
    if cat_means:
        print(f"\n  {cat_name}:")
        print(f"    短语数: {len(cat_means)}")
        print(f"    平均 sim: {np.mean(cat_means):.4f}")
        print(f"    std: {np.std(cat_means):.4f}")
        print(f"    范围: [{min(cat_means):.4f}, {max(cat_means):.4f}]")

all_means = [s["mean_sim"] for s in phrase_stats.values()]
print(f"\n  全部:")
print(f"    短语数: {len(all_means)}")
print(f"    平均 sim: {np.mean(all_means):.4f}")
print(f"    范围: [{min(all_means):.4f}, {max(all_means):.4f}]")

print(f"\n所有图表已保存到: {OUTPUT_DIR}")
