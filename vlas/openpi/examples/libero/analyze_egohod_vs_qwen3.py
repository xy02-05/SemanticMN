"""
对比分析 EgoHOD vs Qwen3 的 LIBERO 文本 token 表征，并诊断 causal attention 导致的「虚假一致性」。

核心发现：
  两个模型都使用 **causal (单向) attention**，所以句首 token 只能看到前面的 token。
  "pick up" 在21个句子中 sim=1.0 不是因为模型理解了"pick up"这个动作，
  而是因为 causal attention 使得句首 token 根本看不到后面的上下文。

分析内容：
1. EgoHOD 的短语级分析（与 Qwen3 对比）
2. 诊断 causal attention 的影响：按 token 在句子中的相对位置分析相似度
3. 量化「虚假一致性」的范围
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path("/root/data/xuyuan1")
OUTPUT_DIR = BASE_DIR / "Codes/mirror_neuron/vlas/openpi/results/libero/word_embedding_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 加载两个模型的数据
# ============================================================

# EgoHOD
ego_data = np.load(BASE_DIR / "dataset/embedding/libero_egohod_text_features.npz")
with open(BASE_DIR / "dataset/embedding/libero_egohod_text_index.json") as f:
    ego_index = json.load(f)

# Qwen3
qw_data = np.load(BASE_DIR / "dataset/embedding/libero_qwen3_text_features.npz")
with open(BASE_DIR / "dataset/embedding/libero_qwen3_text_index.json") as f:
    qw_index = json.load(f)

N = 40
print(f"EgoHOD: token_embeddings {ego_data['token_embeddings'].shape}, dim={ego_data['token_embeddings'].shape[-1]}")
print(f"Qwen3 : token_embeddings {qw_data['token_embeddings'].shape}, dim={qw_data['token_embeddings'].shape[-1]}")


# ============================================================
# 2. 定义短语列表（复用之前的）
# ============================================================

OBJECT_NOUNS = [
    "black bowl", "bowl", "plate", "basket", "stove", "cabinet",
    "drawer", "mug", "white mug", "cream cheese", "alphabet soup",
    "tomato sauce", "chocolate pudding", "wine bottle", "cookie box",
    "moka pot", "ramekin", "wooden cabinet",
]
VERB_PHRASES = ["pick up", "put", "place", "open", "close", "turn on", "push"]
SPATIAL_PHRASES = [
    "on the plate", "in the basket", "on the stove",
    "next to", "between", "on top of",
    "on the ramekin", "on the cookie box",
]


# ============================================================
# 3. 通用短语匹配函数
# ============================================================

def get_user_text_tokens_egohod(sent_idx, data):
    """EgoHOD: CLIP tokenizer，格式为 [SOT] text tokens [EOT]，SOT=49406, EOT=49407"""
    valid_len = int(data['token_lengths'][sent_idx])
    ids = data['input_ids'][sent_idx, :valid_len].tolist()
    te = data['token_embeddings'][sent_idx]
    # 跳过 SOT (pos=0) 和 EOT (最后一个)
    texts = []
    positions = []
    embs = []
    for pos in range(1, valid_len - 1):  # 跳过 SOT 和 EOT
        texts.append(None)  # 占位，后面用 id 解码
        positions.append(pos)
        embs.append(te[pos])
    return ids[1:valid_len-1], positions, embs


def get_user_text_tokens_qwen3(sent_idx, data):
    """Qwen3: chat template，用户文本在第二个 <|im_start|> 之后"""
    valid_len = int(data['token_lengths'][sent_idx])
    ids = data['input_ids'][sent_idx, :valid_len].tolist()
    te = data['token_embeddings'][sent_idx]
    IM_START = 151644
    im_starts = [i for i, x in enumerate(ids) if x == IM_START]
    text_start = im_starts[1] + 3
    IM_END = 151645
    text_end = ids.index(IM_END, im_starts[1] + 1)
    token_ids = ids[text_start:text_end]
    positions = list(range(text_start, text_end))
    embs = [te[pos] for pos in positions]
    return token_ids, positions, embs


def find_phrase_by_text(phrase, sentence_text, token_ids, positions, embs, tokenizer_decode_fn):
    """
    通过原始文本定位短语，然后映射到 token 位置。
    tokenizer_decode_fn: 将 token_id 列表解码为文本的函数。
    """
    # 将 token 文本拼接
    token_texts = [tokenizer_decode_fn(tid) for tid in token_ids]
    full_text = "".join(token_texts)
    phrase_lower = phrase.lower()
    full_lower = full_text.lower()

    # 字符→token index 映射
    char_to_tok = []
    for tok_idx, t in enumerate(token_texts):
        char_to_tok.extend([tok_idx] * len(t))

    results = []
    search_start = 0
    while True:
        pos = full_lower.find(phrase_lower, search_start)
        if pos == -1:
            break
        # 词边界检查
        if pos > 0 and full_lower[pos - 1] not in (" ", "\t"):
            search_start = pos + 1
            continue
        end_pos = pos + len(phrase_lower)
        if end_pos < len(full_lower) and full_lower[end_pos] not in (" ", "\t", ""):
            search_start = pos + 1
            continue

        tok_start = char_to_tok[pos]
        tok_end = char_to_tok[min(end_pos - 1, len(char_to_tok) - 1)]
        phrase_embs = [embs[i] for i in range(tok_start, tok_end + 1)]
        avg_emb = np.mean(phrase_embs, axis=0)
        # 计算短语在句子中的相对位置 (0=句首, 1=句尾)
        rel_pos = (tok_start + tok_end) / 2 / max(len(token_ids) - 1, 1)
        results.append({"embedding": avg_emb, "rel_pos": rel_pos})
        search_start = end_pos

    return results


# ============================================================
# 4. 对两个模型分别提取短语 embedding
# ============================================================

def analyze_model(model_name, data, index_info, get_tokens_fn, decode_fn):
    """对一个模型执行全部短语分析"""
    all_phrases = (
        [(p, "Object Noun") for p in OBJECT_NOUNS] +
        [(p, "Verb Phrase") for p in VERB_PHRASES] +
        [(p, "Spatial Phrase") for p in SPATIAL_PHRASES]
    )

    phrase_stats = {}
    for phrase, category in all_phrases:
        occurrences = []
        for sent_idx in range(N):
            token_ids, positions, embs = get_tokens_fn(sent_idx, data)
            sentence_text = index_info[sent_idx]["text"]
            matches = find_phrase_by_text(phrase, sentence_text, token_ids, positions, embs, decode_fn)
            for m in matches:
                occurrences.append((sent_idx, m["embedding"], m["rel_pos"]))

        sent_set = set(o[0] for o in occurrences)
        if len(sent_set) >= 2:
            embs_arr = np.stack([o[1] for o in occurrences])
            sim_matrix = cosine_similarity(embs_arr)
            triu = np.triu_indices(len(embs_arr), k=1)
            pairwise = sim_matrix[triu]

            phrase_stats[phrase] = {
                "category": category,
                "n_occ": len(occurrences),
                "n_sents": len(sent_set),
                "mean_sim": float(pairwise.mean()),
                "std_sim": float(pairwise.std()),
                "min_sim": float(pairwise.min()),
                "max_sim": float(pairwise.max()),
                "avg_rel_pos": float(np.mean([o[2] for o in occurrences])),
            }

    return phrase_stats


# CLIP tokenizer decode
import clip
clip_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
clip_decoder = clip_tokenizer.decoder
# CLIP tokenizer: 每个 token id 映射到 BPE token 文本
def clip_decode(tid):
    if tid in clip_decoder:
        raw = clip_decoder[tid]
        # CLIP BPE 用 </w> 表示词尾，解码时替换为空格
        return raw.replace("</w>", " ")
    return ""

# Qwen3 tokenizer decode
from transformers import AutoTokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR / "dataset/Qwen3-VL-Embedding-8B"))
def qwen_decode(tid):
    return qwen_tokenizer.decode([tid])


print("\n正在分析 EgoHOD ...")
ego_stats = analyze_model("EgoHOD", ego_data, ego_index, get_user_text_tokens_egohod, clip_decode)

print("正在分析 Qwen3 ...")
qw_stats = analyze_model("Qwen3", qw_data, qw_index, get_user_text_tokens_qwen3, qwen_decode)


# ============================================================
# 5. 打印对比表
# ============================================================

# 找到两个模型都有的短语
common_phrases = sorted(set(ego_stats.keys()) & set(qw_stats.keys()))

print(f"\n{'=' * 110}")
print(f"{'短语':<25} {'类别':<15} {'EgoHOD_sim':>12} {'Qwen3_sim':>12} {'差值':>8} {'EgoHOD_pos':>12} {'Qwen3_pos':>12}")
print(f"{'=' * 110}")

for cat in ["Verb Phrase", "Object Noun", "Spatial Phrase"]:
    cat_phrases = [p for p in common_phrases if ego_stats[p]["category"] == cat]
    cat_phrases.sort(key=lambda p: -(ego_stats[p]["mean_sim"] + qw_stats[p]["mean_sim"]) / 2)
    for phrase in cat_phrases:
        e = ego_stats[phrase]
        q = qw_stats[phrase]
        diff = q["mean_sim"] - e["mean_sim"]
        print(f"{phrase:<25} {cat:<15} {e['mean_sim']:>12.4f} {q['mean_sim']:>12.4f} {diff:>+8.4f} "
              f"{e['avg_rel_pos']:>12.3f} {q['avg_rel_pos']:>12.3f}")
    print("-" * 110)


# ============================================================
# 6. 可视化：EgoHOD vs Qwen3 对比柱状图
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(18, 14))
cat_order = ["Object Noun", "Verb Phrase", "Spatial Phrase"]

for ax_idx, cat in enumerate(cat_order):
    ax = axes[ax_idx]
    cat_phrases = [p for p in common_phrases if ego_stats[p]["category"] == cat]
    cat_phrases.sort(key=lambda p: -(ego_stats[p]["mean_sim"] + qw_stats[p]["mean_sim"]) / 2)

    x = np.arange(len(cat_phrases))
    width = 0.35

    ego_means = [ego_stats[p]["mean_sim"] for p in cat_phrases]
    qw_means = [qw_stats[p]["mean_sim"] for p in cat_phrases]

    bars1 = ax.bar(x - width / 2, ego_means, width, label="EgoHOD (CLIP)", color="#FF7043", alpha=0.85)
    bars2 = ax.bar(x + width / 2, qw_means, width, label="Qwen3-VL-Emb", color="#42A5F5", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_phrases, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"[{cat}] Cross-Sentence Similarity: EgoHOD vs Qwen3", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

plt.suptitle("EgoHOD vs Qwen3: Same Phrase in Different Sentences", fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "egohod_vs_qwen3_comparison.png", dpi=150)
print(f"\n对比图已保存: {OUTPUT_DIR / 'egohod_vs_qwen3_comparison.png'}")


# ============================================================
# 7. 【关键诊断】Causal Attention 导致的虚假一致性
# ============================================================

print("\n\n" + "=" * 90)
print("【关键诊断】Causal Attention 的影响")
print("=" * 90)

print("""
两个模型都使用 causal (单向) attention：
  - EgoHOD: CLIP text encoder, use_bidirectional_lm=False
  - Qwen3-VL-Embedding: GPT-style causal LM

因此 token 在句中的位置决定了它能「看到」多少上下文：
  - 句首 token → 只能看到 system prompt / SOT → 表征与后文无关（虚假一致）
  - 句末 token → 能看到完整前文 → 表征反映了真实上下文
""")

# 按短语的相对位置分组，看相似度与位置的关系
print("--- 短语的句中相对位置 vs 跨句子相似度 ---")
print("  (rel_pos=0 表示句首, rel_pos=1 表示句尾)")

for model_name, stats in [("EgoHOD", ego_stats), ("Qwen3", qw_stats)]:
    positions = []
    similarities = []
    labels = []
    for phrase, s in stats.items():
        positions.append(s["avg_rel_pos"])
        similarities.append(s["mean_sim"])
        labels.append(phrase)

    # 计算相关系数
    corr = np.corrcoef(positions, similarities)[0, 1]
    print(f"\n  {model_name}: 位置-相似度相关系数 = {corr:.4f}")
    print(f"    句首 (pos<0.3): ", end="")
    head = [(p, s) for p, s in zip(positions, similarities) if p < 0.3]
    if head:
        print(f"n={len(head)}, avg_sim={np.mean([s for _, s in head]):.4f}")
    print(f"    句中 (0.3≤pos≤0.7): ", end="")
    mid = [(p, s) for p, s in zip(positions, similarities) if 0.3 <= p <= 0.7]
    if mid:
        print(f"n={len(mid)}, avg_sim={np.mean([s for _, s in mid]):.4f}")
    print(f"    句尾 (pos>0.7): ", end="")
    tail = [(p, s) for p, s in zip(positions, similarities) if p > 0.7]
    if tail:
        print(f"n={len(tail)}, avg_sim={np.mean([s for _, s in tail]):.4f}")


# 可视化：位置 vs 相似度散点图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (model_name, stats) in enumerate([("EgoHOD", ego_stats), ("Qwen3", qw_stats)]):
    ax = axes[ax_idx]
    cat_colors = {"Object Noun": "#4CAF50", "Verb Phrase": "#2196F3", "Spatial Phrase": "#FF9800"}

    for phrase, s in stats.items():
        color = cat_colors[s["category"]]
        ax.scatter(s["avg_rel_pos"], s["mean_sim"], c=color, s=60, alpha=0.8, edgecolors="white")
        ax.annotate(phrase, (s["avg_rel_pos"], s["mean_sim"]), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Relative Position in Sentence (0=start, 1=end)")
    ax.set_ylabel("Cross-Sentence Cosine Similarity")
    ax.set_title(f"{model_name}: Position vs Similarity", fontsize=12, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.4, 1.05)

    # 添加 legend
    for cat, color in cat_colors.items():
        ax.scatter([], [], c=color, label=cat, s=60)
    ax.legend(fontsize=8)

plt.suptitle("Causal Attention Effect: Sentence Position vs Cross-Sentence Similarity", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "position_vs_similarity.png", dpi=150)
print(f"\n位置-相似度图已保存: {OUTPUT_DIR / 'position_vs_similarity.png'}")


# ============================================================
# 8. 总结统计
# ============================================================

print("\n\n" + "=" * 90)
print("【总体对比统计】")
print("=" * 90)

for cat in ["Object Noun", "Verb Phrase", "Spatial Phrase"]:
    ego_cat = [s["mean_sim"] for s in ego_stats.values() if s["category"] == cat]
    qw_cat = [s["mean_sim"] for s in qw_stats.values() if s["category"] == cat]
    print(f"\n  {cat}:")
    if ego_cat:
        print(f"    EgoHOD: n={len(ego_cat)}, avg={np.mean(ego_cat):.4f}, range=[{min(ego_cat):.4f}, {max(ego_cat):.4f}]")
    if qw_cat:
        print(f"    Qwen3 : n={len(qw_cat)}, avg={np.mean(qw_cat):.4f}, range=[{min(qw_cat):.4f}, {max(qw_cat):.4f}]")

ego_all = [s["mean_sim"] for s in ego_stats.values()]
qw_all = [s["mean_sim"] for s in qw_stats.values()]
print(f"\n  Overall:")
print(f"    EgoHOD: avg={np.mean(ego_all):.4f}")
print(f"    Qwen3 : avg={np.mean(qw_all):.4f}")

print(f"\n所有图表已保存到: {OUTPUT_DIR}")
