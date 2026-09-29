"""
线性探针训练: Action Features → Text Embedding 对齐

原理:
  参考 "Embodied Representation Alignment with Mirror Neurons" (Zhu et al.)
  用单层 FC 线性映射 + InfoNCE 对比学习，测试 action representation 是否保持了语义结构。

  如果 pretrained 模型的线性探针 top-1 准确率远高于 fine-tuned 模型，
  说明 fine-tuning 破坏了 action representation 中的语义信息（shortcut learning）。

  双投影架构 (CLIP 风格):
    action: Linear(1024 → D_shared) + L2 norm
    text:   Linear(D_text → D_shared) + L2 norm
  两边各一层 Linear 投影到共享空间，in-batch InfoNCE 对比学习。

数据划分:
  使用 build_split.py 产出的 split.json 做 episode 级划分。
  训练用 train split 特征，评估用 test split 特征。
  所有 40 个 task 都同时出现在 train/test 中（episode 级划分，非 task 级）。

存储:
  每个 (checkpoint, layer) 的线性探针模型 + 评估指标 JSON

用法:
    python train_probe.py --checkpoint pretrained
    python train_probe.py --checkpoint step_30k --text_type qwen3
    python train_probe.py --all  # 对所有 checkpoint 训练
"""
import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import TEXT_EMBEDDINGS, FEATURE_DIR, PROBE_DIR, CHECKPOINTS, LIBERO_SUITES


# ===================== 数据加载 =====================
def load_features(path, granularity="trajectory"):
    """加载 extract_features.py 产出的 .npz
    granularity='trajectory': 轨迹级 mean-pool (features key)
    granularity='chunk': 每帧 chunk 级 (chunk_features key)
    """
    data = np.load(path, allow_pickle=True)
    if granularity == "chunk":
        feat_key, task_key = "chunk_features", "chunk_task_indices"
    else:
        feat_key, task_key = "features", "task_indices"
    return {
        "features": data[feat_key],           # [N, L, D]
        "task_indices": data[task_key],        # [N]
        "layer_indices": data["layer_indices"],# [L]
    }


def load_text_gallery(text_type="qwen3"):
    """
    加载 text embedding gallery。
    LIBERO 有 40 个 task，返回 [40, D_text] 的 embedding 矩阵。
    task_index 0-39 直接对应矩阵行号。
    """
    cfg = TEXT_EMBEDDINGS[text_type]
    data = np.load(cfg["path"])
    embeddings = data[cfg["key"]]  # [40, D_text]
    return embeddings


# ===================== 模型 =====================
class DualAligner(nn.Module):
    """
    双线性投影 (CLIP 风格): action 和 text 各自一层 Linear → 共享空间
    """
    def __init__(self, d_action, d_text, d_shared=512):
        super().__init__()
        self.proj_action = nn.Linear(d_action, d_shared)
        self.proj_text = nn.Linear(d_text, d_shared)
        # text 投影初始化为恒等（如果维度匹配），保留原始 text 结构
        if d_text == d_shared:
            nn.init.eye_(self.proj_text.weight)
            nn.init.zeros_(self.proj_text.bias)

    def forward_action(self, x):
        return self.proj_action(x)

    def forward_text(self, x):
        return self.proj_text(x)


# ===================== 训练 =====================
def train_one_epoch(model, optimizer, train_action, train_text, temperature, batch_size, device):
    """In-batch InfoNCE (CLIP 风格) 训练一个 epoch"""
    model.train()
    perm = torch.randperm(len(train_action))
    total_loss, n_batches = 0.0, 0

    for i in range(0, len(train_action), batch_size):
        idx = perm[i:i + batch_size]
        action_batch = train_action[idx].to(device)
        text_batch = train_text[idx].to(device)

        proj_a = F.normalize(model.forward_action(action_batch), dim=-1)
        proj_t = F.normalize(model.forward_text(text_batch), dim=-1)

        # 相似度矩阵 [B, B]，对角线为正样本
        logits = proj_a @ proj_t.T / temperature
        labels = torch.arange(logits.shape[0], device=device)

        # 双向 InfoNCE
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, action_feats, text_emb, class_indices, text_gallery, device):
    """
    评估: gallery retrieval top-k accuracy + mean cosine similarity
    - action features 投影后与 gallery 中所有 text 计算相似度
    - 检索最近的 text，看是否与 ground truth 匹配
    """
    model.eval()
    gallery_proj = F.normalize(model.forward_text(text_gallery), dim=-1)  # [C, D_shared]

    all_preds, all_cos = [], []
    bs = 512

    for i in range(0, len(action_feats), bs):
        feat_b = action_feats[i:i+bs].to(device)
        text_b = text_emb[i:i+bs].to(device)

        proj_a = F.normalize(model.forward_action(feat_b), dim=-1)
        proj_t = F.normalize(model.forward_text(text_b), dim=-1)

        # gallery retrieval
        sim = proj_a @ gallery_proj.T
        all_preds.append(sim.argsort(dim=1, descending=True).cpu())

        # 与自身 text 的 cosine similarity
        cos = (proj_a * proj_t).sum(dim=-1)
        all_cos.append(cos.cpu())

    sorted_idx = torch.cat(all_preds, dim=0)
    targets = class_indices

    metrics = {}
    for k in [1, 3, 5, 10]:
        if k > gallery_proj.shape[0]:
            continue
        topk = sorted_idx[:, :k]
        correct = (topk == targets.unsqueeze(1)).any(dim=1)
        metrics[f"top{k}"] = correct.float().mean().item()
    metrics["mean_cos_sim"] = torch.cat(all_cos).mean().item()
    return metrics


# ===================== 单层训练 =====================
def train_single_layer(train_feat, train_tasks, test_feat, test_tasks,
                       text_gallery, layer_id, label, args, device, save_dir):
    """
    训练单层的线性探针

    task 级别划分时: test 包含训练时未见的 task（泛化评估）
    episode 级别划分时: train/test 包含相同的 task（插值评估）

    Args:
        train_feat: [N_train, D_action] train split 的轨迹级特征
        train_tasks: [N_train] train split 的 task_index
        test_feat: [N_test, D_action] test split 的轨迹级特征
        test_tasks: [N_test] test split 的 task_index
        text_gallery: [40, D_text] 所有 task 的 text embedding
    """
    D_action = train_feat.shape[1]
    D_text = text_gallery.shape[1]

    # 每条轨迹的 text embedding = gallery[task_index]
    train_text_np = text_gallery[train_tasks]   # [N_train, D_text]
    test_text_np = text_gallery[test_tasks]     # [N_test, D_text]

    train_action = torch.from_numpy(train_feat).float()
    train_text = torch.from_numpy(train_text_np).float()
    test_action = torch.from_numpy(test_feat).float()
    test_text = torch.from_numpy(test_text_np).float()

    train_class = torch.from_numpy(train_tasks).long()
    test_class = torch.from_numpy(test_tasks).long()
    text_gallery_t = torch.from_numpy(text_gallery).float().to(device)

    d_shared = min(D_text, 512)
    model = DualAligner(D_action, D_text, d_shared=d_shared).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

    best_test_top1 = 0.0
    best_epoch = 0
    # epoch >= min_best_epoch 之后才开始选 best（避免早期异常高点）
    min_best_epoch = getattr(args, "min_best_epoch", 50)
    best_after_min = {"top1": 0.0, "epoch": 0}
    save_path = os.path.join(save_dir, f"probe_{label}_layer{layer_id}.pt")
    epoch_history = []

    print(f"\n--- Layer {layer_id}: DualAligner({D_action}→{d_shared}, {D_text}→{d_shared})"
          f" | train={len(train_feat)} test={len(test_feat)} ---")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, optimizer, train_action, train_text,
            args.temperature, args.batch_size, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            train_m = evaluate(model, train_action, train_text, train_class, text_gallery_t, device)
            test_m = evaluate(model, test_action, test_text, test_class, text_gallery_t, device)
            print(f"  Epoch {epoch+1:>3d}: loss={train_loss:.4f}"
                  f"  tr_top1={train_m['top1']:.3f}"
                  f"  test_top1={test_m['top1']:.3f}"
                  f"  test_cos={test_m['mean_cos_sim']:.3f}")

            # 记录每 10 epoch 的结果
            epoch_history.append({
                "epoch": epoch + 1,
                "train_top1": float(train_m["top1"]),
                "test_top1": float(test_m["top1"]),
                "test_top5": float(test_m.get("top5", 0)),
                "test_cos": float(test_m["mean_cos_sim"]),
            })

            # 全局 best（兼容旧逻辑）
            if test_m["top1"] >= best_test_top1:
                best_test_top1 = test_m["top1"]
                best_epoch = epoch + 1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "layer": layer_id, "epoch": epoch + 1,
                    "test_metrics": test_m, "train_metrics": train_m,
                    "d_action": D_action, "d_text": D_text, "d_shared": d_shared,
                }, save_path)

            # epoch >= min_best_epoch 之后的 best
            if (epoch + 1) >= min_best_epoch and test_m["top1"] >= best_after_min["top1"]:
                best_after_min = {"top1": float(test_m["top1"]), "epoch": epoch + 1,
                                  "test_metrics": test_m, "train_metrics": train_m}

    # 加载全局最优并评估
    if best_test_top1 > 0 and os.path.exists(save_path):
        ckpt = torch.load(save_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    final_train = evaluate(model, train_action, train_text, train_class, text_gallery_t, device)
    final_test = evaluate(model, test_action, test_text, test_class, text_gallery_t, device)

    print(f"  Layer {layer_id} best@epoch{best_epoch}:"
          f"  test_top1={final_test['top1']:.4f}  test_top5={final_test.get('top5',0):.4f}"
          f"  | best_after_{min_best_epoch}: epoch{best_after_min['epoch']}"
          f"  test_top1={best_after_min['top1']:.4f}")

    return {
        "layer": int(layer_id),
        "best_epoch": best_epoch,
        "train_metrics": final_train,
        "test_metrics": final_test,
        "best_after_min_epoch": best_after_min.get("epoch", 0),
        "best_after_min_test": best_after_min.get("test_metrics", {}),
        "epoch_history": epoch_history,
        "n_train": len(train_feat),
        "n_test": len(test_feat),
    }


# ===================== 主函数 =====================
def train_probes(checkpoint_name, text_type="qwen3", args=None):
    """对一个 checkpoint 训练所有层的线性探针（加载 train/test split 特征）

    支持三种 feature_mode（与 extract_features.py 对应）：
      - clean       → {ckpt}_{split}.npz
      - rollout     → {ckpt}_{split}_rollout.npz
      - noisy_gt    → {ckpt}_{split}_noisy_gt.npz   (canonical setup)
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 根据 probe_train_mode / probe_test_mode 拼后缀（默认 clean，向后兼容）
    mode_suffix = {"clean": "", "rollout": "_rollout", "noisy_gt": "_noisy_gt"}
    tr_mode = getattr(args, "probe_train_mode", "clean")
    te_mode = getattr(args, "probe_test_mode", "clean")
    train_path = os.path.join(FEATURE_DIR, f"{checkpoint_name}_train{mode_suffix[tr_mode]}.npz")
    test_path = os.path.join(FEATURE_DIR, f"{checkpoint_name}_test{mode_suffix[te_mode]}.npz")

    assert os.path.exists(train_path), f"找不到 train 特征: {train_path}"
    assert os.path.exists(test_path), f"找不到 test 特征: {test_path}"

    granularity = getattr(args, "granularity", "trajectory")
    suite_split = getattr(args, "suite_split", False)

    if suite_split:
        raw_train = load_features(train_path, granularity=granularity)
        raw_test = load_features(test_path, granularity=granularity)
        all_feats = np.concatenate([raw_train["features"], raw_test["features"]], axis=0)
        all_tasks = np.concatenate([raw_train["task_indices"], raw_test["task_indices"]], axis=0)
        goal_tasks = set(LIBERO_SUITES["libero_goal"]["task_indices"])
        tr_mask = np.array([t not in goal_tasks for t in all_tasks])
        te_mask = ~tr_mask
        train_data = {"features": all_feats[tr_mask], "task_indices": all_tasks[tr_mask],
                      "layer_indices": raw_train["layer_indices"]}
        test_data = {"features": all_feats[te_mask], "task_indices": all_tasks[te_mask],
                     "layer_indices": raw_train["layer_indices"]}
        print(f"  Suite split: train {np.unique(train_data['task_indices']).size} tasks, "
              f"test {np.unique(test_data['task_indices']).size} tasks (goal)")
    else:
        train_data = load_features(train_path, granularity=granularity)
        test_data = load_features(test_path, granularity=granularity)

    # chunk 级样本量大，为加速训练做采样（每个 task 最多 max_per_task 个样本）
    if granularity == "chunk":
        max_per_task = getattr(args, "max_chunk_per_task", 500)
        for split_name, split_data in [("train", train_data), ("test", test_data)]:
            feats, tasks = split_data["features"], split_data["task_indices"]
            rng = np.random.RandomState(42)
            keep = []
            for t in np.unique(tasks):
                idx = np.where(tasks == t)[0]
                if len(idx) > max_per_task:
                    idx = rng.choice(idx, max_per_task, replace=False)
                keep.append(idx)
            keep = np.concatenate(keep)
            split_data["features"] = feats[keep]
            split_data["task_indices"] = tasks[keep]
        print(f"  Chunk 采样: train={len(train_data['features'])}, test={len(test_data['features'])}")
    text_gallery = load_text_gallery(text_type)  # [40, D_text]

    layer_indices = train_data["layer_indices"]
    if getattr(args, "layers", None):
        requested = set(args.layers)
        mask = np.array([lid in requested for lid in layer_indices])
        layer_indices = layer_indices[mask]
        train_data["features"] = train_data["features"][:, mask, :]
        test_data["features"] = test_data["features"][:, mask, :]

    print(f"{'='*60}")
    print(f"线性探针训练: {checkpoint_name}")
    print(f"  Train: {train_data['features'].shape}, Test: {test_data['features'].shape}")
    print(f"  Text: {text_gallery.shape}, Layers: {list(layer_indices)}")
    print(f"{'='*60}")

    # save_dir 后缀反映 train+test mode 组合：
    #   clean+clean   → {ckpt}_{text_type}                     （向后兼容旧名）
    #   noisy_gt+noisy_gt → {ckpt}_{text_type}_noisy_gt
    #   noisy_gt+rollout  → {ckpt}_{text_type}_noisy_gt_to_rollout
    #   rollout+rollout   → {ckpt}_{text_type}_rollout
    if tr_mode == "clean" and te_mode == "clean":
        dir_suffix = ""
    elif tr_mode == te_mode:
        dir_suffix = f"_{tr_mode}"
    else:
        dir_suffix = f"_{tr_mode}_to_{te_mode}"
    # chunk 级加 _chunk 后缀区分目录
    gran_suffix = "_chunk" if granularity == "chunk" else ""
    suite_suffix = "_suite" if suite_split else ""
    save_dir = os.path.join(PROBE_DIR, f"{checkpoint_name}_{text_type}{dir_suffix}{gran_suffix}{suite_suffix}")
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    for li_pos, layer_id in enumerate(layer_indices):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_feat = train_data["features"][:, li_pos, :]   # [N_train, D]
        train_tasks = train_data["task_indices"]             # [N_train]
        test_feat = test_data["features"][:, li_pos, :]     # [N_test, D]
        test_tasks = test_data["task_indices"]               # [N_test]

        result = train_single_layer(
            train_feat, train_tasks, test_feat, test_tasks,
            text_gallery, int(layer_id), checkpoint_name,
            args, device, save_dir)
        all_results.append(result)

    # 汇总并保存 JSON
    print(f"\n{'='*60}")
    print(f"结果汇总: {checkpoint_name}")
    print(f"  {'Layer':>5s}  {'Epoch':>5s}  {'tr_top1':>7s}  {'test_top1':>9s}  {'test_cos':>8s}")
    for r in all_results:
        tm, em = r["train_metrics"], r["test_metrics"]
        print(f"  {r['layer']:>5d}  {r['best_epoch']:>5d}"
              f"  {tm['top1']:>7.4f}  {em['top1']:>9.4f}  {em['mean_cos_sim']:>8.4f}")

    summary = {
        "checkpoint": checkpoint_name,
        "text_type": text_type,
        "n_train": int(len(train_data["task_indices"])),
        "n_test": int(len(test_data["task_indices"])),
        "n_tasks": int(len(np.unique(train_data["task_indices"]))),
        "per_layer_results": {
            int(r["layer"]): {
                "best_epoch": r["best_epoch"],
                "train_metrics": {k: float(v) for k, v in r["train_metrics"].items()},
                "test_metrics": {k: float(v) for k, v in r["test_metrics"].items()},
                "best_after_min_epoch": r.get("best_after_min_epoch", 0),
                "best_after_min_test": {k: float(v) for k, v in r.get("best_after_min_test", {}).items()},
                "epoch_history": r.get("epoch_history", []),
            } for r in all_results
        },
    }
    json_path = os.path.join(save_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"结果: {json_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="LIBERO 线性探针训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 名 (如 pretrained, step_5k, ..., 或 all)")
    parser.add_argument("--text_type", type=str, default="qwen3",
                        choices=["qwen3", "egohod"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min_best_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--probe_train_mode", type=str, default="clean",
                        choices=["clean", "rollout", "noisy_gt"],
                        help="probe 训练用的 feature 模式（决定加载 {ckpt}_train{suffix}.npz）")
    parser.add_argument("--probe_test_mode", type=str, default="clean",
                        choices=["clean", "rollout", "noisy_gt"],
                        help="probe 测试用的 feature 模式（决定加载 {ckpt}_test{suffix}.npz）"
                             "；canonical setup 用 noisy_gt train + rollout test")
    parser.add_argument("--granularity", type=str, default="trajectory",
                        choices=["trajectory", "chunk"],
                        help="特征粒度: trajectory (轨迹 mean pool) 或 chunk (每帧)")
    parser.add_argument("--max_chunk_per_task", type=int, default=500,
                        help="chunk 模式下每个 task 最多采样数")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="只训练指定层 (如 --layers 10 或 --layers 0 10 17)")
    parser.add_argument("--suite_split", action="store_true",
                        help="suite 级划分: 3 suites (spatial+object+libero_10) train, goal test")
    args = parser.parse_args()

    mode_suffix = {"clean": "", "rollout": "_rollout", "noisy_gt": "_noisy_gt"}
    tr_suffix = mode_suffix[args.probe_train_mode]

    if args.checkpoint == "all":
        for name in CHECKPOINTS:
            train_path = os.path.join(FEATURE_DIR, f"{name}_train{tr_suffix}.npz")
            if os.path.exists(train_path):
                train_probes(name, args.text_type, args)
            else:
                print(f"跳过 {name}: {train_path} 不存在")
    elif args.checkpoint:
        train_probes(args.checkpoint, args.text_type, args)
    else:
        parser.error("需要 --checkpoint <名称> 或 --checkpoint all")


if __name__ == "__main__":
    main()
