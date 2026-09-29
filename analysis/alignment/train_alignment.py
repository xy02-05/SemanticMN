"""
通用 Action ↔ EgoHOD Text 线性对齐训练

支持 SpatialVLA 和 OpenPI 的标准 npz 特征文件, 自动识别 D_action。

原理:
  双投影架构 (CLIP 风格):
    action: Linear(D_action → D_shared) + L2 norm
    text:   Linear(D_text → D_shared) + L2 norm
  两边各自一层 Linear 投影到共享空间，再做 in-batch InfoNCE 对比学习

数据:
  - action features: 标准 npz 文件 (features [N, L, D], task_labels, task_indices, ...)
  - text embeddings: bridge_text_egohod_proj.npz [21938, 512]
  - 按 task 80/20 划分 train/val (测试泛化到未见 task)

用法:
    python train_alignment.py --feature_path PATH_TO_NPZ
    python train_alignment.py --feature_path PATH --layer 10
    python train_alignment.py --feature_path PATH --epochs 200 --lr 3e-4
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ANALYSIS_DIR = "/root/data/xuyuan1/Codes/analysis"
sys.path.insert(0, ANALYSIS_DIR)

# text embedding 默认路径
# 数据路径 (从 bridge_representation/config.py 同步)
DATA_ROOT = "/root/data/xuyuan1/dataset"
META_DIR = os.path.join(DATA_ROOT, "bridge_orig/bridge_orig_lerobot/meta")
DEFAULT_EGOHOD_PATH = os.path.join(DATA_ROOT, "embedding/bridge_text_egohod_proj.npz")
DEFAULT_TASK_RLDS_PATH = os.path.join(META_DIR, "task_rlds.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "alignment/outputs")


# ===================== 数据加载 =====================

def load_feature_npz(path):
    """加载标准 npz, 返回 features [N,L,D], task_labels [N], task_indices [N], layer_indices [L]"""
    data = np.load(path, allow_pickle=True)
    features = data['features']
    task_labels = data['task_labels']
    layer_indices = data['layer_indices']
    # task_indices: 优先使用 npz 中的, 否则从 task_rlds 查
    task_indices = data['task_indices'] if 'task_indices' in data.files else None
    print(f"Features: {features.shape}, unique_tasks={len(np.unique(task_labels))}")
    return features, task_labels, task_indices, layer_indices


def build_task_indices_from_labels(task_labels, task_rlds_path):
    """当 npz 中没有 task_indices 时, 从 task_rlds.jsonl 查表"""
    text_to_idx = {}
    with open(task_rlds_path) as f:
        for line in f:
            d = json.loads(line)
            text_to_idx[d['task'].lower()] = d['task_index']
    indices = np.array([text_to_idx.get(t.lower(), -1) for t in task_labels])
    valid = indices >= 0
    if valid.sum() < len(indices):
        print(f"  ⚠ {(~valid).sum()} 条轨迹无法映射 task_index, 已过滤")
    return indices, valid


def load_egohod_gallery(unique_task_indices, egohod_path):
    """
    构建 EgoHOD text gallery [C, 512]
    unique_task_indices: 排序后的 task_index 列表
    Returns: gallery [C, D_text], task_idx_to_class {task_idx: class_id}
    """
    egohod_emb = np.load(egohod_path)["embeddings"]  # [21938, 512]
    task_idx_sorted = sorted(unique_task_indices)
    task_idx_to_class = {idx: c for c, idx in enumerate(task_idx_sorted)}
    gallery = np.stack([egohod_emb[idx] for idx in task_idx_sorted])
    print(f"EgoHOD gallery: {gallery.shape} ({len(task_idx_sorted)} tasks)")
    return gallery, task_idx_to_class


def build_task_split(task_indices, train_ratio=0.8, seed=42):
    """按 task 划分 train/val"""
    rng = np.random.RandomState(seed)
    unique_tasks = np.array(sorted(set(task_indices)))
    rng.shuffle(unique_tasks)
    n_train = int(len(unique_tasks) * train_ratio)
    train_tasks = set(unique_tasks[:n_train].tolist())
    val_tasks = set(unique_tasks[n_train:].tolist())
    train_idx = np.array([i for i, t in enumerate(task_indices) if t in train_tasks])
    val_idx = np.array([i for i, t in enumerate(task_indices) if t in val_tasks])
    print(f"Split: {len(train_tasks)} train tasks ({len(train_idx)} traj), "
          f"{len(val_tasks)} val tasks ({len(val_idx)} traj)")
    return train_idx, val_idx, train_tasks, val_tasks


# ===================== 模型 =====================

class DualAligner(nn.Module):
    """
    双线性投影: action 和 text 各自过一层 Linear 投影到共享空间
    action: D_action → D_shared
    text:   D_text   → D_shared
    """
    def __init__(self, d_action, d_text, d_shared=512):
        super().__init__()
        self.proj_action = nn.Linear(d_action, d_shared)
        self.proj_text = nn.Linear(d_text, d_shared)
        # text 投影初始化为恒等矩阵，保留原始 text embedding 结构
        if d_text == d_shared:
            nn.init.eye_(self.proj_text.weight)
            nn.init.zeros_(self.proj_text.bias)

    def forward_action(self, x):
        return self.proj_action(x)

    def forward_text(self, x):
        return self.proj_text(x)


# ===================== 训练 / 评估 =====================

def train_one_epoch(model, optimizer, train_action, train_text_emb,
                    temperature, batch_size, device):
    """
    双投影 in-batch InfoNCE (CLIP 风格) 训练

    原理:
      - action 和 text 各自过 Linear 投影到共享空间
      - 投影后 L2 归一化，计算 cosine 相似度
      - 每个 batch 内，action_i 与 text_i 是正对，其他为负对
      - 双向 CE: action→text + text→action
    """
    model.train()
    perm = torch.randperm(len(train_action))
    total_loss, n_batches = 0.0, 0

    for i in range(0, len(train_action), batch_size):
        idx = perm[i:i + batch_size]
        action_batch = train_action[idx].to(device)   # [B, D_action]
        text_batch = train_text_emb[idx].to(device)    # [B, D_text]

        # 双投影 + L2 归一化
        proj_action = F.normalize(model.forward_action(action_batch), dim=-1)
        proj_text = F.normalize(model.forward_text(text_batch), dim=-1)

        # 相似度矩阵 [B, B]，对角线为正样本
        logits = proj_action @ proj_text.T / temperature
        labels = torch.arange(logits.shape[0], device=device)

        # 双向 InfoNCE
        loss = (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, action_feats, text_emb, class_indices, text_gallery, device):
    """
    评估:
      1. gallery retrieval: action 和 gallery text 各自投影后检索，计算 top-k accuracy
      2. mean cosine: 投影后与自身 text 投影的 cos sim
    """
    model.eval()
    # gallery 也过 text 投影层
    gallery_proj = F.normalize(model.forward_text(text_gallery), dim=-1)

    all_preds, all_cos = [], []
    bs = 1024

    for i in range(0, len(action_feats), bs):
        feat_batch = action_feats[i:i+bs].to(device)
        text_batch = text_emb[i:i+bs].to(device)

        proj_a = F.normalize(model.forward_action(feat_batch), dim=-1)
        proj_t = F.normalize(model.forward_text(text_batch), dim=-1)

        # gallery retrieval
        sim = proj_a @ gallery_proj.T
        all_preds.append(sim.argsort(dim=1, descending=True).cpu())

        # 与自身 text 投影的 cos sim
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

def train_single_layer(layer_feat, traj_text_emb, class_indices,
                       train_idx, val_idx,
                       text_gallery_t, layer_id, args, device, save_dir):
    """
    训练单层的 Linear 对齐投影 (标准 in-batch InfoNCE)

    Args:
        layer_feat: [N, D_action] 每条轨迹的 action 特征
        traj_text_emb: [N, D_text] 每条轨迹对应的 text embedding
        class_indices: [N] 每条轨迹对应的 gallery class index (用于 eval)
        text_gallery_t: [C, D_text] 全 gallery (用于 eval retrieval)
    """
    D_action = layer_feat.shape[1]
    D_text = text_gallery_t.shape[1]

    # 不做输入归一化，让投影层学习完整变换
    train_action = torch.from_numpy(layer_feat[train_idx]).float()
    train_text = torch.from_numpy(traj_text_emb[train_idx]).float()
    train_class = torch.from_numpy(class_indices[train_idx]).long()
    val_action = torch.from_numpy(layer_feat[val_idx]).float()
    val_text = torch.from_numpy(traj_text_emb[val_idx]).float()
    val_class = torch.from_numpy(class_indices[val_idx]).long()

    # 双投影: action 和 text 各自一层 Linear → 共享空间
    model = DualAligner(D_action, D_text, d_shared=D_text).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min)

    best_val_top1 = 0.0
    best_epoch = 0
    save_path = os.path.join(save_dir, f"aligner_{args.label}_layer{layer_id}.pt")

    print(f"\n--- Layer {layer_id}: DualAligner({D_action}→{D_text}, {D_text}→{D_text}), "
          f"in-batch InfoNCE (bs={args.batch_size}) ---")
    sys.stdout.flush()

    for epoch in range(args.epochs):
        # 训练: in-batch InfoNCE，每个 batch 内 action↔text 对比
        train_loss = train_one_epoch(
            model, optimizer, train_action, train_text,
            args.temperature, args.batch_size, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            # 评估: 在全 gallery 上做 retrieval
            train_m = evaluate(model, train_action, train_text,
                               train_class, text_gallery_t, device)
            val_m = evaluate(model, val_action, val_text,
                             val_class, text_gallery_t, device)
            print(f"  Epoch {epoch+1:>3d}: loss={train_loss:.4f}  "
                  f"tr_top1={train_m['top1']:.3f}  "
                  f"val_top1={val_m['top1']:.3f}  "
                  f"val_top5={val_m.get('top5', 0):.3f}  "
                  f"val_cos={val_m['mean_cos_sim']:.3f}")
            sys.stdout.flush()

            if val_m["top1"] > best_val_top1:
                best_val_top1 = val_m["top1"]
                best_epoch = epoch + 1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "layer": layer_id, "epoch": epoch + 1,
                    "val_metrics": val_m, "train_metrics": train_m,
                    "d_action": D_action, "d_text": D_text,
                }, save_path)

            # 定期保存 checkpoint (用于多阶段分析)
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                ep_path = os.path.join(save_dir,
                    f"aligner_{args.label}_layer{layer_id}_ep{epoch+1}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "layer": layer_id, "epoch": epoch + 1,
                    "val_metrics": val_m, "train_metrics": train_m,
                    "d_action": D_action, "d_text": D_text,
                }, ep_path)

    # 加载最优 checkpoint 做最终评估
    if best_val_top1 > 0:
        ckpt = torch.load(save_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # 如果 val_top1 始终为 0，保存最后一个 epoch 的模型
        best_epoch = args.epochs
        torch.save({
            "model_state_dict": model.state_dict(),
            "layer": layer_id, "epoch": args.epochs,
            "d_action": D_action, "d_text": D_text,
        }, save_path)

    final_train = evaluate(model, train_action, train_text,
                           train_class, text_gallery_t, device)
    final_val = evaluate(model, val_action, val_text,
                         val_class, text_gallery_t, device)

    print(f"  Layer {layer_id} best@epoch{best_epoch}: "
          f"val_top1={final_val['top1']:.4f}  val_top5={final_val.get('top5',0):.4f}")
    sys.stdout.flush()

    return {
        "layer": int(layer_id), "best_epoch": best_epoch,
        "train_metrics": final_train, "val_metrics": final_val,
        "save_path": save_path,
    }


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="通用 Action ↔ EgoHOD 线性对齐训练")
    parser.add_argument("--feature_path", type=str, required=True,
                        help="标准 npz 特征文件")
    parser.add_argument("--label", type=str, default=None,
                        help="模型标签 (默认从文件名推断)")
    parser.add_argument("--layer", type=str, default="all",
                        help="训练层: 'all' 或逗号分隔层号如 '10,12,17'")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--egohod_path", type=str, default=DEFAULT_EGOHOD_PATH)
    parser.add_argument("--task_rlds_path", type=str, default=DEFAULT_TASK_RLDS_PATH)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--eta_min", type=float, default=5e-6,
                        help="CosineAnnealing 最低学习率")
    parser.add_argument("--save_every", type=int, default=0,
                        help="每隔多少 epoch 保存一个 checkpoint (0=只保存最优)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 推断标签
    if args.label is None:
        args.label = os.path.basename(args.feature_path).replace(".npz", "").replace("_features", "")

    # ---- 1. 加载特征 ----
    features, task_labels, task_indices, layer_indices = load_feature_npz(args.feature_path)
    layer_list = [int(l) for l in layer_indices]

    # 如果没有 task_indices, 从 task_rlds 查表
    if task_indices is None:
        task_indices, valid_mask = build_task_indices_from_labels(
            task_labels, args.task_rlds_path)
        features = features[valid_mask]
        task_labels = task_labels[valid_mask]
        task_indices = task_indices[valid_mask]

    # 过滤 task_indices < 0 的无效条目
    valid = task_indices >= 0
    if valid.sum() < len(task_indices):
        print(f"  过滤 {(~valid).sum()} 条无效 task_indices")
        features = features[valid]
        task_labels = task_labels[valid]
        task_indices = task_indices[valid]

    # 确定训练层
    if args.layer == "all":
        target_layers = layer_list
    else:
        # 支持逗号分隔多层，如 "10,12,17"
        target_layers = [int(x) for x in args.layer.split(",")]
        for tl in target_layers:
            assert tl in layer_list, f"Layer {tl} not in {layer_list}"

    print("=" * 70)
    print(f"Action ↔ EgoHOD 线性对齐训练: {args.label}")
    print(f"  features: {features.shape}")
    print(f"  layers: {target_layers} (共 {len(target_layers)} 层)")
    print(f"  epochs={args.epochs}, lr={args.lr}, τ={args.temperature}")
    print("=" * 70)

    # ---- 2. 构建每条轨迹的 text embedding + gallery ----
    # 加载全量 EgoHOD embeddings
    egohod_all = np.load(args.egohod_path)["embeddings"]  # [21938, 512]
    D_text = egohod_all.shape[1]
    print(f"EgoHOD embeddings: {egohod_all.shape}")

    # 每条轨迹 → 对应的 text embedding (按 task_index 查表)
    traj_text_emb = egohod_all[task_indices]  # [N, D_text]

    # Gallery 用于评估 retrieval (只包含数据中出现的 task)
    unique_task_indices = sorted(set(task_indices.tolist()))
    gallery, task_idx_to_class = load_egohod_gallery(unique_task_indices, args.egohod_path)
    class_indices = np.array([task_idx_to_class[t] for t in task_indices])

    # 按 task 划分 train/val (测试泛化到未见 task)
    train_idx, val_idx, train_tasks, val_tasks = build_task_split(
        task_indices, train_ratio=args.train_ratio, seed=args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.output_dir, args.label)
    os.makedirs(save_dir, exist_ok=True)
    text_gallery_t = torch.from_numpy(gallery).float().to(device)

    # ---- 3. 逐层训练 (in-batch InfoNCE) ----
    all_results = []
    for layer_id in target_layers:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        li = layer_list.index(layer_id)
        layer_feat = features[:, li, :]  # [N, D_action]

        result = train_single_layer(
            layer_feat, traj_text_emb, class_indices,
            train_idx, val_idx,
            text_gallery_t, layer_id, args, device, save_dir)
        all_results.append(result)

    # ---- 4. 汇总 ----
    print(f"\n{'=' * 70}")
    print(f"所有层对齐结果: {args.label}")
    print(f"  {'Layer':>5s}  {'Epoch':>5s}  {'tr_top1':>7s}  {'tr_top5':>7s}  "
          f"{'val_top1':>8s}  {'val_top5':>8s}  {'val_cos':>7s}")
    for r in all_results:
        tm, vm = r["train_metrics"], r["val_metrics"]
        print(f"  {r['layer']:>5d}  {r['best_epoch']:>5d}  "
              f"{tm['top1']:>7.4f}  {tm.get('top5',0):>7.4f}  "
              f"{vm['top1']:>8.4f}  {vm.get('top5',0):>8.4f}  "
              f"{vm['mean_cos_sim']:>7.4f}")
    print(f"{'=' * 70}")

    # 保存 JSON 汇总
    summary = {
        "label": args.label, "feature_path": args.feature_path,
        "n_total_tasks": len(unique_task_indices),
        "n_train_tasks": len(train_tasks), "n_val_tasks": len(val_tasks),
        "n_train_traj": len(train_idx), "n_val_traj": len(val_idx),
        "d_action": int(features.shape[-1]), "d_text": int(gallery.shape[1]),
        "epochs": args.epochs, "lr": args.lr, "temperature": args.temperature,
        "per_layer_results": {
            int(r["layer"]): {
                "best_epoch": r["best_epoch"],
                "train_metrics": {k: float(v) for k, v in r["train_metrics"].items()},
                "val_metrics": {k: float(v) for k, v in r["val_metrics"].items()},
                "save_path": r["save_path"],
            } for r in all_results
        },
    }
    layer_tag = f"layer{target_layers[0]}" if len(target_layers) == 1 else "all_layers"
    json_path = os.path.join(save_dir, f"alignment_results_{layer_tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"结果: {json_path}")


if __name__ == "__main__":
    main()
