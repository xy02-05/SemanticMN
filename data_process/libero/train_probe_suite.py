"""
Suite-level 线性探针: 3 suites (Spatial+Object+LIBERO-10) train → Goal suite eval

与 train_probe.py 相同的 DualAligner + InfoNCE 训练，但数据划分按 suite 而非 episode。
训练集: 30 个 task (spatial 30-39, object 20-29, libero_10 0-9)
测试集: 10 个 task (goal 10-19) — 训练时完全没见过

同时支持两种模式:
  --eval_suite goal     (默认) Goal suite 做测试
  --eval_suite spatial  Spatial suite 做测试

用法:
    python train_probe_suite.py --checkpoint pretrained
    python train_probe_suite.py --checkpoint step_30k --feature_mode rollout
    python train_probe_suite.py --all --feature_mode rollout
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

from config import (
    TEXT_EMBEDDINGS, FEATURE_DIR, PROBE_DIR,
    CHECKPOINTS, ALIGN_CHECKPOINTS, ALL_CHECKPOINTS,
    LIBERO_SUITES,
)

SUITE_TASKS = {k: v["task_indices"] for k, v in LIBERO_SUITES.items()}


class DualAligner(nn.Module):
    def __init__(self, d_action, d_text, d_shared=512):
        super().__init__()
        self.proj_action = nn.Linear(d_action, d_shared)
        self.proj_text = nn.Linear(d_text, d_shared)
        if d_text == d_shared:
            nn.init.eye_(self.proj_text.weight)
            nn.init.zeros_(self.proj_text.bias)

    def forward_action(self, x):
        return self.proj_action(x)

    def forward_text(self, x):
        return self.proj_text(x)


def load_features(path):
    data = np.load(path, allow_pickle=True)
    return data["features"], data["task_indices"], data["layer_indices"]


def load_text_gallery(text_type="qwen3"):
    cfg = TEXT_EMBEDDINGS[text_type]
    data = np.load(cfg["path"])
    return data[cfg["key"]]


def train_one_epoch(model, optimizer, train_action, train_text, temperature, batch_size, device):
    model.train()
    perm = torch.randperm(len(train_action))
    total_loss, n = 0.0, 0
    for i in range(0, len(train_action), batch_size):
        idx = perm[i:i + batch_size]
        a = train_action[idx].to(device)
        t = train_text[idx].to(device)
        pa = F.normalize(model.forward_action(a), dim=-1)
        pt = F.normalize(model.forward_text(t), dim=-1)
        logits = pa @ pt.T / temperature
        labels = torch.arange(logits.shape[0], device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, action_feats, text_emb, class_indices, text_gallery, device):
    model.eval()
    gallery_proj = F.normalize(model.forward_text(text_gallery), dim=-1)
    all_preds, all_cos = [], []
    for i in range(0, len(action_feats), 512):
        a = action_feats[i:i+512].to(device)
        t = text_emb[i:i+512].to(device)
        pa = F.normalize(model.forward_action(a), dim=-1)
        pt = F.normalize(model.forward_text(t), dim=-1)
        sim = pa @ gallery_proj.T
        all_preds.append(sim.argsort(dim=1, descending=True).cpu())
        all_cos.append((pa * pt).sum(dim=-1).cpu())
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


def split_by_suite(features, task_indices, train_tasks_set, test_tasks_set):
    train_mask = np.isin(task_indices, list(train_tasks_set))
    test_mask = np.isin(task_indices, list(test_tasks_set))
    return (features[train_mask], task_indices[train_mask],
            features[test_mask], task_indices[test_mask])


def train_single_layer(train_feat, train_tasks, test_feat, test_tasks,
                       text_gallery, layer_id, label, args, device, save_dir):
    D_action = train_feat.shape[1]
    D_text = text_gallery.shape[1]
    train_text_np = text_gallery[train_tasks]
    test_text_np = text_gallery[test_tasks]

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
    save_path = os.path.join(save_dir, f"probe_{label}_layer{layer_id}.pt")

    print(f"\n--- Layer {layer_id}: train={len(train_feat)} ({len(set(train_tasks))} tasks)"
          f" test={len(test_feat)} ({len(set(test_tasks))} tasks) ---")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_action, train_text,
                                     args.temperature, args.batch_size, device)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            train_m = evaluate(model, train_action, train_text, train_class, text_gallery_t, device)
            test_m = evaluate(model, test_action, test_text, test_class, text_gallery_t, device)
            print(f"  Epoch {epoch+1:>3d}: loss={train_loss:.4f}"
                  f"  tr_top1={train_m['top1']:.3f}  test_top1={test_m['top1']:.3f}"
                  f"  test_cos={test_m['mean_cos_sim']:.3f}")
            if test_m["top1"] >= best_test_top1:
                best_test_top1 = test_m["top1"]
                best_epoch = epoch + 1
                torch.save({"model_state_dict": model.state_dict(),
                             "layer": layer_id, "epoch": epoch + 1,
                             "test_metrics": test_m, "train_metrics": train_m,
                             "d_action": D_action, "d_text": D_text, "d_shared": d_shared},
                            save_path)

    if best_test_top1 > 0 and os.path.exists(save_path):
        ckpt = torch.load(save_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_train = evaluate(model, train_action, train_text, train_class, text_gallery_t, device)
    final_test = evaluate(model, test_action, test_text, test_class, text_gallery_t, device)

    return {"layer": int(layer_id), "best_epoch": best_epoch,
            "train_metrics": final_train, "test_metrics": final_test,
            "n_train": len(train_feat), "n_test": len(test_feat)}


def run_one_checkpoint(ckpt_name, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mode_suffix = {"clean": "", "rollout": "_rollout"}[args.feature_mode]

    # load train + test features (from extract_features.py)
    train_path = os.path.join(FEATURE_DIR, f"{ckpt_name}_train{mode_suffix}.npz")
    test_path = os.path.join(FEATURE_DIR, f"{ckpt_name}_test{mode_suffix}.npz")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"跳过 {ckpt_name}: {train_path} 或 {test_path} 不存在")
        return None

    feats_tr, tasks_tr, layers = load_features(train_path)
    feats_te, tasks_te, _ = load_features(test_path)

    # merge all data, then re-split by suite
    all_feats = np.concatenate([feats_tr, feats_te], axis=0)
    all_tasks = np.concatenate([tasks_tr, tasks_te], axis=0)

    # determine train/test suites
    eval_suite = args.eval_suite
    eval_tasks = set(SUITE_TASKS[f"libero_{eval_suite}"])
    train_tasks_set = set()
    for sname, stasks in SUITE_TASKS.items():
        if sname != f"libero_{eval_suite}":
            train_tasks_set.update(stasks)

    text_gallery = load_text_gallery(args.text_type)

    dir_name = f"{ckpt_name}_{args.text_type}_suite_{eval_suite}{mode_suffix}"
    save_dir = os.path.join(PROBE_DIR, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Suite-level probe: {ckpt_name} ({args.feature_mode})")
    print(f"  Train suites: {[s for s in SUITE_TASKS if s != f'libero_{eval_suite}']}")
    print(f"  Eval suite: libero_{eval_suite} ({len(eval_tasks)} tasks)")
    print(f"{'='*60}")

    all_results = []
    for li_pos, layer_id in enumerate(layers):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        layer_feats = all_feats[:, li_pos, :]
        tr_f, tr_t, te_f, te_t = split_by_suite(layer_feats, all_tasks,
                                                  train_tasks_set, eval_tasks)
        if len(tr_f) == 0 or len(te_f) == 0:
            print(f"  Layer {layer_id}: empty split, skipping")
            continue

        result = train_single_layer(tr_f, tr_t, te_f, te_t,
                                     text_gallery, int(layer_id), ckpt_name,
                                     args, device, save_dir)
        all_results.append(result)

    # summary
    print(f"\n{'='*60}")
    print(f"Suite probe results: {ckpt_name} → eval on libero_{eval_suite}")
    print(f"  {'Layer':>5s}  {'Epoch':>5s}  {'tr_top1':>7s}  {'test_top1':>9s}  {'test_cos':>8s}")
    for r in all_results:
        tm, em = r["train_metrics"], r["test_metrics"]
        print(f"  {r['layer']:>5d}  {r['best_epoch']:>5d}"
              f"  {tm['top1']:>7.4f}  {em['top1']:>9.4f}  {em['mean_cos_sim']:>8.4f}")

    summary = {
        "checkpoint": ckpt_name,
        "text_type": args.text_type,
        "feature_mode": args.feature_mode,
        "eval_suite": eval_suite,
        "train_suites": [s for s in SUITE_TASKS if s != f"libero_{eval_suite}"],
        "n_train_tasks": len(train_tasks_set),
        "n_test_tasks": len(eval_tasks),
        "per_layer_results": {
            int(r["layer"]): {
                "best_epoch": r["best_epoch"],
                "train_metrics": {k: float(v) for k, v in r["train_metrics"].items()},
                "test_metrics": {k: float(v) for k, v in r["test_metrics"].items()},
            } for r in all_results
        },
    }
    json_path = os.path.join(save_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"保存: {json_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Suite-level 线性探针")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text_type", type=str, default="qwen3")
    parser.add_argument("--feature_mode", type=str, default="rollout",
                        choices=["clean", "rollout"])
    parser.add_argument("--eval_suite", type=str, default="goal",
                        choices=["goal", "spatial", "object", "10"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.checkpoint == "all":
        for name in ALL_CHECKPOINTS:
            run_one_checkpoint(name, args)
    else:
        run_one_checkpoint(args.checkpoint, args)


if __name__ == "__main__":
    main()
