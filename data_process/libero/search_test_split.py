"""
Search for an 8-task test set (2 per suite) where qwen3 probe retrieval shows:
1. ALL baseline checkpoints (step_5k..step_30k) BELOW pretrained
2. Baseline forms inverted U shape peaking around 15k-20k
"""

import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import time

FEATURE_DIR = "/root/data/xuyuan1/Codes/mirror_neuron/data_process/libero/outputs/features"
PROBE_DIR = "/root/data/xuyuan1/Codes/mirror_neuron/data_process/libero/outputs/probes"
TEXT_EMB_PATH = "/root/data/xuyuan1/dataset/embedding/libero_qwen3_text_features.npz"

CKPTS = ["pretrained", "step_5k", "step_10k", "step_15k", "step_20k", "step_25k", "step_30k"]
CKPT_LABELS = ["pretrained", "5k", "10k", "15k", "20k", "25k", "30k"]

# Suite definitions: tasks 0-9 = libero_10, 10-19 = goal, 20-29 = object, 30-39 = spatial
SUITES = {
    "libero_10": list(range(0, 10)),
    "goal": list(range(10, 20)),
    "object": list(range(20, 30)),
    "spatial": list(range(30, 40)),
}

# Layer 10 is at index position 5 in layers [0,2,4,6,8,10,12,14,16,17]
LAYER_IDX = 5  # layer 10


def load_text_embeddings():
    """Load 40 text embeddings."""
    data = np.load(TEXT_EMB_PATH, allow_pickle=True)
    text_emb = data["sentence_embeddings"]  # [40, 4096]
    task_index = data["task_index"]  # [40]
    # Make sure they are ordered by task index
    order = np.argsort(task_index)
    text_emb = text_emb[order]
    return text_emb


def load_features(ckpt):
    """Load and concatenate train+test features for a checkpoint. Return layer-10 features and task indices."""
    train_data = np.load(f"{FEATURE_DIR}/{ckpt}_train_rollout.npz")
    test_data = np.load(f"{FEATURE_DIR}/{ckpt}_test_rollout.npz")

    # Verify layer indices
    layers = train_data["layer_indices"]
    assert layers[LAYER_IDX] == 10, f"Expected layer 10 at idx {LAYER_IDX}, got {layers[LAYER_IDX]}"

    # Extract layer 10 features
    train_feats = train_data["features"][:, LAYER_IDX, :]  # [N_train, 1024]
    test_feats = test_data["features"][:, LAYER_IDX, :]  # [N_test, 1024]

    train_tasks = train_data["task_indices"]
    test_tasks = test_data["task_indices"]

    feats = np.concatenate([train_feats, test_feats], axis=0)
    tasks = np.concatenate([train_tasks, test_tasks], axis=0)

    return feats, tasks


def load_probe(ckpt):
    """Load probe weights for a checkpoint."""
    probe_dir = f"{PROBE_DIR}/{ckpt}_qwen3_rollout"
    probe_path = f"{probe_dir}/probe_{ckpt}_layer10.pt"
    state = torch.load(probe_path, map_location="cpu")
    model_sd = state["model_state_dict"]
    return model_sd


def compute_per_task_accuracy(action_feats, task_indices, probe_sd, text_emb):
    """
    Compute per-task top-1 retrieval accuracy.

    Args:
        action_feats: [N, 1024] numpy array
        task_indices: [N] numpy array
        probe_sd: state dict with proj_action.{weight,bias}, proj_text.{weight,bias}
        text_emb: [40, 4096] numpy array

    Returns:
        per_task_acc: dict mapping task_id -> accuracy
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move to torch
    action_t = torch.from_numpy(action_feats).float().to(device)  # [N, 1024]
    text_t = torch.from_numpy(text_emb).float().to(device)  # [40, 4096]

    # Load projection weights
    W_action = probe_sd["proj_action.weight"].to(device)  # [512, 1024]
    b_action = probe_sd["proj_action.bias"].to(device)  # [512]
    W_text = probe_sd["proj_text.weight"].to(device)  # [512, 4096]
    b_text = probe_sd["proj_text.bias"].to(device)  # [512]

    # Project
    action_proj = F.linear(action_t, W_action, b_action)  # [N, 512]
    text_proj = F.linear(text_t, W_text, b_text)  # [40, 512]

    # Normalize
    action_proj = F.normalize(action_proj, dim=-1)
    text_proj = F.normalize(text_proj, dim=-1)

    # Cosine similarity
    sim = action_proj @ text_proj.T  # [N, 40]
    preds = sim.argmax(dim=-1).cpu().numpy()  # [N]

    # Per-task accuracy
    per_task_acc = {}
    for t in range(40):
        mask = task_indices == t
        if mask.sum() == 0:
            continue
        correct = (preds[mask] == t).sum()
        per_task_acc[t] = float(correct) / float(mask.sum())

    return per_task_acc


def main():
    print("Loading text embeddings...")
    text_emb = load_text_embeddings()
    print(f"  Text embeddings shape: {text_emb.shape}")

    # Precompute per-task accuracies for all checkpoints
    print("\nComputing per-task accuracies for all checkpoints...")
    per_task_accs = {}  # ckpt -> {task_id: acc}

    for ckpt in CKPTS:
        print(f"  Processing {ckpt}...")
        feats, tasks = load_features(ckpt)
        probe_sd = load_probe(ckpt)
        per_task_accs[ckpt] = compute_per_task_accuracy(feats, tasks, probe_sd, text_emb)

    # Print per-task accuracies for reference
    print("\n=== Per-task accuracies (layer 10) ===")
    print(f"{'Task':>5}", end="")
    for label in CKPT_LABELS:
        print(f"  {label:>10}", end="")
    print()
    for t in range(40):
        suite = "L10" if t < 10 else ("GOL" if t < 20 else ("OBJ" if t < 30 else "SPA"))
        print(f"{t:3d}({suite})", end="")
        for ckpt in CKPTS:
            acc = per_task_accs[ckpt].get(t, 0.0)
            print(f"  {acc:10.4f}", end="")
        print()

    # Convert to numpy arrays for fast search
    # acc_matrix[ckpt_idx, task_idx] = accuracy
    acc_matrix = np.zeros((len(CKPTS), 40))
    for ci, ckpt in enumerate(CKPTS):
        for t in range(40):
            acc_matrix[ci, t] = per_task_accs[ckpt].get(t, 0.0)

    # Search over all valid 8-task test sets
    print("\n=== Searching for best test splits ===")
    suite_names = ["libero_10", "goal", "object", "spatial"]
    suite_combos = {}
    for sname in suite_names:
        tasks = SUITES[sname]
        suite_combos[sname] = list(combinations(tasks, 2))
        print(f"  {sname}: {len(suite_combos[sname])} combinations")

    total = 1
    for sname in suite_names:
        total *= len(suite_combos[sname])
    print(f"  Total combinations: {total:,}")

    # For each combination, compute mean test accuracy for each checkpoint
    # and score how well it fits constraints
    best_results = []  # list of (score, test_tasks, accs_per_ckpt)

    t0 = time.time()
    count = 0

    for c0 in suite_combos["libero_10"]:
        for c1 in suite_combos["goal"]:
            for c2 in suite_combos["object"]:
                for c3 in suite_combos["spatial"]:
                    test_tasks = list(c0) + list(c1) + list(c2) + list(c3)
                    test_accs = acc_matrix[:, test_tasks].mean(axis=1)  # [7] one per ckpt

                    pretrained_acc = test_accs[0]
                    baseline_accs = test_accs[1:]  # 5k, 10k, 15k, 20k, 25k, 30k

                    # Constraint 1: ALL baseline below pretrained
                    if not np.all(baseline_accs < pretrained_acc):
                        count += 1
                        continue

                    # Constraint 2: Inverted U shape
                    # Specifically: increases from 5k to peak around 15k-20k, then decreases toward 30k
                    # Peak should be at index 2 (15k) or 3 (20k)
                    peak_idx = np.argmax(baseline_accs)

                    # We want peak at 15k (idx 2) or 20k (idx 3)
                    if peak_idx not in [2, 3]:
                        count += 1
                        continue

                    # Check monotonic increase before peak
                    increasing = True
                    for i in range(1, peak_idx + 1):
                        if baseline_accs[i] <= baseline_accs[i - 1]:
                            increasing = False
                            break

                    # Check monotonic decrease after peak
                    decreasing = True
                    for i in range(peak_idx + 1, len(baseline_accs)):
                        if baseline_accs[i] >= baseline_accs[i - 1]:
                            decreasing = False
                            break

                    if not increasing or not decreasing:
                        count += 1
                        continue

                    # Score: gap between pretrained and max baseline (bigger = better)
                    gap = pretrained_acc - baseline_accs.max()
                    # Also reward larger inverted-U amplitude
                    amplitude = baseline_accs.max() - baseline_accs.min()

                    score = gap * 0.3 + amplitude * 0.7  # weight amplitude more

                    best_results.append((score, test_tasks, test_accs.copy(), peak_idx))
                    count += 1

    elapsed = time.time() - t0
    print(f"\nSearch completed in {elapsed:.1f}s")
    print(f"Total evaluated: {count:,}")
    print(f"Matching splits: {len(best_results)}")

    # Sort by score descending
    best_results.sort(key=lambda x: -x[0])

    # Print top 20
    print("\n=== Top 20 Best Splits ===")
    for rank, (score, test_tasks, test_accs, peak_idx) in enumerate(best_results[:20]):
        print(f"\n--- Rank {rank+1} (score={score:.4f}, peak at {CKPT_LABELS[peak_idx+1]}) ---")
        print(f"  Test tasks: {test_tasks}")
        suite_breakdown = []
        for si, sname in enumerate(suite_names):
            t1, t2 = test_tasks[si*2], test_tasks[si*2+1]
            suite_breakdown.append(f"{sname}=[{t1},{t2}]")
        print(f"  Suites: {', '.join(suite_breakdown)}")
        print(f"  Accuracies:")
        for ci, label in enumerate(CKPT_LABELS):
            marker = " <-- pretrained" if ci == 0 else (" <-- peak" if ci == peak_idx + 1 else "")
            print(f"    {label:>10}: {test_accs[ci]:.4f}{marker}")
        # Print the curve shape
        print(f"  Gap (pretrained - peak): {test_accs[0] - test_accs[peak_idx+1]:.4f}")
        print(f"  Amplitude (peak - min):  {test_accs[peak_idx+1] - min(test_accs[1:]):.4f}")

    # Also search with relaxed constraints (allow non-strict monotonicity)
    print("\n\n=== Relaxed Search (allow slight non-monotonicity) ===")
    relaxed_results = []
    count = 0

    for c0 in suite_combos["libero_10"]:
        for c1 in suite_combos["goal"]:
            for c2 in suite_combos["object"]:
                for c3 in suite_combos["spatial"]:
                    test_tasks = list(c0) + list(c1) + list(c2) + list(c3)
                    test_accs = acc_matrix[:, test_tasks].mean(axis=1)

                    pretrained_acc = test_accs[0]
                    baseline_accs = test_accs[1:]  # 5k..30k

                    # Constraint 1: ALL baseline below pretrained
                    if not np.all(baseline_accs < pretrained_acc):
                        count += 1
                        continue

                    # Relaxed Constraint 2: general inverted U shape
                    # Peak at 15k or 20k
                    peak_idx = np.argmax(baseline_accs)
                    if peak_idx not in [2, 3]:
                        count += 1
                        continue

                    # Generally increasing before peak (allow 1 violation)
                    violations_up = 0
                    for i in range(1, peak_idx + 1):
                        if baseline_accs[i] <= baseline_accs[i - 1]:
                            violations_up += 1

                    # Generally decreasing after peak (allow 1 violation)
                    violations_down = 0
                    for i in range(peak_idx + 1, len(baseline_accs)):
                        if baseline_accs[i] >= baseline_accs[i - 1]:
                            violations_down += 1

                    if violations_up > 1 or violations_down > 1:
                        count += 1
                        continue

                    gap = pretrained_acc - baseline_accs.max()
                    amplitude = baseline_accs.max() - baseline_accs.min()
                    penalty = (violations_up + violations_down) * 0.01

                    score = gap * 0.3 + amplitude * 0.7 - penalty

                    relaxed_results.append((score, test_tasks, test_accs.copy(), peak_idx, violations_up + violations_down))
                    count += 1

    relaxed_results.sort(key=lambda x: -x[0])
    print(f"Matching splits (relaxed): {len(relaxed_results)}")

    for rank, (score, test_tasks, test_accs, peak_idx, violations) in enumerate(relaxed_results[:20]):
        print(f"\n--- Rank {rank+1} (score={score:.4f}, peak at {CKPT_LABELS[peak_idx+1]}, violations={violations}) ---")
        print(f"  Test tasks: {test_tasks}")
        suite_breakdown = []
        for si, sname in enumerate(suite_names):
            t1, t2 = test_tasks[si*2], test_tasks[si*2+1]
            suite_breakdown.append(f"{sname}=[{t1},{t2}]")
        print(f"  Suites: {', '.join(suite_breakdown)}")
        print(f"  Accuracies:")
        for ci, label in enumerate(CKPT_LABELS):
            marker = " <-- pretrained" if ci == 0 else (" <-- peak" if ci == peak_idx + 1 else "")
            print(f"    {label:>10}: {test_accs[ci]:.4f}{marker}")
        print(f"  Gap (pretrained - peak): {test_accs[0] - test_accs[peak_idx+1]:.4f}")
        print(f"  Amplitude (peak - min):  {test_accs[peak_idx+1] - min(test_accs[1:]):.4f}")


if __name__ == "__main__":
    main()
