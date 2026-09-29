"""
为 LIBERO 任务文本离线生成 Qwen3-VL-Embedding 特征。

输出内容：
1. sentence_embeddings: 官方 pooling 句向量 [N, D]
2. cls_embeddings: 与 sentence_embeddings 相同的兼容字段 [N, D]
3. token_embeddings: token 级 hidden states [N, T_max, D]
4. attention_mask: token 有效位 [N, T_max]
5. input_ids: 对应 token id [N, T_max]
6. task_index / task_id / token_lengths
7. index json: 文本与 idx 的映射
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import tqdm

from egovlpv2.model.model_qwen3 import create_qwen3_model


def load_tasks(tasks_path: Path, max_tasks: int | None = None):
    rows = []
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    rows = sorted(rows, key=lambda x: x["task_index"])
    if max_tasks is not None:
        rows = rows[:max_tasks]
    return rows


def pad_sequences(token_list, mask_list, input_id_list, hidden_dim: int):
    token_lengths = [tokens.shape[0] for tokens in token_list]
    max_len = max(token_lengths)
    num_items = len(token_list)

    token_embeddings = np.zeros((num_items, max_len, hidden_dim), dtype=np.float32)
    attention_mask = np.zeros((num_items, max_len), dtype=np.int64)
    input_ids = np.zeros((num_items, max_len), dtype=np.int64)

    for idx, (tokens, mask, ids) in enumerate(zip(token_list, mask_list, input_id_list, strict=True)):
        seq_len = tokens.shape[0]
        token_embeddings[idx, :seq_len] = tokens
        attention_mask[idx, :seq_len] = mask
        input_ids[idx, :seq_len] = ids

    return token_embeddings, attention_mask, input_ids, np.asarray(token_lengths, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Generate LIBERO Qwen3 token embeddings")
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("/root/data/xuyuan1/dataset/physical-intelligence/libero/meta/tasks_with_id.jsonl"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/data/xuyuan1/dataset/Qwen3-VL-Embedding-8B"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("/root/data/xuyuan1/dataset/embedding/libero_qwen3_text_features.npz"),
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=Path("/root/data/xuyuan1/dataset/embedding/libero_qwen3_text_index.json"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks_path, args.max_tasks)
    if not tasks:
        raise ValueError(f"No tasks found in {args.tasks_path}")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)

    model = create_qwen3_model(
        model_path=str(args.model_path),
        device=args.device,
        dtype=torch.bfloat16,
        use_flash_attention=False,
        normalize_embeddings=True,
    )

    sentence_chunks = []
    cls_chunks = []
    token_list = []
    mask_list = []
    input_id_list = []
    index_rows = []

    for begin in tqdm.trange(0, len(tasks), args.batch_size, desc="Generate Qwen3 text embeddings"):
        batch_rows = tasks[begin:begin + args.batch_size]
        batch_texts = [row["task"] for row in batch_rows]
        bundle = model.compute_text_bundle({"text": batch_texts})

        sentence_embeddings = bundle["sentence_embeddings"].detach().float().cpu().numpy()
        cls_embeddings = bundle["cls_embeddings"].detach().float().cpu().numpy()
        token_embeddings = bundle["token_embeddings"].detach().float().cpu().numpy()
        attention_mask = bundle["attention_mask"].detach().cpu().numpy()
        input_ids = bundle["input_ids"].detach().cpu().numpy()

        for row_idx, row in enumerate(batch_rows):
            valid_len = int(attention_mask[row_idx].sum())
            sentence_chunks.append(sentence_embeddings[row_idx])
            cls_chunks.append(cls_embeddings[row_idx])
            token_list.append(token_embeddings[row_idx, :valid_len].astype(np.float32))
            mask_list.append(attention_mask[row_idx, :valid_len].astype(np.int64))
            input_id_list.append(input_ids[row_idx, :valid_len].astype(np.int64))
            index_rows.append(
                {
                    "idx": len(index_rows),
                    "task_index": row["task_index"],
                    "task_id": row["task_id"],
                    "text": row["task"],
                    "token_length": valid_len,
                }
            )

    hidden_dim = token_list[0].shape[-1]
    token_embeddings, attention_mask, input_ids, token_lengths = pad_sequences(
        token_list, mask_list, input_id_list, hidden_dim
    )

    np.savez_compressed(
        args.output_npz,
        sentence_embeddings=np.asarray(sentence_chunks, dtype=np.float32),
        cls_embeddings=np.asarray(cls_chunks, dtype=np.float32),
        token_embeddings=token_embeddings,
        attention_mask=attention_mask,
        input_ids=input_ids,
        task_index=np.asarray([row["task_index"] for row in tasks], dtype=np.int64),
        task_id=np.asarray([row["task_id"] for row in tasks], dtype=np.int64),
        token_lengths=token_lengths,
    )

    with args.output_index.open("w", encoding="utf-8") as f:
        json.dump(index_rows, f, ensure_ascii=False, indent=2)

    print(f"Saved npz to {args.output_npz}")
    print(f"Saved index to {args.output_index}")
    print(f"Generated {len(tasks)} tasks, hidden_dim={hidden_dim}, max_len={token_embeddings.shape[1]}")


if __name__ == "__main__":
    main()
