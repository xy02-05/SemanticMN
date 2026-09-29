#!/usr/bin/env python3

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def validate_embedding(
    embedding_path: Path,
    task_meta: list[dict],
    index_path: Path | None,
) -> tuple[int, int]:
    data = np.load(embedding_path, allow_pickle=True)
    embeddings = data["sentence_embeddings"]
    task_indices = data["task_index"]
    texts = data["texts"] if "texts" in data.files else None

    row_count = len(task_meta)
    expected_indices = np.arange(row_count, dtype=np.int64)
    if embeddings.shape[0] != row_count:
        raise ValueError(f"{embedding_path}: embedding rows != task meta rows")
    if not np.array_equal(task_indices, expected_indices):
        raise ValueError(f"{embedding_path}: task_index is not row-aligned")
    for row_index, row in enumerate(task_meta):
        if row["task_id"] == -1:
            if row["task_index"] != -1:
                raise ValueError(f"invalid task meta row {row_index} must use task_index=-1")
        elif row["task_index"] != row_index:
            raise ValueError(f"task meta task_index mismatch at row {row_index}")

    if texts is not None:
        for row, embedding_text in zip(task_meta, texts):
            if row["task_id"] != -1 and row["task"] != str(embedding_text):
                raise ValueError(
                    f"{embedding_path}: text mismatch at task_index={row['task_index']}"
                )

    if index_path is not None:
        index_rows = json.loads(index_path.read_text())
        if len(index_rows) != row_count:
            raise ValueError(f"{index_path}: index rows != task meta rows")
        for row_index, (row, index_row) in enumerate(zip(task_meta, index_rows)):
            if index_row["idx"] != row_index:
                raise ValueError(f"{index_path}: idx mismatch at row {row_index}")
            if index_row.get("task_index", index_row["idx"]) != row_index:
                raise ValueError(f"{index_path}: task_index mismatch at row {row_index}")
            if row["task_id"] != -1 and index_row["task_id"] != row["task_id"]:
                raise ValueError(f"{index_path}: task_id mismatch at row {row_index}")
            if row["task_id"] != -1 and index_row["text"] != row["task"]:
                raise ValueError(f"{index_path}: text mismatch at row {row_index}")

    data.close()
    return embeddings.shape


def build_dataset_mapping(dataset_root: Path, task_meta: list[dict]) -> list[int]:
    dataset_tasks = load_jsonl(dataset_root / "meta/tasks.jsonl")
    normalized_to_embedding = {}
    for embedding_index, row in enumerate(task_meta):
        normalized_text = row["task"].strip().lower()
        if normalized_text in normalized_to_embedding:
            raise ValueError(f"ambiguous normalized task text: {normalized_text!r}")
        normalized_to_embedding[normalized_text] = embedding_index

    mapping = []
    for dataset_index, row in enumerate(dataset_tasks):
        if row["task_index"] != dataset_index:
            raise ValueError(f"dataset task_index mismatch at row {dataset_index}")
        normalized_text = row["task"].strip().lower()
        if normalized_text not in normalized_to_embedding:
            raise ValueError(
                f"dataset task has no embedding match at task_index={dataset_index}: "
                f"{row['task']!r}"
            )
        mapping.append(normalized_to_embedding[normalized_text])
    return mapping


def validate_parquet_dataset(
    dataset_root: Path,
    task_meta: list[dict],
    dataset_to_embedding: list[int],
) -> Counter:
    info = json.loads((dataset_root / "meta/info.json").read_text())
    total_episodes = int(info["total_episodes"])
    chunk_size = int(info["chunks_size"])
    data_pattern = info["data_path"]
    task_counts = Counter()

    for episode_index in range(total_episodes):
        parquet_path = dataset_root / data_pattern.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        frame = pd.read_parquet(parquet_path, columns=["task_index"])
        task_indices = frame["task_index"].to_numpy(dtype=np.int64)
        if task_indices.min() < 0 or task_indices.max() >= len(dataset_to_embedding):
            raise IndexError(f"{parquet_path}: task_index outside LeRobot tasks.jsonl")
        task_counts.update(task_indices.tolist())

    if sum(task_counts.values()) != int(info["total_frames"]):
        raise ValueError("parquet frame count does not match meta/info.json")
    return task_counts


def validate_positive_mask(
    task_meta: list[dict],
    dataset_to_embedding: list[int],
) -> None:
    all_task_ids = torch.tensor([row["task_id"] for row in task_meta], dtype=torch.long)
    task_ids = all_task_ids[torch.tensor(dataset_to_embedding, dtype=torch.long)]
    valid_ids = task_ids[task_ids != -1]
    counts = Counter(valid_ids.tolist())
    repeated_id = next(task_id for task_id, count in counts.items() if count > 1)
    repeated_indices = (task_ids == repeated_id).nonzero(as_tuple=True)[0][:2]
    invalid_index = (task_ids == -1).nonzero(as_tuple=True)[0][0]
    probe_ids = task_ids[torch.cat([repeated_indices, invalid_index.view(1)])]

    valid = probe_ids != -1
    positive_mask = (probe_ids[:, None] == probe_ids[None, :]) & valid[:, None] & valid[None, :]
    if not positive_mask[0, 1] or not positive_mask[1, 0]:
        raise ValueError("same task_id samples are not treated as mutual positives")
    if positive_mask[-1].any() or positive_mask[:, -1].any():
        raise ValueError("task_id=-1 must not participate in positive pairs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-meta", type=Path, required=True)
    parser.add_argument("--qwen-embedding", type=Path, required=True)
    parser.add_argument("--qwen-index", type=Path, required=True)
    parser.add_argument("--egohod-embedding", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_meta = load_jsonl(args.task_meta)
    qwen_shape = validate_embedding(args.qwen_embedding, task_meta, args.qwen_index)
    egohod_shape = validate_embedding(args.egohod_embedding, task_meta, None)
    dataset_to_embedding = build_dataset_mapping(args.dataset_root, task_meta)
    validate_positive_mask(task_meta, dataset_to_embedding)
    task_counts = validate_parquet_dataset(
        args.dataset_root, task_meta, dataset_to_embedding
    )

    invalid_frames = sum(
        count
        for task_index, count in task_counts.items()
        if task_meta[dataset_to_embedding[task_index]]["task_id"] == -1
    )
    print(
        json.dumps(
            {
                "task_rows": len(task_meta),
                "dataset_task_rows": len(dataset_to_embedding),
                "task_classes": len({row["task_id"] for row in task_meta if row["task_id"] != -1}),
                "invalid_task_rows": sum(row["task_id"] == -1 for row in task_meta),
                "qwen_shape": qwen_shape,
                "egohod_shape": egohod_shape,
                "parquet_frames": sum(task_counts.values()),
                "used_task_indices": len(task_counts),
                "dataset_to_embedding_mapping": "passed",
                "invalid_task_frames": invalid_frames,
                "same_task_positive_mask": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
