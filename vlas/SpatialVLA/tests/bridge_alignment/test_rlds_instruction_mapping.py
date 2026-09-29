#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch


SPATIALVLA_ROOT = Path(__file__).resolve().parents[2]
MIRROR_REPO_ROOT = SPATIALVLA_ROOT.parents[1]
EGO_ROOT = MIRROR_REPO_ROOT / "egovlpv2"

sys.path.insert(0, str(SPATIALVLA_ROOT))
sys.path.insert(0, str(EGO_ROOT))

from data.dataset import load_task_mapping
from egovlpv2.model.model_embedding import EmbeddingModel


tf.config.set_visible_devices([], "GPU")


def decode_text(value: tf.Tensor) -> str:
    raw = value.numpy()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从真实 Bridge RLDS episode 验证指令到 task_index 和 embedding 行的映射"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-file", default="tasks_with_id.jsonl")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.dataset_dir, args.data_root, args.features, args.index):
        if not path.exists():
            raise FileNotFoundError(path)

    lang_to_index, index_to_id = load_task_mapping(
        str(args.data_root), "bridge_orig", args.task_file
    )
    index_rows = json.loads(args.index.read_text(encoding="utf-8"))
    embedding_model = EmbeddingModel(
        embeddings_path=str(args.features),
        index_path=str(args.index),
        device="cpu",
        dtype=torch.float32,
        normalize_embeddings=True,
        preload_to_gpu=False,
    )
    with np.load(args.features) as data:
        source_embeddings = torch.from_numpy(data["sentence_embeddings"]).float()
    source_embeddings = torch.nn.functional.normalize(source_embeddings, dim=-1)

    builder = tfds.builder_from_directory(str(args.dataset_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)

    checked_episodes = 0
    checked_steps = 0
    unique_texts = {}
    invalid_texts = set()
    for episode in dataset.take(args.max_episodes):
        checked_episodes += 1
        for step in episode["steps"]:
            text = decode_text(step["language_instruction"])
            task_index = lang_to_index.get(text.lower(), -1)
            if task_index == -1:
                assert not text.strip() or text.lower() in lang_to_index
                invalid_texts.add(text)
                continue
            task_id = index_to_id[task_index]
            assert task_id >= 0
            index_row = index_rows[task_index]
            assert index_row["task_index"] == task_index
            assert index_row["task_id"] == task_id
            assert index_row["text"].strip().lower() == text.strip().lower()
            unique_texts[text] = task_index
            checked_steps += 1

    assert checked_episodes == args.max_episodes
    assert checked_steps > 0
    assert unique_texts

    task_indices = torch.tensor(list(unique_texts.values()), dtype=torch.long)
    looked_up = embedding_model.compute_text_by_index(task_indices)
    expected = source_embeddings.index_select(0, task_indices)
    assert torch.allclose(looked_up, expected, atol=1e-6, rtol=1e-6)

    print(
        "rlds_mapping_ok "
        f"episodes={checked_episodes} valid_steps={checked_steps} "
        f"unique_instructions={len(unique_texts)} invalid_instructions={len(invalid_texts)}"
    )


if __name__ == "__main__":
    main()
