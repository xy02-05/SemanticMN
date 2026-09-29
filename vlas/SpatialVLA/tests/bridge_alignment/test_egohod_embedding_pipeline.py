#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


SPATIALVLA_ROOT = Path(__file__).resolve().parents[2]
MIRROR_REPO_ROOT = SPATIALVLA_ROOT.parents[1]
EGO_ROOT = MIRROR_REPO_ROOT / "egovlpv2"

sys.path.insert(0, str(SPATIALVLA_ROOT))
sys.path.insert(0, str(EGO_ROOT))

from data.dataset import load_task_mapping
from egovlpv2.model.loss import create_alignment_masks
from egovlpv2.utils.model_data_init import (
    init_alignment_model_components,
    init_egohod_training_components,
    init_embedding_components,
)
from egovlpv2.utils.model_forward import alignment_forward_pass_complete


DEFAULT_WORK_ROOT = Path("/mnt/bn/2d-videos/xy/work/mirror_neuron")
DEFAULT_FEATURES = DEFAULT_WORK_ROOT / "data/embedding/bridge_egohod_proj_text_features.npz"
DEFAULT_INDEX = DEFAULT_WORK_ROOT / "data/embedding/bridge_egohod_proj_text_index.json"
DEFAULT_TASKS = (
    DEFAULT_WORK_ROOT / "data/bridge_orig/bridge_orig_lerobot/meta/tasks_with_id.jsonl"
)
DEFAULT_CLIP_WEIGHT = DEFAULT_WORK_ROOT / "weights/egohod/clip/ViT-L-14-336px.pt"
DEFAULT_EGOHOD_WEIGHT = DEFAULT_WORK_ROOT / "weights/egohod/checkpoints/large_best.pt"

INVALID_LINE_INDICES = (9, 10, 5615, 5800, 6736, 8298, 9841, 20643)
RECOMPUTE_INDICES = (0, 1, 15, 48, 143, 21937)
PIPELINE_LINE_INDICES = (15, 48, 0, 9)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def make_common_sections() -> dict:
    return {
        "optimizer": {
            "type": "AdamW",
            "args": {"lr": 5e-5, "weight_decay": 0.01},
        },
        "loss": {"type": "NormSoftmaxLoss", "args": {}},
        "metrics": [],
        "trainer": {
            "epochs": 1,
            "max_samples_per_epoch": 1,
            "save_dir": "",
            "save_period": 1,
            "verbosity": 1,
            "monitor": "off",
            "early_stop": 1,
            "init_val": False,
            "neptune": False,
            "task_names": "Dual",
            "lora": {"use_lora": False, "lora_rank": 32, "lora_dropout": 0},
        },
        "visualizer": {"type": ""},
    }


def make_egohod_config(clip_weight: Path, egohod_weight: Path) -> dict:
    config = {
        "name": "Bridge_EgoHOD_Embedding_Recompute_Test",
        "n_gpu": 1,
        "model_type": "epic_charades",
        "use_checkpoint": False,
        "arch": {
            "type": "EgoHODModel",
            "args": {
                "video_params": {"model": "SpaceTimeTransformer", "num_frames": 4},
                "text_params": {"model": "clip"},
                "projection_dim": 512,
                "load_checkpoint": str(clip_weight),
                "project_embed_dim": 512,
                "num_frames": 4,
                "use_fast_conv1": True,
                "use_flash_attn": True,
                "context_length": 77,
                "vocab_size": 49408,
                "freeze_temperature": True,
                "egohod_checkpoint_path": str(egohod_weight),
            },
        },
    }
    config.update(make_common_sections())
    return config


def make_embedding_alignment_config(
    features_path: Path,
    index_path: Path,
    use_disentangle: bool = True,
    diff_weight: float = 0.075,
    recon_weight: float = 0.01,
) -> dict:
    config = {
        "name": "Bridge_EgoHOD_Embedding_SpatialVLA_Pipeline_Test",
        "n_gpu": 1,
        "model_type": "embedding",
        "use_checkpoint": False,
        "arch": {
            "type": "EmbeddingModel",
            "args": {
                "embeddings_path": str(features_path),
                "index_path": str(index_path),
                "normalize_embeddings": True,
                "preload_to_gpu": False,
            },
        },
        "alignment": {
            "use_alignment": True,
            "model_type": "AlignmentModel",
            "args": {
                "egovlpv2_dim": 512,
                "openvla_dim": 2304,
                "projection_dim": 512,
                "token_egovlpv2_dim": 512,
                "temperature": 0.07,
                "dropout": 0.0,
                "layer_norm_eps": 1e-5,
                "layer_indices": [10],
                "text_proj_config": {
                    "use_fc_projection": True,
                    "proj_num_layers": 1,
                    "proj_hidden_dim": 1024,
                },
                "action_proj_config": {
                    "use_fc_projection": False,
                    "proj_num_layers": 2,
                    "proj_hidden_dim": 1024,
                },
                "use_feature_bank": False,
                "feature_bank_size": 512,
                "alignment_mode": "task_id",
                "vla_mode": "spatialvla",
                "min_valid_ratio": 0.0,
                "loss_type": "infonce",
                "sigmoid_bias": 0.0,
                "learnable_temperature": False,
                "learnable_bias": False,
                "action_pool_mode": "learnable_query",
                "action_pool_config": {
                    "num_layers": 1,
                    "mlp_hidden_dim": 1024,
                    "dropout": 0.0,
                },
                "use_disentangle": use_disentangle,
                "disentangle_config": {
                    "position": "pre_pool",
                    "encoder_type": "mlp",
                    "hidden_dim": 512,
                    "diff_mode": "cosine",
                    "diff_weight": diff_weight,
                    "recon_weight": recon_weight,
                    "dropout": 0.0,
                },
                "use_distributed_negatives": False,
                "mode_config": {
                    "as2ts": {"enabled": True, "weight": 1.0},
                    "as2tt": {"enabled": False, "weight": 0.0, "temperature": 0.05},
                    "at2tt": {"enabled": False, "weight": 0.0},
                    "at2tt_soft": {
                        "enabled": False,
                        "weight": 0.0,
                        "attn_temperature": 0.02,
                    },
                },
            },
            "optimizer": {
                "type": "AdamW",
                "args": {"lr": 1e-4, "weight_decay": 0.01},
            },
            "training": {
                "enabled": True,
                "log_interval": 10,
                "save_interval": 1000,
                "backbone_update_start_pct": 0.1,
                "stop_last_pct": 0.0,
            },
        },
    }
    config.update(make_common_sections())
    return config


def write_config(config: dict, directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def verify_metadata(features_path: Path, index_path: Path, tasks_path: Path) -> None:
    tasks = load_jsonl(tasks_path)
    index_rows = json.loads(index_path.read_text(encoding="utf-8"))

    with np.load(features_path) as data:
        embeddings = data["sentence_embeddings"]
        task_index = data["task_index"]

    assert embeddings.shape == (21938, 512)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.array_equal(task_index, np.arange(len(task_index)))
    assert len(tasks) == len(index_rows) == len(embeddings)

    invalid_set = set(INVALID_LINE_INDICES)
    for line_index, (task_row, index_row) in enumerate(zip(tasks, index_rows, strict=True)):
        assert index_row["idx"] == line_index
        assert index_row["task_index"] == line_index
        if line_index in invalid_set:
            assert task_row["task_index"] == -1
            assert task_row["task_id"] == -1
            assert not index_row["text"]
            assert np.count_nonzero(embeddings[line_index]) == 0
        else:
            assert task_row["task_index"] == line_index
            assert task_row["task_id"] == index_row["task_id"]
            assert task_row["task"].strip().lower() == index_row["text"].strip().lower()
            assert np.linalg.norm(embeddings[line_index]) > 0.99

    data_root = tasks_path.parents[3]
    data_mix = tasks_path.parents[2].name
    lang_to_index, index_to_id = load_task_mapping(
        str(data_root), data_mix, tasks_path.name
    )
    for line_index in RECOMPUTE_INDICES:
        row = tasks[line_index]
        assert lang_to_index[row["task"].lower()] == line_index
        assert index_to_id[line_index] == row["task_id"]

    for line_index in INVALID_LINE_INDICES:
        row = tasks[line_index]
        if row["task"]:
            assert lang_to_index[row["task"].lower()] == -1
        assert index_to_id[-1] == -1

    print(
        "metadata_ok "
        f"rows={len(tasks)} valid={len(tasks) - len(INVALID_LINE_INDICES)} "
        f"invalid={len(INVALID_LINE_INDICES)}"
    )


def verify_egohod_recompute(
    features_path: Path,
    index_path: Path,
    tasks_path: Path,
    clip_weight: Path,
    egohod_weight: Path,
    device: str,
) -> None:
    if not device.startswith("cuda"):
        raise ValueError("EgoHOD large 权重复算必须使用 CUDA，避免 CPU 测试耗时失控")

    tasks = load_jsonl(tasks_path)
    index_rows = json.loads(index_path.read_text(encoding="utf-8"))
    texts = [tasks[index]["task"] for index in RECOMPUTE_INDICES]

    with tempfile.TemporaryDirectory(prefix="egohod_recompute_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = write_config(
            make_egohod_config(clip_weight, egohod_weight),
            temp_path,
            "egohod_recompute.json",
        )
        components = init_egohod_training_components(
            config_path=str(config_path),
            device=device,
            dtype=torch.bfloat16,
            training_mode=False,
        )
        model = components["model"]
        tokenizer = components["tokenizer"]
        input_ids = tokenizer(texts, truncate=True).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            recomputed = model.compute_text({"input_ids": input_ids})
        recomputed = F.normalize(recomputed.float(), dim=-1).cpu()

    with np.load(features_path) as data:
        all_embeddings = torch.from_numpy(data["sentence_embeddings"]).float()
    all_embeddings = F.normalize(all_embeddings, dim=-1)
    expected = all_embeddings[list(RECOMPUTE_INDICES)]

    cosine = (recomputed * expected).sum(dim=-1)
    max_abs = (recomputed - expected).abs().amax(dim=-1)
    similarity = recomputed @ all_embeddings.T

    for batch_index, task_index in enumerate(RECOMPUTE_INDICES):
        expected_similarity = similarity[batch_index, task_index]
        better_count = int((similarity[batch_index] > expected_similarity + 1e-6).sum())
        top_indices = similarity[batch_index].topk(3).indices.tolist()
        top_texts = [index_rows[index]["text"] for index in top_indices]
        print(
            f"recompute task_index={task_index} cosine={cosine[batch_index]:.6f} "
            f"max_abs={max_abs[batch_index]:.6f} rank={better_count + 1} "
            f"top_indices={top_indices} top_texts={top_texts}"
        )
        assert cosine[batch_index] > 0.999
        assert max_abs[batch_index] < 0.01
        assert better_count < 5

    print(
        f"egohod_recompute_ok min_cosine={cosine.min():.6f} "
        f"max_abs={max_abs.max():.6f}"
    )


def verify_training_pipeline(
    features_path: Path,
    index_path: Path,
    tasks_path: Path,
) -> None:
    tasks = load_jsonl(tasks_path)
    line_indices = list(PIPELINE_LINE_INDICES)
    task_indices = torch.tensor(
        [tasks[index]["task_index"] for index in line_indices], dtype=torch.long
    )
    task_ids = torch.tensor([tasks[index]["task_id"] for index in line_indices], dtype=torch.long)
    texts = [tasks[index]["task"] for index in line_indices]

    positive_mask, valid_mask, valid_pair_mask = create_alignment_masks(task_ids, "task_id")
    assert positive_mask[0, 1] == positive_mask[1, 0] == 1
    assert positive_mask[0].sum() == positive_mask[1].sum() == 2
    assert valid_mask.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert valid_pair_mask[-1].sum() == 0

    with tempfile.TemporaryDirectory(prefix="embedding_pipeline_") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = write_config(
            make_embedding_alignment_config(features_path, index_path),
            temp_path,
            "embedding_alignment.json",
        )
        embedding_components = init_embedding_components(
            config_path=str(config_path),
            device="cpu",
            dtype=torch.float32,
        )
        alignment_components = init_alignment_model_components(
            config_path=str(config_path),
            device="cpu",
            dtype=torch.float32,
        )

        embedding_model = embedding_components["model"]
        looked_up = embedding_model.compute_text_by_index(task_indices)
        with np.load(features_path) as data:
            source = torch.from_numpy(data["sentence_embeddings"]).float()
        source = F.normalize(source, dim=-1)
        assert torch.allclose(looked_up[:3], source[task_indices[:3]], atol=1e-6, rtol=1e-6)
        assert torch.allclose(looked_up[3], source[0], atol=1e-6, rtol=1e-6)

        action_hidden_states = torch.randn(
            27,
            len(line_indices),
            5,
            2304,
            dtype=torch.float32,
            requires_grad=True,
        )
        vla_output = SimpleNamespace(
            action_hidden_states=action_hidden_states,
            hidden_states=None,
        )
        vla_batch = {"lang": texts}
        dist_args = SimpleNamespace(rank=0, world_size=1)

        result = alignment_forward_pass_complete(
            alignment_model=alignment_components["alignment_model"],
            vla_output=vla_output,
            vla_batch=vla_batch,
            vla_tokenizer=None,
            egovlpv2_model=embedding_model,
            egovlpv2_tokenizer=None,
            layer_indices=[10],
            device_id="cpu",
            allgather_fn=embedding_components["allgather"],
            n_gpu=1,
            args=dist_args,
            task_index=task_indices,
            task_ids=task_ids,
            mode="embedding",
        )

        loss = result["loss"]
        loss_dict = result["loss_dict"]
        assert torch.isfinite(loss)
        assert loss.item() >= 0
        assert result["openvla_action_features_shape"] == (4, 1, 5, 2304)
        assert "as2ts_loss" in loss_dict
        assert "disentangle_total_loss" in loss_dict
        assert loss_dict["as2ts_pos_samples"] > 1.0

        loss.backward()
        assert action_hidden_states.grad is not None
        assert torch.isfinite(action_hidden_states.grad).all()
        assert action_hidden_states.grad[10].abs().sum() > 0
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in alignment_components["alignment_model"].parameters()
        )

    print(
        "pipeline_ok "
        f"loss={loss.item():.6f} as2ts={loss_dict['as2ts_loss']:.6f} "
        f"avg_pos={float(loss_dict['as2ts_pos_samples']):.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 Bridge EgoHOD embedding 的权重来源、指令映射和 SpatialVLA 对齐链路"
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--clip-weight", type=Path, default=DEFAULT_CLIP_WEIGHT)
    parser.add_argument("--egohod-weight", type=Path, default=DEFAULT_EGOHOD_WEIGHT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("metadata", "pipeline", "recompute", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.features, args.index, args.tasks):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.mode in ("metadata", "all"):
        verify_metadata(args.features, args.index, args.tasks)
    if args.mode in ("pipeline", "all"):
        verify_training_pipeline(args.features, args.index, args.tasks)
    if args.mode in ("recompute", "all"):
        for path in (args.clip_weight, args.egohod_weight):
            if not path.is_file():
                raise FileNotFoundError(path)
        verify_egohod_recompute(
            args.features,
            args.index,
            args.tasks,
            args.clip_weight,
            args.egohod_weight,
            args.device,
        )


if __name__ == "__main__":
    main()
