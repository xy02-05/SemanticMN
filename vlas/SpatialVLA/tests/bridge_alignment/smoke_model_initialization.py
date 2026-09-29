#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import torch


SPATIALVLA_ROOT = Path(__file__).resolve().parents[2]
MIRROR_REPO_ROOT = SPATIALVLA_ROOT.parents[1]
EGO_ROOT = MIRROR_REPO_ROOT / "egovlpv2"

sys.path.insert(0, str(SPATIALVLA_ROOT))
sys.path.insert(0, str(EGO_ROOT))

from egovlpv2.utils.model_data_init import (
    init_alignment_model_components,
    init_embedding_components,
)
from model import (
    MIMICVLAConfig,
    MIMICVLAModel,
    SpatialVLAConfig,
    SpatialVLAForConditionalGeneration,
    SpatialVLAProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 SpatialVLA 预训练权重及 Bridge embedding/alignment 组件"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--alignment-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not args.alignment_config.is_file():
        raise FileNotFoundError(args.alignment_config)

    processor = SpatialVLAProcessor.from_pretrained(
        args.model_path, local_files_only=True
    )
    spatial_config = SpatialVLAConfig.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    mimic_config = MIMICVLAConfig.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_egovlpv2=False,
        use_alignment=True,
        vlm_loss_weight=0.0,
        alignment_loss_weight=1.0,
        egovlpv2_config_path=str(args.alignment_config),
        vlm_mode="embedding",
    )
    spatial_model = SpatialVLAForConditionalGeneration.from_pretrained(
        args.model_path,
        config=spatial_config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(args.device)
    embedding_components = init_embedding_components(
        str(args.alignment_config),
        device=args.device,
        dtype=torch.bfloat16,
    )
    alignment_components = init_alignment_model_components(
        str(args.alignment_config),
        device=args.device,
        dtype=torch.bfloat16,
    )
    model = MIMICVLAModel(
        config=mimic_config,
        vla_model=spatial_model,
        egovlpv2_components=embedding_components,
        alignment_components=alignment_components,
        vlm_mode="embedding",
    )

    assert model.spatial_vla.config.hidden_size == 2048
    assert model.spatial_vla.config.projection_dim == 2304
    assert model.alignment_layer_indices == [10]
    assert model.alignment_model.loss_type == "infonce"
    assert model.alignment_model.openvla_dim == 2304
    assert model.alignment_loss_weight == 1.0
    assert model.egovlpv2_model.output_dim in (512, 4096)
    assert next(model.spatial_vla.parameters()).device == torch.device(args.device)
    assert next(model.alignment_model.parameters()).device == torch.device(args.device)
    assert next(model.egovlpv2_model.parameters()).device == torch.device(args.device)
    assert processor.action_chunk_size == 4

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        "model_initialization_ok "
        f"text_dim={model.egovlpv2_model.output_dim} "
        f"action_dim={model.alignment_model.openvla_dim} "
        f"layer_indices={model.alignment_layer_indices} "
        f"parameters={total_parameters}"
    )


if __name__ == "__main__":
    main()
