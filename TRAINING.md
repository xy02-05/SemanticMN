# Mirror Neuron training code

This repository keeps only code and lightweight experiment configuration in
Git. Datasets, pretrained weights, checkpoints, logs, caches, and generated
evaluation results stay outside version control.

## Layout

- `egovlpv2/egovlpv2/model/`: shared semantic-alignment, disentanglement, and
  VICReg implementations used by the VLA backbones.
- `egovlpv2/egovlpv2/configs/ft/`: alignment and ablation configurations.
- `vlas/openpi/`: pi0/OpenPI training integration.
- `vlas/SpatialVLA/`: SpatialVLA training integration.
- `vlas/Isaac-GR00T/`: vendored GR00T N1.6 source with the Mirror Neuron
  integration included directly in this snapshot.

## Runtime paths

This first code-only snapshot preserves the paths used by the completed runs.
Before running it on another machine, replace the historical
`/mnt/bn/2d-videos/xy/work/mirror_neuron` and environment paths with local
dataset, weight, output, and environment locations. Those runtime resources
are intentionally not included here.

## pi0 / OpenPI

The canonical launcher is:

```bash
cd vlas/openpi
CONFIG_NAME=pi0_libero_full_qwen_50k_gbs32 \
WANDB_ENABLED=false \
bash scripts/train_libero_50k_pair.sh smoke
```

Use `formal` instead of `smoke` for a full run. Supported configuration names
are validated by the launcher. The 30k-to-50k recovery queue is:

```bash
bash vlas/openpi/scripts/run_libero_30k_to_50k_queue.sh
```

Recovery from a checkpoint without `optimizer.pt` is opt-in through the
corresponding `TrainConfig`; model weights and metadata are still loaded
strictly, while AdamW is restarted with a short learning-rate warmup.

## SpatialVLA

Run one Bridge experiment with an explicit alignment config:

```bash
bash vlas/SpatialVLA/scripts/spatialvla_4b_finetune/train_bridge_embedding_experiment.sh \
  --run-name <run-name> \
  --alignment-config egovlpv2/egovlpv2/configs/ft/<config>.json \
  --cuda-devices 0,1 \
  --master-port 29710
```

The launcher writes a run manifest and a copy of the alignment config beside
the checkpoint before training starts.

## GR00T N1.6

GR00T is a nested repository based on NVIDIA Isaac-GR00T `n1.6.1-release`.
Its Mirror Neuron integration is pinned by the parent Git entry. Run it from
inside `vlas/Isaac-GR00T` with:

```bash
bash examples/SimplerEnv/finetune_mirror_neuron.sh
```

Set `ALIGNMENT_BACKEND`, `ALIGNMENT_CONFIG_PATH`, `ALIGNMENT_WEIGHT`,
`MAX_STEPS`, and storage-related environment variables as needed.

## What must be recorded for a reproducible run

Every reported run must retain:

1. this repository commit and the upstream revision recorded for each VLA;
2. exact training config and launcher snapshot;
3. dataset and pretrained-weight identifiers/checksums;
4. seed, GPU count, global batch size, optimizer, schedule, and step count;
5. checkpoint plus machine-readable run manifest.

Do not commit generated artifacts. Add new local output locations to
`.gitignore` instead of deleting historical files.
