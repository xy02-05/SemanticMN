# Mirror Neuron: code-only snapshot

This directory is a code-only snapshot of the Mirror Neuron training and
representation-analysis workspace. It intentionally excludes datasets,
pretrained weights, checkpoints, extracted features, logs, figures, and
evaluation outputs.

## Directory map

```text
mirror_neuron_final/
├── egovlpv2/                 # shared alignment, DSN and VICReg modules
├── vlas/
│   ├── openpi/               # pi0/OpenPI integration and launchers
│   ├── SpatialVLA/           # SpatialVLA integration and launchers
│   └── Isaac-GR00T/          # vendored GR00T N1.6 Mirror Neuron integration
├── analysis/                 # feature extraction, probes, metrics and plots
├── data_process/             # analysis-oriented preprocessing pipelines
├── scripts/                  # workspace-level training/evaluation helpers
├── requirements/             # environment requirement snapshots
└── TRAINING.md               # training entry points and artifact contract
```

## Source selection

- `egovlpv2/` and `vlas/` come from the inner Mirror Neuron code repository.
- `analysis/` and `data_process/` come from the outer experiment workspace.
- `analysis/openpi/` and `analysis/SpatialVLA/` are retained as historical code
  snapshots because the analysis pipeline differs from the current training
  copies under `vlas/`. They must not be treated as the canonical training
  implementation.
- `vlas/Isaac-GR00T/` is copied as normal source code rather than as a nested
  Git repository, so the custom Mirror Neuron integration is self-contained.

## Excluded content

The copy excludes all nested Git metadata and generated/runtime content,
including `data`, `weights`, `runs`, `logs`, `cache`, `assets`, `checkpoints`,
`results`, `outputs`, media, model tensors, NumPy features, and archives.

No file in the source workspace was moved or deleted while creating this
snapshot.

## Snapshot provenance

The snapshot was assembled on 2026-09-30 from the following working trees and
includes their then-uncommitted source changes:

- Mirror Neuron core: branch `archive`, base commit `cb0fb9b`.
- Analysis pipeline: branch `master`, base commit `d6de7b9`.
- Isaac-GR00T integration: branch `n1.6.1-local`, base commit `9cc220d`.

The outer workspace was not a Git repository. Its datasets and generated
artifacts were used only to identify the source entry points; they were not
copied.

See [TRAINING.md](TRAINING.md) for the current training entry points. Several
historical scripts still contain machine-specific absolute paths; preserve
their recorded behavior until each experiment has a canonical portable
configuration.
