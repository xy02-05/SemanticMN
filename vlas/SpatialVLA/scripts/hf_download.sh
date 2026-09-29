#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
models=(
    google/paligemma2-3b-pt-224
    Intel/zoedepth-nyu-kitti
    IPEC-COMMUNITY/spatialvla-4b-224-pt
    IPEC-COMMUNITY/spatialvla-4b-mix-224-pt
)
mkdir -p ../pretrained
for model in ${models[@]};
do
  echo downloading ${model}...
  huggingface-cli download --resume-download --local-dir-use-symlinks False ${model} \
  --local-dir ../pretrained/$(basename ${model})
done