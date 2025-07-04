#!/usr/bin/env bash
set -e
set -x

BASE_DATA_DIR="/work/nvme/bbsg/xuxalan/datasets"

# Use specified checkpoint path, otherwise, default value
subfolder=${1:-"generation"}
ckpt=${2:-"checkpoints/marigold-v1-0"}

python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --dataset_config config/dataset/data_eth3d.yaml \
    --output_dir output/${subfolder}/eth3d/prediction \
    --processing_res 756 \
    --resample_method bilinear \