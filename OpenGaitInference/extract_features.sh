#!/bin/bash
# Step 1: extract embeddings of every CloGait sequence with each pretrained model.
# Run from the repository root:  bash extract_features.sh
# Output: output/pre_extracted_features/<config-name>.npy
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for CFG in deepgaitv2_gait3d gaitbase_gait3d swingait_gait3d smplgait_gait3d parsinggait_gait3d_parsing; do
    python opengait/pre_extracted_features.py --cfgs configs/clogait/${CFG}.yaml
done
