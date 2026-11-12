#!/bin/bash
# Step 2: gallery/probe evaluation from the extracted embeddings.
# Usage (from the repository root):
#   bash evaluate.sh                # AllViews protocol reported in the paper
#   bash evaluate.sh View3          # same-view protocol, view 1..5
#   bash evaluate.sh CrossView      # all 20 cross-view pairs (all clothing)
# Results are appended to output/results_<protocol>.csv
set -e
PROTOCOL=${1:-AllViews}
PARTITIONS=datasets/CloGait/partitions/${PROTOCOL}
FEATURES=output/pre_extracted_features
RESULTS=output/results_${PROTOCOL}.csv

declare -A MODELS=(
    [DeepGaitV2]=deepgaitv2_gait3d
    [GaitBase]=gaitbase_gait3d
    [SwinGait]=swingait_gait3d
    [SMPLGait]=smplgait_gait3d
    [ParsingGait]=parsinggait_gait3d_parsing
)

for PARTITION in "${PARTITIONS}"/*.json; do
    for MODEL in "${!MODELS[@]}"; do
        echo "== ${MODEL} | $(basename "${PARTITION}")"
        python opengait/evaluate_gallery_probe.py \
            --cfgs "${PARTITION}" \
            --embedding_path "${FEATURES}/${MODELS[$MODEL]}.npy" \
            --results_path "${RESULTS}"
    done
done
