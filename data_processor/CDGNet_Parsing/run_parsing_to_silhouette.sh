#!/bin/bash

# ==============================================================================
# Shell script to convert CDGNet parsing results into binary silhouettes
# (Upper body or Lower body) for OpenGait dataset pretreatment.
# ==============================================================================

# Path to the CDGNet output directory containing 'Pred_parsing_results'
INPUT_PATH="${GAIT_DATA_ROOT:-../data}/cdgnetparsing_output/Pred_parsing_results"

# Body region to extract: 'upper', 'lower', or 'all'
#   upper : Head, Torso, Arms, Hands, Dress (classes: 1, 2, 3, 4, 5, 6, 11)
#   lower : Legs, Feet (classes: 7, 8, 9, 10)
#   all   : Full body silhouette (classes: 1 to 11)
PART='upper'
# Output directory where binary silhouettes (0 and 255) will be saved
OUTPUT_PATH="${GAIT_DATA_ROOT:-../data}/cdgnetparsing_output/silhouettes_${PART}"

# Number of parallel workers
WORKERS=8

# Run conversion
python parsing_to_silhouette.py \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --part "${PART}" \
    --workers "${WORKERS}"
