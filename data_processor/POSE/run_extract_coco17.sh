#!/bin/bash
DATA_ROOT="${GAIT_DATA_ROOT:-../data}"
python3 rtmlib_extract_coco17.py \
    --input "${DATA_ROOT}/person_frames_png" \
    --output "${DATA_ROOT}/pose_coco17" \
    --device cuda \
    --ext png \
    --model rtmpose-l \
    --batch-size 32
