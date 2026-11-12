#!/bin/bash
INPUT_PATH="${GAIT_DATA_ROOT:-./data}/person_frames_png"          
BS=1
GPU_IDS='0'
INPUT_SIZE='256,256'
SNAPSHOT_FROM='./model_best.pth'
NUM_CLASSES=12
OUTPUT_PATH="${GAIT_DATA_ROOT:-./data}/cdgnetparsing_output"
VIS='yes'                                  
HIERARCHICAL='yes'     

# Giới hạn số CPU thread (tránh PyTorch/NumPy chiếm hết toàn bộ core máy)
NUM_THREADS=8
export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export OPENBLAS_NUM_THREADS=${NUM_THREADS}


if [ "$HIERARCHICAL" = "yes" ]; then
    HIER_FLAG='--hierarchical'
else
    HIER_FLAG=''
fi

python inference_gaitdata.py --data-dir ${INPUT_PATH} \
    --gpu ${GPU_IDS} \
    --batch-size ${BS} \
    --input-size ${INPUT_SIZE} \
    --restore-from ${SNAPSHOT_FROM} \
    --num-classes ${NUM_CLASSES} \
    --output-path ${OUTPUT_PATH} \
    --vis ${VIS} \
    ${HIER_FLAG}