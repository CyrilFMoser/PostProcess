#!/bin/bash

script="/mnt/g/Projects/RICS/PostProcess/src/external/PointTransformerV3/Pointcept/pointcept/datasets/preprocessing/sampling_chunking_data.py"
NUM_WORKERS=4
PROCESSED_SCANNETPP_DIR="/mnt/g/Projects/RICS/PostProcess/data/scenesplat/scannetppv2/scenes"

python ${script} --dataset_root ${PROCESSED_SCANNETPP_DIR} --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers ${NUM_WORKERS}
