#!/bin/bash

#cd ../..

# custom config
DATA="/data/wangyijun/dataset"
DATASET=$1
MODEL=CLIP
METHOD=GraphCLIP
TRAINER=GraphCLIP
CFG=rn50_ep25_b256_lr_0_001_adamw_cross

SHOTS=16
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
Device_id=3
# LOADEP=50

for SEED in 1 2 3
do
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/CrossData/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/${MODEL}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."

else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --mode train \
    --device-id ${Device_id} \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
done