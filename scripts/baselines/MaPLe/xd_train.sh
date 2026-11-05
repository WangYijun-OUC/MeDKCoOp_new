#!/bin/bash

#cd ../..

# custom config
DATA="/data/wangyijun/dataset"
DATASET=$1
MODEL=CLIP
METHOD=MaPLe
TRAINER=MaPLe_${MODEL}
CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets

SHOTS=16
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
Device_id=7
# LOADEP=50

for SEED in 1 2 3
do
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/CrossData/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/${METHOD}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."

else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/${CFG}.yaml \
    --mode train \
    --device-id ${Device_id} \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
done