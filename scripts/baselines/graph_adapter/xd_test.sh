#!/bin/bash

#cd ../..

# custom config
DATA="/data/wangyijun/dataset"
DATASET=$1
SEED=$2
MODEL=CLIP
METHOD=GraphCLIP
TRAINER=GraphCLIP
CFG=rn50_ep25_b256_lr_0_001_adamw_cross

SHOTS=16
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
Device_id=3
LOADEP=25

DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/CrossData/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"

else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir /home/wangyijun/ckpt/Biomed_Graph_dcpl/CrossData/BTMRI1/${TRAINER}/${CFG}_${SHOTS}shots/${MODEL}/seed${SEED} \
    --mode test \
    --device-id ${Device_id} \
    --load-epoch ${LOADEP} \
    --eval-only
fi