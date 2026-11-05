#!/bin/bash

#cd ../..

# custom config
DATA="/data/wangyijun/dataset"
DATASET=$1
SEED=$2
MODEL=BiomedCLIP
METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}
CFG=vit_b16_base2new

SHOTS=16
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
Device_id=5

DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/CrossData/evaluation/${TRAINER}_btmri1/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"

else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir /home/wangyijun/ckpt/Biomed_Graph_dcpl/CrossData/BTMRI1/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 100 \
    --mode test \
    --device-id ${Device_id} \
    --eval-only
fi