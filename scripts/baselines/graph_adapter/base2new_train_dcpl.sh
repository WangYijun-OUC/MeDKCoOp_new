#!/bin/bash

#cd ../..

# custom config
DATA="/data/wangyijun/dataset"
MODEL=CLIP
METHOD=GraphCLIP
TRAINER=GraphCLIP

DATASET=$1

Device_id=6
CFG=rn50_ep25_b256_lr_0_001_adamw
# CFG=vit_b16_c2_ep15_batch4_8_2ctx_graph
SHOTS=16
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
# --descriptor-path ${DESCRIPTOR_PATH} \

for SEED in 1 2 3
do
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
# DIR=/home/wangyijun/ckpt/TEST/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."

else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --device-id ${Device_id} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
done