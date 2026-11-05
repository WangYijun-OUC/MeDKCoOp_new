#!/bin/bash

#cd ../..

# custom config
DATA="/data/wangyijun/dataset"
MODEL=PMCCLIP
METHOD=GraphCLIP
TRAINER=GraphCLIP_${MODEL}

DATASET=$1

Device_id=6
CFG=rn50_ep25_b256_lr_0_001_adamw
# CFG=vit_b16_c2_ep15_batch4_8_2ctx_graph
SHOTS=16
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
# --descriptor-path ${DESCRIPTOR_PATH} \
LOADEP=25

for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
# for DATASET in BTMRI
do 
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
    --config-file configs/trainers/${METHOD}/${CFG}.yaml \
    --output-dir ${DIR} \
    --device-id ${Device_id} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/${COMMON_DIR}
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/test_new/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
    echo "Resuming..."

else
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES new
fi
done
done