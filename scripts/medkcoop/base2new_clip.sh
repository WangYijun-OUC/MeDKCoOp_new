#!/bin/bash

#cd ../..

# custom config

DATA="/data/wangyijun/dataset"
TRAINER=GRAPH_DCPL_b07
MODEL=GRAPH_DCPL
# export CUDA_VISIBLE_DEVICES=0,8
# DATASET=$1
LOADEP=50
# CFG=vit_b16_base2new
CFG=vit_b16_0002_b50_randflip
# CFG=rn50_0001_b50_adamw_randflip
SHOTS=16
Device_id=6
# LOADEP=50
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
# --descriptor-path ${DESCRIPTOR_PATH} \

for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
# for DATASET in covid
do 
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
# DIR=/home/wangyijun/ckpt/TEST/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/medkcoop/base_to_novel/${DATASET}.yaml \
    --output-dir ${DIR} \
    --mode train \
    --device-id ${Device_id} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/medkcoop/base_to_novel/${DATASET}.yaml \
    --output-dir ${DIR} \
    --mode train \
    --device-id ${Device_id} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/${COMMON_DIR}
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/test_new/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/medkcoop/base_to_novel/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --mode test \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES new
fi
done
done