#!/bin/bash

# custom config
DATA="/data/wangyijun/dataset"
# DATASET=$2
MODEL=BiomedCLIP
METHOD=Zeroshot
TRAINER=Zeroshot${MODEL}

CFG=vit_b16_base2new
SUB_base=base
SUB_novel=new
SHOTS=16
Device_id=9

for DATASET in CHMNIST covid ctkidney kneexray kvasir lungcolon octmnist retina BTMRI1
# for DATASET in covid
do 
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${METHOD}/${CFG}.yaml \
--output-dir ${DIR} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB_base}

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/train_base/${COMMON_DIR}
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/test_new/${COMMON_DIR}
CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${METHOD}/${CFG}.yaml \
--model-dir ${MODEL_DIR} \
--output-dir ${DIR} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB_novel}
done