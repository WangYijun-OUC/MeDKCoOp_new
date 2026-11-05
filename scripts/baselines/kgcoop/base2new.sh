#!/bin/bash
# custom config

DATA="/data/wangyijun/dataset"
# DATASET=$2
MODEL=CLIP
METHOD=KgCoOp
TRAINER=KgCoOp_${MODEL}
CFG=vit_b16_base2new

SHOTS=16
LOADEP=50
CTP=end
CSC=False
NCTX=4
SUB_base=base
SUB_novel=new
Device_id=1

for SEED in 1 2 3
do
for DATASET in BTMRI CHMNIST covid ctkidney dermamnist kneexray kvasir lungcolon octmnist retina CCBTM BTMRI1
do 
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/train_${SUB_base}/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/${CFG}.yaml \
    --mode train \
    --device-id ${Device_id} \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_base}
fi
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
MODEL_DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/train_${SUB_base}/${COMMON_DIR}
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/test_${SUB_novel}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --mode test \
    --device-id ${Device_id} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_novel}
fi
done
done

