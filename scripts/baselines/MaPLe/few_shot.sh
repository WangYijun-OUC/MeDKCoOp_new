#!/bin/bash

# custom config
DATA="/data/wangyijun/dataset"
# DATASET=$2
NCTX=2
CSC=False
CTP=end
CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets

MODEL=CLIP
METHOD=MaPLe
TRAINER=MaPLe_${MODEL}
Device_id=6

for SHOTS in 16 8 4 2 1
do
for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do
DIR=/data/wangyijun/ckpt/MaPLe/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/${CFG}.yaml  \
            --output-dir ${DIR} \
            TRAINER.MAPLE.N_CTX ${NCTX} \
            TRAINER.MAPLE.CSC ${CSC} \
            TRAINER.MAPLE.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
done
done
done