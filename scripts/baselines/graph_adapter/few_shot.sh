#!/bin/bash

# custom config
DATA="/data/wangyijun/dataset"
MODEL=CLIP
METHOD=GraphCLIP
TRAINER=GraphCLIP

# DATASET=$1
CFG=rn50_ep25_b256_lr_0_001_adamw_cross
# CTP=$3  # class token position (end or middle)
# NCTX=$4  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
# CSC=$6  # class-specific context (False or True)
CSC=False
CTP=end
NCTX=2
Device_id=6

for SHOTS in 16 8 4 2 1
do
for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do
    DIR=/data/wangyijun/ckpt/GraphAdapter/few_shot/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
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
        TRAINER.GRAPHADAPTER.N_CTX ${NCTX} \
        TRAINER.GRAPHADAPTER.CSC ${CSC} \
        TRAINER.GRAPHADAPTER.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
done
done