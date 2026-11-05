#!/bin/bash

# custom config
DATA="/data/wangyijun/dataset"
# DATASET=$2
# SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end
CFG=vit_b16

METHOD=CoCoOp
TRAINER=CoCoOp_${MODEL}
Device_id=6

for SHOTS in 16 8 4 2 1
do
for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do
        DIR=/data/wangyijun/ckpt/coop/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
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
            TRAINER.COCOOP.N_CTX ${NCTX} \
            TRAINER.COCOOP.CSC ${CSC} \
            TRAINER.COCOOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
done
done
done