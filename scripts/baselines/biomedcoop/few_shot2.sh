#!/bin/bash

# custom config
DATA="/data/wangyijun/dataset"
DATASET=$2
SHOTS=4
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end

METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}
Device_id=9

for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do
DIR=/data/wangyijun/ckpt/BiomedCoOp/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml  \
            --output-dir ${DIR} \
            TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
            TRAINER.BIOMEDCOOP.CSC ${CSC} \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
done
done