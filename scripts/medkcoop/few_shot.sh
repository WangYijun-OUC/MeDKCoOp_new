#!/bin/bash

# custom config
DATA="/data/wangyijun/dataset"
TRAINER=MeDKCoOp_Time
MODEL=MeDKCoOp
# export CUDA_VISIBLE_DEVICES=0,8
# DATASET=$1
LOADEP=50
CFG=vit_b16_base2new
# CFG=vit_b16_c2_ep15_batch4_8_2ctx_graph
SHOTS=2
Device_id=7
NCTX=2
CSC=False
CTP=end

for SEED in 1
do
for DATASET in CHMNIST 
# covid ctkidney kvasir lungcolon octmnist retina BTMRI1
# for DATASET in CHMNIST
do 
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl_time/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${MODEL}/few_shot/${DATASET}.yaml  \
    --mode train \
    --device-id ${Device_id} \
    --output-dir ${DIR} \
    TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
    TRAINER.BIOMEDCOOP.CSC ${CSC} \
    TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
else
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${MODEL}/few_shot/${DATASET}.yaml  \
    --mode train \
    --device-id ${Device_id} \
    --output-dir ${DIR} \
    TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
    TRAINER.BIOMEDCOOP.CSC ${CSC} \
    TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
done
done