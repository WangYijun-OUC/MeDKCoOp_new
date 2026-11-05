#!/bin/bash
# custom config

DATA="/data/wangyijun/dataset"
DATASET=$1
MODEL=BiomedCLIP
METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}
CFG=vit_b16_base2new

SHOTS=16
LOADEP=100
CTP=end
CSC=False
NCTX=4
SUB_base=base
SUB_novel=new
Device_id=9

DATASET=BTMRI1
SEED=2

MODEL_DIR=/data/wangyijun/ckpt/BiomedCoOp/few_shot/BTMRI1/shots_16/BiomedCoOp_BiomedCLIP/nctx4_cscFalse_ctpend/seed2
DIR=/data/wangyijun/ckpt/BiomedCoOp/heatmap
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --heatmap True \
    --dataset ${DATASET} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_novel}
else
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --heatmap True \
    --dataset ${DATASET} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_novel}
fi