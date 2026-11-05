#!/bin/bash

# custom config
DATA=/data/wangyijun/dataset
TRAINER=GraphCLIP_v1

DATASET=$1

CFG=vit_b16_base2new  # config file
CTP=end  # class token position (end or middle)
NCTX=2  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
SUB=new
Device_id=6
LOADEP=50

for SEED in 1 2 3
do
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/train_base/${COMMON_DIR}
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
    echo "Resuming..."

else
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/base_to_novel/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
done