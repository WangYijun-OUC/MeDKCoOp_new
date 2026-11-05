#!/bin/bash

#cd ../..

# custom config

DATA="/data/wangyijun/dataset"
MODEL=PUBMEDCLIP
METHOD=DCPL
TRAINER=DCPL_${MODEL}

# DATASET=$1

CFG=vit_b16_c2_ep5_batch4_2ctx
# CFG=vit_b16_c2_ep15_batch4_8_2ctx_graph
SHOTS=16
Device_id=6
LOADEP=15
CTP=end
CSC=False
NCTX=4
SUB_base=base
SUB_novel=new
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
# --descriptor-path ${DESCRIPTOR_PATH} \

for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_${SUB_base}/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
# DIR=/home/wangyijun/ckpt/TEST/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/${CFG}.yaml \
    --output-dir ${DIR} \
    --mode train \
    --device-id ${Device_id} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_base}
fi
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
MODEL_DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_${SUB_base}/${COMMON_DIR}
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/test_${SUB_novel}/${COMMON_DIR}
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