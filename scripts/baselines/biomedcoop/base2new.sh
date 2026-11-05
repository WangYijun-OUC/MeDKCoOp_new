#!/bin/bash
# custom config

DATA="/data/wangyijun/dataset"
DATASET=$1
MODEL=PubMedCLIP
METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}
CFG=vit_b16_base2new

SHOTS=16
LOADEP=50
CTP=end
CSC=False
NCTX=4
SUB_base=base
SUB_novel=new
Device_id=9

for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_base}
fi
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/${COMMON_DIR}
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/test_${SUB_novel}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
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
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_novel}
fi
done
done

