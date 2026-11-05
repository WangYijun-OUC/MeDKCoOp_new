#!/bin/bash

#cd ../..

# custom config
DATA="/data/wangyijun/dataset"
TRAINER=DCPL
MODEL=DCPL_biomedclip

DATASET=$1

CFG=vit_b16_base2new
# CFG=vit_b16_c2_ep15_batch4_8_2ctx_graph
SHOTS=16
LOADEP=50
SUB=new
Device_id=9
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
# --descriptor-path ${DESCRIPTOR_PATH} \

for SEED in 1 2 3
do
COMMON_DIR=${DATASET}/shots_${SHOTS}/${MODEL}/${CFG}/seed${SEED}
MODEL_DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/train_base/${COMMON_DIR}
DIR=/home/wangyijun/ckpt/Biomed_Graph_dcpl/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${MODEL} \
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