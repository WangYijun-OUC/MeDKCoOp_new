#!/bin/bash

#cd ../..

# custom config
DATA="/data/wangyijun/dataset"
DATASET=$1
TRAINER=GRAPH_DCPL_NO_TXT_VIS_GRAPH1
MODEL=MeDKCoOp
CFG=vit_b16_base2new

CFG=vit_b16_base2new
SHOTS=16
# DESCRIPTOR_PATH=descriptors/descriptors_${DATASET}.json  # descriptor path
Device_id=7
# LOADEP=50

for SHOTS in 8 4 2 1
do
for SEED in 1 2 3
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do 
DIR=/data/wangyijun/ckpt/Biomed_Graph_dcpl_rebut/few-shot/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/${MODEL}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."

else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${MODEL}/few_shot/${DATASET}.yaml \
    --mode train \
    --device-id ${Device_id} \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
done
done
done