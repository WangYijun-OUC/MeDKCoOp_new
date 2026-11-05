CFG=vit_b16
DATA="/data/wangyijun/dataset"
# DATASET=$2
MODEL=BiomedCLIP
METHOD=Zeroshot
TRAINER=Zeroshot${MODEL}
Device_id=6

for SHOTS in 16 8 4 2 1
do
for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do
    DIR=/data/wangyijun/ckpt/BiomedCLIP/zero_shot/${DATASET}/shots_${SHOTS}/${METHOD}
    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only
done
done