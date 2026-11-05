#!/usr/bin/env bash
# 16-shot base-to-new：先在 base 类上训练，再用 --eval-only 在 new 类上评测。
#
# 用法:
#   bash scripts/medkcoop/base2new.sh [DATASET] [SEED]
#   GPU=1 SHOTS=16 EPOCH=50 bash scripts/medkcoop/base2new.sh BTMRI1 1
#
# 可用环境变量: DATA / CKPT / GPU / SHOTS / EPOCH / TRAINER / CFG
#               BIOMEDCLIP_DIR / MEDSAM_CKPT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA:-/data/wangyijun/dataset}"
CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp}"
DATASET="${1:-BTMRI1}"
SEED="${2:-1}"
GPU="${GPU:-0}"
SHOTS="${SHOTS:-16}"
EPOCH="${EPOCH:-50}"
TRAINER="${TRAINER:-MeDKCoOp}"
CFG="${CFG:-vit_b16_0002_b50_randflip}"

export CUDA_VISIBLE_DEVICES="$GPU"
export BIOMEDCLIP_DIR="${BIOMEDCLIP_DIR:-/data/wangyijun/model/BiomedCLIP}"
export MEDSAM_CKPT="${MEDSAM_CKPT:-/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth}"

BASE_DIR="$CKPT/base2new/$DATASET/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}"
TRAIN_DIR="$BASE_DIR/train_base"
TEST_DIR="$BASE_DIR/test_new"
mkdir -p "$TRAIN_DIR" "$TEST_DIR"

CONFIG="$ROOT/configs/trainers/medkcoop/base_to_novel/${DATASET}.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${DATASET}.yaml"
[ -f "$CONFIG" ] || { echo "缺少配置: $CONFIG"; exit 1; }
[ -f "$DATASET_CONFIG" ] || { echo "缺少配置: $DATASET_CONFIG"; exit 1; }

run_train () {
  python "$ROOT/train.py" \
    --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
    --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
    --output-dir "$TRAIN_DIR" --mode train \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES base
}

run_eval () {
  python "$ROOT/train.py" \
    --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
    --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
    --output-dir "$TEST_DIR" --model-dir "$TRAIN_DIR" --load-epoch "$EPOCH" \
    --mode test --eval-only \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES new
}

echo "== base2new: dataset=$DATASET seed=$SEED shots=$SHOTS gpu=$GPU trainer=$TRAINER =="
echo "   train -> $TRAIN_DIR"
echo "   test  -> $TEST_DIR"

CKPT_FILE="$TRAIN_DIR/MultiModalPromptLearner/model.pth.tar-${EPOCH}"
if [ -f "$CKPT_FILE" ]; then
  echo "[skip] 已有 checkpoint: $CKPT_FILE"
else
  run_train
fi

run_eval

echo "== 结果 =="
grep -E "\* accuracy:|\* total:" "$TRAIN_DIR/log.txt" | tail -2 || true
grep -E "\* accuracy:|\* total:" "$TEST_DIR/log.txt" | tail -2 || true
