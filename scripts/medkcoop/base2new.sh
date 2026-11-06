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
EVAL_BS="${EVAL_BS:-16}"
TRAINER="${TRAINER:-GRAPH_DCPL_NO_TXT_VIS_GRAPH}"
CFG="${CFG:-vit_b16_0002_b50_randflip}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/trainers/medkcoop}"
PYTHON_BIN="${PYTHON_BIN:-/data/wangyijun/conda/medkcoop/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python

export CUDA_VISIBLE_DEVICES="$GPU"
export BIOMEDCLIP_DIR="${BIOMEDCLIP_DIR:-/data/wangyijun/model/BiomedCLIP}"
export MEDSAM_CKPT="${MEDSAM_CKPT:-/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth}"

BASE_DIR="$CKPT/base2new/$DATASET/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}"
TRAIN_DIR="$BASE_DIR/train_base"
TEST_DIR="$BASE_DIR/test_new"
mkdir -p "$TRAIN_DIR" "$TEST_DIR"

CONFIG="$CONFIG_DIR/base_to_novel/${DATASET}.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${DATASET}.yaml"
[ -f "$CONFIG" ] || { echo "缺少配置: $CONFIG"; exit 1; }
[ -f "$DATASET_CONFIG" ] || { echo "缺少配置: $DATASET_CONFIG"; exit 1; }

run_train () {
  "$PYTHON_BIN" "$ROOT/train.py" \
    --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
    --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
    --output-dir "$TRAIN_DIR" --mode train \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES base \
    DATALOADER.TEST.BATCH_SIZE "$EVAL_BS" OPTIM.MAX_EPOCH "$EPOCH"
}

run_eval () {
  "$PYTHON_BIN" "$ROOT/train.py" \
    --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
    --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
    --output-dir "$TEST_DIR" --model-dir "$TRAIN_DIR" --load-epoch "$EPOCH" \
    --mode test --eval-only \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES new \
    DATALOADER.TEST.BATCH_SIZE "$EVAL_BS"
}

has_accuracy () {
  [ -s "$1" ] && grep -q '\* accuracy:' "$1"
}

find_checkpoint () {
  find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type f -name "model.pth.tar-${EPOCH}" -print -quit
}

CKPT_FILE="$(find_checkpoint)"
if [ -n "$CKPT_FILE" ] && has_accuracy "$TRAIN_DIR/log.txt" && has_accuracy "$TEST_DIR/log.txt"; then
  echo "[skip] base 和 new 指标均已完成: $BASE_DIR"
  exit 0
fi

echo "== base2new: dataset=$DATASET seed=$SEED shots=$SHOTS gpu=$GPU trainer=$TRAINER =="
echo "   train -> $TRAIN_DIR"
echo "   test  -> $TEST_DIR"

if [ -n "$CKPT_FILE" ]; then
  if ! has_accuracy "$TRAIN_DIR/log.txt"; then
    echo "错误: checkpoint 已存在，但 base 指标缺失: $TRAIN_DIR" >&2
    echo "请先核查已有日志和 checkpoint，再选择新的 CKPT 目录重训。" >&2
    exit 1
  fi
  echo "[skip] 已有 checkpoint 和 base 指标: $CKPT_FILE"
else
  run_train
  has_accuracy "$TRAIN_DIR/log.txt" || { echo "错误: 训练结束但 base 指标缺失" >&2; exit 1; }
  CKPT_FILE="$(find_checkpoint)"
  [ -n "$CKPT_FILE" ] || { echo "错误: 训练结束但 epoch $EPOCH 的 checkpoint 缺失" >&2; exit 1; }
fi

run_eval
has_accuracy "$TEST_DIR/log.txt" || { echo "错误: 评测结束但 new 指标缺失" >&2; exit 1; }

echo "== 结果 =="
grep -E "\* accuracy:|\* total:" "$TRAIN_DIR/log.txt" | tail -2 || true
grep -E "\* accuracy:|\* total:" "$TEST_DIR/log.txt" | tail -2 || true
