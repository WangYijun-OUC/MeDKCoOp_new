#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA="${DATA:-/data/wangyijun/dataset}"
CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data/wangyijun/conda/medkcoop/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
TRAINER="${TRAINER:-CoCoOp_BiomedCLIP}"
export CUDA_VISIBLE_DEVICES="$GPU"
has_accuracy() { [[ -s "$1" ]] && grep -q '\\* accuracy:' "$1"; }
checkpoint() { find "$1" -mindepth 2 -maxdepth 3 -type f -name "model.pth.tar-$2" -print -quit 2>/dev/null; }
# Usage: base2new.sh [DATASET=BTMRI1] [SEED=1]
DATASET="${1:-BTMRI1}"
SEED="${2:-1}"
SHOTS="${SHOTS:-16}"
EPOCH="${EPOCH:-50}"
EVAL_BS="${EVAL_BS:-16}"
CONFIG="$ROOT/configs/trainers/CoCoOp/vit_b16_base2new.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${DATASET}.yaml"
RUN_ROOT="$CKPT/cocoop/base2new/$DATASET/shots_${SHOTS}/$TRAINER/seed${SEED}"
TRAIN_DIR="$RUN_ROOT/train_base"
TEST_DIR="$RUN_ROOT/test_new"
[[ -f "$CONFIG" && -f "$DATASET_CONFIG" ]] || { echo "缺少配置: $DATASET" >&2; exit 2; }
if [[ -n "$(checkpoint "$TRAIN_DIR" "$EPOCH")" ]] && has_accuracy "$TRAIN_DIR/log.txt" && has_accuracy "$TEST_DIR/log.txt"; then echo "[skip] $RUN_ROOT"; exit 0; fi
if [[ -n "$(checkpoint "$TRAIN_DIR" "$EPOCH")" ]]; then
  has_accuracy "$TRAIN_DIR/log.txt" || { echo "checkpoint 存在但 base 指标缺失: $TRAIN_DIR" >&2; exit 1; }
else
  mkdir -p "$TRAIN_DIR"
  "$PYTHON_BIN" "$ROOT/train.py" --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
    --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" --output-dir "$TRAIN_DIR" --mode train \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES base DATALOADER.TEST.BATCH_SIZE "$EVAL_BS" OPTIM.MAX_EPOCH "$EPOCH"
  has_accuracy "$TRAIN_DIR/log.txt" || { echo "base 指标缺失: $TRAIN_DIR" >&2; exit 1; }
  [[ -n "$(checkpoint "$TRAIN_DIR" "$EPOCH")" ]] || { echo "checkpoint 缺失: $TRAIN_DIR" >&2; exit 1; }
fi
mkdir -p "$TEST_DIR"
"$PYTHON_BIN" "$ROOT/train.py" --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
  --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" --output-dir "$TEST_DIR" \
  --model-dir "$TRAIN_DIR" --load-epoch "$EPOCH" --mode test --eval-only \
  DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES new DATALOADER.TEST.BATCH_SIZE "$EVAL_BS"
has_accuracy "$TEST_DIR/log.txt" || { echo "new 指标缺失: $TEST_DIR" >&2; exit 1; }
