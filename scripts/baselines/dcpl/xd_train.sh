#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA="${DATA:-./data}"
CKPT="${CKPT:-./output}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
TRAINER="${TRAINER:-DCPL_biomedclip}"
export CUDA_VISIBLE_DEVICES="$GPU"
has_accuracy() { [[ -s "$1" ]] && grep -q '\* accuracy:' "$1"; }
checkpoint() { find "$1" -mindepth 2 -maxdepth 3 -type f -name "model.pth.tar-$2" -print -quit 2>/dev/null; }
# Usage: xd_train.sh [SOURCE=BTMRI1] [SEED=1]
DATASET="${1:-BTMRI1}"
SEED="${2:-1}"
SHOTS="${SHOTS:-16}"
EPOCH="${EPOCH:-100}"
EVAL_BS="${EVAL_BS:-16}"
CONFIG="$ROOT/configs/trainers/DCPL/few_shot/${DATASET}.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${DATASET}.yaml"
DIR="$CKPT/dcpl/cross_data/source/$DATASET/shots_${SHOTS}/$TRAINER/seed${SEED}"
[[ -f "$CONFIG" && -f "$DATASET_CONFIG" ]] || { echo "缺少来源配置: $DATASET" >&2; exit 2; }
if [[ -n "$(checkpoint "$DIR" "$EPOCH")" ]]; then
  if has_accuracy "$DIR/log.txt"; then echo "[skip] $DIR"; exit 0; fi
  echo "checkpoint 存在但指标缺失: $DIR" >&2; exit 1
fi
mkdir -p "$DIR"
"$PYTHON_BIN" "$ROOT/train.py" --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
  --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" --output-dir "$DIR" --mode train \
  DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES all DATALOADER.TEST.BATCH_SIZE "$EVAL_BS" OPTIM.MAX_EPOCH "$EPOCH"
has_accuracy "$DIR/log.txt" || { echo "来源指标缺失: $DIR" >&2; exit 1; }
[[ -n "$(checkpoint "$DIR" "$EPOCH")" ]] || { echo "来源 checkpoint 缺失: $DIR" >&2; exit 1; }
