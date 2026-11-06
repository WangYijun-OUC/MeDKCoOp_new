#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA="${DATA:-/data/wangyijun/dataset}"
CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data/wangyijun/conda/medkcoop/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
TRAINER="${TRAINER:-CoOp_BiomedCLIP}"
export CUDA_VISIBLE_DEVICES="$GPU"
has_accuracy() { [[ -s "$1" ]] && grep -q '\* accuracy:' "$1"; }
checkpoint() { find "$1" -mindepth 2 -maxdepth 3 -type f -name "model.pth.tar-$2" -print -quit 2>/dev/null; }
# Usage: xd_test.sh TARGET SEED [SOURCE=BTMRI1]
DATASET="${1:?需指定目标数据集}"
SEED="${2:-1}"
SOURCE="${3:-BTMRI1}"
SHOTS="${SHOTS:-16}"
EPOCH="${EPOCH:-100}"
EVAL_BS="${EVAL_BS:-16}"
CONFIG="$ROOT/configs/trainers/CoOp/vit_b16.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${DATASET}.yaml"
MODEL_DIR="${MODEL_DIR:-$CKPT/coop/cross_data/source/$SOURCE/shots_${SHOTS}/$TRAINER/seed${SEED}}"
DIR="$CKPT/coop/cross_data/evaluation/$SOURCE/$DATASET/shots_${SHOTS}/$TRAINER/seed${SEED}"
[[ -f "$CONFIG" && -f "$DATASET_CONFIG" ]] || { echo "缺少目标配置: $DATASET" >&2; exit 2; }
[[ -n "$(checkpoint "$MODEL_DIR" "$EPOCH")" ]] || { echo "来源 checkpoint 缺失: $MODEL_DIR" >&2; exit 2; }
if has_accuracy "$DIR/log.txt"; then echo "[skip] $DIR"; exit 0; fi
mkdir -p "$DIR"
"$PYTHON_BIN" "$ROOT/train.py" --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
  --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" --output-dir "$DIR" \
  --model-dir "$MODEL_DIR" --load-epoch "$EPOCH" --mode test --eval-only \
  DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES all DATALOADER.TEST.BATCH_SIZE "$EVAL_BS"
has_accuracy "$DIR/log.txt" || { echo "目标指标缺失: $DIR" >&2; exit 1; }
