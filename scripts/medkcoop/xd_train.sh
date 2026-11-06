#!/usr/bin/env bash
# Cross-data source training. Usage: xd_train.sh [SOURCE=BTMRI1] [SEED=1].
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${1:-BTMRI1}"
SEED="${2:-1}"
DATA="${DATA:-/data/wangyijun/dataset}"
CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp}"
GPU="${GPU:-0}"
SHOTS="${SHOTS:-16}"
EPOCH="${EPOCH:-100}"
TRAINER="${TRAINER:-GRAPH_DCPL_NO_TXT_VIS_GRAPH}"
PYTHON_BIN="${PYTHON_BIN:-/data/wangyijun/conda/medkcoop/bin/python}"
RUN_TAG="${RUN_TAG:-}"
TAG_PREFIX="${RUN_TAG:+$RUN_TAG/}"
EXTRA_OPTS="${EXTRA_OPTS:-}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
CONFIG="$ROOT/configs/trainers/medkcoop/few_shot/${SOURCE}.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${SOURCE}.yaml"
DIR="$CKPT/cross_data/${TAG_PREFIX}source/$SOURCE/shots_${SHOTS}/$TRAINER/seed${SEED}"
[[ -f "$CONFIG" && -f "$DATASET_CONFIG" ]] || { echo "缺少来源配置: $SOURCE" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"
if [[ -d "$DIR" ]]; then
  CKPT_FILE="$(find "$DIR" -mindepth 2 -maxdepth 2 -type f -name "model.pth.tar-${EPOCH}" -print -quit)"
  if [[ -n "$CKPT_FILE" ]]; then
    if [[ -s "$DIR/log.txt" ]] && grep -q '\* accuracy:' "$DIR/log.txt"; then
      echo "[skip] 已完成: $DIR"; exit 0
    fi
    echo "checkpoint 已存在但指标缺失: $DIR" >&2; exit 1
  fi
fi
mkdir -p "$DIR"
"$PYTHON_BIN" "$ROOT/train.py" --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
  --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
  --output-dir "$DIR" --mode train \
  DATASET.NUM_SHOTS "$SHOTS" OPTIM.MAX_EPOCH "$EPOCH"
grep -q '\* accuracy:' "$DIR/log.txt" || { echo "来源训练缺少指标: $DIR" >&2; exit 1; }
