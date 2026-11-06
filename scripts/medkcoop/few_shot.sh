#!/usr/bin/env bash
# Train and evaluate one few-shot run: DATASET SEED SHOTS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA:-/data/wangyijun/dataset}"
CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp}"
DATASET="${1:-BTMRI1}"
SEED="${2:-1}"
SHOTS="${3:-16}"
GPU="${GPU:-0}"
TRAINER="${TRAINER:-GRAPH_DCPL_NO_TXT_VIS_GRAPH}"
EPOCH="${EPOCH:-100}"
CFG="${CFG:-vit_b16_base2new}"
CONFIG="$ROOT/configs/trainers/medkcoop/few_shot/${DATASET}.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${DATASET}.yaml"
DIR="$CKPT/few_shot/$DATASET/shots_${SHOTS}/$TRAINER/$CFG/seed${SEED}"
[[ -f "$CONFIG" && -f "$DATASET_CONFIG" ]] || { echo "缺少配置: $DATASET" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"
CKPT_FILE=""
if [[ -d "$DIR" ]]; then
  CKPT_FILE="$(find "$DIR" -mindepth 2 -maxdepth 2 -type f -name "model.pth.tar-${EPOCH}" -print -quit)"
fi
if [[ -n "$CKPT_FILE" ]]; then
  if [[ -s "$DIR/log.txt" ]] && grep -q '\* accuracy:' "$DIR/log.txt"; then
    echo "[skip] 已完成: $DIR"
    exit 0
  fi
  echo "checkpoint 已存在但指标缺失，请核查后使用新的 CKPT 目录重训: $DIR" >&2
  exit 1
fi
mkdir -p "$DIR"
python "$ROOT/train.py" \
  --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
  --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
  --output-dir "$DIR" --mode train \
  DATASET.NUM_SHOTS "$SHOTS" OPTIM.MAX_EPOCH "$EPOCH"
grep -q '\* accuracy:' "$DIR/log.txt" || { echo "训练结束但指标缺失: $DIR" >&2; exit 1; }
