#!/usr/bin/env bash
# Cross-data target evaluation. Usage: xd_test.sh TARGET SEED [SOURCE=BTMRI1].
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${1:?需指定目标数据集}"
SEED="${2:-1}"
SOURCE="${3:-BTMRI1}"
DATA="${DATA:-/data/wangyijun/dataset}"
CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp}"
GPU="${GPU:-0}"
SHOTS="${SHOTS:-16}"
EPOCH="${EPOCH:-100}"
TRAINER="${TRAINER:-GRAPH_DCPL_NO_TXT_VIS_GRAPH}"
PYTHON_BIN="${PYTHON_BIN:-/data/wangyijun/conda/medkcoop/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
MODEL_DIR="${MODEL_DIR:-$CKPT/cross_data/source/$SOURCE/shots_${SHOTS}/$TRAINER/seed${SEED}}"
CONFIG="$ROOT/configs/trainers/medkcoop/few_shot/${TARGET}.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${TARGET}.yaml"
DIR="$CKPT/cross_data/evaluation/$SOURCE/$TARGET/shots_${SHOTS}/$TRAINER/seed${SEED}"
[[ -f "$CONFIG" && -f "$DATASET_CONFIG" ]] || { echo "缺少目标配置: $TARGET" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] && [[ -n "$(find "$MODEL_DIR" -mindepth 2 -maxdepth 2 -type f -name "model.pth.tar-${EPOCH}" -print -quit)" ]] || { echo "来源模型缺失: $MODEL_DIR" >&2; exit 1; }
if [[ -s "$DIR/log.txt" ]] && grep -q '\* accuracy:' "$DIR/log.txt"; then
  echo "[skip] 已完成: $DIR"; exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$DIR"
"$PYTHON_BIN" "$ROOT/train.py" --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
  --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
  --output-dir "$DIR" --model-dir "$MODEL_DIR" --load-epoch "$EPOCH" \
  --mode test --eval-only \
  DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES all \
  TRAINER.GRAPHDCPL.TRANSFER_EVAL True
grep -q '\* accuracy:' "$DIR/log.txt" || { echo "目标评测缺少指标: $DIR" >&2; exit 1; }
