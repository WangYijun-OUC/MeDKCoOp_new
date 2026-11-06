#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA="${DATA:-/data/wangyijun/dataset}"
CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp}"
GPU="${GPU:-0}"
EVAL_BS="${EVAL_BS:-16}"
PYTHON_BIN="${PYTHON_BIN:-/data/wangyijun/conda/medkcoop/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
TRAINER="${TRAINER:-ZeroshotBiomedCLIP}"
export CUDA_VISIBLE_DEVICES="$GPU"
has_accuracy() { [[ -s "$1" ]] && grep -q '\* accuracy:' "$1"; }
run_eval() {
  local dataset="$1" config="$2" output="$3" subset="$4" seed="$5"
  local ds_config="$ROOT/configs/datasets/${dataset}.yaml"
  [[ -f "$config" && -f "$ds_config" ]] || { echo "缺少配置: $dataset" >&2; exit 2; }
  if has_accuracy "$output/log.txt"; then echo "[skip] $output"; return; fi
  mkdir -p "$output"
  "$PYTHON_BIN" "$ROOT/train.py" --root "$DATA" --seed "$seed" --trainer "$TRAINER" \
    --dataset-config-file "$ds_config" --config-file "$config" --output-dir "$output" \
    --mode test --eval-only DATASET.SUBSAMPLE_CLASSES "$subset" DATALOADER.TEST.BATCH_SIZE "$EVAL_BS"
  has_accuracy "$output/log.txt" || { echo "指标缺失: $output" >&2; exit 1; }
}
DATASET="${1:?需指定目标数据集}"
SEED="${2:-1}"
CONFIG="$ROOT/configs/trainers/Zeroshot/vit_b16.yaml"
run_eval "$DATASET" "$CONFIG" "$CKPT/zeroshot/cross_data/$DATASET/$TRAINER/seed${SEED}" all "$SEED"
