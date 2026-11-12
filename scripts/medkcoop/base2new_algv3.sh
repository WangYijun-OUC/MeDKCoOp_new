#!/usr/bin/env bash
# 改进版 MeDKCoOp（alg_v3）的 16-shot base-to-new：训练 base 类 → eval-only 评测 new 类。
#
# 与 base2new.sh 的区别只在默认值：本脚本预置了改进算法的全部配置，
# 直接运行即可复现 docs/RESULTS.md 第 5 节的数值。
#
# 改进点（相对论文原实现）：
#   N_CTX=8        learnable context 从 2 扩到 8（原实现梯度被切断，实际未训练）
#   LR=1e-2        学习率 4 倍（原 2.5e-3，16-shot 下每 epoch 仅 8 步，原值学不动）
#   LEARN_BETA     beta 从固定 0.3 改为可学习 sigmoid 标量
#   GRAD_IMGPROTO  图像原型去掉 .data，参与梯度
#   TRAIN_VISUAL_NET  visual_net 纳入 optimizer（原实现有梯度但不更新）
#
# 用法:
#   bash scripts/medkcoop/base2new_algv3.sh                 # 默认 BTMRI1, seed 1
#   bash scripts/medkcoop/base2new_algv3.sh BTMRI1 2        # 指定数据集与 seed
#   GPU=4 EPOCH=50 bash scripts/medkcoop/base2new_algv3.sh CHMNIST 1
#
# 可用环境变量: DATA / CKPT / GPU / SHOTS / EPOCH / EVAL_BS / CFG
#               NCTX / LR / PYTHON_BIN / BIOMEDCLIP_DIR / MEDSAM_CKPT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA:-/data/wangyijun/dataset}"
CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp_alg}"
DATASET="${1:-BTMRI1}"
SEED="${2:-1}"
GPU="${GPU:-0}"
SHOTS="${SHOTS:-16}"
EPOCH="${EPOCH:-50}"
# N_CTX=8 时 MedSAM 在 eval batch 16 会 OOM（实测），默认降到 8
EVAL_BS="${EVAL_BS:-8}"
TRAINER="${TRAINER:-MeDKCoOpFix}"
CFG="${CFG:-vit_b16_0002_b50_randflip}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/trainers/medkcoop}"
PYTHON_BIN="${PYTHON_BIN:-/data/wangyijun/conda/medkcoop/bin/python}"
RUN_TAG="${RUN_TAG:-alg_v3}"
NCTX="${NCTX:-8}"
LR="${LR:-0.01}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python

export CUDA_VISIBLE_DEVICES="$GPU"
export BIOMEDCLIP_DIR="${BIOMEDCLIP_DIR:-/data/wangyijun/model/BiomedCLIP}"
export MEDSAM_CKPT="${MEDSAM_CKPT:-/data/wangyijun/model/medsam/sam_vit_b_01ec64.pth}"
# MedSAM 特征缓存（按 256 维均值向量，约 1.7 KB/图）；分块 2 张以避免显存峰值
export MEDSAM_CACHE="${MEDSAM_CACHE:-1}"
export MEDSAM_CACHE_DIR="${MEDSAM_CACHE_DIR:-/data/wangyijun/ckpt/medsam_cache}"
export MEDSAM_CHUNK="${MEDSAM_CHUNK:-2}"

# 改进算法的 5 个开关
ALG_OPTS="TRAINER.GRAPHDCPL.N_CTX ${NCTX} OPTIM.LR ${LR} \
TRAINER.GRAPHDCPL.LEARN_BETA True \
TRAINER.GRAPHDCPL.GRAD_IMGPROTO True \
TRAINER.GRAPHDCPL.TRAIN_VISUAL_NET True"

BASE_DIR="$CKPT/base2new/${RUN_TAG:+$RUN_TAG/}$DATASET/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}"
TRAIN_DIR="$BASE_DIR/train_base"
TEST_DIR="$BASE_DIR/test_new"

CONFIG="$CONFIG_DIR/base_to_novel/${DATASET}.yaml"
DATASET_CONFIG="$ROOT/configs/datasets/${DATASET}.yaml"
[ -f "$CONFIG" ] || { echo "缺少配置: $CONFIG" >&2; exit 1; }
[ -f "$DATASET_CONFIG" ] || { echo "缺少配置: $DATASET_CONFIG" >&2; exit 1; }

run_train () {
  "$PYTHON_BIN" "$ROOT/train.py" \
    --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
    --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
    --output-dir "$TRAIN_DIR" --mode train \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES base \
    DATALOADER.TEST.BATCH_SIZE "$EVAL_BS" OPTIM.MAX_EPOCH "$EPOCH" $ALG_OPTS
}

run_eval () {
  "$PYTHON_BIN" "$ROOT/train.py" \
    --root "$DATA" --seed "$SEED" --trainer "$TRAINER" \
    --dataset-config-file "$DATASET_CONFIG" --config-file "$CONFIG" \
    --output-dir "$TEST_DIR" --model-dir "$TRAIN_DIR" --load-epoch "$EPOCH" \
    --mode test --eval-only \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES new \
    DATALOADER.TEST.BATCH_SIZE "$EVAL_BS" $ALG_OPTS
}

has_accuracy () {
  [ -s "$1" ] && grep -q '\* accuracy:' "$1"
}

find_checkpoint () {
  [ -d "$TRAIN_DIR" ] || return 0
  find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type f -name "model.pth.tar-${EPOCH}" -print -quit 2>/dev/null || true
}

CKPT_FILE="$(find_checkpoint)"
if [ -n "$CKPT_FILE" ] && has_accuracy "$TRAIN_DIR/log.txt" && has_accuracy "$TEST_DIR/log.txt"; then
  echo "[skip] base 和 new 指标均已完成: $BASE_DIR"
  exit 0
fi

echo "== base2new_algv3: dataset=$DATASET seed=$SEED shots=$SHOTS gpu=$GPU =="
echo "   trainer=$TRAINER  N_CTX=$NCTX  lr=$LR  eval_bs=$EVAL_BS"
echo "   train -> $TRAIN_DIR"
echo "   test  -> $TEST_DIR"

if [ -n "$CKPT_FILE" ]; then
  if ! has_accuracy "$TRAIN_DIR/log.txt"; then
    echo "错误: checkpoint 已存在，但 base 指标缺失: $TRAIN_DIR" >&2
    echo "请删除该目录后重训，或换一个 CKPT 目录。" >&2
    exit 1
  fi
  echo "[skip] 已有 checkpoint 和 base 指标: $CKPT_FILE"
else
  mkdir -p "$TRAIN_DIR"
  run_train
  has_accuracy "$TRAIN_DIR/log.txt" || { echo "错误: 训练结束但 base 指标缺失" >&2; exit 1; }
  CKPT_FILE="$(find_checkpoint)"
  [ -n "$CKPT_FILE" ] || { echo "错误: 训练结束但 epoch $EPOCH 的 checkpoint 缺失" >&2; exit 1; }
fi

mkdir -p "$TEST_DIR"
run_eval
has_accuracy "$TEST_DIR/log.txt" || { echo "错误: 评测结束但 new 指标缺失" >&2; exit 1; }

echo "== 结果（$DATASET seed$SEED）=="
echo -n "  base: "; grep '\* accuracy:' "$TRAIN_DIR/log.txt" | tail -1 | sed 's/.*accuracy: //'
echo -n "  new : "; grep '\* accuracy:' "$TEST_DIR/log.txt" | tail -1 | sed 's/.*accuracy: //'
