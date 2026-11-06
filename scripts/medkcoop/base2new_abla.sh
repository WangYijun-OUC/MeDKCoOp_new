#!/usr/bin/env bash
# Table 4 knowledge and text-intervention ablations.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINERS=(GRAPH_DCPL_NO_VIS_BIAS GRAPH_DCPL_NO_TXT_BIAS GRAPH_DCPL_NO_MEDSAM GRAPH_DCPL_GENERAL GRAPH_DCPL_INCORRECT GRAPH_DCPL_OUTOFDOMAIN)
for trainer in "${TRAINERS[@]}"; do
  for seed in 1 2 3; do
    for dataset in BTMRI1 CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do
      CKPT="${CKPT:-/data/wangyijun/ckpt/MeDKCoOp}/ablation" TRAINER="$trainer" \
        bash "$ROOT/scripts/medkcoop/base2new.sh" "$dataset" "$seed"
    done
  done
done
