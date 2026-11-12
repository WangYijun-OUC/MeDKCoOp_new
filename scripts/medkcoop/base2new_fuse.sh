#!/usr/bin/env bash
# Table 5 component-combination ablations.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINERS=(GRAPH_DCPL_NO_TXT_VIS_FUSE GRAPH_DCPL_NO_VIS_FUSE_GRAPH GRAPH_DCPL_NO_VIS_GRAPH GRAPH_DCPL_NO_TXT_VIS_GRAPH1 GRAPH_DCPL_NO_TXT_GRAPH GRAPH_DCPL_NO_TXT_FUSE_GRAPH GRAPH_DCPL_NO_FUSE)
for trainer in "${TRAINERS[@]}"; do
  for seed in 1 2 3; do
    for dataset in BTMRI1 CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do
      CKPT="$CKPT/ablation" TRAINER="$trainer" \
        bash "$ROOT/scripts/medkcoop/base2new.sh" "$dataset" "$seed"
    done
  done
done
