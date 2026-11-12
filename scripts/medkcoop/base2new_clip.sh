#!/usr/bin/env bash
# Fig. 4 backbone ablations. PubMedCLIP and PMC-CLIP require their local weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINERS=(GRAPH_DCPL_CLIP GRAPH_DCPL_PUBMEDCLIP GRAPH_DCPL_PMCCLIP)
for trainer in "${TRAINERS[@]}"; do
  for seed in 1 2 3; do
    for dataset in BTMRI1 CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do
      CKPT="$CKPT/backbone_ablation" TRAINER="$trainer" \
        bash "$ROOT/scripts/medkcoop/base2new.sh" "$dataset" "$seed"
    done
  done
done
