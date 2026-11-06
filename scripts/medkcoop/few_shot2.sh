#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for shots in 1 2 4 8; do
  for seed in 1 2 3; do
    for dataset in BTMRI1 CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do
      bash "$ROOT/scripts/medkcoop/few_shot.sh" "$dataset" "$seed" "$shots"
    done
  done
done
