#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for seed in 1 2 3; do
  for dataset in BTMRI1 CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do
    bash "$ROOT/scripts/medkcoop/few_shot.sh" "$dataset" "$seed" 16
  done
done
