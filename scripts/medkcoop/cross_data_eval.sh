#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${SOURCE:-BTMRI1}"
for seed in 1 2 3; do
  bash "$ROOT/scripts/medkcoop/xd_train.sh" "$SOURCE" "$seed"
  for target in CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do
    bash "$ROOT/scripts/medkcoop/xd_test.sh" "$target" "$seed" "$SOURCE"
  done
done
