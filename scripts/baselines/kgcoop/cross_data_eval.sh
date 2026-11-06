#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${SOURCE:-BTMRI1}"
for seed in 1 2 3; do
  bash "$HERE/xd_train.sh" "$SOURCE" "$seed"
  for target in CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do bash "$HERE/xd_test.sh" "$target" "$seed" "$SOURCE"; done
done
