#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for seed in 1 2 3; do
  for target in CHMNIST covid ctkidney lungcolon retina octmnist kvasir; do bash "$HERE/xd_test.sh" "$target" "$seed"; done
done
