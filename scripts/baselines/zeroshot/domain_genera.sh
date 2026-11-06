#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for seed in 1 2 3; do
  for target in CHMNISTv2 covidv2 ctkidneyv2 lungcolonv2 retinav2 octmnistv2 kvasirv2; do bash "$HERE/xd_test.sh" "$target" "$seed"; done
done
