#!/usr/bin/env bash
# Convenience scheduler for maintainers. Public reproduction remains in each method directory.
# Usage: run_all_methods.sh TASK [DATASET=BTMRI1] [SEED=1] [SHOTS=16]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?base2new|few_shot|cross_data|domain_generalization}"
DATASET="${2:-BTMRI1}"
SEED="${3:-1}"
SHOTS="${4:-16}"
METHOD_DIRS=(biomedcoop coop cocoop kgcoop MaPLe dcpl graph_adapter zeroshot)

for method in "${METHOD_DIRS[@]}"; do
  case "$TASK" in
    base2new) bash "$ROOT/scripts/baselines/$method/base2new.sh" "$DATASET" "$SEED" ;;
    few_shot) bash "$ROOT/scripts/baselines/$method/few_shot.sh" "$DATASET" "$SEED" "$SHOTS" ;;
    cross_data) bash "$ROOT/scripts/baselines/$method/cross_data_eval.sh" ;;
    domain_generalization) bash "$ROOT/scripts/baselines/$method/domain_genera.sh" ;;
    *) echo "未知任务: $TASK" >&2; exit 2 ;;
  esac
done
