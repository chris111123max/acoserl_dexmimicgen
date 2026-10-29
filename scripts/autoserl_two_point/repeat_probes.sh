#!/usr/bin/env bash
set -euo pipefail
ARM=$1
for SEED in 1 2 3; do
  EXTRA=()
  if [[ "$ARM" == B ]]; then EXTRA=(--prefix 58); fi
  python -u scripts/autoserl_two_point/probe.py --case "$ARM" --seed "$SEED" --offset-mm "$((SEED+2))" --label accepted "${EXTRA[@]}" > "outputs/autoserl_two_point/accepted_${ARM}_s${SEED}.log" 2>&1
done
