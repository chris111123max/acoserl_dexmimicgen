#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
: "${ACOSERL_ACCEPTANCE_FILE:?Set the externally generated current-source acceptance JSON}"
: "${ACOSERL_HOLD_CERTIFICATE:?Set the externally generated static-hold qualification JSON}"
TAG=${1:-autoserl_reviewed_$(date +%Y%m%d_%H%M%S)}
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
source scripts/ascend16/activate.sh
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs/autoserl_baseline
# The Python learner holds the process-wide NPU flock for its full lifetime.
# No detached duplicate child is created inside this script.
printf '%s\n' "$TAG" > logs/autoserl_baseline/formal.tag
printf '%s\n' "$$" > logs/autoserl_baseline/learner.pid
exec python -u scripts/ascend16/train.py --mode formal --tag "$TAG" \
  --gate "$ACOSERL_ACCEPTANCE_FILE" \
  --static-hold-certificate "$ACOSERL_HOLD_CERTIFICATE" \
  --utd 1 --visual resnet18_spatial84 --supervisor-version baseline \
  --demo-original-quota 96 --demo-positive-quota 2 --demo-admission official \
  > "logs/autoserl_baseline/$TAG.log" 2>&1
