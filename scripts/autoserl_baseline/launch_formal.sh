#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"

# A certificate authorizes Recovery only. Ordinary RLPD and independent Guide
# have their own runtime safety guards and do not require the P0 static Hold.
METHOD=${ACOSERL_INTERVENTION_METHOD:-independent_window}
TAG=${1:-autoserl_${METHOD}_$(date +%Y%m%d_%H%M%S)}
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid training tag" >&2; exit 2; }
EXTRA=()
case "$METHOD" in
  plain_rlpd|independent_window)
    if [[ -n "${ACOSERL_ACCEPTANCE_FILE:-}" || -n "${ACOSERL_HOLD_CERTIFICATE:-}" ]]; then
      echo "Only full_acoserl uses external Recovery credentials" >&2
      exit 2
    fi
    ;;
  full_acoserl)
    : "${ACOSERL_ACCEPTANCE_FILE:?Current-source physical acceptance is required for Recovery}"
    : "${ACOSERL_HOLD_CERTIFICATE:?Current-source static Hold certificate is required for Recovery}"
    EXTRA+=(--gate "$ACOSERL_ACCEPTANCE_FILE"
            --static-hold-certificate "$ACOSERL_HOLD_CERTIFICATE")
    ;;
  *)
    echo "Unknown ACo-SERL training mode: $METHOD" >&2; exit 2 ;;
esac

source scripts/ascend16/activate.sh
export PYTHONDONTWRITEBYTECODE=1
TESTING_ROOT=$(CDPATH= cd -- "$ROOT/../.." && pwd)/testing
LOG_DIR="$TESTING_ROOT/logs/autoserl_baseline"
RUN_DIR="$TESTING_ROOT/runs/formal"
mkdir -p "$LOG_DIR" "$RUN_DIR"
printf '%s\n' "$TAG" > "$LOG_DIR/formal.tag"
printf '%s\n' "$$" > "$LOG_DIR/$TAG.pid"
exec python -u scripts/ascend16/train.py --mode formal --tag "$TAG" \
  --intervention-method "$METHOD" \
  --output-root "$RUN_DIR" \
  "${EXTRA[@]}" \
  --utd 1 --visual resnet18_spatial84 --supervisor-version baseline \
  --demo-original-quota 96 --demo-positive-quota 2 --demo-admission official \
  > "$LOG_DIR/$TAG.log" 2>&1
