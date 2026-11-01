#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
METHOD=${ACOSERL_INTERVENTION_METHOD:-independent_window}
TAG=${1:-guide_grasp_v2_$(date +%Y%m%d_%H%M%S)}
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid training tag" >&2; exit 2; }
EXTRA=()
case "$METHOD" in
  plain_rlpd|independent_window)
    if [[ -n "${ACOSERL_ACCEPTANCE_FILE:-}" || -n "${ACOSERL_HOLD_CERTIFICATE:-}" ]]; then
      echo "Only full_acoserl uses external Recovery credentials" >&2; exit 2
    fi
    ;;
  full_acoserl)
    : "${ACOSERL_ACCEPTANCE_FILE:?Current-source acceptance required for Recovery}"
    : "${ACOSERL_HOLD_CERTIFICATE:?Current-source static Hold certificate required}"
    EXTRA+=(--gate "$ACOSERL_ACCEPTANCE_FILE" --static-hold-certificate "$ACOSERL_HOLD_CERTIFICATE")
    ;;
  *) echo "Unknown training mode: $METHOD" >&2; exit 2 ;;
esac
case "${ACOSERL_GRASP_GUIDE:-1}" in
  1) EXTRA+=(--grasp-guide) ;;
  0) EXTRA+=(--no-grasp-guide) ;;
  *) echo "ACOSERL_GRASP_GUIDE must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "${ACOSERL_CHECKPOINT_MIRROR_ROOT:-}" ]]; then
  EXTRA+=(--checkpoint-mirror-root "$ACOSERL_CHECKPOINT_MIRROR_ROOT")
fi
source scripts/ascend16/activate.sh
export PYTHONDONTWRITEBYTECODE=1
TESTING_ROOT=$(CDPATH= cd -- "$ROOT/../.." && pwd)/testing
RUN_DIR=${ACOSERL_OUTPUT_ROOT:-$TESTING_ROOT/runs/formal}
LOG_DIR="$RUN_DIR/launcher_logs"
mkdir -p "$LOG_DIR"
if [[ -e "$RUN_DIR/$TAG" || -e "$LOG_DIR/$TAG.log" || -e "$LOG_DIR/$TAG.pid" ]]; then
  echo "Run tag already exists; old artifacts will not be overwritten" >&2; exit 2
fi
set -o noclobber
printf '%s\n' "$$" > "$LOG_DIR/$TAG.pid"
printf 'RUN_ROOT=%s\nLOG=%s\n' "$RUN_DIR" "$LOG_DIR/$TAG.log"
# The learner records the actual mount and store identity. A directory name
# alone is not a persistence guarantee; use a verified mounted volume here.
exec python -u scripts/ascend16/train.py --mode formal --tag "$TAG" \
  --intervention-method "$METHOD" --output-root "$RUN_DIR" "${EXTRA[@]}" \
  --checkpoint-seconds "${ACOSERL_CHECKPOINT_SECONDS:-3600}" \
  --no-grasp-budget "${ACOSERL_NO_GRASP_BUDGET:-10000}" \
  --utd 1 --visual resnet18_spatial84 --supervisor-version baseline \
  --demo-original-quota 96 --demo-positive-quota 2 --demo-admission official \
  > "$LOG_DIR/$TAG.log" 2>&1
