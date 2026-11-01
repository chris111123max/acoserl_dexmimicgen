#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
source scripts/ascend16/activate.sh
export PYTHONDONTWRITEBYTECODE=1
BASE=${ACOSERL_OUTPUT_ROOT:-$(dirname "$ROOT")/training_runs/acoserl_dexmimicgen/simple/formal}
METHOD=${ACOSERL_SIMPLE_METHOD:-autoserl}
TAG=${1:-simple_${METHOD}_$(date +%Y%m%d_%H%M%S)}
[[ "$METHOD" == autoserl || "$METHOD" == rlpd ]] || { echo 'Invalid simple method' >&2; exit 2; }
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid tag' >&2; exit 2; }
# Resolve relative paths before starting Python; generated files never belong
# inside the tracked checkout. No credentials or proof JSON are fabricated.
BASE=$(realpath -m -- "$BASE")
case "$BASE/" in "$ROOT/"*) echo 'Output must be outside the repository' >&2; exit 2 ;; esac
mkdir -p "$BASE/launcher_logs"
[[ ! -e "$BASE/$TAG" && ! -e "$BASE/launcher_logs/$TAG.log" && ! -e "$BASE/launcher_logs/$TAG.pid" ]] || {
  echo 'Existing run tag will not be overwritten' >&2; exit 2;
}
EXTRA=()
if [[ -n "${ACOSERL_CHECKPOINT_MIRROR_ROOT:-}" ]]; then
  EXTRA+=(--checkpoint-mirror-root "$ACOSERL_CHECKPOINT_MIRROR_ROOT")
fi
set -o noclobber
printf '%s\n' "$$" > "$BASE/launcher_logs/$TAG.pid"
printf 'RUNTIME=simple_bimanual_autoserl_v1\nRUN=%s\nLOG=%s\n' "$BASE/$TAG" "$BASE/launcher_logs/$TAG.log"
exec python -u scripts/autoserl_baseline/train_simple.py --mode formal --method "$METHOD" \
  --tag "$TAG" --output-root "$BASE" --visual resnet18_spatial84 --utd 1 \
  --checkpoint-seconds "${ACOSERL_CHECKPOINT_SECONDS:-3600}" \
  --no-grasp-budget "${ACOSERL_NO_GRASP_BUDGET:-10000}" "${EXTRA[@]}" \
  > "$BASE/launcher_logs/$TAG.log" 2>&1
