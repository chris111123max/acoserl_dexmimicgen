#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p logs/ascend16
exec 9>/tmp/acoserl_dexmimicgen_formal.lock
flock -n 9 || { echo 'Formal learner already owns lock'; exit 1; }
source scripts/ascend16/activate.sh
TAG=${1:-formal_$(date +%Y%m%d_%H%M%S)}
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
printf '%s\n' "$$" > logs/ascend16/formal.pid
printf '%s\n' "$TAG" > logs/ascend16/formal.tag
printf '%s\n' "$(date -Is) launching tag=$TAG utd=1 device=npu:0 workers=16"
python -u scripts/ascend16/train.py --mode formal --tag "$TAG" --utd 1 > "logs/ascend16/$TAG.log" 2>&1 9>&- &
LEARNER=$!
printf '%s\n' "$LEARNER" > logs/ascend16/learner.pid
set +e
wait "$LEARNER"
STATUS=$?
printf '%s\n' "$STATUS" > "logs/ascend16/$TAG.exit"
exit "$STATUS"
