#!/usr/bin/env bash
set -euo pipefail
cd /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
mkdir -p logs/autoserl_baseline
exec 9>/tmp/acoserl_dexmimicgen_formal.lock
flock -n 9 || { echo 'Existing formal learner owns NPU lock'; exit 1; }
source scripts/ascend16/activate.sh
TAG=${1:-autoserl_baseline_formal_$(date +%Y%m%d_%H%M%S)}
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
python - <<'PY'
from pathlib import Path
from acoserl.baseline_gate import require_baseline
r=Path.cwd();require_baseline(r/'docs/acoserl_foundation/ACCEPTANCE_AUTOSERL_BASELINE.json',r)
PY
printf '%s\n' "$TAG" > logs/autoserl_baseline/formal.tag
printf '%s\n' "$$" > logs/autoserl_baseline/launcher.pid
python -u scripts/ascend16/train.py --mode formal --tag "$TAG" --gate docs/acoserl_foundation/ACCEPTANCE_AUTOSERL_BASELINE.json --utd 1 --visual resnet18_spatial84 --supervisor-version baseline --demo-original-quota 96 --demo-positive-quota 2 --demo-admission official > "logs/autoserl_baseline/$TAG.log" 2>&1 9>&- &
LEARNER=$!
printf '%s\n' "$LEARNER" > logs/autoserl_baseline/learner.pid
trap 'kill -TERM "$LEARNER" 2>/dev/null || true; wait "$LEARNER" || true' TERM INT
set +e
wait "$LEARNER"
STATUS=$?
printf '%s\n' "$STATUS" > "logs/autoserl_baseline/$TAG.exit"
python scripts/ascend16/plot_curves.py --tag "$TAG" > "logs/autoserl_baseline/$TAG.curves.log" 2>&1 || true
if [[ -f "outputs/ascend16/$TAG/evaluation_workers/env16.jsonl" ]]; then
 python scripts/autoserl_baseline/summarize_evaluation.py --run "outputs/ascend16/$TAG" > "logs/autoserl_baseline/$TAG.evaluation_summary.log" 2>&1 || true
fi
exit "$STATUS"
