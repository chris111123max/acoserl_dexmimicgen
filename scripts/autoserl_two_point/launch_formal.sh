#!/usr/bin/env bash
set -euo pipefail
cd /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
mkdir -p logs/autoserl_two_point
exec 9>/tmp/acoserl_dexmimicgen_formal.lock
flock -n 9 || { echo 'Existing formal learner owns lock'; exit 1; }
source scripts/ascend16/activate.sh
TAG=${1:-autoserl_two_point_$(date +%Y%m%d_%H%M%S)}
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
printf '%s\n' "$TAG" > logs/autoserl_two_point/formal.tag
printf '%s\n' "$$" > logs/autoserl_two_point/launcher.pid
python -u scripts/ascend16/train.py --mode formal --tag "$TAG" --utd 1 --gate docs/autoserl_two_point/deployment_gate.json > "logs/autoserl_two_point/$TAG.log" 2>&1 9>&- &
PID=$!
printf '%s\n' "$PID" > logs/autoserl_two_point/learner.pid
set +e
wait "$PID"
STATUS=$?
printf '%s\n' "$STATUS" > "logs/autoserl_two_point/$TAG.exit"
exit "$STATUS"
