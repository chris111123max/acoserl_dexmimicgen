#!/usr/bin/env bash
set -euo pipefail
cd /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
mkdir -p logs/pretrained_visual
exec 9>/tmp/acoserl_dexmimicgen_formal.lock
flock -n 9 || { echo 'Existing formal learner owns NPU lock'; exit 1; }
source scripts/ascend16/activate.sh
TAG=${1:-pretrained_visual_$(date +%Y%m%d_%H%M%S)}
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
VISUAL=$(python -c 'import json;print(json.load(open("docs/pretrained_visual/deployment_gate.json"))["selected_visual"])')
printf '%s\n' "$TAG" > logs/pretrained_visual/formal.tag
printf '%s\n' "$$" > logs/pretrained_visual/launcher.pid
python -u scripts/ascend16/train.py --mode formal --tag "$TAG" --utd 1 --visual "$VISUAL" --gate docs/pretrained_visual/deployment_gate.json > "logs/pretrained_visual/$TAG.log" 2>&1 9>&- &
PID=$!
printf '%s\n' "$PID" > logs/pretrained_visual/learner.pid
set +e
wait "$PID"
STATUS=$?
printf '%s\n' "$STATUS" > "logs/pretrained_visual/$TAG.exit"
exit "$STATUS"
