#!/usr/bin/env bash
set -euo pipefail
source /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen/scripts/ascend16/activate.sh
TASK_TAG=${1:?supply a new unique output tag}
TASK_PY="$ACOSERL_ROOT/.venv_autoserl_npu/bin/python"
"$TASK_PY" scripts/acoserl_foundation/validate_data.py --tag "${TASK_TAG}_data"
"$TASK_PY" scripts/acoserl_foundation/validate_contracts.py
"$TASK_PY" scripts/acoserl_foundation/validate_stage_gates.py --tag "${TASK_TAG}_stages"
for TASK_CASE in Recover_A Recover_B Guide_A Guide_B free risk; do
 "$TASK_PY" scripts/acoserl_foundation/validate_physics.py --case "$TASK_CASE" --tag "${TASK_TAG}_${TASK_CASE}"
done
