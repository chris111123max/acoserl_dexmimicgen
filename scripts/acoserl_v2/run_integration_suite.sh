#!/usr/bin/env bash
set -euo pipefail
cd /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
exec 9>/tmp/acoserl_dexmimicgen_formal.lock
flock -n 9
source scripts/ascend16/activate.sh
TASK_PY="$ACOSERL_ROOT/.venv_autoserl_npu/bin/python"
TASK_PREFIX=${1:?unique tag prefix required}
run_case() {
 local task_method=$1 task_admission=$2 task_steps=$3 task_suffix=$4
 "$TASK_PY" -u scripts/ascend16/train.py --mode integration --tag "${TASK_PREFIX}_${task_suffix}" --utd 1 --visual resnet18_spatial84 --intervention-method "$task_method" --demo-admission "$task_admission" --demo-positive-quota 2 --integration-steps "$task_steps" --integration-seconds 1200 > "outputs/acoserl_v2/${TASK_PREFIX}_${task_suffix}.stdout" 2>&1
}
run_case full_acoserl official "${2:-512}" full_official
run_case plain_rlpd official 64 plain
run_case independent_window official 64 window
run_case window_recovery official 64 recovery
run_case full_acoserl safe_handback "${3:-128}" full_safe
