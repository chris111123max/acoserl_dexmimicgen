# Existing CPU environments are never modified.
export ACOSERL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ACOSERL_ROOT/.venv_autoserl_npu/bin/activate"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit}/set_env.sh"
export PYTHONPATH="$ACOSERL_ROOT/src:$ACOSERL_ROOT/third_party/residual-offpolicy-rl:$ACOSERL_ROOT/third_party/lerobot-resfit:$ACOSERL_ROOT/third_party/robosuite:$ACOSERL_ROOT/third_party/dexmimicgen${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ACOSERL_ROOT/third_party/system_deps<LOCAL_BUILD_PATH>"
source "$ACOSERL_ROOT/scripts/bimanual_autoserl/activate_render.sh"
export PYTHONNOUSERSITE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=disabled PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HF_HOME="$ACOSERL_ROOT/cache/huggingface" NUMBA_CACHE_DIR="$ACOSERL_ROOT/cache/numba_npu"
export RESFIT_ROOT="$ACOSERL_ROOT/third_party/residual-offpolicy-rl"
cd "$ACOSERL_ROOT"
export LP_NUM_THREADS=1 MESA_GLTHREAD=false
