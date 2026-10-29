export ACOSERL_ROOT=/data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
source "$ACOSERL_ROOT/.venv_resfit/bin/activate"
export RESFIT_ROOT="$ACOSERL_ROOT/third_party/residual-offpolicy-rl"
export PIP_CACHE_DIR="$ACOSERL_ROOT/cache/pip"
export HF_HOME="$ACOSERL_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$ACOSERL_ROOT/cache/resfit"
export TMPDIR="$ACOSERL_ROOT/cache/resfit/tmp"
export NUMBA_CACHE_DIR="$ACOSERL_ROOT/cache/numba_resfit"
export CACHE_DIR="$ACOSERL_ROOT/outputs/resfit"
export WANDB_DIR="$ACOSERL_ROOT/logs/resfit_wandb"
export WANDB_MODE=disabled
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RESFIT_ROOT:$ACOSERL_ROOT/src:$ACOSERL_ROOT/third_party/lerobot-resfit"
export LD_LIBRARY_PATH="$ACOSERL_ROOT/third_party/system_deps/root/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ACOSERL_ROOT"
