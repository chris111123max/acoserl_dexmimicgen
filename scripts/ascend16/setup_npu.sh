#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PYTHON=${ACOSERL_PYTHON:-python3.10}
[[ ! -e "$ROOT/.venv_autoserl_npu" ]] || { echo 'Existing NPU runtime preserved; choose a fresh checkout to rebuild.'; exit 1; }
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,10)'
python3 scripts/bootstrap_sources.py
"$PYTHON" -m venv .venv_autoserl_npu
source .venv_autoserl_npu/bin/activate
python -m pip install setuptools==79.0.1 wheel==0.45.1
python -m pip install torch==2.3.1 torch-npu==2.3.1 torchvision==0.18.1 numpy==1.23.3
python - <<'PY'
from pathlib import Path
rows=Path('configs/ascend16/npu_runtime_freeze.txt').read_text().splitlines()
rows=[r for r in rows if r and not r.startswith(('torch==','torch-npu==','torchvision==','tensordict @','torchrl @'))]
Path('.venv_autoserl_npu/build-requirements.txt').write_text('\n'.join(rows)+'\n')
PY
python -m pip install -r .venv_autoserl_npu/build-requirements.txt
python - <<'PY'
from pathlib import Path
import json,subprocess
lock=json.loads(Path('third_party.lock.json').read_text())
for dep in lock['npu_build_sources']:
 p=Path('third_party')/dep['name']
 if not p.exists():subprocess.run(['git','clone','--no-checkout',dep['url'],str(p)],check=True)
 subprocess.run(['git','-C',str(p),'checkout','--detach',dep['commit']],check=True)
 subprocess.run(['python','-m','pip','install','--no-build-isolation',str(p.resolve())],check=True)
PY
python -m pip install --no-deps -e .
printf '%s\n' 'NPU runtime created. Activate it and validate this machine before explicit deployment authorization.'
