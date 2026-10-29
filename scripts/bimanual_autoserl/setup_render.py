"""Restore pinned Debian rendering libraries into a private directory, never apt install."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
records=json.loads((ROOT/'docs/bimanual_autoserl/runtime.json').read_text())['private_native_dependencies']
base=ROOT/'third_party/system_deps/autoserl_render';debdir=base/'debs';debdir.mkdir(parents=True,exist_ok=True)
ready=all((base/'root/usr/lib/aarch64-linux-gnu'/name).exists() for name in ['libOSMesa.so.8','libGL.so.1'])
for record in records:
 path=debdir/record['file']
 if not path.exists():
  fields=dict(line.split(': ',1) for line in record['package_metadata'].splitlines())
  subprocess.run(['apt-get','download',fields['Package']+'='+fields['Version']],cwd=debdir,check=True)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],path
 if not ready:subprocess.run(['dpkg-deb','-x',str(path),str(base/'root')],check=True)
print('Private render runtime ready; source scripts/bimanual_autoserl/activate_render.sh')
