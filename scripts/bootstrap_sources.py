"""Restore only pinned core source repositories; never launches training."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git',*map(str,args)],text=True).strip()
def main():
 lock=json.loads((ROOT/'third_party.lock.json').read_text())
 subprocess.run(['git','-C',str(ROOT),'submodule','update','--init'],check=True)
 for dep in lock['repositories']:
  path=ROOT/dep['path']
  assert git('-C',path,'rev-parse','HEAD')==dep['commit'],dep['name']+' commit mismatch'
  if dep.get('patch'):
   patch=ROOT/dep['patch']
   if subprocess.run(['git','-C',str(path),'apply','--reverse','--check',str(patch)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:continue
   subprocess.run(['git','-C',str(path),'apply','--check',str(patch)],check=True)
   subprocess.run(['git','-C',str(path),'apply',str(patch)],check=True)
 print('Four exact core repositories ready. No environments or training started.')
if __name__=='__main__':main()
