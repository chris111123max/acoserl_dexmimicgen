"""Fetch only official public weights and a pinned official DINOv2 source tree."""
import hashlib,json,subprocess,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DINO_REV='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
FILES={
 'dinov2_vits14_pretrain.pth':('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth','b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'),
 'resnet18-f37072fd.pth':('https://download.pytorch.org/models/resnet18-f37072fd.pth','f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec')}
FILES['squeezenet1_1-b8a52dc0.pth']=('https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth','b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b')
if __name__=='__main__':
 repo=ROOT/'third_party/dinov2'
 if not repo.exists():
  subprocess.run(['git','clone','https://github.com/facebookresearch/dinov2.git',str(repo)],check=True)
  subprocess.run(['git','checkout',DINO_REV],cwd=repo,check=True)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==DINO_REV,'existing DINO source differs; do not overwrite'
 folder=ROOT/'cache/pretrained_visual';folder.mkdir(parents=True,exist_ok=True)
 for name,(url,expected) in FILES.items():
  path=folder/name
  if not path.exists():
   temp=path.with_suffix('.partial');urllib.request.urlretrieve(url,temp);assert hashlib.sha256(temp.read_bytes()).hexdigest()==expected;temp.rename(path)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
  print(name,expected,path.stat().st_size)
