"""Try the alternate official Google download host; never use a weight mirror."""
import urllib.request,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
url='https://drive.usercontent.google.com/download?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-&export=download&confirm=t'
result={'url':url}
try:
 r=urllib.request.urlopen(url,timeout=10);result.update(status=r.status,content_type=r.headers.get('content-type'))
 p=ROOT/'cache/pretrained_visual/r3m_resnet18.pt.partial'
 with p.open('wb') as f:
  while True:
   b=r.read(1024*1024)
   if not b:break
   f.write(b)
 result.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
except Exception as e:result.update(error_type=type(e).__name__,error=str(e))
(ROOT/'outputs/pretrained_visual/r3m_download_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
