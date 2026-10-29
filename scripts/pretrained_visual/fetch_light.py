import urllib.request,hashlib,json
from pathlib import Path
url='https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth';p=Path('cache/pretrained_visual/squeezenet1_1-b8a52dc0.pth');urllib.request.urlretrieve(url,p)
r={'model':'squeezenet1_1_imagenet','url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};Path('outputs/pretrained_visual/light_download.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
