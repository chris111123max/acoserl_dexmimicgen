"""Resumable range acceleration of an existing official Hub partial file.
Uses the official repository through HF_ENDPOINT; verifies full LFS SHA256 before
publishing. All workers write disjoint ranges into one file (no full copies).
"""
import argparse, concurrent.futures, hashlib, json, os, threading, time
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]

def main(name, workers):
    record = next(x for x in json.loads((ROOT/'docs/hf_manifest.json').read_text()) if x['path']=='generated/'+name)
    size, sha = record['size'], record['lfs']['oid']
    final = ROOT/'datasets/generated'/name
    state = ROOT/'datasets/.cache'/ (name+'.ranges.json')
    state.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(final)
    if state.exists():
        progress = json.loads(state.read_text())
        partial = Path(progress['partial'])
    else:
        matches = list((ROOT/'datasets/.cache/huggingface/download/generated').glob('*.'+sha+'.incomplete'))
        partial = matches[0] if matches else ROOT/'datasets/.cache'/(name+'.incomplete')
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > size:
            raise ValueError('Oversized partial')
        progress = {'partial': str(partial), 'sha256': sha, 'size': size, 'prefix': prefix, 'done': []}
        state.write_text(json.dumps(progress))
    if progress['sha256'] != sha or progress['size'] != size:
        raise ValueError('Manifest changed; refusing to reuse partial')
    endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    url = endpoint+'/datasets/MimicGen/dexmimicgen_datasets/resolve/main/generated/'+name
    # Resolve once to the official signed CDN URL, avoiding one mirror resolve
    # request per chunk (which can trigger HTTP 429).
    for attempt in range(8):
        response = requests.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=(15,60))
        if response.status_code == 429:
            response.close()
            time.sleep(30)
            continue
        response.raise_for_status()
        url = response.url
        response.close()
        break
    else:
        raise RuntimeError('Mirror rate limit persists; all progress retained')
    lock = threading.Lock()
    fd = os.open(partial, os.O_RDWR|os.O_CREAT, 0o644)
    chunk = 64*1024*1024
    ranges = [(s,min(s+chunk,size)-1) for s in range(progress['prefix'],size,chunk)]
    def fetch(pair):
        start, end = pair
        if start in progress['done']:
            return
        for attempt in range(6):
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15,60)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('Server did not honor the exact range: '+str(r.headers.get('Content-Range')))
                    pos = start
                    for data in r.iter_content(1024*1024):
                        if pos+len(data)>end+1:
                            raise ValueError('Oversized range')
                        view = memoryview(data)
                        while view:
                            n = os.pwrite(fd, view, pos)
                            if n <= 0: raise IOError('Short write')
                            pos += n
                            view = view[n:]
                    if pos != end+1:
                        raise IOError('Incomplete range')
                with lock:
                    os.fsync(fd)
                    progress['done'].append(start)
                    tmp = state.with_suffix('.tmp')
                    tmp.write_text(json.dumps(progress))
                    tmp.replace(state)
                    print(name, len(progress['done']), '/',len(ranges),'ranges', flush=True)
                return
            except Exception as e:
                print(name, start, 'attempt',attempt+1,repr(e),flush=True)
                if attempt==5: raise
                time.sleep(min(2**attempt,20))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, ranges))
    finally:
        os.close(fd)
    digest = hashlib.sha256()
    with partial.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''): digest.update(block)
    if partial.stat().st_size != size or digest.hexdigest()!=sha:
        raise ValueError('SHA256 validation failed; partial and range state retained')
    final.parent.mkdir(parents=True,exist_ok=True)
    partial.rename(final)
    progress['complete'] = True
    state.write_text(json.dumps(progress,indent=2))
    print('COMPLETE',str(final),size,sha,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('filename')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    main(args.filename,args.workers)
