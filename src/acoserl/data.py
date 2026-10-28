"""Read-only dataset access. No implicit multi-demonstration buffer."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

def metadata(path):
    with h5py.File(path, 'r') as f:
        return json.loads(f['data'].attrs['env_args'])

def load_demo(path, demo_id):
    with h5py.File(path, 'r') as f:
        g = f['data'][demo_id]
        result = {'demo_id': demo_id, 'actions': g['actions'][()], 'states': g['states'][()],
                  'model': g.attrs['model_file'], 'ep_meta': g.attrs.get('ep_meta'),
                  'metadata': json.loads(f['data'].attrs['env_args'])}
        result['obs'] = {k: v[()] for k, v in g.get('obs', {}).items() if isinstance(v, h5py.Dataset) and v.ndim <= 2}
        return result

def inspect(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    report = {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    if expected:
        report['expected_bytes'] = expected['size']
        report['expected_sha256'] = expected['lfs']['oid']
        report['integrity_ok'] = report['bytes'] == expected['size'] and report['sha256'] == expected['lfs']['oid']
        if not report['integrity_ok']:
            raise ValueError(f'Integrity failure: {path}')
    with h5py.File(path, 'r') as f:
        report['metadata'] = json.loads(f['data'].attrs['env_args'])
        report['episodes'] = {}
        for name in sorted(f['data'], key=lambda x: int(x.split('_')[-1])):
            g = f['data'][name]
            fields = {}
            def visit(k, v):
                if isinstance(v, h5py.Dataset):
                    fields[k] = {'shape': list(v.shape), 'dtype': str(v.dtype)}
            g.visititems(visit)
            n = len(g['actions'])
            if len(g['states']) != n:
                raise ValueError(f'{name}: action/state length mismatch')
            report['episodes'][name] = {'length': n, 'fields': fields, 'attributes': list(g.attrs)}
        report['episode_count'] = len(report['episodes'])
        report['root_attributes'] = list(f.attrs)
        report['data_attributes'] = list(f['data'].attrs)
        report['mask_keys'] = list(f.get('mask', {}))
    return report
