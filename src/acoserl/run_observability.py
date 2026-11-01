"""Small durable run records and optional checkpoint replication.

A path name is NOT proof of persistence. Record the actual Linux mount and a
store identity, and disclose when a mirror is on the same device. No records
or old checkpoints are removed. Copying is opt-in and outside the hot loop.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mount_record(path):
    path = Path(path).resolve()
    entries = []
    try:
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, right = line.split(' - ', 1)
            columns, tail = left.split(), right.split()
            mount = Path(columns[4].replace('\\040', ' ').replace('\\134', '\\'))
            if path == mount or mount in path.parents:
                entries.append((len(mount.parts), dict(mount_point=str(mount),
                    mount_root=columns[3], device=columns[2], filesystem=tail[0],
                    source=tail[1], options=columns[5])))
    except (OSError, ValueError, IndexError):
        pass
    result = max(entries, key=lambda x: x[0])[1] if entries else dict(filesystem='unknown')
    result.update(path=str(path), st_dev=path.stat().st_dev,
                  free_bytes=shutil.disk_usage(path).free,
                  persistence_verified=False)
    result['warning'] = ('output is on a container/temporary filesystem' if
        result['filesystem'] in ('overlay', 'tmpfs', 'ramfs') else
        'mount recorded; retention across instances still requires storage-provider confirmation')
    return result


class RunStore:
    def __init__(self, out, source_commit, mirror_root=None):
        self.out = Path(out).resolve()
        self.root = self.out.parent
        marker = self.root / '.acoserl_store_id.json'
        expected = os.environ.get('ACOSERL_EXPECTED_STORE_ID')
        if expected and (not marker.exists() or json.loads(marker.read_text()).get('store_id') != expected):
            raise RuntimeError('configured output store is missing or has changed identity')
        try:
            with marker.open('x') as stream:
                json.dump(dict(store_id=str(uuid.uuid4()), created=time.time()), stream)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
        self.identity = json.loads(marker.read_text())
        self.mirror_root = Path(mirror_root).resolve() if mirror_root else None
        if self.mirror_root is not None:
            self.mirror_root.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint = None
        self.record = dict(run_id=str(uuid.uuid4()), source_commit=source_commit,
                           hostname=socket.gethostname(), pid=os.getpid(),
                           output=str(self.out), store=self.identity,
                           storage=mount_record(self.out), created=time.time())
        if self.mirror_root is not None:
            self.record['mirror'] = mount_record(self.mirror_root)
            self.record['mirror_on_different_device'] = self.out.stat().st_dev != self.mirror_root.stat().st_dev
        atomic_json(self.out / 'storage_manifest.json', self.record)
        print('RUN_STORAGE', json.dumps(self.record), flush=True)

    def heartbeat(self, counters, health, timing, elapsed, dynamic_rows, admission, checkpoint=None):
        if checkpoint is not None:
            self.last_checkpoint = str(checkpoint)
        status = dict(status='running', timestamp=time.time(), counters=counters,
            health=health, timing=timing, elapsed_seconds=elapsed,
            dynamic_demo_rows=dynamic_rows, dynamic_admission=admission,
            checkpoint=self.last_checkpoint, store_id=self.identity['store_id'],
            free_bytes=shutil.disk_usage(self.out).free)
        atomic_json(self.out / 'live_status.json', status)
        atomic_json(self.root / 'latest_run.json', dict(self.record, latest_status=str(self.out / 'live_status.json'),
                                                       latest_checkpoint=self.last_checkpoint))

    def checkpoint(self, source):
        self.last_checkpoint = str(source)
        receipt = dict(source=str(source), timestamp=time.time(), mirrored=False)
        if self.mirror_root is not None:
            dest = self.mirror_root / self.out.name / Path(source).name
            stage = dest.with_name(dest.name + '.partial')
            started = time.perf_counter()
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() or stage.exists():
                    raise FileExistsError('mirror destination already exists')
                shutil.copytree(source, stage)
                # Preserve incomplete copies on failure, never announce them as complete.
                for file in stage.rglob('*'):
                    if file.is_file():
                        with file.open('rb') as stream:
                            os.fsync(stream.fileno())
                os.replace(stage, dest)
                receipt.update(mirrored=True, destination=str(dest))
            except OSError as error:
                receipt['error'] = str(error)
            receipt['mirror_seconds'] = time.perf_counter() - started
        atomic_json(self.out / ('replica_' + Path(source).name + '.json'), receipt)
        return receipt


def task_progress_stop(health, budget=10000):
    """A full Demo buffer is not task progress. Preserve physical success semantics."""
    if budget <= 0:
        return False
    d = health
    stable = sum(d.get('stable_grasp_episodes', {}).values())
    return bool(d.get('valid_steps', 0) >= budget and d.get('ends', 0) >= 32 and
                stable == 0 and not d.get('first_assembly', 0) and not d.get('full_success', 0))
