"""External, source-bound qualification receipts; never edit production JSON.

A bounded requalification trial authorizes only the SAME recorded static P0
pose/grasp domain while all numerical, retention and substep limits remain on.
It is not production authorization and cannot be used by the learner. Passing
physical traces can issue a new certificate under the external testing folder.
"""
from __future__ import annotations
import copy
import hashlib
import json
import subprocess
import sys
import importlib.metadata
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def external_path(path):
    p = Path(path).expanduser().resolve()
    if p == ROOT or ROOT in p.parents:
        raise ValueError('qualification/test output must be outside the repository')
    return p


def source_manifest():
    files = []
    for name in ('src', 'scripts'):
        files.extend((ROOT/name).rglob('*.py'))
    for pattern in ('*.json', '*.npz'):
        files.extend((ROOT/'configs').rglob(pattern))
    semantics=ROOT/'docs/bimanual_autoserl/strict_semantics.json'
    if semantics.exists():files.append(semantics)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files)) if '__pycache__' not in p.parts}


def dependency_manifest():
    base = ROOT/'third_party/residual-offpolicy-rl'
    if not base.exists():
        raise FileNotFoundError('ResFiT checkout missing')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(base), *args], stderr=subprocess.STDOUT)
    versions={}
    for name in ('numpy','scipy','mujoco','robosuite','torch','torch-npu','torchrl','tensordict'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return dict(python=sys.version.split()[0],versions=versions,head=git('rev-parse', 'HEAD').decode().strip(),
                tracked_diff_sha256=hashlib.sha256(git('diff', 'HEAD', '--')).hexdigest(),
                status=git('status', '--porcelain').decode())


def _matched_entry(manager, active, state, entry, proof):
    if not entry['allowed'] or manager.contact_assessment(state)['category'] != 'Independent':
        return None
    if any(state['actual_grasps']['AB'[active]].values()): return None
    pi, arm = 1-active, 'AB'[1-active]
    pose = state['poses'][pi]
    for domain in proof['entries']:
        if domain['passive_arm'] != arm or state['actual_grasps'][arm] != domain['grasps']: continue
        if np.linalg.norm(pose['position']-np.asarray(domain['pose']['position'])) > .002: continue
        if Rotation.from_matrix(pose['rotation']@np.asarray(domain['pose']['rotation']).T).magnitude() > .02: continue
        if (entry['linear_speed_m_s'] > domain['max_linear_speed_m_s'] or
                entry['angular_speed_rad_s'] > domain['max_angular_speed_rad_s']): continue
        okay = True
        for obj, held in domain['grasps'].items():
            if not held: continue
            now = state['objects'][obj]; old = domain['objects'][obj]
            rel = pose['rotation'].T@(np.asarray(now['position'])-pose['position'])
            old_rel = np.asarray(domain['pose']['rotation']).T@(np.asarray(old['position'])-domain['pose']['position'])
            r = pose['rotation'].T@Rotation.from_quat(np.roll(now['quaternion_wxyz'], -1)).as_matrix()
            old_r = np.asarray(domain['pose']['rotation']).T@Rotation.from_quat(np.roll(old['quaternion_wxyz'], -1)).as_matrix()
            okay &= np.linalg.norm(rel-old_rel) <= manager.cfg['relative_grasp_motion_m']
            okay &= Rotation.from_matrix(r@old_r.T).magnitude() <= manager.cfg['passive_object_rotation_drift_rad']
        bound = (entry['predicted_peak_position_m'] + entry['linear_speed_m_s']/manager.env.control_freq
                 + domain['observed_peak_eef_m'])
        if okay and bound < manager.cfg['passive_position_drift_m']:
            return dict(entry, allowed=True, calibrated_domain=domain['name'],
                        conservative_peak_position_m=float(bound),
                        validated_error_bound_m=domain['observed_peak_eef_m'],
                        calibration_horizon_controls=manager.cfg['max_recovery_steps'],
                        reason='external_source_bound_static_hold',
                        qualification_scope='recorded static P0 pose/grasp only')
    return None


def external_hold_entry(manager, active, state):
    from acoserl.bimanual_core import LegacyBimanualAutoSERL
    trial = getattr(manager, '_static_hold_trial', None)
    path = manager.manager_cfg.get('static_hold_certificate')
    if trial is None and not path: return None
    entry = LegacyBimanualAutoSERL.passive_hold_entry(manager, active, state)
    if trial is not None:
        if not isinstance(trial, StaticHoldTrial) or not trial.open or trial.active != active:
            return dict(entry, allowed=False, reason='invalid_static_hold_trial')
        proof = trial.domain
        result = _matched_entry(manager, active, state, entry, proof)
        if result is not None:
            trial.matched_domain = result['calibrated_domain']
            return dict(result, production_qualified=False, requalification_trial=True)
        return dict(entry, allowed=False, reason='requalification_outside_original_domain')
    path = external_path(path)
    cache = getattr(manager, '_external_hold_cache', None)
    if cache is None or cache[0] != str(path):
        proof = json.loads(path.read_text())
        if proof.get('protocol') != 'static_hold_requalification_v1' or not proof.get('passed'):
            return dict(entry, allowed=False, reason='external_hold_certificate_not_passed')
        if proof.get('source_sha256') != source_manifest() or proof.get('dependency') != dependency_manifest():
            return dict(entry, allowed=False, reason='external_hold_certificate_source_mismatch')
        for receipt in proof['receipts']:
            p = external_path(receipt['path'])
            if hashlib.sha256(p.read_bytes()).hexdigest() != receipt['sha256']:
                return dict(entry, allowed=False, reason='external_hold_receipt_changed')
        cache = (str(path), proof); manager._external_hold_cache = cache
    result = _matched_entry(manager, active, state, entry, cache[1])
    return (dict(result, production_qualified=True, requalification_trial=False) if result
            else dict(entry, allowed=False, reason='outside_external_static_hold_domain'))


class StaticHoldTrial:
    """Explicit bounded physical requalification API for external test scripts.

    Prepare a recorded P0 domain using real actions BEFORE opening the trial.
    Call observe after every real control. close writes a receipt, not a gate.
    No learner, synthetic transition or direct state restoration is authorized.
    """
    def __init__(self, manager, active, output):
        if active not in (0, 1): raise ValueError('active must be 0 or 1')
        self.manager, self.active = manager, active
        self.output = external_path(output)
        self.domain = json.loads((ROOT/'configs/bimanual_autoserl/pregrasp_hold_baseline.json').read_text())
        self.manifest = source_manifest(); self.dependency = dependency_manifest()
        self.open = False; self.rows = []; self.matched_domain = None;self.proposals=0

    def __enter__(self):
        if getattr(self.manager, '_static_hold_trial', None) is not None:
            raise RuntimeError('nested qualification trial')
        self.open = True; self.manager._static_hold_trial = self
        return self

    def observe(self, runner, decision, after):
        if not self.open: raise RuntimeError('trial is closed')
        attempt = copy.deepcopy(runner.last_attempt)
        if attempt is None: raise ValueError('no actual physics attempt')
        if self.rows and abs(attempt['start_time']-self.rows[-1]['attempt']['physical_time'])>1e-10:
            raise ValueError('noncontiguous physical trial')
        if self.rows and attempt['start_time']==self.rows[-1]['attempt']['start_time']:
            raise ValueError('duplicate physics attempt in qualification')
        if runner.physical_writes_attempted:
            raise RuntimeError('physical state write attempted during qualification')
        self.rows.append(dict(attempt=attempt, decision=copy.deepcopy(decision), after=copy.deepcopy(after)))
        if len(self.rows) > self.manager.cfg['max_recovery_steps']+40:
            raise RuntimeError('bounded qualification control budget exceeded')

    def __exit__(self, typ, error, tb):
        self.open = False; self.manager._static_hold_trial = None
        real = [r for r in self.rows if r['attempt'].get('complete')]
        mode = 'Recover_'+'AB'[self.active]
        controls = [r for r in real if r['decision'].get('mode') == mode]
        handbacks = [r for r in real if r['after'].get('safe_handback') and
                     r['after'].get('handback_arm') == 'AB'[self.active]]
        force = max((s['robot_force_N'] for r in self.rows for s in r['attempt'].get('substeps', [])), default=0.)
        passed = bool(typ is None and self.matched_domain and controls and handbacks and
                      len(real) == len(self.rows) and self.manager.stop_reason is None and
                      force <= self.manager.cfg['robot_force_stop_N'] and source_manifest() == self.manifest)
        baseline=copy.deepcopy(self.manager.baseline)
        result = dict(protocol='static_hold_trial_v1', passed=passed, arm='AB'[self.active],
                      domain=self.matched_domain, source_sha256=self.manifest,
                      dependency=self.dependency, actual_controls=len(real),
                      recovery_controls=len(controls), handbacks=len(handbacks),
                      force_peak_N=force,baseline=baseline, error=str(error) if error else None, rows=self.rows)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open('x') as f:
            json.dump(result, f, indent=2, default=lambda v:v.tolist() if hasattr(v, 'tolist') else str(v))
        return False


def issue_static_hold_certificate(receipts, output):
    """Issue only from >=3 real passing trials per arm, at this exact source."""
    manifest, dependency = source_manifest(), dependency_manifest()
    counts = {'A': 0, 'B': 0}; records = []; seen=set(); tested_domains=set(); observed={}
    for name in receipts:
        path = external_path(name); raw = path.read_bytes(); value = json.loads(raw)
        if (value.get('protocol') != 'static_hold_trial_v1' or not value.get('passed') or
                value.get('source_sha256') != manifest or value.get('dependency') != dependency or
                value.get('actual_controls', 0) <= 0 or value.get('handbacks', 0) <= 0):
            raise ValueError('invalid physical receipt: '+str(path))
        digest=hashlib.sha256(raw).hexdigest()
        if str(path) in seen or digest in seen:raise ValueError('duplicate qualification receipt')
        seen.update((str(path),digest));tested_domains.add(value['domain'])
        passive=1-'AB'.index(value['arm'])
        reference=np.asarray(value['baseline']['poses'][passive]['position'])
        peak=max((float(np.linalg.norm(np.asarray(step['positions'][passive])-reference))
                  for row in value['rows'] if row['decision'].get('mode','').startswith('Recover_')
                  for step in row['attempt'].get('substeps',[])),default=0.)
        observed[value['domain']]=max(observed.get(value['domain'],0.),peak)
        counts[value['arm']] += 1
        records.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    if min(counts.values()) < 3: raise ValueError('need three distinct recorded trials per arm')
    template = json.loads((ROOT/'configs/bimanual_autoserl/pregrasp_hold_baseline.json').read_text())
    entries=[copy.deepcopy(d) for d in template['entries'] if d['name'] in tested_domains]
    for d in entries:d['observed_peak_eef_m']=max(d['observed_peak_eef_m'],observed.get(d['name'],0.))
    certificate = dict(protocol='static_hold_requalification_v1', passed=True,
                       source_sha256=manifest, dependency=dependency, receipts=records,
                       counts=counts, entries=entries)
    output = external_path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(certificate, f, indent=2)
    return certificate
