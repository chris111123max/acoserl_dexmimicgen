"""Single-task AutoSERL launch gate; advanced12+4 recovery is not a prerequisite."""
import json,hashlib
from pathlib import Path
from acoserl.v3_gate import CONTROL_FILES
REQUIRED=('official_core_connected','real_grasp_phases_verified','single_demo_strict_protocol_verified',
          'executed_actions_replay_verified','enabled_recovery_domains_verified','unverified_recovery_disabled',
          'legal_grasp_assembly_contact_verified','original_safety_limits_verified','npu16_real_updates_verified',
          'checkpoint_reload_update_verified','no_systemic_collection_deadlock')
def source_files():
    return sorted(set(['src/acoserl/'+n for n in CONTROL_FILES]+[
        'src/acoserl/collection_health.py','src/acoserl/transition_semantics.py','src/acoserl/action_provenance.py','src/acoserl/pregrasp_hold.py','src/acoserl/baseline_gate.py',
        'scripts/ascend16/train.py','configs/bimanual_autoserl/baseline.json',
        'configs/bimanual_autoserl/pregrasp_hold_baseline.json','configs/bimanual_autoserl/v1.json',
        'configs/bimanual_autoserl/recovery_points.json','configs/bimanual_autoserl/reference_trace.npz',
        'configs/ascend16/formal.json']))
def require_baseline(path,root):
    root=Path(root);gate=json.loads(Path(path).read_text())
    assert gate.get('collector_protocol')=='autoserl_baseline_v1'
    missing=[k for k in REQUIRED if not gate.get('checks',{}).get(k,{}).get('passed',False)]
    assert not missing,'Baseline checks pending: '+', '.join(missing)
    for key in REQUIRED:
        paths=gate['checks'][key].get('paths',[])
        assert paths and all((root/p).exists() for p in paths),'Missing evidence: '+key
    for name in source_files():
        assert gate.get('tested_source_sha256',{}).get(name)==hashlib.sha256((root/name).read_bytes()).hexdigest(),'Untested baseline file: '+name
    cfg=json.loads((root/'configs/bimanual_autoserl/baseline.json').read_text())
    assert cfg['v3_enabled'] and cfg['baseline_mode'] and not cfg['development_validation']
    assert not any(cfg[k]['enabled'] for k in ('joint','passive_brake','carry_local'))
    assert gate.get('passed') and gate.get('formal_training_approved')
    assert gate.get('selected_utd')==1 and gate.get('selected_visual')=='resnet18_spatial84'
    return gate
