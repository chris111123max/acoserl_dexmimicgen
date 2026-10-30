"""Mandatory evidence gate for changed V3 collection/control semantics."""
import hashlib,json
from pathlib import Path

MANDATORY=("source_commit_verified","contact_failure_reproduced",
    "contact_failure_root_cause_fixed","first_piece_progress_protected",
    "handoff_protocol_verified","contact_recovery_12_of_12_passed",
    "contact_actor_continuation_verified","independent_contact_cases_verified",
    "passive_hold_prediction_verified","safety_thresholds_unchanged",
    "dynamic_demo_provenance_verified","safety_projection_excluded_from_demo",
    "nstep_checkpoint_verified","real_actor_intervention_verified","npu_16env_integration_verified")
CONTROL_FILES=("manager_v3.py","progress_guard.py","hold_envelope.py","manager_v2.py","local_recovery.py",
    "contact_modes.py","bimanual_autoserl.py","intervention_manager.py","two_point_supervisor.py",
    "dynamic_demo.py","transition_semantics.py","isolated_nstep.py","ascend_data.py",
    "autoserl_env.py","parallel_autoserl.py","ascend_checkpoint.py","v3_gate.py")

def require_v3_acceptance(path,root):
    root=Path(root);gate=json.loads(Path(path).read_text())
    assert gate.get("collector_protocol")=="acoserl_v3","V3 cannot use legacy acceptance"
    missing=[name for name in MANDATORY if gate.get(name) is not True
             or gate.get("checks",{}).get(name,{}).get("passed") is not True]
    assert not missing,"V3 mandatory gates pending: "+", ".join(missing)
    for name in MANDATORY:
        paths=gate["checks"][name].get("paths",[])
        assert paths and all((root/p).exists() for p in paths),"Missing evidence: "+name
    expected=gate.get("tested_source_sha256",{})
    files=["src/acoserl/"+name for name in CONTROL_FILES]+["scripts/ascend16/train.py"]
    for name in files:
        digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        assert expected.get(name)==digest,"Untested V3 source: "+name
    assert gate.get("passed") is True and gate.get("formal_training_approved") is True
    assert gate.get("selected_utd")==1 and gate.get("selected_visual")=="resnet18_spatial84"
    return gate
