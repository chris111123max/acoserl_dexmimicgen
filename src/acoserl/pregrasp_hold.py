"""Reuse measured task-specific P0 passive holds; no new recovery planner."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from acoserl.object_geometry import object_error
ROOT=Path(__file__).resolve().parents[2]

def qualified_entry(manager,active,state,entry):
    p=ROOT/'configs/bimanual_autoserl/pregrasp_hold_baseline.json'
    if not p.exists() or not entry['allowed']:return None
    proof=json.loads(p.read_text())
    development=bool(manager.manager_cfg.get('development_validation',False))
    if not proof.get('passed') and not development:return None
    if proof.get('passed') and proof.get('source_sha256')!=manager.v3_fingerprint() and not development:return None
    pi=1-active;arm='AB'[pi]
    if manager.contact_assessment(state)['category']!='Independent':return None
    if any(state['actual_grasps']['AB'[active]].values()):return None
    for domain in proof['entries']:
        if domain['passive_arm']!=arm or state['actual_grasps'][arm]!=domain['grasps']:continue
        pose=state['poses'][pi]
        if np.linalg.norm(pose['position']-np.asarray(domain['pose']['position']))>.002:continue
        if Rotation.from_matrix(pose['rotation']@np.asarray(domain['pose']['rotation']).T).magnitude()>.02:continue
        if entry['linear_speed_m_s']>domain['max_linear_speed_m_s'] or entry['angular_speed_rad_s']>domain['max_angular_speed_rad_s']:continue
        okay=True
        for name,held in domain['grasps'].items():
            if not held:continue
            relative=pose['rotation'].T@(np.asarray(state['objects'][name]['position'])-pose['position'])
            old=domain['objects'][name];old_pose=domain['pose']
            previous=np.asarray(old_pose['rotation']).T@(np.asarray(old['position'])-old_pose['position'])
            current_r=pose['rotation'].T@Rotation.from_quat(np.roll(state['objects'][name]['quaternion_wxyz'],-1)).as_matrix()
            old_r=np.asarray(old_pose['rotation']).T@Rotation.from_quat(np.roll(old['quaternion_wxyz'],-1)).as_matrix()
            okay &= np.linalg.norm(relative-previous)<=manager.cfg['relative_grasp_motion_m']
            okay &= Rotation.from_matrix(current_r@old_r.T).magnitude()<=manager.cfg['passive_object_rotation_drift_rad']
        if not okay:continue
        # Whole-control-period latency, plus measured residual at this exact
        # passive pose/grasp. No extrapolation to moving or unknown carry states.
        position=entry['predicted_peak_position_m']+entry['linear_speed_m_s']/manager.env.control_freq+domain['observed_peak_eef_m']
        if position>=manager.cfg['passive_position_drift_m']:continue
        return {**entry,'allowed':True,'reason':'qualified_task_specific_pregrasp_hold',
            'calibrated_domain':domain['name'],'validated_error_bound_m':domain['observed_peak_eef_m'],
            'conservative_peak_position_m':float(position),'calibration_horizon_controls':manager.cfg['max_recovery_steps'],
            'qualification_scope':'static measured P0 hold pose/grasp only; actual2ms retention checks remain active',
            'development_qualification':bool(development and proof.get('source_sha256')!=manager.v3_fingerprint()) or not proof.get('passed',False)}
    return None
