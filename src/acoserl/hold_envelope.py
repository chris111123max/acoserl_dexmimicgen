"""Fail-closed lookup of empirically calibrated captured-pose hold domains."""
import copy,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]

def evaluate(state,entry,cfg,envelope):
    result=copy.deepcopy(entry);arm=entry["passive_arm"];pi="AB".index(arm)
    if envelope.get("version")!="bounded_empirical_passive_hold_v3r3":
        return {**result,"allowed":False,"reason":"passive_hold_calibration_protocol_mismatch","calibrated_domain":None}
    candidates=[]
    for name,domain in envelope.get("domains",{}).items():
        if not name.endswith(":"+arm) or not domain.get("passed"):continue
        if state["actual_grasps"]!=domain["actual_grasps"]:continue
        if bool(state["official_metrics"]["first_piece_assembled"])!=domain["first_piece_assembled"]:continue
        positions=np.asarray([p["position"] for p in state["poses"]])
        selected=(pi,) if domain.get("requires_independent_active",False) else (0,1)
        if any(np.any(positions[i]<domain["eef_min_m"][i]) or np.any(positions[i]>domain["eef_max_m"][i]) for i in selected):continue
        if any(Rotation.from_matrix(np.asarray(state["poses"][i]["rotation"])@np.asarray(domain["rotations"][i]).T).magnitude()>domain["rotation_domain_rad"]
               for i in selected):continue
        if entry["linear_speed_m_s"]>domain["max_linear_speed_m_s"] or entry["angular_speed_rad_s"]>domain["max_angular_speed_rad_s"]:continue
        relative_ok=True
        for obj,expected in domain["relative_grasp_positions"].items():
            eef=state["poses"][pi]
            actual=np.asarray(eef["rotation"]).T@(np.asarray(state["objects"][obj]["position"])-eef["position"])
            relative_ok &= np.linalg.norm(actual-expected)<=cfg["relative_grasp_motion_m"]
            orientation=np.asarray(eef["rotation"]).T@Rotation.from_quat(np.roll(state["objects"][obj]["quaternion_wxyz"],-1)).as_matrix()
            relative_ok &= Rotation.from_matrix(orientation@np.asarray(domain["relative_grasp_rotations"][obj]).T).magnitude()<=cfg["passive_object_rotation_drift_rad"]
        if relative_ok:candidates.append((name,domain))
    if not envelope.get("passed") or len(candidates)!=1:
        result.update(allowed=False,reason="passive_hold_outside_calibrated_domain",calibrated_domain=None,
                      validated_error_bound_m=None)
        return result
    name,domain=candidates[0]
    position=entry["predicted_peak_position_m"]+domain["position_error_bound_m"]
    rotation=entry["predicted_peak_rotation_rad"]+domain["rotation_error_bound_rad"]
    objects={obj:value+domain["object_position_error_bound_m"]
             for obj,value in entry["object_position_predictions_m"].items()}
    object_rotation=entry["predicted_peak_rotation_rad"]+domain["object_rotation_error_bound_rad"]
    allowed=(entry["allowed"] and position<cfg["passive_position_drift_m"]
             and rotation<cfg["passive_rotation_drift_rad"]
             and all(v<cfg["passive_object_position_drift_m"] for v in objects.values())
             and (not objects or object_rotation<cfg["passive_object_rotation_drift_rad"]))
    result.update(allowed=bool(allowed),reason="eligible_calibrated_hold" if allowed else "calibrated_hold_margin_insufficient",
        calibrated_domain=name,validated_error_bound_m=domain["position_error_bound_m"],
        validated_rotation_error_bound_rad=domain["rotation_error_bound_rad"],
        conservative_peak_position_m=position,conservative_peak_rotation_rad=rotation,
        conservative_object_position_m=objects,calibration_horizon_controls=domain["horizon_controls"],
        conservative_object_rotation_rad=object_rotation,
        requires_independent_active=domain.get("requires_independent_active",False),
        certificate_scope="empirical bounded pose/grasp/speed domain; no global guarantee")
    return result
