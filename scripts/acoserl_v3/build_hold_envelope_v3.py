"""Calibrate EEF AND held-object error, with a measured finite hold horizon."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/"outputs/acoserl_v3/hold_calibration_v1"
carry=ROOT/"outputs/acoserl_v3/hold_carry_rotation_v2"
late=ROOT/"outputs/acoserl_v3/hold_late_carry_b_v1"
rows=[]
for directory in (base,carry,late):
    for r in json.loads((directory/"summary.json").read_text()):
        if directory==base and r["spec"]["domain"]=="carry":continue
        r["evidence"]=str((directory/f"case_{r['case']:02d}/result.json").relative_to(ROOT))
        # Empty passive hands have no held-object rotation to bound.
        if "actual_peak_held_object_rotation_rad" not in r:
            assert not any(r["entry_state"]["actual_grasps"]["AB"[r["spec"]["passive"]]].values())
            r["actual_peak_held_object_rotation_rad"]=0.
        rows.append(r)
result=dict(version="bounded_empirical_passive_hold_v3r3",passed=False,
    selected_physical_cases=30,calibration_cases=18,independent_validation_cases=12,
    sources=[str(d.relative_to(ROOT)) for d in (base,carry,late)],
    control_hz=20,physics_dt_s=.002,domains={},
    position_reserve_m=.0001,rotation_reserve_rad=.001,
    scope="Empirical passive pose/grasp/speed envelope; active independence must be measured. No universal guarantee.")
for kind in ("free","carry","contact","carry_late"):
 for pi in (0,1):
    data=[r for r in rows if r["spec"]["domain"]==kind and r["spec"]["passive"]==pi]
    if not data:continue
    cal=[r for r in data if r["spec"]["split"]=="calibration"]
    valid=[r for r in data if r["spec"]["split"]=="validation"]
    assert len(cal)==3 and len(valid)==2
    ref=cal[0]["entry_state"];positions=np.asarray([[p["position"] for p in r["entry_state"]["poses"]] for r in cal])
    def residual(actual,pred):return max(0.,max(r[actual]-pred(r) for r in cal))
    object_nominal=lambda r:max(r["entry"]["object_position_predictions_m"].values(),default=0.)
    proof=dict(passed=False,horizon_controls=cal[0]["spec"].get("hold_controls",80),
        position_error_bound_m=residual("actual_peak_position_m",lambda r:r["nominal_position_m"])+.0001,
        rotation_error_bound_rad=residual("actual_peak_rotation_rad",lambda r:r["entry"]["predicted_peak_rotation_rad"])+.001,
        object_position_error_bound_m=residual("actual_peak_held_object_m",object_nominal)+.0001,
        object_rotation_error_bound_rad=residual("actual_peak_held_object_rotation_rad",lambda r:r["entry"]["predicted_peak_rotation_rad"])+.001,
        max_linear_speed_m_s=max(r["entry"]["linear_speed_m_s"] for r in cal),
        max_angular_speed_rad_s=max(r["entry"]["angular_speed_rad_s"] for r in cal),
        eef_min_m=(positions.min(0)-.002).tolist(),eef_max_m=(positions.max(0)+.002).tolist(),
        rotations=[p["rotation"] for p in ref["poses"]],rotation_domain_rad=.01,
        actual_grasps=ref["actual_grasps"],first_piece_assembled=ref["official_metrics"]["first_piece_assembled"],
        relative_grasp_positions={},relative_grasp_rotations={},requires_independent_active=kind!="contact",
        calibration_paths=[r["evidence"] for r in cal],validation=[])
    for obj,held in ref["actual_grasps"]["AB"[pi]].items():
        if not held:continue
        eef=ref["poses"][pi];obj_pose=ref["objects"][obj]
        proof["relative_grasp_positions"][obj]=(np.asarray(eef["rotation"]).T@(np.asarray(obj_pose["position"])-eef["position"])).tolist()
        proof["relative_grasp_rotations"][obj]=(np.asarray(eef["rotation"]).T@Rotation.from_quat(np.roll(obj_pose["quaternion_wxyz"],-1)).as_matrix()).tolist()
    for r in valid:
        pred=r["entry"];upper=pred["predicted_peak_position_m"]+proof["position_error_bound_m"]
        obj_upper=object_nominal(r)+proof["object_position_error_bound_m"]
        rot_upper=pred["predicted_peak_rotation_rad"]+proof["rotation_error_bound_rad"]
        object_rot_upper=pred["predicted_peak_rotation_rad"]+proof["object_rotation_error_bound_rad"]
        okay=(r["complete"] and r["physical_writes"]==0
            and r["actual_peak_position_m"]<=upper<.002
            and r["actual_peak_rotation_rad"]<=rot_upper<.03
            and r["actual_peak_held_object_m"]<=obj_upper<.003
            and r["actual_peak_held_object_rotation_rad"]<=object_rot_upper<.035
            and r["first_piece_peak_m"]<.002 and r["max_force_N"]<=80)
        proof["validation"].append(dict(path=r["evidence"],passed=bool(okay),
            position_upper_m=upper,object_upper_m=obj_upper,object_rotation_upper_rad=object_rot_upper))
    proof["passed"]=all(v["passed"] for v in proof["validation"])
    result["domains"][kind+":"+"AB"[pi]]=proof
result["passed"]=all(p["passed"] for p in result["domains"].values())
result["source_sha256"]={str((d/"summary.json").relative_to(ROOT)):hashlib.sha256((d/"summary.json").read_bytes()).hexdigest() for d in (base,carry,late)}
out=ROOT/"configs/bimanual_autoserl/hold_envelope_v3r3.json";assert not out.exists()
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(dict(passed=result["passed"],domains={k:dict(passed=v["passed"],horizon=v["horizon_controls"],eef_error=v["position_error_bound_m"],object_error=v["object_position_error_bound_m"],object_rotation_error=v["object_rotation_error_bound_rad"]) for k,v in result["domains"].items()})))
assert result["passed"]
