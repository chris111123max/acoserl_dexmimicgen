"""Fit empirical hold-error envelopes from calibration rows only, then validate."""
import json,argparse,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def build(tag):
    source=ROOT/"outputs/acoserl_v3"/tag
    rows=json.loads((source/"summary.json").read_text());domains={};validation=[]
    for row in rows:
        key=row["spec"]["domain"]+":"+"AB"[row["spec"]["passive"]]
        if row["spec"]["split"]!="calibration":continue
        domains.setdefault(key,[]).append(row)
    result=dict(version="bounded_empirical_passive_hold_v3",source=str(source.relative_to(ROOT)),
        calibration_cases=15,independent_validation_cases=10,horizon_controls=80,control_hz=20,
        physics_dt_s=.002,position_reserve_m=.0001,rotation_reserve_rad=.001,
        note="Measured calibration maximum underprediction plus explicit reserve; not a universal physics certificate.",domains={})
    for key,cal in domains.items():
        assert all(r["complete"] and r["physical_writes"]==0 for r in cal)
        error=max(0.,max(r["actual_peak_position_m"]-r["nominal_position_m"] for r in cal))
        angular=max(0.,max(r["actual_peak_rotation_rad"]-r["entry"]["predicted_peak_rotation_rad"] for r in cal))
        poses=np.asarray([[p["position"] for p in r["entry_state"]["poses"]] for r in cal])
        relative={}
        passive="AB".index(key[-1])
        reference=cal[0]["entry_state"]
        for name,held in reference["actual_grasps"][key[-1]].items():
            if held:
                ee=reference["poses"][passive]
                relative[name]=(np.asarray(ee["rotation"]).T@(np.asarray(reference["objects"][name]["position"])-ee["position"])).tolist()
        proof=dict(position_error_bound_m=error+result["position_reserve_m"],
            rotation_error_bound_rad=angular+result["rotation_reserve_rad"],
            max_linear_speed_m_s=max(r["entry"]["linear_speed_m_s"] for r in cal),
            max_angular_speed_rad_s=max(r["entry"]["angular_speed_rad_s"] for r in cal),
            eef_min_m=(poses.min(0)-.002).tolist(),eef_max_m=(poses.max(0)+.002).tolist(),
            rotations=[p["rotation"] for p in reference["poses"]],rotation_domain_rad=.01,
            actual_grasps=reference["actual_grasps"],relative_grasp_positions=relative,
            first_piece_assembled=reference["official_metrics"]["first_piece_assembled"],
            calibration_cases=[r["case"] for r in cal],validation_cases=[],passed=False)
        tests=[r for r in rows if r["spec"]["split"]=="validation" and r["spec"]["domain"]+":"+"AB"[r["spec"]["passive"]]==key]
        checks=[]
        for r in tests:
            okay=(r["complete"] and r["physical_writes"]==0
                  and r["actual_peak_position_m"]<=r["nominal_position_m"]+proof["position_error_bound_m"]
                  and r["actual_peak_rotation_rad"]<=r["entry"]["predicted_peak_rotation_rad"]+proof["rotation_error_bound_rad"]
                  and r["actual_peak_position_m"]<.002 and r["actual_peak_held_object_m"]<.003
                  and r["first_piece_peak_m"]<.002 and r["max_force_N"]<=80)
            checks.append(okay);proof["validation_cases"].append(r["case"])
            validation.append(dict(domain=key,case=r["case"],passed=bool(okay),actual_m=r["actual_peak_position_m"],upper_m=r["nominal_position_m"]+proof["position_error_bound_m"]))
        proof["passed"]=bool(checks and all(checks));result["domains"][key]=proof
    result["validation"]=validation;result["passed"]=all(d["passed"] for d in result["domains"].values())
    result["source_sha256"]=hashlib.sha256((source/"summary.json").read_bytes()).hexdigest()
    out=ROOT/"configs/bimanual_autoserl/hold_envelope_v3.json"
    assert not out.exists(),"Never silently replace a calibrated envelope"
    out.write_text(json.dumps(result,indent=2)+"\n")
    (source/"envelope_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(passed=result["passed"],domains={k:dict(position_bound_m=v["position_error_bound_m"],max_speed=v["max_linear_speed_m_s"],passed=v["passed"]) for k,v in result["domains"].items()})))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);a=p.parse_args();build(a.tag)
