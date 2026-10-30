"""Evaluate the frozen empirical envelope on real saved calibration/validation states."""
import json
from pathlib import Path
from acoserl.hold_envelope import evaluate
ROOT=Path(__file__).resolve().parents[2]
config=json.loads((ROOT/"configs/bimanual_autoserl/v1.json").read_text())
envelope=json.loads((ROOT/"configs/bimanual_autoserl/hold_envelope_v3.json").read_text())
source=ROOT/envelope["source"];rows=json.loads((source/"summary.json").read_text());checks=[]
for r in sorted(rows,key=lambda r:r["case"]):
    gate=evaluate(r["entry_state"],r["entry"],config,envelope)
    safe=(r["actual_peak_position_m"]<=gate.get("conservative_peak_position_m",0)
          and r["actual_peak_rotation_rad"]<=gate.get("conservative_peak_rotation_rad",0)
          and all(r["actual_peak_held_object_m"]<=bound for bound in gate.get("conservative_object_position_m",{}).values()))
    checks.append(dict(case=r["case"],split=r["spec"]["split"],domain=gate.get("calibrated_domain"),
                       allowed=gate["allowed"],predictions_cover_actual=bool(safe)))
result=dict(passed=all(r["allowed"] and r["predictions_cover_actual"] for r in checks),checks=checks,
    calibration_cases=15,independent_validation_cases=10,physical_source=str(source))
(source/"runtime_envelope_validation.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result));assert result["passed"]
