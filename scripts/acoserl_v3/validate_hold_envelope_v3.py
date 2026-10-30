"""Runtime domain coverage and all-constraint prediction checks on real held-out rows."""
import json
from pathlib import Path
from acoserl.hold_envelope import evaluate
ROOT=Path(__file__).resolve().parents[2]
cfg=json.loads((ROOT/"configs/bimanual_autoserl/v1.json").read_text())
e=json.loads((ROOT/"configs/bimanual_autoserl/hold_envelope_v3r3.json").read_text())
checks=[]
for d in e["domains"].values():
 for path in d["calibration_paths"]+[v["path"] for v in d["validation"]]:
  r=json.loads((ROOT/path).read_text());g=evaluate(r["entry_state"],r["entry"],cfg,e)
  held_rotation=r.get("actual_peak_held_object_rotation_rad",0.)
  covered=(r["actual_peak_position_m"]<=g.get("conservative_peak_position_m",0)
    and r["actual_peak_rotation_rad"]<=g.get("conservative_peak_rotation_rad",0)
    and all(r["actual_peak_held_object_m"]<=p for p in g.get("conservative_object_position_m",{}).values())
    and (not g.get("conservative_object_position_m") or held_rotation<=g["conservative_object_rotation_rad"]))
  checks.append(dict(path=path,split=r["spec"]["split"],allowed=g["allowed"],covered=bool(covered),domain=g.get("calibrated_domain")))
result=dict(passed=all(x["allowed"] and x["covered"] for x in checks),cases=checks,
 independent_validation_cases=sum(x["split"]=="validation" for x in checks),bounded_empirical_only=True)
out=ROOT/"outputs/acoserl_v3/hold_envelope_v3r3_validation.json";assert not out.exists();out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result));assert result["passed"]
