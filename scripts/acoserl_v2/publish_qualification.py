"""Publish empirical, narrow qualifications; failed domains stay disabled."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];base=ROOT/"outputs/acoserl_v2"
def read_group(tag,kind,indices):
 return [(base/tag/(kind+"_"+str(i)+".json"),json.loads((base/tag/(kind+"_"+str(i)+".json")).read_text())) for i in indices]
def measurements(items):
 objpeak=0.;speedpeak=0.;both=True;shield_speed=0.;capture_position=0.;capture_rotation=0.
 trace=np.load(ROOT/"configs/bimanual_autoserl/reference_trace.npz")
 for path,d in items:
  both &= not d.get("passive_empty",False)
  speedpeak=max(speedpeak,d.get("passive_entry",{}).get("linear_speed_m_s",0))
  rows=[json.loads(s) for s in path.with_suffix(".jsonl").read_text().splitlines()]
  previous=None;baseline=None;active=0 if d["kind"].endswith("A") else 1
  for row in rows:
   if row["mode"].startswith("Recover_"):
    if baseline is None:
     baseline=previous
     if baseline and d["kind"]!="Contact_B":
      capture_position=max(capture_position,float(np.linalg.norm(np.asarray(baseline["poses"][active]["position"])-trace["positions"][80,active])))
      capture_rotation=max(capture_rotation,float(Rotation.from_matrix(np.asarray(baseline["poses"][active]["rotation"])@trace["rotations"][80,active].T).magnitude()))
    if baseline:
     pi="B" if active==0 else "A"
     for obj,held in baseline["grasps"][pi].items():
      if held:objpeak=max(objpeak,float(np.linalg.norm(np.array(row["objects"][obj]["position"])-baseline["objects"][obj]["position"])))
   else:
    if baseline is not None:baseline=None
   for risk in row.get("decision",{}).get("policy_risks",[]):shield_speed=max(shield_speed,risk.get("linear_speed_m_s",0))
   previous=row
 return dict(object_peak_m=objpeak,capture_position_max=capture_position,capture_rotation_max=capture_rotation,entry_speed_max=speedpeak,requires_both_holding=both,shield_speed_max=shield_speed)
direct={}
combined=[]
for arm in ("A","B"):
 items=read_group("local_physics_v3","Carry_"+arm,range(6))+read_group("hold_domain_v1","Carry_"+arm,range(6))
 summary=measurements(items);valid=all(d["passed"] and d.get("actor_steps")==5 and d["physical_writes"]==0 and d["max_passive_m"]<=.002 for _,d in items) and summary["object_peak_m"]<=.003
 direct[arm]=dict(passed=valid,trials=len(items),**summary,paths=[str(p.relative_to(ROOT)) for p,_ in items])
 combined+=items
brakes=read_group("brake_actor_v1","Carry_B",range(12));bm=measurements(brakes)
brake_pass=all(d["passed"] and d.get("brake_steps",0)>=3 and not d["passive_entry"]["allowed"] and d["actor_steps"]==5 and d["physical_writes"]==0 and d["max_passive_m"]<=.002 for _,d in brakes) and bm["object_peak_m"]<=.003
contacts=read_group("local_physics_v3","Contact_B",range(12))
contact_pass=all(d["passed"] and d["actor_steps"]==5 and d["physical_writes"]==0 for _,d in contacts)
local_hash=hashlib.sha256((ROOT/"src/acoserl/local_recovery.py").read_bytes()).hexdigest()
manager_hash=hashlib.sha256((ROOT/"src/acoserl/manager_v2.py").read_bytes()).hexdigest()
speed=[d["passive_entry"]["linear_speed_m_s"] for _,d in brakes]
all_shield=measurements(combined+brakes)
proof={
 "version":"acoserl_v2_empirical_qualified_domains",
 "carry_local":dict(passed=all(d["passed"] for d in direct.values()),controller_sha256=local_hash,
   requires_both_holding=True,reference_index=80,
   target_poses={a:dict(position=np.load(ROOT/"configs/bimanual_autoserl/reference_trace.npz")["positions"][80,i].tolist(),rotation=np.load(ROOT/"configs/bimanual_autoserl/reference_trace.npz")["rotations"][80,i].tolist()) for i,a in enumerate(("A","B"))},
   direct_hold_max_speed={a:d["entry_speed_max"] for a,d in direct.items()},
   capture_position_max={"A":direct["A"]["capture_position_max"],"B":max(direct["B"]["capture_position_max"],bm["capture_position_max"])},
   capture_rotation_max={"A":direct["A"]["capture_rotation_max"],"B":max(direct["B"]["capture_rotation_max"],bm["capture_rotation_max"])},per_arm=direct,
   excluded="passive empty handback failed; all other reference neighborhoods unqualified"),
 "passive_brake":dict(passed=brake_pass,controller_sha256=local_hash,
   speed_intervals={"B":[min(speed),max(speed)]},trials=12,**bm,
   excluded="active A braking and all other speed/phase domains"),
 "joint":dict(passed=contact_pass,controller_sha256=local_hash,
   mode="B second-insertion vertical 3mm unload; A completed and held static",
   trials=12,passed_trials=sum(d["passed"] for _,d in contacts),
   failed_cases=[str(p.relative_to(ROOT)) for p,d in contacts if not d["passed"]],
   reason="requires all 12 complete physical+Actor handback trials; no general coupled recovery"),
 "joint_grip_shield":dict(passed=all(d["passed"] for d in direct.values()) and brake_pass,
   controller_sha256=manager_hash,max_speed_m_s=all_shield["shield_speed_max"],
   domain="independent correct two-object grasps, >=10 stable grasp steps; only retain both closed gripper commands; Actor pose commands unchanged except workspace clipping",
   not_a_coupled_recovery=True),
 "formal_training_approved":False}
p=ROOT/"configs/bimanual_autoserl/qualification_v2.json"
p.write_text(json.dumps(proof,indent=2)+"\n")
(base/"qualification_published.json").write_text(json.dumps(proof,indent=2)+"\n")
print(json.dumps({k:v.get("passed") for k,v in proof.items() if isinstance(v,dict)}),flush=True)
