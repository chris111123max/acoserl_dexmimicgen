"""Real contact classification and hard collision rejection; no RL training."""
import json,sys,argparse
from pathlib import Path
import numpy as np
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.autoserl_env import StrictTaskProtocol
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"scripts/assembly_feasibility"))
from common import setup,snapshot,servo,save,SafetyStop
parser=argparse.ArgumentParser();parser.add_argument("--tag",required=True);args=parser.parse_args()
out=ROOT/"outputs/acoserl_v2"/args.tag;out.mkdir(exist_ok=False)
records=[];result={}
for case in ("demonstration","robot_collision","unknown_coupling"):
 E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False);R.safety=M.substep_safety
 item=dict(passed=False,physical_writes=0);rows=[];P=StrictTaskProtocol()
 try:
  if case=="demonstration":
   for t,action in enumerate(D["actions"]):
    final,decision=M.choose(action)
    assert final is not None,M.stop_reason
    s=R.step(final);after=M.after_step();success=P.observe(E)
    assessment=M.contact_assessment(s)
    rows.append(dict(frame=t,assessment=assessment,metrics=s["official_metrics"],indices=M.indices()))
    assert M.mode!="Safe_Stop",M.stop_reason
    if success["success"]:break
   item.update(passed=success["success"],actual_steps=t+1,categories=sorted(set(x["assessment"]["category"] for x in rows)))
  else:
   # All contact approaches are physical OSC actions from the canonical reset.
   if case=="unknown_coupling":
    for t in range(110):
     final,_=M.choose(D["actions"][t]);R.step(final);M.after_step()
    baseline=snapshot(E);targets=baseline["poses"];grips=[1.,1.]
    targets[1]=M.pose(1,214)
   else:
    targets=snapshot(E)["poses"];grips=[-1.,-1.]
    center=(targets[0]["position"]+targets[1]["position"])/2
    for target in targets:target["position"]=center.copy()
   for t in range(350):
    action=servo(E,targets,grips,max_position_step=.006,position_gain=2)
    final,decision=M.choose(action)
    if final is None:
     item.update(passed=True,reason=M.stop_reason,refused_before_step=True,actual_steps=t,contact=M.contacts.read());break
    try:state=R.step(final)
    except SafetyStop as error:
     item.update(passed=True,reason=str(error),partial_step=True,actual_steps=t,contact=M.contacts.read());break
    M.after_step()
    rows.append(dict(frame=t,action=final,assessment=M.contact_assessment()))
    if M.mode=="Safe_Stop":
     item.update(passed=True,reason=M.stop_reason,full_step_boundary=True,actual_steps=t+1);break
   if case=="unknown_coupling":
    item["actual_unknown_coupling_reached"]=str(item.get("reason","")).startswith("unqualified_dynamic_contact_chain")
 except BaseException:
  import traceback
  item["error"]=traceback.format_exc()
 finally:
  item["physical_writes"]=R.physical_writes_attempted
  item["final_snapshot"]=snapshot(E)
  save(out/(case+".json"),item);save(out/(case+"_trace.json"),rows)
  E.close();result[case]=item
save(out/"result.json",result);print(json.dumps(result,default=str),flush=True)
