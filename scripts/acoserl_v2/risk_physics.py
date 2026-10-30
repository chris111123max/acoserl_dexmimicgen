"""Directed risk execution checks, physically reached states, unchanged limits."""
import json,copy,sys,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"scripts/assembly_feasibility"))
from common import setup,snapshot,servo,save
from acoserl.bimanual_autoserl import BimanualAutoSERL
parser=argparse.ArgumentParser();parser.add_argument("--tag",required=True);args=parser.parse_args()
out=ROOT/"outputs/acoserl_v2"/args.tag;out.mkdir(exist_ok=False)
result={}
for case in ("workspace_A","workspace_B","both_workspace","open_gripper_A","passive_dynamic"):
 E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False);R.safety=M.substep_safety;rows=[]
 record=dict(passed=False)
 try:
  def step(action):
   final,decision=M.choose(action)
   if final is None:return None,decision
   state=R.step(final);after=M.after_step();rows.append(dict(action=final,decision=copy.deepcopy(decision),after=after,state=state))
   if M.mode=="Safe_Stop":raise RuntimeError(M.stop_reason)
   return state,decision
  initial=snapshot(E);targets=copy.deepcopy(initial["poses"]);grips=[-1.,-1.]
  if "workspace" in case:
   arms=[0,1] if case=="both_workspace" else [0 if case.endswith("A") else 1]
   for ai in arms:
    axis=2 if ai==0 else 1
    targets[ai]["position"][axis]=M.positions[:,ai,axis].max()+M.cfg["free_workspace_margin_m"]-.006
   for t in range(300):
    state,_=step(servo(E,targets,grips,max_position_step=.006,position_gain=2))
    if state is None:raise RuntimeError(M.stop_reason)
    if max(np.linalg.norm(state["poses"][ai]["position"]-targets[ai]["position"]) for ai in arms)<.001:break
   for t in range(10):step(servo(E,targets,grips,max_position_step=.006,position_gain=2))
   proposed=servo(E,targets,grips)
   for ai in arms:
    ctl=E.robots[ai].part_controllers["right"]
    proposed[7*ai:7*ai+3]=ctl.origin_ori.T@(np.array([0,0,.012]) if ai==0 else np.array([0,.012,0]))/ctl.output_max[:3]
  elif case=="open_gripper_A":
   for action in D["actions"][:51]:step(action)
   state=snapshot(E);proposed=servo(E,state["poses"],[-1.,-1.])
   direction=np.asarray(state["objects"]["piece_1"]["position"])-state["poses"][0]["position"]
   direction*=.004/max(np.linalg.norm(direction),1e-9)
   ctl=E.robots[0].part_controllers["right"];proposed[:3]=ctl.origin_ori.T@direction/ctl.output_max[:3];proposed[6]=-1
  else:
   for action in D["actions"][:80]:step(action)
   state=snapshot(E);targets=copy.deepcopy(state["poses"]);grips=[1.,1.]
   for t in range(50):step(servo(E,targets,grips,max_position_step=.002,position_gain=2))
   targets[0]["position"][2]+=.08
   for t in range(10):
    step(servo(E,targets,grips,max_position_step=.02,position_gain=2))
    if not M.passive_hold_entry(1,snapshot(E))["allowed"]:break
   gate=M.passive_hold_entry(1,snapshot(E));record["passive_entry"]=gate
   proposed=servo(E,targets,grips)
  state=snapshot(E);risks=M.action_risks(proposed,state)
  record["risk_before"]=risks;M.cfg["intervention_enabled"]=True
  final,decision=M.choose(proposed)
  record.update(decision=decision,proposed_action=proposed)
  if case=="both_workspace":record["passed"]=final is None and M.mode=="Safe_Stop"
  elif case=="passive_dynamic":record["passed"]=not record["passive_entry"]["allowed"]
  else:
   expected_ai=1 if case=="workspace_B" else 0
   record["passed"]=risks[expected_ai]["risk"] and (final is None or not np.array_equal(final,proposed))
   if final is not None:
    state=R.step(final);after=M.after_step()
    record.update(executed_action=final,after=after,force_N=M.contacts.read()["max_robot_contact_force_N"])
 except BaseException:
  import traceback
  record["passed"]=False;record["error"]=traceback.format_exc()
 finally:
  record.update(physical_writes=R.physical_writes_attempted,actual_complete_steps=len(rows))
  save(out/(case+".json"),record);save(out/(case+"_trace.json"),rows);result[case]=record;E.close()
save(out/"result.json",result);print(json.dumps(result,default=str),flush=True)
