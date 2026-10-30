"""Directed passive deceleration: physically reach a failed direct-hold gate."""
import sys,json,copy,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"scripts/assembly_feasibility"))
from common import setup,snapshot,servo,save
from acoserl.bimanual_autoserl import BimanualAutoSERL
parser=argparse.ArgumentParser();parser.add_argument("--tag",required=True);args=parser.parse_args()
out=ROOT/"outputs/acoserl_v2"/args.tag;out.mkdir(exist_ok=False)
results=[]
for trial in range(3):
 E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False);R.safety=M.substep_safety
 result=dict(trial=trial,passed=False,actor_handoff_verified=False);rows=[]
 try:
  def step(policy):
   action,decision=M.choose(policy)
   if action is None:raise RuntimeError(M.stop_reason)
   state=R.step(action);after=M.after_step();rows.append(dict(after=after,action=action,state=state))
   if M.mode=="Safe_Stop":raise RuntimeError(M.stop_reason)
   return state,after
  for action in D["actions"][:80]:step(action)
  targets=copy.deepcopy(snapshot(E)["poses"]);targets[1]["position"][trial%2]+=.004
  for t in range(60):step(servo(E,targets,[1,1],max_position_step=.002,position_gain=2))
  targets[0]["position"][2]+=.08
  gate=None
  for t in range(14):
   step(servo(E,targets,[1,1],max_position_step=.014+.001*trial,position_gain=2))
   gate=M.passive_hold_entry(1,snapshot(E))
   if not gate["allowed"]:break
  result["entry_gate"]=gate
  if gate["allowed"]:raise RuntimeError("failed_to_reach_braking_domain")
  allowed,reason=M.start_local(1,80,validation=True)
  if not allowed:raise RuntimeError(reason)
  result["brake_selected"]=M.local.stage=="Brake"
  for t in range(85):
   state,after=step(servo(E,targets,[1,1]))
   if after.get("safe_handback"):
    result.update(passed=True,brake_steps=after.get("brake_steps"),local_handback=True);break
 except BaseException:
  import traceback
  result["error"]=traceback.format_exc()
 finally:
  result.update(physical_writes=R.physical_writes_attempted,actual_steps=len(rows))
  save(out/(str(trial)+".json"),result);save(out/(str(trial)+"_trace.json"),rows);E.close()
 results.append(result)
save(out/"result.json",results);print(json.dumps(results),flush=True)
