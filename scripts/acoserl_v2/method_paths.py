"""Four real method paths on one paired, action-reached pregrasp stagnation state."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,servo,save
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.intervention_manager import method_config
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
out=ROOT/'outputs/acoserl_v2'/a.tag;out.mkdir(exist_ok=False)
results={}
for method in ('plain_rlpd','independent_window','window_recovery','full_acoserl'):
 E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False);M.manager_cfg=method_config(method);R.safety=M.substep_safety
 rows=[];result=dict(method=method,passed=False)
 def step(policy):
  final,decision=M.choose(policy)
  if final is None:raise RuntimeError(M.stop_reason)
  state=R.step(final);after=M.after_step()
  rows.append(dict(mode=decision['mode'],decision=copy.deepcopy(decision),after=after,action=final,poses=state['poses'],objects=state['objects'],grasps=state['actual_grasps'],contact=M.contacts.read()))
  if M.mode=='Safe_Stop':raise RuntimeError(M.stop_reason)
  return state,after
 try:
  for action in D['actions'][:38]:step(action)
  hold=copy.deepcopy(snapshot(E)['poses']);grips=[-1.,-1.]
  for _ in range(40):step(servo(E,hold,grips,max_position_step=.002))
  start=snapshot(E);M.cfg['intervention_enabled']=True;mode_seen=set()
  for _ in range(180 if method in ('window_recovery','full_acoserl') else 24):
   state,after=step(servo(E,hold,grips,max_position_step=.002));mode_seen.add(rows[-1]['mode'])
   if after.get('safe_handback'):break
  required=method in ('window_recovery','full_acoserl')
  result.update(modes=sorted(mode_seen),handback=bool(after.get('safe_handback')),initial_poses=start['poses'],initial_grasps=start['actual_grasps'],actual_steps=len(rows),last_after=after,decision_stats=M.decision_stats)
  result['passed']=bool(after.get('safe_handback')) and any(s.startswith('Recover_') for s in mode_seen) if required else all(s=='Normal' for s in mode_seen)
 except BaseException:
  import traceback
  result['error']=traceback.format_exc()
 finally:
  result['physical_writes']=R.physical_writes_attempted;results[method]=result;save(out/(method+'.json'),result);save(out/(method+'_trace.json'),rows);E.close()
save(out/'result.json',results);print(json.dumps({k:v['passed'] for k,v in results.items()}),flush=True)
