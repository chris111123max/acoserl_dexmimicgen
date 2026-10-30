"""Full canonical demo through current hard safety; never a claim of Actor success."""
import argparse,json,copy,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,save,SafetyStop
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.autoserl_env import StrictTaskProtocol
from resfit.rl_finetuning.utils.normalization import ActionScaler
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--diagnose-support',action='store_true');a=p.parse_args()
out=ROOT/'outputs/autoserl_baseline'/a.tag;out.mkdir(exist_ok=False)
E,D,R=setup();m=BimanualAutoSERL(E,intervention_enabled=False);m.manager_cfg['v3_enabled']=True;m.manager_cfg['baseline_mode']=True
first_partial=None;i=-1;support_samples=[]
from acoserl.progress_guard import velocities
def monitor(contact):
 global first_partial
 reason=m.substep_safety(contact)
 if a.diagnose_support and m.progress_guard.anchor is not None:
  s=snapshot(E);check=m.progress_guard.check(s)
  support_samples.append(dict(time=s['sim_time'],frame=i,check=check,velocities=velocities(E)['objects']['piece_1'],contacts=[e for e in contact['edges'] if 'piece_1' in (e['a'],e['b'])],pending=m.pending_release_valid(s),grasp=s['actual_grasps']))
  if reason=='completed_piece1_support_lost':return None
 if reason and first_partial is None:
  s=snapshot(E);first_partial=dict(frame=i,reason=reason,state=s,velocities=velocities(E),contact=contact,protection=m.progress_guard.check(s))
 return reason
R.safety=monitor
n=json.loads((ROOT/'docs/ascend16/normalization.json').read_text());scale=ActionScaler(torch.tensor(n['effective_action_min']),torch.tensor(n['effective_action_max']),action_scale=0,min_range_per_dim=.1)
m.action_bounds=(scale.limits.min.numpy(),scale.limits.max.numpy());protocol=StrictTaskProtocol()
rows=[];result=dict(passed=False,source='demonstration_not_Actor',recorded_actions=len(D['actions']),physical_writes=0,projection_steps=0)
try:
 for i,raw in enumerate(D['actions']):
  policy=scale.unscale(scale.scale(torch.tensor(raw,dtype=torch.float32))).numpy()
  final,decision=m.choose(policy)
  if final is None:raise RuntimeError('pre-step refusal:'+str(m.stop_reason))
  action=policy if decision['mode']=='Normal' else scale.unscale(scale.scale(torch.tensor(final,dtype=torch.float32))).numpy()
  R.step(action);after=m.after_step();state=snapshot(E);strict=protocol.observe(E)
  result['projection_steps']+=int(decision['mode'].startswith('Constrained_'))
  rows.append(dict(frame=i,demonstration_action=raw,actual_action=action,decision=copy.deepcopy(decision),after=after,state=state,contact=m.contacts.read(),gripper_commands=[r.gripper['right'].current_action for r in E.robots],strict=strict))
  if i%50==0:print('frame',i,'mode',decision['mode'],'task',state['official_metrics'],flush=True)
  if m.mode=='Safe_Stop':raise RuntimeError(m.stop_reason)
  if strict['success']:result['passed']=True;break
except BaseException as error:
 result['error']=traceback.format_exc()
 if isinstance(error,SafetyStop):m.stop('physics_substep_safety:'+str(error))
finally:
 result.update(complete_controls=len(rows),attempted_controls=i+1,first_partial=first_partial,physical_writes=R.physical_writes_attempted,stop_reason=m.stop_reason,
  final_metrics=rows[-1]['state']['official_metrics'] if rows else None,final_strict=rows[-1]['strict'] if rows else None,
  observed_grasps={arm:any(r['state']['actual_grasps'][arm][obj] for r in rows) for arm,obj in [('A','piece_1'),('B','piece_2')]},
  controller_fingerprint=m.v3_fingerprint())
 result['support_diagnostic_only']=a.diagnose_support;save(out/'support_samples.json',support_samples);save(out/'result.json',result);save(out/'controls.json',rows);E.close();print(json.dumps(result,default=str),flush=True)
