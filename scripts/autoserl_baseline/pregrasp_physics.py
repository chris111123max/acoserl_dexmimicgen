"""Four bounded real P0/window tests; task-controller proposals, not Actor success."""
import os,argparse,copy,json,sys,traceback,multiprocessing as mp,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def case(value):
 tag,arm,kind,production=value
 os.environ['ASCEND_RT_VISIBLE_DEVICES']=''
 import torch
 torch.set_num_threads(1);sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
 from common import setup,snapshot,servo,save,SafetyStop
 from acoserl.bimanual_autoserl import BimanualAutoSERL
 from acoserl.intervention_manager import method_config
 from resfit.rl_finetuning.utils.normalization import ActionScaler
 out=ROOT/'outputs/autoserl_baseline'/tag/(kind+'_'+arm);out.mkdir(parents=True,exist_ok=False)
 E,D,R=setup();m=BimanualAutoSERL(E,intervention_enabled=False)
 m.manager_cfg={**method_config('full_acoserl'),'v3_enabled':True,'baseline_mode':True,'development_validation':not production,
                'carry_local':{'enabled':False},'joint':{'enabled':False},'passive_brake':{'enabled':False}}
 n=json.loads((ROOT/'docs/ascend16/normalization.json').read_text());scale=ActionScaler(torch.tensor(n['effective_action_min']),torch.tensor(n['effective_action_max']),action_scale=0,min_range_per_dim=.1)
 m.action_bounds=(scale.limits.min.numpy(),scale.limits.max.numpy());R.safety=m.substep_safety
 ai='AB'.index(arm);prefix=30 if arm=='A' else 58
 complete=0;rows=[];result=dict(passed=False,arm=arm,kind=kind,proposal_source='controlled_test_controller_not_Actor')
 def execute(action,supervise=False):
  nonlocal complete
  policy=scale.unscale(scale.scale(torch.tensor(action,dtype=torch.float32))).numpy()
  before=snapshot(E)
  if supervise:
   final,decision=m.choose(policy)
   if final is None:raise RuntimeError('refused:'+str(m.stop_reason))
  else:final=policy;decision=dict(mode='DiagnosticPreparation',policy_action=policy.tolist())
  physical=policy if decision['mode'] in ('Normal','DiagnosticPreparation') else scale.unscale(scale.scale(torch.tensor(final,dtype=torch.float32))).numpy()
  R.step(physical);complete+=1
  after=m.after_step() if supervise else {}
  rows.append(dict(control=complete-1,before=before,after_state=snapshot(E),decision=copy.deepcopy(decision),action=physical,after=after,contact=m.contacts.read()))
  if m.mode=='Safe_Stop':raise RuntimeError(m.stop_reason)
  return after
 try:
  for action in D['actions'][:prefix]:execute(action)
  state=snapshot(E);targets=copy.deepcopy(state['poses']);grips=[1. if any(state['actual_grasps'][a].values()) else -1. for a in 'AB']
  if kind=='window':targets[ai]=m.pose(ai,10 if ai==0 else 20)
  else:
   tangent=m.positions[prefix+1,ai]-m.positions[prefix,ai]
   targets[ai]['position']+=.003*tangent/max(np.linalg.norm(tangent),1e-9)
  for _ in range(100):execute(servo(E,targets,grips,max_position_step=.006,position_gain=2))
  m.reset((prefix,prefix));m.cfg['intervention_enabled']=True
  result['entry']=snapshot(E);result['hold_gate']=m.passive_hold_entry(ai,snapshot(E));seen=[];handback=False;post=0
  for _ in range(240):
   if m.handoff_phase=='Handoff_Ready':
    now=snapshot(E);targets=copy.deepcopy(now['poses']);grips=[1. if any(now['actual_grasps'][a].values()) else -1. for a in 'AB']
   after=execute(servo(E,targets,grips,max_position_step=.006,position_gain=2),True)
   mode=rows[-1]['decision']['mode'];seen.append(mode)
   if after.get('safe_handback'):
    handback=True;now=snapshot(E);targets=copy.deepcopy(now['poses']);grips=[1. if any(now['actual_grasps'][a].values()) else -1. for a in 'AB']
   elif handback:post+=1
   if handback and post>=5:break
  held=snapshot(E)['actual_grasps'][arm][m.points['arms'][arm]['object']]
  required='Guide_'+arm if kind=='window' else 'Recover_'+arm
  result.update(passed=handback and required in seen and (held if kind=='recovery' else True),
                handback=handback,post_handback_steps=post,observed_modes=sorted(set(seen)),active_grasp=held)
 except BaseException:result['error']=traceback.format_exc()
 finally:
  result.update(complete_controls=complete,physical_writes=R.physical_writes_attempted,stop_reason=m.stop_reason,
      stats=m.v3_stats,decision_stats=m.decision_stats,controller_fingerprint=m.v3_fingerprint())
  save(out/'result.json',result);save(out/'controls.json',rows);E.close()
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--kind',choices=['window','recovery']);p.add_argument('--production',action='store_true');a=p.parse_args()
 out=ROOT/'outputs/autoserl_baseline'/a.tag;out.mkdir(exist_ok=False)
 work=[(a.tag,arm,kind,a.production) for arm in ('A','B') for kind in ('window','recovery') if not a.kind or kind==a.kind]
 (out/'cases.json').write_text(json.dumps(work))
 with mp.get_context('spawn').Pool(4) as pool:
  results=[]
  for r in pool.imap_unordered(case,work):results.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('entry','controller_fingerprint')}),flush=True)
 (out/'summary.json').write_text(json.dumps(results,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
