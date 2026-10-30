"""Canonical0 real-action construction of independent simultaneous workspace risks."""
import sys,copy,json,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,servo,save
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.intervention_manager import method_config
E,D,R=setup();m=BimanualAutoSERL(E,intervention_enabled=False);m.manager_cfg.update({**method_config('full_acoserl'),'v3_enabled':True,'baseline_mode':True});R.safety=m.substep_safety
out=ROOT/'outputs/autoserl_intervention_fix/independent_workspace_v1';out.mkdir(exist_ok=False);rows=[];result={'passed':False,'physical_writes':0}
try:
 start=snapshot(E);goals=copy.deepcopy(start['poses'])
 for ai in range(2):goals[ai]['position'][2]=float(m.positions[:,ai,2].max()+m.cfg['free_workspace_margin_m']-.009)
 for t in range(200):
  action=servo(E,goals,[-1.,-1.],max_position_step=.004,position_gain=1.)
  final,decision=m.choose(action)
  if final is None:raise RuntimeError('preparation_refused:'+str(m.stop_reason))
  physical=R.step(final);after=m.after_step();state=snapshot(E)
  margins=[float(m.positions[:,i,2].max()+m.cfg['free_workspace_margin_m']-state['poses'][i]['position'][2]) for i in range(2)]
  rows.append(dict(action=final,state=state,margins=margins,contact=physical['contacts'],force=physical['substep_max_external_force_N']))
  if all(.001<x<.010 for x in margins) and t>10:break
 else:raise RuntimeError('could_not_reach_joint_workspace_screen_without_physical_shortcut')
 state=snapshot(E);policy=np.zeros(14);policy[[0,7]]=.2;policy[[2,9]]=1.
 before=m.action_risks(policy,state);assert all(x['risk'] and 'workspace' in x['categories'] for x in before),before
 corrected,reason=m._project_independent_risks(policy,state,before,[0,1]);result.update(before=state,policy=policy,risks=before,reason=reason,corrected=corrected)
 if corrected is not None:
  after_risks=m.action_risks(corrected,state);assert not any(r['risk'] for r in after_risks)
  assert abs(corrected[0])+abs(corrected[7])>1e-6,'both motions must not be zeroed'
  physical=R.step(corrected);result.update(after=snapshot(E),after_risks=after_risks,physical=physical,passed=True)
except BaseException:result['error']=traceback.format_exc()
finally:
 result['physical_writes']=R.physical_writes_attempted;save(out/'result.json',result);save(out/'prefix.json',rows);E.close()
