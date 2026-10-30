"""Replay real logged action prefixes from canonical0 and physically probe old risk proposals."""
import json,sys,copy,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,save,SafetyStop
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.intervention_manager import method_config
from resfit.rl_finetuning.utils.normalization import ActionScaler
run=ROOT/'outputs/ascend16/autoserl_baseline_formal_20261030_133346';out=ROOT/'outputs/autoserl_intervention_fix/old_risk_replay_v1';out.mkdir(exist_ok=False)
cases={}
for f in sorted((run/'workers').glob('env[0-9][0-9].jsonl')):
 ep=None;prefix=[]
 for line in f.open():
  d=json.loads(line)
  if d['event']!='step':continue
  if ep!=d['episode_id']:ep=d['episode_id'];prefix=[]
  if d.get('supervisor_stop_reason')=='simultaneous_high_confidence_risks_no_joint_qualification':
   dec=d['auto_intervention'];risks=dec['policy_risks'];key=' | '.join(','.join(r.get('categories',[])) for r in risks)
   if key not in cases:cases[key]=dict(env=d['env_id'],episode=ep,prefix=copy.deepcopy(prefix),policy=dec['policy_action'],old_risks=risks)
  if d.get('executed'):prefix.append(d['executed_action_physical'])
 if len(cases)>=2:break
save(out/'selected.json',cases)
results=[]
for key,c in cases.items():
 E,D,R=setup();m=BimanualAutoSERL(E,intervention_enabled=False);m.manager_cfg.update({**method_config('full_acoserl'),'v3_enabled':True,'baseline_mode':True});R.safety=m.substep_safety
 norm=json.loads((ROOT/'docs/ascend16/normalization.json').read_text());scale=ActionScaler(torch.tensor(norm['effective_action_min']),torch.tensor(norm['effective_action_max']),action_scale=0,min_range_per_dim=.1);m.action_bounds=(scale.limits.min.numpy(),scale.limits.max.numpy())
 result={'old_risk_class':key,'env':c['env'],'episode':c['episode'],'prefix_controls':len(c['prefix']),'physical_writes':0}
 try:
  for a in c['prefix']:
   a=np.asarray(a);final,decision=m.choose(a)
   if final is None:raise RuntimeError('prefix supervisor refused:'+str(m.stop_reason))
   R.step(a);m.after_step()
  state=snapshot(E);policy=np.asarray(c['policy']);risks=m.action_risks(policy,state);active=[i for i,x in enumerate(risks) if x['risk']]
  corrected,reason=m._project_independent_risks(policy,state,risks,active) if active else (policy,'new_predictor_no_risk')
  result.update(before=state,old_risks=c['old_risks'],new_risks=risks,projection_result=reason,policy=policy,corrected=corrected,contact_before=m.contacts.read())
  if corrected is not None:
   again=m.action_risks(corrected,state);assert not any(x['risk'] for x in again);result['rechecked_risks']=again
   result['physical_result']=R.step(corrected);result['after']=snapshot(E);result['completed_action']=True
 except BaseException:result['error']=traceback.format_exc()
 finally:
  result['physical_writes']=R.physical_writes_attempted;save(out/(str(len(results))+'.json'),result);E.close();results.append(result)
save(out/'summary.json',results)
