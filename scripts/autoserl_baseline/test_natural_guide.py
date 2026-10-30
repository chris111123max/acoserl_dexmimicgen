"""Canonical0 random exploration; no prefix, no mid-episode state setters."""
import sys,json,argparse,copy,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,save
from acoserl.bimanual_autoserl import BimanualAutoSERL,DEFAULT
from acoserl.intervention_manager import method_config
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--seed',type=int,default=1);p.add_argument('--steps',type=int,default=128);p.add_argument('--amplitude',type=float,default=1.0);a=p.parse_args()
out=ROOT/'outputs/autoserl_intervention_fix'/a.tag;out.mkdir(exist_ok=False)
E,D,R=setup();m=BimanualAutoSERL(E,{**DEFAULT,'manager':{**method_config('full_acoserl'),'v3_enabled':True,'baseline_mode':True,'guide_policy_enabled':True,'carry_local':{'enabled':False},'joint':{'enabled':False},'passive_brake':{'enabled':False}}},True);R.safety=m.substep_safety
rng=np.random.default_rng(a.seed);rows=[];error=None
try:
 for i in range(a.steps):
  policy=rng.uniform(-a.amplitude,a.amplitude,14);before=snapshot(E);policy_risks=m.action_risks(policy,before);action,decision=m.choose(policy)
  if action is None:break
  final_risks=m.action_risks(action,before);physics=R.step(action);after=m.after_step();rows.append(dict(i=i,before=before,policy=policy,policy_risks=policy_risks,executed=action,final_risks=final_risks,decision=copy.deepcopy(decision),after=after,state=snapshot(E),contact=physics['contacts'],substep_max_external_force_N=physics['substep_max_external_force_N'],substep_contact_edges=physics['substep_contact_edges']))
  if m.mode=='Safe_Stop':break
except BaseException:error=traceback.format_exc()
finally:
 save(out/'controls.json',rows);save(out/'result.json',dict(seed=a.seed,complete_controls=len(rows),physical_writes=R.physical_writes_attempted,error=error,stop=m.stop_reason,source='canonical0 random actual controller proposals',guide_steps=sum(r['decision']['mode'].startswith('Guide_') for r in rows),handbacks=sum(bool(r['after'].get('safe_handback')) for r in rows)));E.close()
