"""Controlled real-motion approach to a table: independent physical protection."""
import sys,copy,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,servo,save,SafetyStop
from acoserl.bimanual_autoserl import BimanualAutoSERL
p=argparse.ArgumentParser();p.add_argument('--label',default='safety_real_contact_fixed');a=p.parse_args()
E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False);base=snapshot(E);goals=copy.deepcopy(base['poses']);goals[0]['position'][2]-=.30;rows=[];result={'passed':False}
R.safety=lambda c:'force_limit_80N' if c['max_robot_contact_force_N']>80 else 'coupled_contact' if c['dynamic_A_B_coupled'] else None
try:
 for i in range(200):
  proposed=servo(E,goals,[-1,-1],max_position_step=.006,position_gain=2);action,decision=M.choose(proposed)
  if action is None:result.update(passed=True,reason=M.stop_reason,refused_before_step=True,steps=len(rows),contacts=M.contacts.read());break
  try:r=R.step(action)
  except SafetyStop as e:result.update(passed=True,reason=str(e),partial_step=True,steps=len(rows),contacts=M.contacts.read());break
  after=M.after_step();rows.append({'action':action,'snapshot':r,'after':after})
 result['physical_writes']=R.physical_writes_attempted
finally:save(ROOT/'outputs/autoserl_two_point'/(a.label+'.json'),{**result,'rows':rows});E.close()
print({k:v for k,v in result.items() if k!='contacts'},flush=True)
