"""Directed real demo execution and stage qualification; no training."""
import json,sys,argparse
from pathlib import Path
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.autoserl_env import StrictTaskProtocol
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,save
parser=argparse.ArgumentParser();parser.add_argument('--tag',default='stage_gates_v2');args=parser.parse_args()
E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False)
M.manager_cfg['autonomy_enabled']=False;P=StrictTaskProtocol()
R.safety=lambda c:'force_limit' if c['max_robot_contact_force_N']>80 else 'coupled' if c['dynamic_A_B_coupled'] else None
out=ROOT/'outputs/acoserl_foundation'/args.tag;out.mkdir(exist_ok=False)
result={'passed':False,'checks':[]}
try:
 previous=[0,0]
 for t,proposed in enumerate(D['actions']):
  action,decision=M.choose(proposed)
  assert action is not None,M.stop_reason
  state=R.step(action);after=M.after_step();success=P.observe(E)
  assert all(i>=j for i,j in zip(M.indices(),previous));previous=M.indices()
  if t+1 in (80,170,214,224):
   coop=M.collaboration(state,M.contacts.read())
   gates={a:M.eligible(i,state,M.contacts.read(),'safety_recovery',M.points['arms'][a]['p0']) for i,a in enumerate(['A','B'])}
   result['checks'].append({'state':t+1,'collaboration':coop,'eligible':gates,'completed':M.completed.copy(),'indices':M.indices(),'metrics':state['official_metrics']})
   if t+1==80:assert not any(x[0] for x in gates.values())
   assert not M.advanced_qualification('carry_local')['allowed']
   assert not M.advanced_qualification('joint')['allowed']
  assert M.mode!='Safe_Stop',M.stop_reason
  if success['success']:break
 result.update(passed=success['success'],actual_control_steps=t+1,physical_writes=R.physical_writes_attempted,strict_success=success['success'],final_progress=M.indices())
except BaseException:
 import traceback
 result['error']=traceback.format_exc()
finally:
 save(out/'result.json',result);E.close()
print(json.dumps(result),flush=True)
