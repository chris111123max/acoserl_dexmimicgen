import sys,json,copy,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,save,snapshot,SafetyStop
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.autoserl_env import StrictTaskProtocol
p=argparse.ArgumentParser();p.add_argument('--label',default='full_demo_fixed');args=p.parse_args()
E,D,R=setup();M=BimanualAutoSERL(E);M.cfg['development_probe']=True;P=StrictTaskProtocol();records=[];result={'status':'running'}
R.safety=lambda c:'force' if c['max_robot_contact_force_N']>80 else 'coupled' if c['dynamic_A_B_coupled'] else None
try:
 for i,a in enumerate(D['actions']):
  final,decision=M.choose(a)
  if final is None:result['stop_reason']=M.stop_reason;break
  r=R.step(final);after=M.after_step();success=P.observe(E)
  records.append({'index':i,'decision':copy.deepcopy(decision),'after':after,'success':success,'snapshot':r,'executed_action':final})
  if i==170:
   result['assembly_barrier']={str(ai):M.eligible(ai,snapshot(E),M.contacts.read(),'safety_recovery',M.points['arms'][arm]['p0']) for ai,arm in enumerate(['A','B'])}
  if M.mode=='Safe_Stop':result['stop_reason']=M.stop_reason;break
 result.update(status='completed',steps=len(records),strict_success=P.streak>=3,final=snapshot(E),events=M.events,physical_writes=R.physical_writes_attempted)
except BaseException:
 import traceback
 result.update(status='error',error=traceback.format_exc())
finally:
 result['records']=records;save(ROOT/'outputs/autoserl_two_point'/(args.label+'.json'),result);E.close()
print({k:v for k,v in result.items() if k not in ['records','final','events']},flush=True)
