"""Reach asynchronous pick states by real actions, including a held passive object."""
import sys,copy,json,argparse,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,servo,snapshot,save,ARMS
from acoserl.bimanual_autoserl import BimanualAutoSERL
p=argparse.ArgumentParser();p.add_argument('--active',choices=ARMS,required=True);p.add_argument('--passive-held',action='store_true');a=p.parse_args();E,D,R=setup();M=BimanualAutoSERL(E);ai=ARMS.index(a.active);pi=1-ai;records=[];total=0;result={'active':a.active,'passive_held':a.passive_held,'passed':False}
R.safety=lambda c:'force' if c['max_robot_contact_force_N']>80 else 'coupled' if c['dynamic_A_B_coupled'] else None
try:
 initial=snapshot(E);targets=copy.deepcopy(initial['poses']);grips=[-1.,-1.]
 if a.passive_held:
  for t in range(M.points['arms'][ARMS[pi]]['p1']):
   action=servo(E,targets,grips,max_position_step=.006,position_gain=2);action[pi*7:pi*7+7]=D['actions'][t,pi*7:pi*7+7];R.step(action);total+=1
  state=snapshot(E);assert state['actual_grasps'][ARMS[pi]][M.points['arms'][ARMS[pi]]['object']]
  targets[pi]=copy.deepcopy(state['poses'][pi]);grips[pi]=1.
 targets[ai]=M.pose(ai,M.points['arms'][a.active]['p0'])
 for k in range(120):R.step(servo(E,targets,grips,max_position_step=.006,position_gain=2));total+=1
 progress=[0,0];progress[ai]=M.points['arms'][a.active]['p0'];progress[pi]=M.points['arms'][ARMS[pi]]['p1'] if a.passive_held else 0;M.reset(tuple(progress));result['baseline']=snapshot(E)
 for k in range(450):
  action,decision=M.choose(servo(E,targets,grips,max_position_step=.006,position_gain=2))
  if action is None:result['stop_reason']=M.stop_reason;break
  r=R.step(action);total+=1;after=M.after_step();records.append({'decision':copy.deepcopy(decision),'after':after,'snapshot':r,'action':action})
  if after.get('handback') and decision['mode']=='Recover_'+a.active and M.recovery_kind=='safety_recovery':result['passed']=True;break
  if M.mode=='Safe_Stop':result['stop_reason']=M.stop_reason;break
 result.update(total_actual_steps=total,final=snapshot(E),events=M.events,physical_writes=R.physical_writes_attempted)
except BaseException:
 import traceback
 result['error']=traceback.format_exc()
finally:save(ROOT/'outputs/autoserl_two_point'/f'parked_{a.active}_passiveheld{a.passive_held}.json',{**result,'records':records});E.close()
print({k:v for k,v in result.items() if k not in ['baseline','final']},flush=True)
