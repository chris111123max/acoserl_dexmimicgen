"""Bounded real physics cases. All prefixes, perturbations and recovery use actions."""
import sys,copy,json,argparse,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,servo,save,errors,objects,ARMS,SafetyStop,Contacts
from acoserl.bimanual_autoserl import BimanualAutoSERL,DEFAULT
p=argparse.ArgumentParser();p.add_argument('--case',choices=['A','B','free'],required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--label',default='probe');p.add_argument('--prefix',type=int);p.add_argument('--offset-mm',type=float,default=0.);a=p.parse_args()
np.random.seed(a.seed);env,demo,runner=setup();records=[];result={'case':a.case,'seed':a.seed,'label':a.label,'status':'running'}
try:
 manager=BimanualAutoSERL(env,{**DEFAULT,'development_probe':True});result['points']=manager.points;result['offset_mm']=a.offset_mm
 runner.safety=lambda c:'robot_force_limit' if c['max_robot_contact_force_N']>80 else 'dynamic_coupling' if c['dynamic_A_B_coupled'] else None
 if a.case=='free':prefix=0
 else:prefix=manager.points['arms'][a.case]['p0'] if a.prefix is None else a.prefix
 for j in range(prefix):runner.step(demo['actions'][j])
 result['setup_prefix']=prefix;result['preparation_steps']=100 if a.case!='free' else 0
 initial=snapshot(env)
 if a.case!='free':
  import mujoco
  ai=ARMS.index(a.case);obj=manager.points['arms'][a.case]['object'];m=env.sim.model
  gaps=[float(mujoco.mj_geomDistance(m._model,env.sim.data._data,m.geom_name2id(g),m.geom_name2id(o),1.,None)) for g in env.robots[ai].gripper['right'].contact_geoms for o in getattr(env,obj).contact_geoms]
  result['p0_measured_min_geom_clearance_m']=min(gaps)
 targets=copy.deepcopy(initial['poses']);grips=[1. if any(initial['actual_grasps'][arm].values()) else -1. for arm in ARMS]
 # Preparatory quasistatic hold is real motion, and precedes the recovery baseline.
 if a.case!='free':
  ai=ARMS.index(a.case);t=manager.positions[min(prefix+1,manager.n_states-1),ai]-manager.positions[prefix,ai];targets[ai]['position']+=a.offset_mm*.001*t/max(np.linalg.norm(t),1e-9)
  for j in range(100):runner.step(servo(env,targets,grips,max_position_step=.006,position_gain=2))
  result['preparation_initial']=initial;result['preparation_final']=snapshot(env)
 manager.reset((prefix,prefix));baseline=snapshot(env);result['recovery_baseline']=baseline
 passive_max=0.;steps=0;handbacks=[]
 for j in range(450 if a.case!='free' else 160):
  policy=servo(env,targets,grips,max_position_step=.006,position_gain=2)
  final,decision=manager.choose(policy)
  if final is None:
   result['stop_reason']=manager.stop_reason;break
  r=runner.step(final);after=manager.after_step();steps+=1
  records.append({'control_step':j,'action':final,'policy_action':policy,'decision':copy.deepcopy(decision),'after':after,'snapshot':r})
  if a.case!='free' and decision['mode']=='Recover_'+a.case:passive_max=max(passive_max,after.get('passive_error',{}).get('position_m',0.))
  if after.get('handback'):
   handbacks.append(manager.events[-1]);print('HANDBACK',handbacks[-1],flush=True)
   if a.case!='free' and manager.events[-1]['arm']==a.case and manager.events[-1]['kind']=='safety_recovery':break
  if manager.mode=='Safe_Stop':result['stop_reason']=manager.stop_reason;break
 if handbacks and a.case!='free' and any(h['arm']==a.case and h['kind']=='safety_recovery' for h in handbacks):
  held_state=snapshot(env);hold=copy.deepcopy(held_state['poses']);hg=[1. if any(held_state['actual_grasps'][arm].values()) else -1. for arm in ARMS];result['handback_policy_steps']=0
  for k in range(5):
   policy=servo(env,hold,hg,max_position_step=.006,position_gain=2);final,decision=manager.choose(policy)
   if final is None:result['stop_reason']=manager.stop_reason;break
   runner.step(final);after=manager.after_step();steps+=1;result['handback_policy_steps']+=1
 result.update(status='completed',actual_control_steps=steps,physical_total_control_steps=prefix+(100 if a.case!='free' else 0)+steps,handbacks=handbacks,events=manager.events,passive_max_position_drift_m=passive_max,final=snapshot(env),direct_physical_writes=runner.physical_writes_attempted)
 if a.case!='free':result['passed']=any(x['arm']==a.case and x['kind']=='safety_recovery' for x in handbacks) and result['final']['actual_grasps'][a.case][manager.points['arms'][a.case]['object']] and passive_max<=.002 and not result.get('stop_reason')
 else:result['passed']=steps>=20 and result.get('stop_reason')!='no_validated_independent_phase'
except BaseException as error:
 import traceback
 result.update(status='error',error=traceback.format_exc());print(result['error'],flush=True)
finally:
 result['records']=records;save(ROOT/'outputs/autoserl_two_point'/f'{a.label}_{a.case}_s{a.seed}.json',result);env.close()
print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ['records','final','recovery_baseline','preparation_initial','preparation_final']}),flush=True)
