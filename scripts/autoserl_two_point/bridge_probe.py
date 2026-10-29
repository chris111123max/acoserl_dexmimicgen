"""Real image environment + unchanged ActionScaler; verify recorded executed actions."""
from pathlib import Path
import argparse,json,copy,numpy as np,torch
from acoserl.parallel_autoserl import WorkerFactory,AutoSERLWorker
from acoserl.bimanual_autoserl import snapshot,servo,ARMS
ROOT=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--tag',default='bridge');a=p.parse_args();out=ROOT/'outputs/autoserl_two_point'/f'{a.tag}_{a.arm}';out.mkdir(exist_ok=False)
norm=json.loads((ROOT/'docs/ascend16/normalization.json').read_text());w=AutoSERLWorker(WorkerFactory(0,20261028,norm,str(out),True));M=w.bridge.manager;M.cfg['development_probe']=True;M.cfg['intervention_enabled']=False;rows=[];result={'arm':a.arm,'passed':False}
def step_physical(action):
 normalized=w.scaler.scale(torch.tensor(action,dtype=torch.float32)).numpy();obs,r,t,tr,info=w.step(normalized);x=json.loads(info['payload']);rows.append(x)
 if x['executed']:
  inverse=w.scaler.unscale(torch.tensor(x['executed_action'])).numpy();assert np.max(abs(inverse-np.asarray(x['executed_action_physical'])))<2e-6
  assert obs['observation.state'].shape==(18,) and all(v.shape==(3,84,84) and v.dtype==np.uint8 for k,v in obs.items() if k!='observation.state')
 if t or tr:raise RuntimeError(x.get('supervisor_stop_reason') or 'episode_boundary')
 return x
try:
 w.reset();prefix=M.points['arms'][a.arm]['p0'] if a.arm=='A' else 58
 for action in M.trace['actions'][:prefix]:step_physical(action)
 state=snapshot(w.w.env);targets=copy.deepcopy(state['poses']);grips=[1. if any(state['actual_grasps'][arm].values()) else -1. for arm in ARMS]
 for j in range(100):step_physical(servo(w.w.env,targets,grips,max_position_step=.006,position_gain=2))
 M.reset((prefix,prefix));M.cfg['intervention_enabled']=True
 for j in range(400):
  x=step_physical(servo(w.w.env,targets,grips,max_position_step=.006,position_gain=2))
  if (x.get('recovery_result') or {}).get('handback') and x['auto_intervention']['mode']=='Recover_'+a.arm:
   result['passed']=True;break
 result.update(complete_physics_steps=w.stats['real_steps'],events=M.events,final_grasps=snapshot(w.w.env)['actual_grasps'],max_normalized_action=max(np.max(np.abs(x['executed_action'])) for x in rows if x['executed']))
except BaseException:
 import traceback
 result['error']=traceback.format_exc()
finally:
 result['rows']=rows;(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');w.close()
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
