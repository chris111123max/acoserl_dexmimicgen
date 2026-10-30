"""Bounded real Actor collection (no optimizer updates), admitting the same true rows in both demo modes."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
 out=ROOT/'outputs/acoserl_v2'/a.tag;out.mkdir(exist_ok=False);deadline=time.time()+600
 import torch,torch_npu,gymnasium as gym
 from acoserl.parallel_autoserl import WorkerFactory
 from acoserl.intervention_manager import method_config
 from acoserl.ascend_data import offline,replay,raw_row
 from acoserl.dynamic_demo import DynamicDemo
 from acoserl.isolated_nstep import IsolatedNStep
 from acoserl.pretrained_visual import make_agent
 from acoserl.single_demo_rlpd import IMAGES
 from resfit.rl_finetuning.config.rlpd import QAgentConfig
 from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
 torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.config.allow_internal_format=False;torch.set_num_threads(2)
 torch.manual_seed(20261027);torch.npu.manual_seed_all(20261027);np.random.seed(20261027);random.seed(20261027)
 off,_,_,norm=offline(128);on=replay(256,128);stream=IsolatedNStep(3,.99)
 banks={mode:DynamicDemo(capacity=256,admission_mode=mode,original_quota=96,positive_quota=2) for mode in ('official','safe_handback')}
 agent=make_agent((3,84,84),(18,),14,IMAGES,QAgentConfig(device='npu:0'),'resnet18_spatial84')
 model=torch.load(a.checkpoint/'model.pt',map_location='cpu',weights_only=False);agent.load_state_dict(model['agent_state_dict'])
 vector=gym.vector.AsyncVectorEnv([WorkerFactory(0,20261027,norm,str(out/'worker'),True,manager_config=method_config('full_acoserl'))],context='spawn',autoreset_mode=gym.vector.AutoresetMode.DISABLED)
 result=dict(optimizer_updates=0,real_steps=0,partial_attempts=0,handbacks=0,post_handback_actor_steps=0,post_handback_unmodified_policy_steps=0,checkpoint=str(a.checkpoint),rows_are_real=True,passed=False)
 previous=None
 try:
  obs,_=vector.reset(seed=20261027)
  for _ in range(96):
   if time.time()>deadline:break
   data={k:torch.from_numpy(v).to('npu:0') for k,v in obs.items()}
   with torch.no_grad(),eval_mode(agent):action=agent.act(data,eval_mode=False,stddev=.1,cpu=True).numpy()
   nxt,r,t,tr,info=vector.step(action);meta=json.loads(info['payload'][0]);hand=bool((meta.get('recovery_result') or {}).get('safe_handback'))
   if meta['executed']:
    row=raw_row({k:torch.from_numpy(v[0]) for k,v in obs.items()},meta['executed_action'],{k:torch.from_numpy(v[0]) for k,v in nxt.items()},r[0],t[0],tr[0],0,meta['episode_id'],meta['frame'],metadata=meta)
    for item in stream.append(row):on.add(item)
    for bank in banks.values():bank.observe(row,handback=hand,safe=not bool(meta.get('supervisor_stop_reason')))
    result['real_steps']+=1
    if previous is not None and previous['episode_id']==meta['episode_id'] and (previous.get('recovery_result') or {}).get('safe_handback'):
     result['post_handback_actor_steps']+=1
     result['post_handback_unmodified_policy_steps']+=int(meta['action_source']=='Policy')
   else:
    result['partial_attempts']+=int(bool(meta.get('partial_physics_step')))
    for item in stream.close(0,bootstrap=bool(meta['bootstrap_allowed']),boundary_kind=meta['boundary_kind']):on.add(item)
    for bank in banks.values():bank.close_episode(0,bool(meta['bootstrap_allowed']),meta['boundary_kind'])
   result['handbacks']+=int(hand)
   previous=meta;obs=nxt
   if t[0] or tr[0]:obs,_=vector.reset()
   if result['handbacks'] and result['post_handback_unmodified_policy_steps'] and all(len(b.buffer)>0 for b in banks.values()):break
  for item in stream.truncate(0):on.add(item)
  for name,bank in banks.items():
   bank.close_episode(0,True,6);bank.save(out/name)
   batch=torch.cat([on.sample(128),bank.sample_demo(off,128)],0);assert batch.shape[0]==256
   result[name]=dict(rows=len(bank.buffer),stats=bank.stats,sampling=bank.last_sample)
  on.dumps(out/'online_buffer')
  result['online_rows']=len(on);result['passed']=result['handbacks']>0 and result['post_handback_actor_steps']>0 and all(result[n]['rows']>0 for n in banks)
 finally:
  vector.close();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
