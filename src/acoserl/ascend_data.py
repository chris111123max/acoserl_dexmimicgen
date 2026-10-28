"""Single-demo 224-row strict protocol, with preassembled n=3 returns."""
import json
from pathlib import Path
import torch
from torchrl.data import ReplayBuffer,LazyTensorStorage,RandomSampler
from acoserl.single_demo_rlpd import demo_arrays,scalers,normalize,transition,IMAGES,ROOT
from acoserl.isolated_nstep import IsolatedNStep

def replay(capacity,batch_size):
 return ReplayBuffer(storage=LazyTensorStorage(capacity,device='cpu'),sampler=RandomSampler(),batch_size=batch_size,pin_memory=False)
def raw_row(obs,action,nxt,reward,term,trunc,env,episode,frame,source=1):
 r=transition(obs,torch.as_tensor(action),nxt,float(reward),bool(term),bool(trunc),source,frame,episode)[0]
 r['env_id']=torch.tensor(env,dtype=torch.long);return r

def offline(batch_size,gamma=.99):
 semantics=json.loads((ROOT/'docs/bimanual_autoserl/strict_semantics.json').read_text())
 assert semantics['effective_transitions']==224 and semantics['protocol']=='assembly_real_grippers_stable3_v1'
 std,scale,stats=scalers(224);state,images,actions=demo_arrays();rb=replay(224,batch_size);assembler=IsolatedNStep(3,gamma)
 obs=[normalize({'observation.state':state[t],**{k:v[t] for k,v in images.items()}},std) for t in range(225)]
 for t in range(224):
  r=raw_row(obs[t],scale.scale(torch.from_numpy(actions[t])),obs[t+1],semantics['reward_at_recorded_state'][t+1],t==223,False,-1,0,t,0)
  for out in assembler.append(r):rb.add(out)
 assert assembler.pending()==0 and len(rb)==224
 assert rb[:]['next','terminated'].sum()==3 and (rb[:]['next','reward']>0).sum()==3
 return rb,std,scale,stats
