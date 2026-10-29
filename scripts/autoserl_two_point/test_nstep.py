"""Adversarial interleaved streams + independent serial and upstream oracles."""
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from acoserl.isolated_nstep import IsolatedNStep
from resfit.rl_finetuning.utils.rb_transforms import _multi_step_func

def row(e,ep,t,length,term=False,trunc=False):
 obs=torch.tensor([e,ep,t],dtype=torch.float32)
 return TensorDict({'env_id':torch.tensor(e),'episode':torch.tensor(ep),'frame':torch.tensor(t),'action':torch.tensor([float(t)]),
  'obs':TensorDict({'marker':obs},[]),'next':TensorDict({'obs':TensorDict({'marker':obs+torch.tensor([0,0,1])},[]),
  'reward':torch.tensor(float(e+10*ep+t)),'done':torch.tensor(term or trunc),'terminated':torch.tensor(term),'truncated':torch.tensor(trunc)},[])},[])

def main():
 n=IsolatedNStep();emitted=[];sequences={};boundaries={};rng=np.random.default_rng(89)
 # Includes one/two-frame tails, terminals, timeouts and cuts before next action.
 for ep in range(3):
  lengths={e:1+(e+ep)%7 for e in range(16)}
  for t in range(7):
   for e in rng.permutation(16):
    if t>=lengths[e]:continue
    last=t==lengths[e]-1;kind=(e+ep)%3
    r=row(e,ep,t,lengths[e],last and kind==0,last and kind==1)
    sequences.setdefault((e,ep),[]).append(r.clone());emitted.extend(n.append(r))
    if last and kind==2:
     sequences[e,ep][-1]['next','done'].fill_(True);sequences[e,ep][-1]['next','truncated'].fill_(True)
     emitted.extend(n.truncate(e))
 assert n.pending()==0
 seen=set();compared=0
 for out in emitted:
  e,ep,t=[int(out[k]) for k in ['env_id','episode','frame']];seq=sequences[e,ep];m=min(3,len(seq)-t);last=seq[t+m-1]
  assert (e,ep,t) not in seen;seen.add((e,ep,t))
  expected=sum(.99**j*float(seq[t+j]['next','reward']) for j in range(m))
  assert abs(float(out['next','reward'])-expected)<2e-5
  assert torch.equal(out['next','obs','marker'],last['next','obs','marker'])
  assert int(out['steps_to_next_obs'])==m and abs(float(out['gamma'])-.99**m)<1e-7
  assert bool(out['nonterminal'])==(not bool(last['next','terminated']))
  for k in ['done','terminated','truncated']:assert torch.equal(out['next',k],last['next',k])
  compared+=1
 assert compared==sum(len(s) for s in sequences.values())
 # Official single-stream 3-step reward/next observation/gamma/terminal oracle.
 seq=[row(0,0,t,8,t==7,False) for t in range(8)]
 official=_multi_step_func(torch.stack(seq),done_key='done',done_keys=['done','terminated','truncated'],reward_keys=['reward'],mask_key=None,n_steps=3,gamma=.99)
 assembly=IsolatedNStep();ours=[]
 for r in seq:ours+=assembly.append(r)
 for t,o in enumerate(ours):
  for key in [('next','reward'),('next','obs','marker'),'gamma','nonterminal']:
   torch.testing.assert_close(o[key],official[t][key])
 # Existing accepted n_step=1 uses terminated as bootstrap boundary.
 seq=[row(0,0,t,2,False,t==1) for t in range(2)]
 official=_multi_step_func(torch.stack(seq),done_key='terminated',done_keys=['done','terminated','truncated'],reward_keys=['reward'],mask_key=None,n_steps=1,gamma=.99)
 assembly=IsolatedNStep(1);ours=[]
 for r in seq:ours+=assembly.append(r)
 for t,o in enumerate(ours):
  for key in [('next','reward'),('next','obs','marker'),'gamma','nonterminal']:torch.testing.assert_close(o[key],official[t][key])
 result={'passed':True,'workers':16,'episodes':48,'real_rows':compared,'output_rows':len(seen),'n_step':3,
  'interleaving_and_episode_boundaries_verified':True,'terminated_disables_bootstrap':True,'truncated_and_safety_cut_bootstrap':True,
  'short_tails_flushed':True,'no_fake_safety_transition':True,'single_stream_official_n3_oracle':True,'accepted_n1_timeout_oracle':True}
 Path('outputs/autoserl_two_point/nstep_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
