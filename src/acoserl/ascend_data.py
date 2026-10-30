"""Single-demo 224-row strict protocol, with preassembled n=3 returns."""
import json
from pathlib import Path
import torch
from torchrl.data import ReplayBuffer,LazyTensorStorage,RandomSampler
from acoserl.single_demo_rlpd import demo_arrays,scalers,normalize,transition,IMAGES,ROOT
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.transition_semantics import boundary,STAGES,source_code

def replay(capacity,batch_size):
 return ReplayBuffer(storage=LazyTensorStorage(capacity,device='cpu'),sampler=RandomSampler(),batch_size=batch_size,pin_memory=False)
def raw_row(obs,action,nxt,reward,term,trunc,env,episode,frame,source=1,metadata=None):
 r=transition(obs,torch.as_tensor(action),nxt,float(reward),bool(term),bool(trunc),source,frame,episode)[0]
 r['env_id']=torch.tensor(env,dtype=torch.long)
 m=metadata or {}
 mode=m.get('auto_intervention',{}).get('mode','Normal')
 progress=m.get('auto_intervention',{}).get('indices',[0,0])
 stage=m.get('stage','unknown')
 kind,boot=boundary(term,trunc,bool(m.get('success',term)),m.get('supervisor_stop_reason'))
 for key,val in dict(action_source=6 if source==0 else (12 if m.get('action_source')=='DiagnosticPreparation' else source_code(mode)),stage_code=STAGES.index(stage) if stage in STAGES else 0,takeover_id=m.get('takeover_id',0)).items():
  r[key]=torch.tensor(val,dtype=torch.long)
 r['reference_progress']=torch.tensor(progress,dtype=torch.long)
 # Three distinct normalized actions share the executed action's coordinate domain.
 # Missing quality evidence fails closed for dynamic expert admission.
 policy=m.get('policy_action',action)
 proposal=m.get('intervention_action_normalized',action)
 r['policy_action_proposed']=torch.as_tensor(policy,dtype=torch.float32).clone()
 r['intervention_action_proposed']=torch.as_tensor(proposal,dtype=torch.float32).clone()
 quality=m.get('task_quality',{})
 r['quality_validated']=torch.tensor(source==0 or bool(quality.get('eligible',False)))
 r['quality_error_before']=torch.tensor(float(quality.get('error_before_m',0.)),dtype=torch.float32)
 r['quality_error_after']=torch.tensor(float(quality.get('error_after_m',0.)),dtype=torch.float32)
 r['quality_handoff_wait']=torch.tensor(bool(quality.get('handoff_wait',False)))
 r['quality_handoff_confirmed']=torch.tensor(bool(quality.get('handoff_confirmed',False)))
 from acoserl.action_provenance import describe
 roles=describe(int(r['action_source']),r['policy_action_proposed'].numpy(),r['action'].numpy(),
                m.get('auto_intervention',{}).get('active_arm'),bool(r['quality_validated']))
 for key,value in roles.items():r[key]=torch.tensor(value,dtype=torch.long if key.endswith('sources') else torch.bool)
 r['complete_physics']=torch.tensor(True)
 r['next','bootstrap_allowed']=torch.tensor(bool(m.get('bootstrap_allowed',boot)))
 r['next','boundary_kind']=torch.tensor(int(m.get('boundary_kind',kind)))
 return r

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
