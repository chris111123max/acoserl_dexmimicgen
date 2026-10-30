"""Durable learner/replay checkpoints; physics resumes only at NEW episodes."""
import os,random,json,time
from pathlib import Path
import numpy as np
import torch
from resfit.rl_finetuning.utils.checkpoint import save_checkpoint

def rng_state():return {'torch':torch.get_rng_state(),'npu':torch.npu.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}
def restore_rng(r):
 torch.set_rng_state(r['torch']);torch.npu.set_rng_state_all(r['npu']);np.random.set_state(r['numpy']);random.setstate(r['python'])

def save(directory,agent,on,off,assembler,pool,counters,config,normalization,dynamic=None):
 directory=Path(directory);assert not directory.exists();stage=directory.with_name(directory.name+'.partial');stage.mkdir(parents=True,exist_ok=False)
 torch.npu.synchronize()
 save_checkpoint(agent,stage/'model.pt',counters['actual_interactions'],config=config,normalization=normalization)
 on.dumps(stage/'online_buffer');off.dumps(stage/'offline_buffer')
 if dynamic is not None:dynamic.save(stage/'dynamic_demo')
 workers=pool.checkpoint_states();rng=rng_state()
 torch.save({'counters':counters.copy(),'assembler':assembler,'rng':rng,'workers':workers,'config':config,'normalization':normalization},stage/'collector.pt')
 (stage/'metadata.json').write_text(json.dumps({'complete':True,'time':time.time(),'counters':counters,'online_rows':len(on),'offline_rows':len(off),
  'dynamic_rows':len(dynamic.buffer) if dynamic is not None else 0,'pending_real_rows':assembler.pending(),'resume_boundary':'pending real tails truncated; reset all workers into NEW episodes; no mid-physics restoration'},indent=2)+'\n')
 os.replace(stage,directory);return directory

def load(directory,agent,on,off,normalization,dynamic=None):
 directory=Path(directory);assert json.loads((directory/'metadata.json').read_text())['complete']
 model=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
 state=torch.load(directory/'collector.pt',map_location='cpu',weights_only=False)
 assert state['normalization']==normalization
 agent.load_state_dict(model['agent_state_dict'])
 for name,value in model['optimizer_state_dict'].items():getattr(agent,name).load_state_dict(value)
 for name,value in model.get('scheduler_state_dict',{}).items():getattr(agent,name).load_state_dict(value)
 on.loads(directory/'online_buffer');off.loads(directory/'offline_buffer')
 if dynamic is not None:
  assert (directory/'dynamic_demo').is_dir(),'checkpoint has no dynamic intervention state'
  dynamic.load(directory/'dynamic_demo',discard_pending=True)
 for key,value in agent.state_dict().items():assert torch.equal(value.detach().cpu(),model['agent_state_dict'][key]),key
 assembler=state['assembler']
 for env in list(assembler.queues):
  for row in assembler.truncate(env):on.add(row)
 return state,assembler

def tree_digest(value):
 import hashlib
 from collections.abc import Mapping
 h=hashlib.sha256()
 def visit(x):
  if isinstance(x,torch.Tensor):
   a=x.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
  elif isinstance(x,Mapping) or hasattr(x,'items'):
   for k,v in sorted(x.items(),key=lambda kv:str(kv[0])):h.update(repr(k).encode());visit(v)
  elif isinstance(x,(list,tuple)):
   for v in x:visit(v)
  else:h.update(repr(x).encode())
 visit(value);return h.hexdigest()

def verify_roundtrip(directory,agent,on,off,normalization,dynamic):
 model=torch.load(Path(directory)/'model.pt',map_location='cpu',weights_only=False)
 names=list(model['optimizer_state_dict'])
 before=dict(parameters=tree_digest(agent.state_dict()),optimizers=tree_digest({n:getattr(agent,n).state_dict() for n in names}),online=tree_digest(on[:]),offline=tree_digest(off[:]),dynamic=tree_digest(dynamic.buffer[:]) if dynamic is not None and len(dynamic.buffer) else None)
 sample_before=torch.cat([on.sample(128).clone(),dynamic.sample_demo(off,128) if dynamic is not None else off.sample(128).clone()],0)
 state,assembler=load(directory,agent,on,off,normalization,dynamic)
 restore_rng(state['rng'])
 after=dict(parameters=tree_digest(agent.state_dict()),optimizers=tree_digest({n:getattr(agent,n).state_dict() for n in names}),online=tree_digest(on[:]),offline=tree_digest(off[:]),dynamic=tree_digest(dynamic.buffer[:]) if dynamic is not None and len(dynamic.buffer) else None)
 assert before==after,(before,after)
 sample_after=torch.cat([on.sample(128).clone(),dynamic.sample_demo(off,128) if dynamic is not None else off.sample(128).clone()],0)
 assert tree_digest(sample_before)==tree_digest(sample_after),'restored sampling mismatch'
 # Return to the exact checkpoint sampling/RNG counters before real continuation.
 state,assembler=load(directory,agent,on,off,normalization,dynamic);restore_rng(state['rng'])
 return state,assembler,dict(parameters_exact=True,optimizers_exact=True,replays_exact=True,sampling_exact=True,rng_restored=True,pending=assembler.pending())
