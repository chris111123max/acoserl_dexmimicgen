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
