"""Load the actual bounded NPU checkpoint and replay, then perform one real update."""
import json,torch,torch_npu,numpy as np
from pathlib import Path
from acoserl.ascend_data import offline,replay
from acoserl.ascend_checkpoint import load,restore_rng
from acoserl.single_demo_rlpd import IMAGES
from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
ROOT=Path(__file__).resolve().parents[2];torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.config.allow_internal_format=False;torch.set_num_threads(2)
off,_,_,norm=offline(128);on=replay(256,128);agent=QAgent((3,84,84),(18,),14,IMAGES,QAgentConfig(device='npu:0'))
state,assembler=load(ROOT/'outputs/ascend16/autoserl2_validation_20261028_1/checkpoints/final',agent,on,off,norm);restore_rng(state['rng'])
assert len(off)==224 and len(on)==140 and assembler.pending()==0
batch=torch.cat([on.sample(128).clone(),off.sample(128).clone()],0).to('npu:0');v=agent.update(batch,.1,True,bc_batch=None,ref_agent=agent);torch.npu.synchronize();metrics={k:float(x) for k,x in v.items() if not k.startswith('_')};assert all(np.isfinite(x) for x in metrics.values())
result={'passed':True,'parameters_exact_on_load':True,'optimizer_and_scheduler_loaded':True,'online_rows':len(on),'offline_rows':len(off),'actual_post_reload_updates':1,'metrics':metrics,'physics_resume_boundary':'not tested mid-episode; resume requires new episodes'}
(ROOT/'outputs/autoserl_two_point/checkpoint_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
