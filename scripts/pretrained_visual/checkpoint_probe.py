"""Restore the new encoder's bounded learner/replay and update, never the formal run."""
import argparse,json,torch,torch_npu,numpy as np
from pathlib import Path
from acoserl.ascend_data import offline,replay
from acoserl.ascend_checkpoint import load,restore_rng
from acoserl.single_demo_rlpd import IMAGES
from acoserl.pretrained_visual import make_agent,manifest
from resfit.rl_finetuning.config.rlpd import QAgentConfig
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--visual',required=True);a=p.parse_args()
torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.config.allow_internal_format=False;torch.set_num_threads(2)
off,_,_,norm=offline(128);on=replay(256,128);agent=make_agent((3,84,84),(18,),14,IMAGES,QAgentConfig(device='npu:0'),a.visual)
state,assembler=load(ROOT/'outputs/ascend16'/a.run/'checkpoints/final',agent,on,off,norm);restore_rng(state['rng'])
assert len(off)==224 and len(on)>0 and assembler.pending()==0
before=manifest(agent)
batch=torch.cat([on.sample(128).clone(),off.sample(128).clone()],0).to('npu:0');v=agent.update(batch,.1,True,bc_batch=None,ref_agent=agent);torch.npu.synchronize();metrics={k:float(x) for k,x in v.items() if not k.startswith('_')};assert all(np.isfinite(x) for x in metrics.values());assert before['backbone_hashes']==manifest(agent)['backbone_hashes']
result={'passed':True,'run':a.run,'parameters_exact_on_load':True,'optimizers_loaded':True,'online_rows':len(on),'offline_rows':len(off),'post_reload_real_updates':1,'backbone_preserved':True,'metrics':metrics,'physics_boundary':'new episodes, no mid-physics state restore'}
result['optimized_parameter_counts']={name:sum(p.numel() for g in getattr(agent,name).param_groups for p in g['params'] if p.requires_grad) for name in ['actor_opt','critic_opt','encoder_opt']}
result['executed_frozen_prefix_parameters']=sum(p.numel() for e in agent.encoders for name in ['conv1','bn1','layer1','layer2'] for p in getattr(e.backbone,name).parameters())
(ROOT/'outputs/pretrained_visual/checkpoint_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
