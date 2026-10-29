"""Real CPU/NPU feature parity and unchanged official 256-row RLPD updates."""
import argparse,json,time,traceback,contextlib
from pathlib import Path
import torch,torch_npu,numpy as np
from acoserl.ascend_data import offline
from acoserl.pretrained_visual import make_agent,manifest,digest
from acoserl.single_demo_rlpd import IMAGES
from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();result={'name':a.name,'passed':False}
try:
 torch.set_num_threads(2);torch.manual_seed(20261028);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.config.allow_internal_format=False
 off,_,_,_=offline(128);batch=torch.cat([off.sample(128).clone(),off.sample(128).clone()],0)
 with (ROOT/f'logs/pretrained_visual/{a.name}_model.txt').open('w') as f,contextlib.redirect_stdout(f):agent=make_agent((3,84,84),(18,),14,IMAGES,QAgentConfig(device='cpu'),a.name)
 before=manifest(agent);result['manifest']=before
 image=batch['obs',IMAGES[0]][:2].float()/255.
 with torch.no_grad():cpu=agent.encoders[0](image,flatten=False)
 agent.to('npu:0');agent.train(True)
 with torch.no_grad():npu=agent.encoders[0](image.to('npu:0'),flatten=False).cpu()
 result.update(cpu_npu_max_abs=float((cpu-npu).abs().max()),cpu_npu_relative_l2=float((cpu-npu).norm()/cpu.norm().clamp_min(1e-8)),shape=list(cpu.shape),spatial_variation=float(cpu.var(dim=1).mean()))
 assert torch.isfinite(npu).all() and result['cpu_npu_relative_l2']<.01 and result['spatial_variation']>1e-8
 data=batch.to('npu:0');infer={k:data['obs',k][:16] for k in IMAGES+['observation.state']}
 for _ in range(2):
  with torch.no_grad(),eval_mode(agent):agent.act(infer,eval_mode=True,cpu=False)
 torch.npu.synchronize();t=time.perf_counter()
 for _ in range(5):
  with torch.no_grad(),eval_mode(agent):agent.act(infer,eval_mode=True,cpu=False)
 torch.npu.synchronize();result['inference16_seconds']=(time.perf_counter()-t)/5
 t=time.perf_counter()
 for _ in range(5):
  with torch.no_grad():agent._encode(infer,augment=False)
 torch.npu.synchronize();result['encoding_3x16_seconds']=(time.perf_counter()-t)/5
 snapshots={k:p.detach().cpu().clone() for k,p in agent.named_parameters() if p.requires_grad}
 updates=[]
 for i in range(6):
  t=time.perf_counter();v=agent.update(data,.1,True,bc_batch=None,ref_agent=agent);torch.npu.synchronize();values={k:float(x) for k,x in v.items() if not k.startswith('_')};assert all(np.isfinite(list(values.values())))
  updates.append({'seconds':time.perf_counter()-t,**values})
 result['updates']=updates;result['steady_update_seconds']=sum(x['seconds'] for x in updates[2:])/4
 deltas={k:float((p.detach().cpu()-snapshots[k]).abs().max()) for k,p in agent.named_parameters() if k in snapshots}
 result['changed_parameter_tensors']={group:sum(v>0 for k,v in deltas.items() if k.startswith(group)) for group in ['actor.','critic.','encoders.']}
 assert all(result['changed_parameter_tensors'].values())
 result['adapter_grad_norms']=[float(e.adapter.weight.grad.norm()) for e in agent.encoders if hasattr(e,'adapter')]
 after=manifest(agent);result['backbones_unchanged']=before['backbone_hashes']==after['backbone_hashes'];assert result['backbones_unchanged']
 assert all(not p.requires_grad and p.grad is None for e in agent.encoders if hasattr(e,'backbone') for p in e.backbone.parameters())
 result['backbones_eval']=all(not e.backbone.training for e in agent.encoders if hasattr(e,'backbone'));assert result['backbones_eval']
 result['peak_npu_allocated_bytes']=torch.npu.max_memory_allocated();result['passed']=True
except Exception:result['error']=traceback.format_exc();print(result['error'],flush=True)
(ROOT/f'outputs/pretrained_visual/benchmark_{a.name}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
