"""CPU-only audit of actual Adam states; distinguish targets from optimized weights."""
import json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'outputs/ascend16/pretrained_spatial84_validation_20261028_1/checkpoints/final/model.pt'
s=torch.load(p,map_location='cpu',weights_only=False)
counts={n:sum(v['exp_avg'].numel() for v in opt['state'].values() if 'exp_avg' in v) for n,opt in s['optimizer_state_dict'].items()}
parameters=s['agent_state_dict'];prefix=sum(v.numel() for k,v in parameters.items() if k.startswith('encoders.') and '.backbone.' in k and k.split('.backbone.')[1].split('.')[0] in ['conv1','bn1','layer1','layer2'] and not any(x in k for x in ['running_mean','running_var','num_batches_tracked']))
r={'optimized_parameters_with_real_Adam_state':counts,'total_optimized':sum(counts.values()),'executed_frozen_prefix_parameters':prefix,'note':'Official actor_target/critic_target retain their requires_grad flags but are not optimizer groups; updated by official target averaging. No flags/algorithm changed.'}
(ROOT/'docs/pretrained_visual/parameter_counts.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
