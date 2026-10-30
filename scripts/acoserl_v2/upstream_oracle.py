"""Specific autonomy/capability and upstream return-oracle checks."""
import json,sys,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/autoserl_two_point'))
from test_nstep import row
from resfit.rl_finetuning.utils.rb_transforms import _multi_step_func
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.intervention_manager import InterventionManagerMixin,MANAGER_DEFAULT
seq=[row(0,0,t,8,t==7,False) for t in range(8)]
reference=_multi_step_func(torch.stack(seq),done_key='done',done_keys=['done','terminated','truncated'],reward_keys=['reward'],mask_key=None,n_steps=3,gamma=.99)
a=IsolatedNStep();ours=[]
for r in seq:ours.extend(a.append(r))
for t,o in enumerate(ours):
 for k in [('next','reward'),('next','obs','marker'),'gamma','nonterminal']:torch.testing.assert_close(o[k],reference[t][k])

result={'passed':True,'upstream_n3_rows_exact':len(ours)}
(ROOT/'outputs/acoserl_v2/upstream_n3.json').write_text(json.dumps(result,indent=2))
print(result)
