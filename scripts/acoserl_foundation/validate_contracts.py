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
m=object.__new__(InterventionManagerMixin)
m.manager_cfg=copy.deepcopy(MANAGER_DEFAULT);m.cfg={'fixed_initial_state':True,'termination_intervention_steps':10}
m.episode_phase_steps={('A','approach_pregrasp'):20};m.episode_policy_steps={('A','approach_pregrasp'):20}
m.episode_completed={('A','approach_pregrasp')};m.episode_touched=set()
m.autonomy={};m.clean_full_episodes=0;m.auto_disabled=False;m.termination_log=[]
m.intervention_control_steps=5;m.episode_interventions=1;m.decision_stats={'handbacks':1}
for i in range(8):
 r=m.episode_result(True,full_policy_episode=True)
 assert not m.auto_disabled and not r['autonomous_success']
assert m.stage_mastered('A','approach_pregrasp')
assert not m.stage_mastered('A','independent_carry')
m.manager_cfg['joint']={'enabled':True,'validated':True}
assert not m.advanced_qualification('joint')['allowed']
m.intervention_control_steps=0;m.episode_interventions=0
for i in range(4):
 m.episode_result(True,full_policy_episode=True);assert not m.auto_disabled
m.episode_result(True,full_policy_episode=True);assert m.auto_disabled
result={'passed':True,'upstream_n3_rows_exact':len(ours),'assisted_success_does_not_disable':True,'full_clean_successes_required':5,'advanced_config_cannot_bypass_gate':True,'carry_stage_autonomy_disabled':True}
p=ROOT/'outputs/acoserl_foundation/contract_acceptance.json';p.write_text(json.dumps(result,indent=2));print(result)
