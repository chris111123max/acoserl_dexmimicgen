"""P0 data, sampling, bootstrap, restart and shared-policy evidence contracts."""
import json,sys,tempfile
from pathlib import Path
import torch
from acoserl.ascend_data import offline,raw_row
from acoserl.dynamic_demo import DynamicDemo
from acoserl.global_capability import GlobalCapability
from acoserl.intervention_manager import method_config
ROOT=Path(__file__).resolve().parents[2]
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v3_data_contracts');args=parser.parse_args()
assert args.tag.replace('_','').isalnum()
out=ROOT/'outputs/acoserl_v3'/args.tag;out.mkdir(exist_ok=False)
original,*_=offline(128)
def row(t,episode=0,takeover=1,done=False,source="Recover_A"):
 r=original[0].clone()
 for key in ("gamma","steps_to_next_obs","nonterminal"):r.pop(key)
 r["frame"].fill_(t);r["episode"].fill_(episode);r["env_id"].fill_(0)
 r["quality_validated"].fill_(True);r["quality_error_before"].fill_(.01);r["quality_error_after"].fill_(.009)
 from acoserl.transition_semantics import source_code
 r["action_source"].fill_(source_code(source));r["takeover_id"].fill_(takeover)
 from acoserl.action_provenance import describe
 roles=describe(source_code(source),r["action"].numpy(),r["action"].numpy(),quality=True)
 for k,v in roles.items():r[k]=torch.tensor(v,dtype=torch.long if k.endswith("sources") else torch.bool)
 r["next","reward"].fill_(float(done));r["next","done"].fill_(done)
 r["next","terminated"].fill_(done);r["next","truncated"].fill_(False)
 r["next","bootstrap_allowed"].fill_(not done)
 return r
results=[]
for mode in ("official","safe_handback"):
 d=DynamicDemo(capacity=224,admission_mode=mode)
 for t in range(6):d.observe(row(t),safe=t!=5)
 d.close_episode(0,False,4)
 assert len(d.buffer)==(5 if mode=="official" else 0)
 if mode=="official":assert bool(d.buffer[-1]["nonterminal"]), "excluded unsafe row is a discontinuity; do not invent its terminal next state"
 d=DynamicDemo(capacity=224,admission_mode=mode)
 for t in range(3):d.observe(row(t),handback=t==2)
 assert len(d.buffer)==3 and bool(d.buffer[-1]["next","truncated"]) and bool(d.buffer[-1]["nonterminal"])
 d.observe(row(2),handback=True);assert len(d.buffer)==3
 d.observe(row(3,source="Normal"))
 for t in range(8,10):d.observe(row(t,takeover=2),handback=t==9)
 assert len(d.buffer)==5
 for r in d.buffer[:]:assert int(r["steps_to_next_obs"])<=3
 checkpoint=out/mode;d.save(checkpoint)
 restored=DynamicDemo(capacity=224,admission_mode=mode);restored.load(checkpoint)
 for key in d.buffer[:].keys(True,True):assert torch.equal(d.buffer[:][key],restored.buffer[:][key])
 a=d.sample_demo(original,128);b=restored.sample_demo(original,128)
 for key in a.keys(True,True):assert torch.equal(a[key],b[key]),key
 for quota in (64,96,128):
  for positive in (0,2,4):
   bank=DynamicDemo(capacity=224,admission_mode=mode,original_quota=quota,positive_quota=positive)
   for t in range(3):bank.observe(row(t),handback=t==2)
   batch=bank.sample_demo(original,128)
   assert int((batch["source"]==0).sum())==quota
   assert int((batch["source"]==2).sum())==128-quota
   assert int((batch["next","reward"]>0).sum())>=positive
   results.append(dict(mode=mode,quota=quota,positive_min=positive,actual=bank.last_sample))
 # Mid-intervention checkpoint: preserve complete official rows exactly once,
 # reject unfinished safe-handback rows at the new physics episode.
 pending=DynamicDemo(capacity=224,admission_mode=mode)
 for t in range(2):pending.observe(row(t))
 pending.save(out/(mode+"_pending"))
 loaded=DynamicDemo(capacity=224,admission_mode=mode);loaded.load(out/(mode+"_pending"))
 assert len(loaded.buffer)==(2 if mode=="official" else 0) and not loaded.pending
# Contract-only constructed rows: a projection breaks a task correction segment.
for mode in ("official","safe_handback"):
 d=DynamicDemo(capacity=224,admission_mode=mode)
 for t in range(2):d.observe(row(t))
 d.observe(row(2,source="Constrained_A"),handback=True)
 for t in (3,4):d.observe(row(t,takeover=2),handback=t==4)
 d.close_episode(0,True,6)
 assert len(d.buffer)==(4 if mode=="official" else 2)
 assert 2 not in d.buffer[:]["frame"].tolist()
 assert d.stats["safety_projection_rows"]==d.stats["safety_projection_excluded_rows"]==1
 for r in d.buffer[:]:assert int(r["steps_to_next_obs"])<=2

# An actual first Actor row can confirm the preceding recovery without itself
# becoming expert. Stabilization is observed, omitted and never bridges n-step.
for mode in ("official","safe_handback"):
 d=DynamicDemo(capacity=224,admission_mode=mode)
 for t in (0,1):d.observe(row(t))
 hold=row(2);hold["quality_validated"].fill_(False);hold["quality_handoff_wait"].fill_(True)
 d.observe(hold)
 d.observe(row(3))
 policy=row(4,source="Normal");policy["quality_handoff_confirmed"].fill_(True);d.observe(policy,handback=True)
 assert len(d.buffer)==3 and set(d.buffer[:]["frame"].tolist())=={0,1,3}
 assert max(d.buffer[:]["steps_to_next_obs"].tolist())<=2
 assert d.stats["quality_wait_rows"]==1
 # Qualified guarded handoff can confirm prior task rows while excluding the
 # projected confirming action itself. Pure projection cannot fake handoff.
 for confirmed in (False,True):
  x=DynamicDemo(capacity=224,admission_mode="safe_handback")
  x.observe(row(0));x.observe(row(1))
  shield=row(2,source="Constrained_B");shield["quality_handoff_confirmed"].fill_(confirmed)
  x.observe(shield,handback=True)
  assert len(x.buffer)==(2 if confirmed else 0)
  assert not len(x.buffer) or not bool((x.buffer[:]["action_source"]==10).any())
 # A claimed eligible row without measured error reduction must be rejected.
 bad=row(5,takeover=2);bad["quality_error_after"].fill_(.011)
 d.observe(bad,handback=True)
 assert len(d.buffer)==3
 # A missing physical row cannot be treated as qualified stabilization.
 pending=DynamicDemo(capacity=224,admission_mode="safe_handback")
 pending.observe(row(0));pending.observe(row(2,source="Normal"),handback=True)
 assert len(pending.buffer)==0

g=GlobalCapability()
for e in range(5):
 payload=dict(env_id=e,episode_id=0,episode_outcome=dict(autonomous_success=True,
              stage_events={"A:approach_pregrasp":dict(stage_policy_completed=True,stage_stably_completed=True)}))
 g.observe(payload);g.observe(payload)
 assert g.state["episodes"]==e+1
 assert g.state["disable"]==(e==4)
for method in ("plain_rlpd","independent_window","window_recovery","full_acoserl"):
 cfg=method_config(method)
 assert cfg["dynamic_demo_enabled"]==(method=="full_acoserl")
result=dict(passed=True,admission_modes=2,sampling_combinations=results,original_rows=len(original),
            original_positive_rows=int((original[:]["next","reward"]>0).sum()),
            checkpoint_sampling_exact=True,no_intervention_gap_crossing=True,shared_policy_capability=True,safety_projection_excluded=True,unsafe_task_rows_excluded=True,contract_data_not_new_physics=True)
(out/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
