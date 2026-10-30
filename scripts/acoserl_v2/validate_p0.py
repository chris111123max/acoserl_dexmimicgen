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
parser=argparse.ArgumentParser();parser.add_argument('--tag',default='p0_contracts');args=parser.parse_args()
assert args.tag.replace('_','').isalnum()
out=ROOT/'outputs/acoserl_v2'/args.tag;out.mkdir(exist_ok=False)
original,*_=offline(128)
def row(t,episode=0,takeover=1,done=False,source="Recover_A"):
 r=original[0].clone()
 for key in ("gamma","steps_to_next_obs","nonterminal"):r.pop(key)
 r["frame"].fill_(t);r["episode"].fill_(episode);r["env_id"].fill_(0)
 from acoserl.transition_semantics import source_code
 r["action_source"].fill_(source_code(source));r["takeover_id"].fill_(takeover)
 r["next","reward"].fill_(float(done));r["next","done"].fill_(done)
 r["next","terminated"].fill_(done);r["next","truncated"].fill_(False)
 r["next","bootstrap_allowed"].fill_(not done)
 return r
results=[]
for mode in ("official","safe_handback"):
 d=DynamicDemo(capacity=224,admission_mode=mode)
 for t in range(6):d.observe(row(t),safe=t!=5)
 d.close_episode(0,False,4)
 assert len(d.buffer)==(6 if mode=="official" else 0)
 if mode=="official":assert not bool(d.buffer[-1]["nonterminal"])
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
            checkpoint_sampling_exact=True,no_intervention_gap_crossing=True,shared_policy_capability=True)
(out/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
