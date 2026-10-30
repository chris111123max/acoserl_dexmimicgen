"""Audit actual RGB recovery rows, executed-action provenance and expert admission."""
import json,argparse
from pathlib import Path
import numpy as np,torch
from acoserl.ascend_data import offline,replay
from acoserl.dynamic_demo import DynamicDemo
from acoserl.transition_semantics import TASK_INTERVENTION_SOURCES,SAFETY_PROJECTION_SOURCES,source_code
from resfit.rl_finetuning.utils.normalization import ActionScaler
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--tag",required=True);a=p.parse_args()
source=ROOT/"outputs/acoserl_v3"/a.source;out=ROOT/"outputs/acoserl_v3"/a.tag;out.mkdir(exist_ok=False)
raw=torch.load(source/"raw_transitions.pt",weights_only=False,map_location="cpu")
logs={r["control"]:r for r in json.loads((source/"controls.json").read_text())}
norm=json.loads((ROOT/"docs/ascend16/normalization.json").read_text())
scale=ActionScaler(torch.tensor(norm["effective_action_min"]),torch.tensor(norm["effective_action_max"]),action_scale=0,min_range_per_dim=.1)
byframe={int(r["frame"]):r for r in raw}
for i,row in enumerate(raw):
 r=logs[int(row["frame"])];decision=r["decision"]
 assert bool(row["complete_physics"]) and source_code(decision["mode"])==int(row["action_source"])
 for field,value in (("action",r["executed"]),("policy_action_proposed",r["policy"]),
                     ("intervention_action_proposed",decision["intervention_action_proposed"])):
  expected=scale.scale(torch.tensor(value,dtype=torch.float32))
  assert torch.max(abs(row[field]-expected))<2e-6,(i,field)
 if i+1<len(raw):
  assert int(raw[i+1]["frame"])==int(row["frame"])+1
  for key in row["obs"].keys():assert torch.equal(row["next","obs",key],raw[i+1]["obs",key]),(i,key)
 assert float(row["next","reward"])==0. and not row["next","terminated"]
# Explicit provenance migration for HISTORICAL audit only, never checkpoint
# continuation. Physical actions/observations/rewards remain byte-for-byte.
from acoserl.action_provenance import describe
from acoserl.isolated_nstep import IsolatedNStep
for row in raw:
 roles=describe(int(row['action_source']),row['policy_action_proposed'].numpy(),row['action'].numpy(),quality=bool(row['quality_validated']))
 for k,v in roles.items():row[k]=torch.tensor(v,dtype=torch.long if k.endswith('sources') else torch.bool)
original,*_=offline(128);on=replay(128,128);stream=IsolatedNStep(3,.99)
for row in raw:
 for item in stream.append(row):on.add(item)
for item in stream.truncate(0):on.add(item)
results={}
for mode in ("official","safe_handback"):
 bank=DynamicDemo(capacity=128,admission_mode=mode)
 for row in raw:
  after=logs[int(row['frame'])]['recovery_result']
  bank.observe(row,handback=bool(after.get('safe_handback')),safe=not after.get('failure'))
 bank.close_episode(0,True,6)
 assert len(bank.buffer)>0
 for row in bank.buffer[:]:
  assert int(row["action_source"]) in TASK_INTERVENTION_SOURCES
  assert bool(row["quality_validated"]) and row["quality_error_before"]>row["quality_error_after"]
  first=byframe[int(row["frame"])];last=byframe[int(row["frame"])+int(row["steps_to_next_obs"])-1]
  assert torch.equal(row["action"],first["action"])
  for key in row["obs"].keys():
   assert torch.equal(row["obs",key],first["obs",key])
   assert torch.equal(row["next","obs",key],last["next","obs",key])
 bank.save(out/mode)
 other=DynamicDemo(capacity=128,admission_mode=mode);other.load(out/mode,discard_pending=False)
 demo=bank.sample_demo(original,128);again=other.sample_demo(original,128)
 for key in demo.keys(True,True):assert torch.equal(demo[key],again[key]),key
 mixed=torch.cat([on.sample(128).clone(),demo],0)
 assert mixed.batch_size==torch.Size([256])
 assert (demo["source"]==0).sum()==bank.original_quota
 assert (demo["source"]==2).sum()==128-bank.original_quota
 assert (demo["next","reward"]>0).sum()>=bank.positive_quota
 formal=DynamicDemo(capacity=128,admission_mode=mode,original_quota=96,positive_quota=2)
 for row in raw:
  after=logs[int(row["frame"])]["recovery_result"]
  formal.observe(row,handback=bool(after.get("safe_handback")),safe=not after.get("failure"))
 formal.close_episode(0,True,6)
 assert len(formal.buffer)>0
 formal_batch=formal.sample_demo(original,128)
 assert (formal_batch["source"]==0).sum()==96 and (formal_batch["source"]==2).sum()==32
 assert (formal_batch["next","reward"]>0).sum()>=2
 formal.save(out/(mode+"_formal_quota"))
 results[mode]=dict(expert_rows=len(bank.buffer),sampling=bank.last_sample,
      formal_quota_sampling=formal.last_sample,source_counts=bank.stats["accepted_action_sources"],projection_excluded=bank.stats["safety_projection_excluded_rows"],
      admitted_error_reduction_m=bank.stats["accepted_error_reduction_m"],reload_sampling_exact=True)
result=dict(passed=True,source=str(source),real_raw_rows=len(raw),online_nstep_rows=len(on),original_rows=len(original),
 three_action_chain_verified=True,actual_next_rgb_continuity_verified=True,expert_projection_rows=0,
 reward_unchanged=True,explicit_historical_per_arm_metadata_rebuild=True,admission_modes=results,new_physics_steps=0,new_optimizer_updates=0)
(out/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
