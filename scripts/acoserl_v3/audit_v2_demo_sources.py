"""Audit real V2 Replay against executed worker records; do not infer provenance from buffer size."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from acoserl.ascend_data import replay
from acoserl.dynamic_demo import DynamicDemo
from acoserl.transition_semantics import source_family
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
 out=ROOT/'outputs/acoserl_v3'/args.tag;out.mkdir(parents=True,exist_ok=False)
 src=ROOT/'outputs/acoserl_v2/actor_stream_v1'
 logs=[json.loads(s) for s in (src/'worker/env00.jsonl').read_text().splitlines()]
 logs=[x for x in logs if x.get('event')=='step'];actual=[x for x in logs if x.get('executed')]
 identity=lambda x:(int(x['env_id']),int(x['episode_id']),int(x['frame']))
 index={identity(x):x for x in actual};assert len(index)==len(actual)
 on=replay(256,None);on.loads(src/'online_buffer')
 observed=Counter();mismatches=[]
 for row in on[:]:
  key=tuple(int(row[k]) for k in ('env_id','episode','frame'));log=index[key]
  observed[source_family(row['action_source'])]+=1
  if not np.allclose(row['action'].numpy(),log['executed_action'],atol=2e-6,rtol=0):mismatches.append(key)
 assert not mismatches
 banks={}
 for mode in ('official','safe_handback'):
  old=replay(256,None);old.loads(src/mode/'buffer');counts=Counter();err=[]
  for row in old[:]:
   key=tuple(int(row[k]) for k in ('env_id','episode','frame'));log=index[key]
   counts[source_family(row['action_source'])]+=1
   if not np.allclose(row['action'].numpy(),log['executed_action'],atol=2e-6,rtol=0):err.append(key)
  fresh=DynamicDemo(capacity=256,admission_mode=mode)
  for row in sorted(list(on[:]),key=lambda r:tuple(int(r[k]) for k in ('env_id','episode','frame'))):
   fresh.observe(row,handback=True)
  assert len(fresh.buffer)==0 and fresh.stats['safety_projection_excluded_rows']==counts['SafetyProjection']
  assert not err
  rejected_legacy=False
  try:DynamicDemo(capacity=256,admission_mode=mode).load(src/mode)
  except ValueError as e:rejected_legacy='provenance protocol' in str(e)
  assert rejected_legacy
  banks[mode]=dict(old_rows=len(old),actual_source_counts=dict(counts),v3_rows=len(fresh.buffer),v3_stats=fresh.stats,legacy_contamination_import_rejected=rejected_legacy)
 result=dict(passed=True,proposal_count=len(logs),complete_steps=len(actual),not_executed=len(logs)-len(actual),actual_action_sources=dict(Counter(x['action_source'] for x in actual)),online_replay_source_families=dict(observed),online_rows=len(on),executed_action_matches=True,modes=banks,new_physics_steps=0,scope='historical real-data audit and exclusion regression; no new Guide/Recover physics qualification')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
