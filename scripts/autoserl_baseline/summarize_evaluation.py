"""Stage metrics from real evaluation worker logs; no synthetic success labels."""
import json,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
log=a.run/'evaluation_workers/env16.jsonl'
if not log.exists():raise SystemExit('No actual evaluation log')
episodes={};training_steps={}
for line in log.open():
 row=json.loads(line)
 if row['event']=='reset_ready' and row['episode']==0:training_steps[row['pid']]=row['seed']-9000000
 if row['event']=='step':episodes.setdefault((row['pid'],row['episode_id']),[]).append(row)
result=[]
for (pid,ep),rows in sorted(episodes.items()):
 record=dict(worker_pid=pid,training_step=training_steps.get(pid),episode=ep,length=sum(bool(r['executed']) for r in rows),grasp_reached={'A':False,'B':False},stable_grasp={'A':False,'B':False},policy_only_stable_grasp={'A':False,'B':False},both_stably_held=False,first_assembly=False,second_assembly_geometry=False,full_success=False,projection_steps=0,reason=None)
 histories=[[],[]];direct=[[],[]]
 for r in rows:
  if not r['executed']:record['reason']=r.get('supervisor_stop_reason');continue
  assert not r['action_source'].startswith(('Guide_','Recover_')),'active recovery in evaluation'
  record['projection_steps']+=int(r['action_source'].startswith('Constrained_'))
  stable_now=[]
  for ai,(arm,obj) in enumerate([('A','piece_1'),('B','piece_2')]):
   held=r['supervisor_grasps'][arm][obj];record['grasp_reached'][arm]|=held
   if held:
    relative=np.asarray(r['eef_rotations'][ai]).T@(np.asarray(r['supervisor_objects'][obj]['position'])-r['eef_positions'][ai])
    histories[ai].append(relative);histories[ai]=histories[ai][-10:]
    roles=r['arm_control'];direct[ai].append(roles['pose_sources'][ai]==0 and roles['gripper_sources'][ai]==0);direct[ai]=direct[ai][-10:]
   else:histories[ai]=[];direct[ai]=[]
   stable=len(histories[ai])==10 and max(np.linalg.norm(v-histories[ai][0]) for v in histories[ai])<.0005
   stable_now.append(stable);record['stable_grasp'][arm]|=stable;record['policy_only_stable_grasp'][arm]|=stable and all(direct[ai])
  record['both_stably_held']|=all(stable_now)
  record['first_assembly']|=bool(r['task_metrics']['first_piece_assembled'])
  record['second_assembly_geometry']|=bool(r['task_metrics']['task'])
  record['full_success']|=bool(r.get('success'))
  if r.get('supervisor_stop_reason'):record['reason']=r['supervisor_stop_reason']
 result.append(record)
output=dict(episodes=result,episodes_count=len(result),fixed_canonical_initial_state=True,independent_initial_physical_states=False,full_success_rate=sum(r['full_success'] for r in result)/max(len(result),1),a_stable_grasp_rate=sum(r['stable_grasp']['A'] for r in result)/max(len(result),1),b_stable_grasp_rate=sum(r['stable_grasp']['B'] for r in result)/max(len(result),1),both_stable_hold_rate=sum(r['both_stably_held'] for r in result)/max(len(result),1),first_assembly_rate=sum(r['first_assembly'] for r in result)/max(len(result),1),second_assembly_geometry_rate=sum(r['second_assembly_geometry'] for r in result)/max(len(result),1),note='Geometry stage reach is not full strict success; safety-constrained and policy-only grasp metrics are separate. Bounded/incomplete diagnostic episodes are not formal20-episode performance evaluations.')
(a.run/'stage_evaluation_metrics.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
