"""Propose points from measured contacts/grasps; actual recovery verification is separate."""
import json,numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[2];d=json.loads((root/'outputs/autoserl_two_point/demo_audit.json').read_text());rows=d['rows'];z=np.load(root/'outputs/autoserl_two_point/reference.npz');points={}
for ai,(arm,obj) in enumerate([('A','piece_1'),('B','piece_2')]):
 first_contact=next(i for i,r in enumerate(rows) if any({e['a'],e['b']}=={arm,obj} for e in r['contact']['edges']))
 first_grasp=next(i for i,r in enumerate(rows) if r['actual_grasps'][arm][obj])
 closing=np.flatnonzero(z['actions'][:first_contact,[6,13][ai]]>0);p0=int(closing[0])
 assert not rows[p0]['actual_grasps'][arm][obj] and p0<first_contact
 p1=None
 for i in range(first_grasp+2,len(rows)):
  block=rows[i-2:i+1]
  if all(r['actual_grasps'][arm][obj] and np.allclose(r['gripper_command'][ai],[-1,1]) for r in block):
   rel=[np.asarray(r['poses'][ai]['rotation']).T@(np.asarray(r['objects'][obj]['position'])-np.asarray(r['poses'][ai]['position'])) for r in block]
   if max(np.linalg.norm(x-rel[0]) for x in rel)<.001:p1=i;break
 points[arm]={'object':obj,'p0':p0,'p1':p1,'window_length':p1-p0,'first_contact':first_contact,'first_grasp':first_grasp,'validated':False,'basis':'P0: pre-action open gripper, first commanded closure, no measured target contact or grasp; P1: three consecutive closed-command actual-grasp states, relative object translation stable <1mm. Candidate until real return/replay acceptance.'}
assembly=[]
for pair in [{'base','piece_1'},{'piece_1','piece_2'}]:
 ids=[i for i,r in enumerate(rows) if any({e['a'],e['b']}==pair for e in r['contact']['edges'])];assembly.append({'objects':sorted(pair),'first_contact':min(ids) if ids else None,'last_contact':max(ids) if ids else None,'unilateral_recovery_validated':False,'joint_recovery_validated':False,'barrier':'No unilateral replay across assembly contact; do not rewind completed piece_1. No invented joint recovery.'})
# P0 is revised after proximity.py; do not overwrite validated configuration silently.
config={'task':'TwoArmThreePieceAssembly','demo':'demo_0','recorded_actions':d['recorded_actions'],'recorded_states':d['recorded_states'],'reference_states':len(rows),'reference_path':'outputs/autoserl_two_point/reference.npz','arms':points,'assembly':assembly,'reset_protocol':'Only canonical demo frame0; every intermediate state reached by actual control actions'}
(root/'outputs/autoserl_two_point/initial_candidates.json').write_text(json.dumps(config,indent=2)+'\n')
(root/'configs/bimanual_autoserl/reference_grasps.json').write_text(json.dumps({'records':[{'index':i,'actual_grasps':r['actual_grasps'],'official_metrics':r['official_metrics']} for i,r in enumerate(rows)]},indent=2)+'\n')
print(json.dumps(config,indent=2))
