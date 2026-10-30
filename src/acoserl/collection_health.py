"""Observational collection-health checks with sustained-grasp and intervention gates.

A transient physical grasp is not proof of task progress, and a moving policy
is not proof the AutoSERL intervention loop is alive.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation

class CollectionHealth:
 def __init__(self,state=None):
  defaults=dict(valid_steps=0,proposals=0,nonzero_motion_steps=0,
   policy_motion_steps=0,projection_steps=0,task_intervention_steps=0,
   safe_handbacks=0,ends=0,short_ends=0,grasps={'A':0,'B':0},
   first_assembly=0,full_success=0,reasons={},stages={},
   grasp_streaks={'A':0,'B':0},stable_grasp_episodes={'A':0,'B':0},
   grasp_episode_markers={'A':None,'B':None},dynamic_demo_rows=0,guide_candidates=0,candidate_refusals={},takeovers=0)
  self.data=copy.deepcopy(defaults)
  if state:self.data.update(copy.deepcopy(state))
 def observe(self,p):
  d=self.data;d['proposals']+=1
  decision=p.get('auto_intervention') or {}
  rejected=decision.get('deferred') or []
  d['guide_candidates']+=len(rejected)+int(bool(decision.get('trigger_reason')))
  for candidate in rejected:
   key=str(candidate.get('reason','unknown'));d['candidate_refusals'][key]=d['candidate_refusals'].get(key,0)+1
  d['takeovers']+=int(decision.get('recovery_step')==1 and str(decision.get('mode','')).startswith(('Guide_','Recover_')))
  if not p.get('executed'):return
  d['valid_steps']+=1;source=p.get('action_source','Policy')
  d['nonzero_motion_steps']+=int(p.get('nonzero_physical_motion',False))
  d['policy_motion_steps']+=int(source=='Policy' and p.get('nonzero_physical_motion',False))
  d['projection_steps']+=int(source.startswith('Constrained_'))
  d['task_intervention_steps']+=int(source.startswith(('Guide_','Recover_')))
  d['safe_handbacks']+=int(bool((p.get('recovery_result') or {}).get('safe_handback')))
  episode=(p.get('env_id'),p.get('episode_id'))
  # Per-arm streaks must be tracked by environment; batch workers interleave.
  by_env=d.setdefault('grasp_streaks_by_env',{})
  streak=by_env.setdefault(str(p.get('env_id')),{'A':0,'B':0,'episode':None})
  if streak.get('episode')!=p.get('episode_id'):
   streak.update(A=0,B=0,episode=p.get('episode_id'),counted_A=False,counted_B=False,relative_A=[],relative_B=[])
  for arm,obj in [('A','piece_1'),('B','piece_2')]:
   held=bool(p.get('supervisor_grasps',{}).get(arm,{}).get(obj,False))
   d['grasps'][arm]+=int(held) # historical step-occurrence counter
   streak[arm]=streak[arm]+1 if held else 0
   d['grasp_streaks'][arm]=max(d['grasp_streaks'][arm],streak[arm])
   history=streak.setdefault('relative_'+arm,[])
   ai='AB'.index(arm)
   try:
    pose=p['eef_positions'][ai];rotation=np.asarray(p['eef_rotations'][ai]);item=p['supervisor_objects'][obj]
    rel=rotation.T@(np.asarray(item['position'])-np.asarray(pose))
    rr=rotation.T@Rotation.from_quat(np.roll(item['quaternion_wxyz'],-1)).as_matrix()
    if held:history.append({'position':rel.tolist(),'rotation':rr.tolist()});del history[:-10]
    else:history.clear()
   except (KeyError,IndexError,TypeError,ValueError):history.clear()
   stable=(len(history)==10 and max(np.linalg.norm(np.asarray(v['position'])-history[0]['position']) for v in history)<.0005
           and max(Rotation.from_matrix(np.asarray(v['rotation'])@np.asarray(history[0]['rotation']).T).magnitude() for v in history)<=.035)
   if streak[arm]>=10 and stable and not streak.get('counted_'+arm,False):
    d['stable_grasp_episodes'][arm]+=1
    d['grasp_episode_markers'][arm]=list(episode)
    streak['counted_'+arm]=True
  d['first_assembly']+=int(p.get('task_metrics',{}).get('first_piece_assembled',False))
  d['full_success']+=int(p.get('success',False))
  stage=p.get('stage','unknown');d['stages'][stage]=d['stages'].get(stage,0)+1
 def episode(self,p):
  d=self.data;d['ends']+=1;d['short_ends']+=int(p.get('frame',0)<3)
  reason=p.get('supervisor_stop_reason')
  if reason:d['reasons'][reason]=d['reasons'].get(reason,0)+1
 def set_dynamic_demo_rows(self,n):
  self.data['dynamic_demo_rows']=int(n)
 def stall(self):
  d=self.data;n=max(d['valid_steps'],1)
  stable=sum(d['stable_grasp_episodes'].values())
  no_task_progress=not (stable or d['first_assembly'] or d['full_success'])
  reset_deadlock=d['ends']>=16 and d['short_ends']/max(1,d['ends'])>.9
  projection_deadlock=d['projection_steps']/n>.95 or d['policy_motion_steps']/n<.01
  intervention_deadlock=(d['valid_steps']>=10000 and no_task_progress and d['dynamic_demo_rows']==0)
  if d['dynamic_demo_rows']>0:intervention_diagnosis='demo_available_actor_not_yet_successful' if no_task_progress else 'progress_observed'
  elif d['task_intervention_steps']>0:intervention_diagnosis='intervention_without_dynamic_demo'
  elif d['guide_candidates']>0:intervention_diagnosis='candidates_rejected'
  else:intervention_diagnosis='no_candidates_observed'
  early_termination_deadlock=(d['ends']>=32 and
                              d['reasons'].get('simultaneous_high_confidence_risks_no_joint_qualification',0)/d['ends']>.5)
  return dict(stop=d['valid_steps']>=10000 and
              ((no_task_progress and (reset_deadlock or projection_deadlock))
               or intervention_deadlock or (no_task_progress and early_termination_deadlock)),
    no_task_progress=no_task_progress,reset_deadlock=reset_deadlock,
    projection_deadlock=projection_deadlock,
    intervention_deadlock=intervention_deadlock,intervention_diagnosis=intervention_diagnosis,
    early_termination_deadlock=early_termination_deadlock)
 def snapshot(self):return copy.deepcopy(self.data)
