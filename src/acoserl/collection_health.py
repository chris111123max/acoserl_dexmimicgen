"""Observational collection health; never changes rewards or safety thresholds."""
import copy
class CollectionHealth:
 def __init__(self,state=None):
  self.data=copy.deepcopy(state) if state else dict(valid_steps=0,proposals=0,nonzero_motion_steps=0,policy_motion_steps=0,projection_steps=0,task_intervention_steps=0,safe_handbacks=0,ends=0,short_ends=0,grasps={'A':0,'B':0},first_assembly=0,full_success=0,reasons={},stages={})
 def observe(self,p):
  d=self.data;d['proposals']+=1
  if not p.get('executed'):return
  d['valid_steps']+=1;source=p.get('action_source','Policy')
  d['nonzero_motion_steps']+=int(p.get('nonzero_physical_motion',False))
  d['policy_motion_steps']+=int(source=='Policy' and p.get('nonzero_physical_motion',False))
  d['projection_steps']+=int(source.startswith('Constrained_'))
  d['task_intervention_steps']+=int(source.startswith(('Guide_','Recover_')))
  d['safe_handbacks']+=int(bool((p.get('recovery_result') or {}).get('safe_handback')))
  for arm,obj in [('A','piece_1'),('B','piece_2')]:d['grasps'][arm]+=int(p.get('supervisor_grasps',{}).get(arm,{}).get(obj,False))
  d['first_assembly']+=int(p.get('task_metrics',{}).get('first_piece_assembled',False));d['full_success']+=int(p.get('success',False))
  stage=p.get('stage','unknown');d['stages'][stage]=d['stages'].get(stage,0)+1
 def episode(self,p):
  d=self.data;d['ends']+=1;d['short_ends']+=int(p.get('frame',0)<3)
  reason=p.get('supervisor_stop_reason')
  if reason:d['reasons'][reason]=d['reasons'].get(reason,0)+1
 def stall(self):
  d=self.data;n=max(d['valid_steps'],1)
  no_task_progress=not (sum(d['grasps'].values()) or d['first_assembly'])
  reset_deadlock=d['ends']>=16 and d['short_ends']/max(1,d['ends'])>.9
  projection_deadlock=d['projection_steps']/n>.95 or d['policy_motion_steps']/n<.01
  return dict(stop=d['valid_steps']>=10000 and no_task_progress and (reset_deadlock or projection_deadlock),no_task_progress=no_task_progress,reset_deadlock=reset_deadlock,projection_deadlock=projection_deadlock)
 def snapshot(self):return copy.deepcopy(self.data)
