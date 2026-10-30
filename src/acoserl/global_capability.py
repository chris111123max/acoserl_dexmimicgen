"""Coordinator evidence for a shared Actor, never a worker-local authority."""
import copy
class GlobalCapability:
 def __init__(self,state=None,stage_required=5,full_required=5):
  self.state=copy.deepcopy(state or {'stage_counts':{},'clean_full':0,'disable':False,'episodes':0,'stage_metrics':{}})
  self.stage_required=stage_required;self.full_required=full_required;self.seen=set();self.actor_only={}
 def observe(self,payload,actor_controlled=True):
  identity=(payload['env_id'],payload['episode_id'])
  self.actor_only[identity]=self.actor_only.get(identity,True) and actor_controlled
  outcome=payload.get('episode_outcome')
  if not outcome or identity in self.seen:return
  self.seen.add(identity);self.state['episodes']+=1
  valid=self.actor_only.pop(identity)
  for key,event in outcome.get('stage_events',{}).items():
   metric=self.state['stage_metrics'].setdefault(key,{})
   for name,value in event.items():metric[name]=metric.get(name,0)+int(bool(value))
   qualifies=valid and event.get('stage_policy_completed') and event.get('stage_stably_completed')
   self.state['stage_counts'][key]=self.state['stage_counts'].get(key,0)+1 if qualifies else 0
  clean=valid and not payload.get('supervisor_stop_reason') and outcome.get('autonomous_success',False)
  self.state['clean_full']=self.state['clean_full']+1 if clean else 0
  self.state['disable']=self.state['disable'] or self.state['clean_full']>=self.full_required
 def snapshot(self):return copy.deepcopy(self.state)
