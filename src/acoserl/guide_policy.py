"""Free-space Guide/Policy ownership; P0/P1 continues through qualified Hold.

Explicit pre-action screens plus unchanged actual substep safety. Development
qualification is separate from production acceptance. No joint contact recovery.
"""
import copy
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from acoserl.two_point_supervisor import snapshot,errors,ARMS
from acoserl.osc_prediction import predict,command_for_displacement

class GuidePolicyMixin:
 def reset(self,initial_progress=(0,0)):
  super().reset(initial_progress);self.policy_guide=None;self.guide_handoff_pending=None
 def independent_empty(self,state):
  contact=self.contacts.read()
  return (self.contact_assessment(state,contact)['category']=='Independent'
   and not self.collaboration(state,contact)['coupled']
   and not any(any(x.values()) for x in state['actual_grasps'].values())
   and not any(e['a'] in ARMS or e['b'] in ARMS for e in contact['edges']))
 def joint_free_screen(self,action,state):
  if not self.independent_empty(state):return False,'not_independent_empty_free_space'
  risks=self.action_risks(action,state)
  if any(x['risk'] for x in risks):return False,'remaining_action_risk'
  # Swept relative EEF segment, and conservative whole-arm geom clearance.
  nominal=[predict(self.env,i,action) for i in range(2)]
  delta=[p['displacement'] for p in nominal];relative=state['poses'][0]['position']-state['poses'][1]['position'];motion=delta[0]-delta[1]
  t=np.clip(-relative@motion/max(motion@motion,1e-12),0,1)
  if np.linalg.norm(relative+t*motion)<.08:return False,'relative_eef_swept_corridor'
  owners=self.contacts.owners;model=self.env.sim.model;data=self.env.sim.data
  ids=[[g for g,a in owners.items() if a==arm] for arm in ARMS]
  budget=[np.linalg.norm(n['displacement'])+.15*np.linalg.norm(n['angular_displacement']) for n in nominal]
  for ai in range(2):
   for g in ids[ai]:
    others=[h for h,owner in owners.items() if owner!=ARMS[ai] and h!=g]
    # Static collisions remain handled by existing workspace and real substeps;
    # table/task approach uses all actual collision-enabled task geoms below.
    for h in others:
     if not ((int(model.geom_contype[g])&int(model.geom_conaffinity[h])) or (int(model.geom_contype[h])&int(model.geom_conaffinity[g]))):continue
     bound=budget[ai]+(budget[1-ai] if owners[h] in ARMS else 0.)+.003
     if np.linalg.norm(data.geom_xpos[g]-data.geom_xpos[h])-model.geom_rbound[g]-model.geom_rbound[h]>bound:continue
     gap=float(mujoco.mj_geomDistance(model._model,data._data,g,h,bound,None))
     if gap<bound:return False,'whole_arm_swept_clearance'
  return True,'independent_joint_nominal_screen'
 def constrain_free(self,policy,state):
  action=np.asarray(policy).copy()
  if not self.independent_empty(state):return None,'not_independent_free_space'
  for ai,risk in enumerate(self.action_risks(action,state)):
   if not risk['risk']:continue
   if any(c not in ('workspace','open_gripper_unplanned_approach') for c in risk['categories']):return None,'unsupported_free_risk'
   if risk.get('gripper_contact',{}).get('command_reversal'):return None,'near_contact_reversal'
   position=state['poses'][ai]['position'];dp=np.asarray(risk['displacement'])
   lo=self.positions[:,ai].min(0)-self.cfg['free_workspace_margin_m'];hi=self.positions[:,ai].max(0)+self.cfg['free_workspace_margin_m']
   dp=np.clip(dp,lo+.003-position,hi-.003-position)
   dp[((position-lo<self.manager_cfg['risk_margin_m'])&(dp<0))|((hi-position<self.manager_cfg['risk_margin_m'])&(dp>0))]=0
   if 'gripper_contact' in risk:
    normal=np.asarray(risk['gripper_contact']['normal']);dp-=max(0.,float(dp@normal))*normal
   action=command_for_displacement(self.env,ai,action,dp)
  if hasattr(self,'action_bounds'):action=np.clip(action,*self.action_bounds)
  okay,reason=self.joint_free_screen(action,state)
  return (action if okay else None),reason
 def eligible_recovery(self,ai,state,contact,kind,target):
  return super().eligible(ai,state,contact,kind,target)
 def eligible_guide(self,ai,state,target,policy):
  if self.completed[ai] or target is None:return False,'completed_or_no_target'
  lo,hi,_=self.legal_bounds(ai,state)
  if not lo<=target<=hi or target>=self.points['arms'][ARMS[ai]]['first_contact']:return False,'guide_event_barrier'
  if not self.independent_empty(state):return False,'guide_requires_independent_empty'
  proposal=self.guide_command(ai,target,policy,state)
  safe,reason=self.constrain_free(proposal,state)
  return safe is not None,reason
 def guide_command(self,ai,target,policy,state):
  targets=copy.deepcopy(state['poses']);targets[ai]=self.pose(ai,target)
  grips=[1. if any(state['actual_grasps'][arm].values()) else -1. for arm in ARMS]
  command=np.asarray(policy).copy();controlled=self.command(targets,grips);command[7*ai:7*ai+7]=controlled[7*ai:7*ai+7]
  return np.clip(command,*self.action_bounds) if hasattr(self,'action_bounds') else command
 def choose(self,policy_action):
  if not self.manager_cfg.get('guide_policy_enabled',False):return super().choose(policy_action)
  policy=np.asarray(policy_action,dtype=float);state=snapshot(self.env)
  danger=self.physical_danger(state,self.contacts.read())
  if danger:self.last_decision={'policy_action':policy.tolist()};self.stop(danger);return self.refuse()
  if self.mode=='Safe_Stop':return self.refuse()
  if self.policy_guide is None and self.mode=='Normal' and self.cfg['intervention_enabled'] and not self.auto_disabled:
   for ai in range(2):
    measure=self.window_measure(ai,state,record_history=False)
    if self.cooldown[ai] or measure['distance_m'] is None or measure['distance_m']<=self.cfg['trigger_position_m'] or not measure['direction_ok']:continue
    okay,reason=self.eligible_guide(ai,state,measure['pipoint'],policy)
    if not okay:continue
    self.active=ai;self.policy_guide=dict(ai=ai,target=measure['pipoint'],age=0,stable=0);self.takeover_sequence+=1;self.takeover_id=self.takeover_sequence
    self.mode='Guide_'+ARMS[ai];self.targets=copy.deepcopy(state['poses']);self.targets[ai]=self.pose(ai,measure['pipoint']);self.grips=[-1.,-1.];self.baseline=copy.deepcopy(state)
    self.episode_interventions+=1;self.decision_stats['starts']+=1;self.handoff_phase=None
    break
  if self.policy_guide is not None:
   guide=self.policy_guide;ai=guide['ai'];self.steps+=1;self.active=ai
   proposal=self.guide_command(ai,guide['target'],policy,state);action,reason=self.constrain_free(proposal,state)
   self.last_decision=dict(mode='Guide_'+ARMS[ai],policy_action=policy.tolist(),indices=self.indices(),takeover_id=self.takeover_id,active_arm=ARMS[ai],guide_policy=True,guide_target=guide['target'],joint_screen=reason,recovery_step=guide['age']+1,stage='approach_pregrasp',execute=action is not None)
   if action is None:self.stop('guide_joint_screen:'+reason);return self.refuse()
   roles=[0,0];roles[ai]=1;other=1-ai
   self.last_decision['pose_sources']=roles.copy();self.last_decision['gripper_sources']=roles.copy()
   if np.any(abs(action[7*ai:7*ai+6]-proposal[7*ai:7*ai+6])>1e-6):self.last_decision['pose_sources'][ai]=4
   if abs(action[7*ai+6]-proposal[7*ai+6])>1e-6:self.last_decision['gripper_sources'][ai]=4
   if np.any(abs(action[7*other:7*other+6]-policy[7*other:7*other+6])>1e-6):self.last_decision['pose_sources'][other]=4
   if abs(action[7*other+6]-policy[7*other+6])>1e-6:self.last_decision['gripper_sources'][other]=4
   self.episode_touched.add((ARMS[ai],'approach_pregrasp'))
   return self._protected(action,self.last_decision,policy)
  risks=self.action_risks(policy,state)
  if len([x for x in risks if x['risk']])>1 and self.mode=='Normal' and self.independent_empty(state):
   corrected,reason=self.constrain_free(policy,state)
   if corrected is not None:
    action,decision=super().choose(corrected);decision['policy_action']=policy.tolist()
    if action is not None and decision['mode']=='Normal':decision.update(mode='Constrained_Joint',projection_reason='independent_free_joint_screen',executed_action=action.tolist())
    return action,decision
  return super().choose(policy)
 def retention(self,state):
  if self.policy_guide is not None:
   return (None if self.independent_empty(state) else 'guide_free_space_domain_changed'),{'passive_control':'Policy','hold_drift_limit_applicable':False}
  return super().retention(state)
 def after_step(self):
  if self.policy_guide is None:
   result=super().after_step()
   if self.guide_handoff_pending is not None and self.mode!='Safe_Stop':
    result.update(handback=True,safe_handback=True);self.last_decision['takeover_id']=self.guide_handoff_pending;self.guide_handoff_pending=None;self.decision_stats['handbacks']+=1
    result.setdefault('task_quality',{})['handoff_confirmed']=True
   return result
  guide=self.policy_guide;ai=guide['ai'];self.mode='Normal';self.projected=True;self.projected_arms=[ai]
  result=super().after_step();guide['age']+=1;self.intervention_control_steps+=1
  e=errors(snapshot(self.env)['poses'][ai],self.pose(ai,guide['target']))
  guide['stable']=guide['stable']+1 if e['position_m']<=self.cfg['exit_position_m'] and e['rotation_rad']<=self.cfg['exit_rotation_rad'] else 0
  if self.mode=='Safe_Stop':return result
  if guide['stable']>=self.cfg['stable_exit_steps']:
   self.policy_guide=None;self.guide_handoff_pending=self.takeover_id;self.cooldown[ai]=self.cfg['cooldown_steps'];self.history[ai].clear()
  elif guide['age']>=self.cfg['max_recovery_steps']:self.stop('guide_timeout')
  else:self.mode='Guide_'+ARMS[ai]
  result.update(guide_policy=True,guide_error=e,guide_controls=guide['age'])
  return result
