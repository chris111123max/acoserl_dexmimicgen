"""Conservative simulator supervisor inspired by AutoSERL; no RL implementation.

The P0 OSC helper operates in world error -> base-frame normalized deltas.
Privileged object/contact observations stay here and never enter Actor inputs.
"""
from pathlib import Path
import copy,json,sys
from collections import deque
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import poses,objects,snapshot,Contacts,servo,errors,ARMS,SafetyStop,ActionOnly
from acoserl.object_geometry import object_error
from acoserl.trajectories import load_single_reference
DEFAULT=json.loads((ROOT/'configs/bimanual_autoserl/v1.json').read_text())
# Explicitly logged in checkpoint configuration; False is only for legacy diagnosis.
DEFAULT['passive_hold_entry_gate_enabled']=True
SURVEY=json.loads((ROOT/'configs/bimanual_autoserl/reference_grasps.json').read_text())['records']


def strict_geometry(env):
 """Keep original geometric predicate; require real GripperModel grasp checks."""
 official=bool(env._check_success())
 held={a:{o:bool(env._check_grasp(r.gripper['right'],getattr(env,o).contact_geoms))
       for o in ['piece_1','piece_2']} for a,r in zip(ARMS,env.robots)}
 return official and not any(v for g in held.values() for v in g.values()),official,held


def direction_ok(tangent,to_target,eps=1e-6,minimum=0):
 a=float(np.linalg.norm(tangent));b=float(np.linalg.norm(to_target))
 if a<eps:return True,'stationary_reference_use_pose_and_object_gate'
 if b<eps:return True,'already_at_target'
 cosine=float(np.clip(np.dot(tangent,to_target)/(a*b),-1,1))
 return cosine>=minimum,cosine

class LegacyBimanualAutoSERL:
 def __init__(self,env,config=None,intervention_enabled=None):
  self.env=env;self.cfg=copy.deepcopy(DEFAULT if config is None else config)
  if intervention_enabled is not None:self.cfg['intervention_enabled']=intervention_enabled
  self.reference=load_single_reference(ROOT/'datasets/single_demo/two_arm_three_piece_assembly_demo_0.hdf5')
  self.contacts=Contacts(env);self.clean_successes=0;self.auto_disabled=False;self.termination_log=[];self.reset()
 def reset(self,initial_progress=(0,0)):
  self.cursors=[self.reference.cursors[k] for k in ['trajectory_A','trajectory_B']]
  for c,i in zip(self.cursors,initial_progress):c.seek(i)
  self.mode='Normal';self.age=0;self.cooldown=[0,0];self.stable=0;self.active=None
  self.baseline=None;self.targets=None;self.events=[];self.steps=0;self.episode_interventions=0
  self.history=[deque(maxlen=self.cfg['stagnation_window']) for _ in range(2)]
  self.stop_reason=None;self.last_decision=None
 def indices(self):return [c.index for c in self.cursors]
 def pose(self,ai,idx):
  a=self.reference.arms[['trajectory_A','trajectory_B'][ai]]
  return {'position':a['eef_position'][idx], 'rotation':Rotation.from_quat(a['eef_quaternion_xyzw'][idx]).as_matrix()}
 def ref_object(self,name,idx):
  m=self.reference.object_poses[name][idx]
  return {'position':m[:3,3],'quaternion_wxyz':np.roll(Rotation.from_matrix(m[:3,:3]).as_quat(),1)}
 def support_evidence(self,name):
  # Small voxel bodies can distribute weight across many contacts below the P0
  # 0.1 N dynamic-graph threshold. Aggregate measured support relative to real weight.
  model=self.env.sim.model._model;data=self.env.sim.data._data
  gravity=np.asarray(model.opt.gravity);g=float(np.linalg.norm(gravity));up=-gravity/max(g,1e-12)
  weight=float(model.body_subtreemass[self.env.obj_body_id[name]]*g);support=0.;evidence=[]
  for i in range(self.env.sim.data.ncon):
   c=self.env.sim.data.contact[i];owners={self.contacts.owners[int(c.geom1)],self.contacts.owners[int(c.geom2)]}
   if name not in owners or not owners.intersection({'base','static'}):continue
   force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
   vertical=float(max(force[0],0)*abs(np.dot(np.asarray(c.frame[:3]),up)));support+=vertical
   evidence.append({'owners':sorted(owners),'normal_force_N':float(force[0]),'vertical_force_N':vertical,'distance_m':float(c.dist)})
  return {'weight_N':weight,'vertical_support_N':support,'supported':bool(weight>0 and support>=self.cfg['placed_support_weight_fraction']*weight),'contacts':evidence}
 def phase(self,state,contact):
  self.last_support_evidence=None
  if contact['dynamic_A_B_coupled']:return 'coupled','dynamic_A_B_contact_chain'
  if contact['max_robot_contact_force_N']>self.cfg['robot_force_stop_N']:return 'unknown','robot_force_limit'
  g=state['actual_grasps'];correctA=g['A']['piece_1'];correctB=g['B']['piece_2']
  # Contact geometry is also checked below the force-graph reporting threshold.
  for i in range(self.env.sim.data.ncon):
   c=self.env.sim.data.contact[i];owners={self.contacts.owners[int(c.geom1)],self.contacts.owners[int(c.geom2)]}
   if c.dist>0:continue
   for grasps in g.values():
    for held_name,held in grasps.items():
     if held and held_name in owners and (owners-{held_name}).intersection({'base','piece_1','piece_2','static'}):
      return 'assembly_contact','held_object_has_environment_contact'
  p1=state['objects']['piece_1'];base=state['objects']['base']
  placed=np.linalg.norm(np.asarray(p1['position'])[:2]-np.asarray(base['position'])[:2])<.02
  support=self.support_evidence('piece_1')
  self.last_support_evidence=support
  p1_contact=support['supported']
  velocity=np.asarray(self.env.sim.data.cvel[self.env.obj_body_id['piece_1']])
  at_rest=np.linalg.norm(velocity[3:])<=self.cfg['placed_object_linear_speed_m_s'] and np.linalg.norm(velocity[:3])<=self.cfg['placed_object_angular_speed_rad_s']
  finished_A=not any(g['A'].values()) and correctB and placed and p1_contact and at_rest and .02<np.asarray(p1['position'])[2]-np.asarray(base['position'])[2]<.05
  # A released piece_1 resting in base is a completed assembly, not a constraint on B.
  # Any contact involving B's piece_2 with that assembly still forbids unilateral recovery.
  for e in contact['edges']:
   pair={e['a'],e['b']}
   for arm in ARMS:
    if arm in pair:
     for obj in pair.intersection({'base','piece_1','piece_2'}):
      if not g[arm][obj]:return 'assembly_contact','robot_contacts_nonheld_task_object'
   if pair=={'base','piece_1'} and finished_A:continue
   if pair<={'base','piece_1','piece_2'} or 'base' in pair and bool(pair & {'A','B'}):return 'assembly_contact','object_assembly_or_gripper_base_contact'
  g=state['actual_grasps'];correctA=g['A']['piece_1'];correctB=g['B']['piece_2']
  if any(g[a][o] for a in ARMS for o in ['base','piece_1','piece_2'] if (a,o) not in [('A','piece_1'),('B','piece_2')]):return 'unknown','unexpected_grasp_relation'
  if correctA and correctB:return 'independent_carry','separate_objects_no_assembly_contact'
  # A may already have placed piece_1. Use real grasp status and geometric placement.
  p1=state['objects']['piece_1'];base=state['objects']['base']
  placed=np.linalg.norm(np.asarray(p1['position'])[:2]-np.asarray(base['position'])[:2])<.02
  if finished_A:return 'A_finished_B_carry','piece1_geometric_placement_and_A_empty'
  return 'unknown','no_validated_independent_phase'
 def window(self,ai,state):
  cursor=self.cursors[ai].index;candidates=[];actual_grasp=state['actual_grasps'][ARMS[ai]]
  for j in range(cursor,min(229,cursor+self.cfg['window_forward']+1)):
   # Legal phase / grasp / gripper sign; no reference release crossing through nearest point.
   refgrasp=SURVEY[j]['actual_grasps'][ARMS[ai]]
   if actual_grasp!=refgrasp:continue
   g0=np.sign(self.reference.joint_actions[cursor,[6,13][ai]])
   if np.sign(self.reference.joint_actions[j,[6,13][ai]])!=g0:continue
   obj_errors={};valid=True
   for name,held in actual_grasp.items():
    if held or name=='piece_1' and state['official_metrics']['first_piece_assembled'] and not any(state['actual_grasps']['A'].values()):
     oe=object_error(state['objects'][name],self.ref_object(name,j));obj_errors[name]=oe
     valid &= oe['position_m']<=self.cfg['object_window_position_m'] and oe['rotation_rad']<=self.cfg['object_window_rotation_rad']
   if not valid:continue
   pe=errors(state['poses'][ai],self.pose(ai,j))
   score=pe['position_m']/.002+pe['rotation_rad']/.035+sum(o['position_m']/.012+o['rotation_rad']/.15 for o in obj_errors.values())
   candidates.append({'index':j,'score':score,'pose_error':pe,'object_errors':obj_errors})
  return sorted(candidates,key=lambda x:x['score'])
 def stop(self,reason):
  self.mode='Safe_Stop';self.stop_reason=reason
  self.events.append({'step':self.steps,'event':'safe_stop','reason':reason,'indices':self.indices()})
 def passive_hold_entry(self,active,state):
  """Screen the proposed passive hold before control ownership changes.

  A captured pose is not a stationary state. For the verified critically damped
  OSC, v0/(e*sqrt(kp)) estimates the peak free transient; one physics step is
  added for sampled-pose latency. This is a local eligibility screen, not a
  nonlinear/contact safety certificate. Existing retention limits are unchanged.
  """
  pi=1-active;robot=self.env.robots[pi];ctl=robot.part_controllers['right']
  kp=np.asarray(ctl.kp);kd=np.asarray(ctl.kd)
  if kp.shape!=(6,) or not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp<=0) or not np.allclose(kd,2*np.sqrt(kp)) or ctl.impedance_mode!='fixed':
   return {'allowed':False,'passive_arm':ARMS[pi],'reason':'unverified_passive_controller_damping'}
  site=self.env.sim.model.site_id2name(robot.eef_site_id['right'])
  velocity=np.asarray(self.env.sim.data.get_site_xvelp(site));angular=np.asarray(self.env.sim.data.get_site_xvelr(site))
  if not np.isfinite(velocity).all() or not np.isfinite(angular).all():
   return {'allowed':False,'passive_arm':ARMS[pi],'reason':'nonfinite_passive_velocity'}
  speed=float(np.linalg.norm(velocity));angular_speed=float(np.linalg.norm(angular))
  latency=float(self.env.sim.model._model.opt.timestep)
  position_prediction=speed*(1/(np.e*np.sqrt(kp[:3].min()))+latency)
  rotation_prediction=angular_speed*(1/(np.e*np.sqrt(kp[3:].min()))+latency)
  held=[obj for obj,yes in state['actual_grasps'][ARMS[pi]].items() if yes]
  gripper_target=np.asarray(robot.gripper['right'].current_action)
  closed=not held or gripper_target.shape==(2,) and np.allclose(gripper_target,[-1.,1.],atol=1e-6,rtol=0)
  object_predictions={}
  for obj in held:
   radius=float(np.linalg.norm(np.asarray(state['objects'][obj]['position'])-state['poses'][pi]['position']))
   object_predictions[obj]=float(position_prediction+radius*rotation_prediction)
  checks={'position':bool(position_prediction<=self.cfg['passive_position_drift_m']),
          'rotation':bool(rotation_prediction<=self.cfg['passive_rotation_drift_rad']),
          'object_position':all(v<=self.cfg['passive_object_position_drift_m'] for v in object_predictions.values()),
          'object_rotation':bool(not held or rotation_prediction<=self.cfg['passive_object_rotation_drift_rad']),
          'held_gripper_command_closed':bool(closed)}
  return {'allowed':all(checks.values()),'passive_arm':ARMS[pi],'reason':'eligible' if all(checks.values()) else 'passive_hold_entry_not_quasistatic',
    'linear_speed_m_s':speed,'angular_speed_rad_s':angular_speed,'physics_latency_s':latency,
    'predicted_peak_position_m':float(position_prediction),'predicted_peak_rotation_rad':float(rotation_prediction),
    'object_position_predictions_m':object_predictions,'gripper_current_action':gripper_target.tolist(),'checks':checks,
    'position_limit_m':self.cfg['passive_position_drift_m'],'object_position_limit_m':self.cfg['passive_object_position_drift_m']}
 def command(self,targets,grips):
  return servo(self.env,targets,grips,max_position_step=self.cfg['position_command_cap_m'],
   max_rotation_step=self.cfg['rotation_command_cap_rad'],position_gain=self.cfg['position_gain'],rotation_gain=self.cfg['rotation_gain'])
 def choose(self,policy_action):
  policy_action=np.asarray(policy_action,dtype=float);assert policy_action.shape==(14,) and np.isfinite(policy_action).all()
  self.steps+=1
  if not self.cfg['intervention_enabled'] or self.auto_disabled:
   self.last_decision={'mode':'Normal','enabled':False,'indices':self.indices(),'policy_action':policy_action.tolist(),'executed_action':policy_action.tolist()}
   return policy_action.copy(),self.last_decision
  state=snapshot(self.env);contact=self.contacts.read();phase,reason=self.phase(state,contact)
  windows=[self.window(i,state) for i in range(2)]
  self.last_decision={'mode_before':self.mode,'phase':phase,'phase_reason':reason,'indices':self.indices(),'support_evidence':getattr(self,'last_support_evidence',None),
    'window_candidates':windows,'policy_action':policy_action.tolist(),'handback':False}
  if self.mode=='Normal':
   triggers=[]
   for ai in range(2):
    c=self.cursors[ai];self.history[ai].append(state['poses'][ai]['position'].copy())
    if self.cooldown[ai]>0:self.cooldown[ai]-=1
    if not windows[ai]:
     self.last_decision.setdefault('per_arm',[]).append({'arm':ARMS[ai],'window_match':False,'cooldown':self.cooldown[ai]})
     # Unknown task phase is never used to guess a distant global reference.
     if phase not in ['independent_carry','A_finished_B_carry']:self.stop(reason)
     else:self.stop('no_phase_and_object_consistent_window_'+ARMS[ai])
     continue
    candidate=windows[ai][0];j=candidate['index'];pe=candidate['pose_error']
    if pe['position_m']<=self.cfg['advance_position_m'] and pe['rotation_rad']<=self.cfg['advance_rotation_rad'] and j>c.index:
     c.advance(min(self.cfg['advance_max'],j-c.index))
    history=self.history[ai];stagnant=len(history)==history.maxlen and np.linalg.norm(history[-1]-history[0])<self.cfg['stagnation_motion_m']
    self.last_decision.setdefault('per_arm',[]).append({'arm':ARMS[ai],'window_match':True,'pose_error':pe,'stagnant':bool(stagnant),'cooldown':self.cooldown[ai]})
    trigger=pe['position_m']>self.cfg['trigger_position_m'] or pe['rotation_rad']>self.cfg['trigger_rotation_rad'] or stagnant and pe['position_m']>self.cfg['stagnation_error_m']
    if trigger and not self.cooldown[ai]:triggers.append((ai,candidate,'stagnation' if stagnant else 'window_distance'))
   if self.mode!='Safe_Stop' and triggers:
    if phase not in ['independent_carry','A_finished_B_carry']:self.stop(reason)
    elif len(triggers)>1:self.stop('both_arms_outside_validated_single_arm_recovery_envelope')
    else:
     ai,candidate,trigger=triggers[0];j=candidate['index'];pe=candidate['pose_error'];target=self.pose(ai,j)
     tangent=self.pose(ai,min(j+1,228))['position']-target['position']
     allowed,cosine=direction_ok(tangent,target['position']-state['poses'][ai]['position'],self.cfg['direction_epsilon_m'],self.cfg['direction_cosine_min'])
     # P0 supports returning to the last verified point even if it is behind the forward tangent.
     # This is a bounded safety-recovery fallback, never a cursor rewind.
     if not allowed:j=self.cursors[ai].index;target=self.pose(ai,j);trigger+=':verified_local_safety_anchor'
     pe=errors(state['poses'][ai],target)
     if pe['position_m']>self.cfg['max_capture_position_m'] or pe['rotation_rad']>self.cfg['max_capture_rotation_rad']:self.stop('deviation_exceeds_verified_capture_range')
     else:
      entry=self.passive_hold_entry(ai,state)
      self.last_decision['passive_hold_entry']=entry
      if self.cfg.get('passive_hold_entry_gate_enabled',True) and not entry['allowed']:
       self.stop(entry['reason']+':'+ARMS[1-ai])
       self.last_decision.update(mode='Safe_Stop',reason=self.stop_reason,execute=False)
       return None,self.last_decision
      self.recovery_index=j
      self.active=ai;self.baseline=state;self.targets=copy.deepcopy(state['poses']);self.targets[ai]=target
      self.grips=[1. if any(state['actual_grasps'][a].values()) else float(self.reference.joint_actions[self.cursors[i].index,[6,13][i]]) for i,a in enumerate(ARMS)]
      self.mode='Recover_'+ARMS[ai];self.age=0;self.stable=0;self.episode_interventions+=1
      self.events.append({'step':self.steps,'event':'start','mode':self.mode,'reason':trigger,'direction':cosine,'target_index':j,'indices':self.indices(),'phase':phase})
  if self.mode.startswith('Recover_'):
   if phase not in ['independent_carry','A_finished_B_carry']:self.stop('recovery_phase_invalid:'+reason)
   if self.mode!='Safe_Stop':return self._recover_decision(state)
  if self.mode=='Safe_Stop':
   # Signal stop before physical execution; caller explicitly truncates the preceding real transition.
   self.last_decision.update(mode=self.mode,reason=self.stop_reason,execute=False)
   return None,self.last_decision
  self.last_decision.update(mode='Normal',enabled=True,indices=self.indices(),executed_action=policy_action.tolist(),execute=True)
  return policy_action.copy(),self.last_decision
 def _recover_decision(self,state):
  action=self.command(self.targets,self.grips);self.age+=1
  self.last_decision.update(mode=self.mode,indices=self.indices(),recovery_step=self.age,arm_modes=['Recover' if i==self.active else 'Hold' for i in range(2)],cooldown=self.cooldown.copy(),executed_action=action.tolist(),execute=True)
  return action,self.last_decision
 def after_step(self):
  if not self.mode.startswith('Recover_'):return {'handback':False}
  state=snapshot(self.env);phase,reason=self.phase(state,self.contacts.read());ai=self.active;pi=1-ai
  failure=None
  if phase not in ['independent_carry','A_finished_B_carry']:failure='phase_invalid:'+reason
  if state['actual_grasps']!=self.baseline['actual_grasps']:failure='grasp_relation_changed'
  pe=errors(state['poses'][pi],self.baseline['poses'][pi])
  if pe['position_m']>self.cfg['passive_position_drift_m'] or pe['rotation_rad']>self.cfg['passive_rotation_drift_rad']:failure='passive_pose_changed'
  for name,held in self.baseline['actual_grasps'][ARMS[pi]].items():
   if held:
    oe=object_error(state['objects'][name],self.baseline['objects'][name])
    if oe['position_m']>self.cfg['passive_object_position_drift_m'] or oe['rotation_rad']>self.cfg['passive_object_rotation_drift_rad']:failure='passive_object_changed'
  if object_error(state['objects']['base'],self.baseline['objects']['base'])['position_m']>self.cfg['base_drift_m']:failure='base_displaced'
  if self.baseline['official_metrics']['first_piece_assembled']:
   if not state['official_metrics']['first_piece_assembled']:failure='first_piece_progress_lost'
   oe=object_error(state['objects']['piece_1'],self.baseline['objects']['piece_1'])
   if oe['position_m']>self.cfg['assembled_object_drift_m'] or oe['rotation_rad']>self.cfg['assembled_object_rotation_drift_rad']:failure='completed_piece1_displaced'
  active_object_valid=True;active_object_errors={}
  for obj,held in state['actual_grasps'][ARMS[ai]].items():
   if held:
    oe=object_error(state['objects'][obj],self.ref_object(obj,self.recovery_index));active_object_errors[obj]=oe
    if oe['position_m']>self.cfg['object_window_position_m'] or oe['rotation_rad']>self.cfg['object_window_rotation_rad']:failure='active_object_outside_capture_envelope'
    active_object_valid &= oe['position_m']<=self.cfg['active_object_exit_position_m'] and oe['rotation_rad']<=self.cfg['active_object_exit_rotation_rad']
  e=errors(state['poses'][ai],self.targets[ai]);okay=e['position_m']<=self.cfg['exit_position_m'] and e['rotation_rad']<=self.cfg['exit_rotation_rad'] and active_object_valid
  self.stable=self.stable+1 if okay and failure is None else 0
  result={'handback':False,'error':e,'passive_error':pe,'phase':phase,'retention_failure':failure,'active_object_exit_valid':bool(active_object_valid),'active_object_errors':active_object_errors}
  if failure:self.stop(failure)
  elif self.stable>=self.cfg['stable_exit_steps']:
   mode=self.mode;self.mode='Normal';self.cooldown[ai]=self.cfg['cooldown_steps'];result['handback']=True
   self.events.append({'step':self.steps,'event':'handback','from':mode,'indices':self.indices(),'recovery_steps':self.age,'error':e,'phase':phase})
  elif self.age>=self.cfg['max_recovery_steps']:self.stop('recovery_timeout_no_safe_handback')
  return result
 def episode_result(self,stable_full_success,had_intervention=None,full_policy_episode=False):
  intervened=self.episode_interventions>0 if had_intervention is None else had_intervention
  clean=bool(stable_full_success and not intervened and full_policy_episode)
  self.clean_successes=self.clean_successes+1 if clean else 0
  self.auto_disabled=self.clean_successes>=self.cfg['disable_after_clean_success_episodes']
  row={'stable_full_success':bool(stable_full_success),'had_intervention':bool(intervened),'full_policy_episode':bool(full_policy_episode),'consecutive_clean_successes':self.clean_successes,'auto_disabled':self.auto_disabled}
  self.termination_log.append(row);return row


from acoserl.two_point_supervisor import TwoPointMixin

from acoserl.intervention_manager import InterventionManagerMixin

class BimanualAutoSERL(InterventionManagerMixin,TwoPointMixin,LegacyBimanualAutoSERL):
 """Section-3.3 windows and two-point recovery; inherit verified OSC/hold gates."""
 pass
