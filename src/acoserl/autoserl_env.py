"""Explicit-reset bridge: executed normalized actions only, no fake transitions."""
import numpy as np
import torch
from acoserl.transition_semantics import boundary,Boundary
from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper
from acoserl.bimanual_autoserl import BimanualAutoSERL,strict_geometry,DEFAULT,SafetyStop

class StrictTaskProtocol:
 version='assembly_real_grippers_stable3_v1'
 def __init__(self,stable_steps=3):self.required=stable_steps;self.reset()
 def reset(self):self.streak=0
 def observe(self,raw):
  strict,official,held=strict_geometry(raw)
  self.streak=self.streak+1 if strict else 0
  stable=self.streak>=self.required
  return {'success':stable,'strict_instant_success':strict,'legacy_success':official,
          'strict_success_streak':self.streak,'actual_grasps':held,'reward_protocol':self.version}

class StrictAssemblyEnv(RobosuiteGymWrapper):
 def __init__(self,**kwargs):
  self.protocol=StrictTaskProtocol();self.action_runner=None;super().__init__('TwoArmThreePieceAssembly',**kwargs)
 def reset(self,**kwargs):
  self.protocol.reset();return super().reset(**kwargs)
 def step(self,action):
  if self.action_runner is None:
   obs,legacy_reward,legacy_term,legacy_trunc,info=super().step(action)
  else:
   self.action_runner.step(action)
   obs=self._process_obs(self.env._get_observations(force_update=True))
   legacy_reward=float(self.env.reward());info={};self.episode_steps+=1
  result=self.protocol.observe(self.env);term=result['success']
  trunc=bool(self.env.timestep>=self.env.horizon) and not term
  self.episode_steps=int(self.env.timestep)
  return obs,float(term),term,trunc,{**info,**result,'legacy_reward':legacy_reward,'episode_steps':self.episode_steps}

class SafeStopBeforeStep(RuntimeError):
 def __init__(self,decision):self.decision=decision;super().__init__(str(decision.get('reason')))

class UnsafePartialStep(SafeStopBeforeStep):
 """Some, but not all, live physics substeps advanced; no complete row exists."""
 pass

class InterventionBridge:
 """One manager per env; supervision uses physical controller coordinates."""
 def __init__(self,env,action_scaler,intervention_enabled=True,initial_progress=(0,0),config=None):
  self.env=env;self.scaler=action_scaler
  self.manager=BimanualAutoSERL(env.env,config,intervention_enabled)
  self.manager.action_bounds=(self.scaler.limits.min.numpy().copy(),self.scaler.limits.max.numpy().copy())
  self.manager.reset(initial_progress)
 def reset(self,**kwargs):
  obs,info=self.env.reset(**kwargs);self.manager.reset();return obs,info
 def step(self,policy_action):
  policy=torch.as_tensor(policy_action,dtype=torch.float32).cpu()
  physical=self.scaler.unscale(policy).numpy()
  final,decision=self.manager.choose(physical)
  if final is None:raise SafeStopBeforeStep(decision)
  minimum=self.scaler.limits.min.numpy();maximum=self.scaler.limits.max.numpy()
  if np.any(final<minimum-1e-6) or np.any(final>maximum+1e-6):
   self.manager.stop('recovery_action_outside_offline_normalization_domain')
   raise SafeStopBeforeStep({'mode':'Safe_Stop','reason':self.manager.stop_reason,'proposed_action':final.tolist()})
  if decision['mode']=='Normal':
   assert np.array_equal(final,physical)
   executed_normalized=policy.clone();inverse=physical.copy()
  else:
   executed_normalized=self.scaler.scale(torch.tensor(final,dtype=torch.float32))
   inverse=self.scaler.unscale(executed_normalized).numpy()
  assert np.max(abs(inverse-final))<2e-6
  start_time=float(self.env.env.sim.data.time)
  try:
   obs,reward,term,trunc,info=self.env.step(inverse)
  except SafetyStop as error:
   self.manager.stop('physics_substep_safety:'+str(error))
   partial=bool(float(self.env.env.sim.data.time)>start_time)
   payload={**decision,'interrupted_mode':decision['mode'],'mode':'Safe_Stop','reason':self.manager.stop_reason,
     'partial_physics_step':partial,'physical_time':float(self.env.env.sim.data.time),
     'qpos_partial':self.env.env.sim.data.qpos.tolist(),'proposed_action':inverse.tolist(),
     'physics_failure_evidence':getattr(error,'evidence',{})}
   # Preview can reject BEFORE the first live substep. Do not count a shadow
   # calculation as an attempted live physical interaction.
   raise (UnsafePartialStep(payload) if partial else SafeStopBeforeStep(payload)) from error
  after=self.manager.after_step()
  proposal=np.asarray(decision.get('intervention_action_proposed',final),dtype=np.float32)
  info['intervention_action_normalized']=self.scaler.scale(torch.from_numpy(proposal)).numpy()
  info['task_quality']=after.get('task_quality',{})
  info['strict_task_success']=bool(info['success'])
  info['success']=bool(info['success'] and self.manager.stop_reason is None)
  reward=float(info['success'])
  kind,bootstrap=boundary(term,trunc,info['success'],self.manager.stop_reason)
  if self.manager.mode=='Safe_Stop' and not info['success']:
   term=kind==Boundary.PHYSICAL_FAILURE;trunc=not term
  runner=self.env.action_runner
  if runner is not None and runner.last_attempt is not None:
   attempt=runner.last_attempt
   decision['physics_execution']=dict(complete=attempt['complete'],
       executed_substeps=attempt['executed_substeps'],
       max_robot_force_N=max((s['robot_force_N'] for s in attempt['substeps']),default=0.),
       preview=runner.preview.audit() if runner.preview else None)
  info.update(boundary_kind=int(kind),bootstrap_allowed=bootstrap)
  info.update({'auto_intervention':decision,'recovery_result':after,
    'policy_action_normalized':policy.numpy(),'policy_action_physical':physical,
    'executed_action_physical':inverse,'executed_action_normalized':executed_normalized,
    'supervisor_mode_after':self.manager.mode,'supervisor_stop_reason':self.manager.stop_reason})
  return obs,reward,term,trunc,info
