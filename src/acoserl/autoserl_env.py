"""Explicit-reset, one-environment integration and versioned strict task semantics."""
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
 """A monitored 2-ms integration step advanced; no full transition may be invented."""
 pass

class InterventionBridge:
 """Each instance owns one manager; vector auto-reset is intentionally unsupported.

 Incoming and stored actions use official normalized coordinates. Supervision
 uses physical controller input coordinates. It does not change Actor observations.
 """
 def __init__(self,env,action_scaler,intervention_enabled=True,initial_progress=(0,0),config=None):
  self.env=env;self.scaler=action_scaler
  self.manager=BimanualAutoSERL(env.env,config,intervention_enabled)
  # Recovery commands obey the same physical action domain as policy and offline rows.
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
   # Preserve exact legacy numerical behavior: do not scale/unscale policy actions twice.
   assert np.array_equal(final,physical)
   executed_normalized=policy.clone();inverse=physical.copy()
  else:
   executed_normalized=self.scaler.scale(torch.tensor(final,dtype=torch.float32))
   inverse=self.scaler.unscale(executed_normalized).numpy()
  assert np.max(abs(inverse-final))<2e-6
  # Execute the same float32 controller inputs represented in Replay Buffer.
  try:
   obs,reward,term,trunc,info=self.env.step(inverse)
  except SafetyStop as error:
   self.manager.stop('physics_substep_safety:'+str(error))
   raise UnsafePartialStep({**decision,'interrupted_mode':decision['mode'],'mode':'Safe_Stop','reason':self.manager.stop_reason,
    'partial_physics_step':True,'physical_time':float(self.env.env.sim.data.time),
    'qpos_partial':self.env.env.sim.data.qpos.tolist(),'proposed_action':inverse.tolist()}) from error
  after=self.manager.after_step()
  kind,bootstrap=boundary(term,trunc,info['success'],self.manager.stop_reason)
  if self.manager.mode=='Safe_Stop' and not info['success']:
   term=kind==Boundary.PHYSICAL_FAILURE;trunc=not term
  info.update(boundary_kind=int(kind),bootstrap_allowed=bootstrap)
  info.update({'auto_intervention':decision,'recovery_result':after,
   'policy_action_normalized':policy.numpy(),'policy_action_physical':physical,
   'executed_action_physical':inverse,'executed_action_normalized':executed_normalized,
   'supervisor_mode_after':self.manager.mode,'supervisor_stop_reason':self.manager.stop_reason})
  return obs,reward,term,trunc,info
