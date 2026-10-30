"""Four asynchronously stepped groups; only environment workers own MuJoCo.

Gymnasium's parent-side dummy factory supplies spaces only, avoiding a seventeenth
simulator and keeping all sixteen actual simulators out of the NPU process.
"""
import os,time,json,sys,multiprocessing,copy,random
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
IMAGES=['observation.images.'+k for k in ['agentview','robot0_eye_in_hand','robot1_eye_in_hand']]

def spaces():
 return gym.spaces.Dict({'observation.state':gym.spaces.Box(-np.inf,np.inf,(18,),np.float32),
  **{k:gym.spaces.Box(0,255,(3,84,84),np.uint8) for k in IMAGES}})
class SpaceOnly(gym.Env):
 observation_space=spaces();action_space=gym.spaces.Box(-1,1,(14,),np.float32)
 metadata={'render_modes':['rgb_array'],'render_fps':20};render_mode='rgb_array'
 def close(self):pass

@dataclass
class WorkerFactory:
 env_id:int
 seed:int
 normalization:dict
 directory:str
 intervention_enabled:bool=True
 resume:dict=None
 manager_config:dict=None
 def __call__(self):
  if multiprocessing.current_process().name=='MainProcess':return SpaceOnly()
  return AutoSERLWorker(self)

class AutoSERLWorker(gym.Env):
 metadata={'render_modes':['rgb_array'],'render_fps':20};render_mode='rgb_array'
 def __init__(self,cfg):
  for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
  os.environ['LP_NUM_THREADS']='1';os.environ['MESA_GLTHREAD']='false'
  os.environ['ASCEND_RT_VISIBLE_DEVICES']='';os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa'
  import torch
  torch.set_num_threads(1);torch.manual_seed(cfg.seed);random.seed(cfg.seed)
  assert 'torch_npu' not in sys.modules,'worker initialized NPU plugin'
  from acoserl.autoserl_env import StrictAssemblyEnv,InterventionBridge
  from acoserl.bimanual_autoserl import DEFAULT
  from acoserl.single_demo_rlpd import normalize
  from resfit.rl_finetuning.utils.normalization import StateStandardizer,ActionScaler
  self.torch=torch;self.normalize=normalize;self.cfg=cfg;self.env_id=cfg.env_id;self.directory=Path(cfg.directory)
  self.directory.mkdir(parents=True,exist_ok=True);self.log=self.directory/f'env{self.env_id:02d}.jsonl'
  self.stats={'env_id':self.env_id,'pid':os.getpid(),'seed':cfg.seed,'episodes':0,'resets':0,'real_steps':0,'proposals':0,'interventions':0,'recoveries':0,'safe_stops':0,'reasons':{},'episode_length_sum':0,'episode_length_histogram':{},'assisted_successes':0,'autonomous_successes':0,'intervention_steps':0}
  self.event('construct_start');np.random.seed(cfg.seed)
  self.w=StrictAssemblyEnv(camera_size=84,env_id=self.env_id)
  s=cfg.normalization;self.standardizer=StateStandardizer(torch.tensor(s['state_mean']),torch.tensor(s['state_std_population']),min_std=.1)
  self.scaler=ActionScaler(torch.tensor(s['effective_action_min']),torch.tensor(s['effective_action_max']),action_scale=0,min_range_per_dim=.1)
  self.bridge=InterventionBridge(self.w,self.scaler,intervention_enabled=cfg.intervention_enabled,config={**DEFAULT,'manager':cfg.manager_config or {}})
  self.action_space=gym.spaces.Box(-1,1,(14,),np.float32);self.observation_space=spaces()
  self.preparing=False;self.prepared_episode=False;self.episode=-1;self.frame=0;self.dead=True;self.forced=False;self.runner=None
  if cfg.resume:
   for key in self.stats:
    if key not in ['pid','env_id','seed'] and key in cfg.resume:self.stats[key]=copy.deepcopy(cfg.resume[key])
   self.episode=int(cfg.resume['episode_id'])
   self.bridge.manager.set_global_capability(cfg.resume.get('global_capability',{}))
   self.bridge.manager.autonomy=copy.deepcopy(cfg.resume.get('stage_autonomy',{}))
   self.bridge.manager.clean_full_episodes=int(cfg.resume.get('clean_full_episodes',0))
   self.bridge.manager.takeover_sequence=int(cfg.resume.get('takeover_sequence',0))
   self.bridge.manager.clean_successes=int(cfg.resume.get('clean_successes',0));self.bridge.manager.auto_disabled=bool(cfg.resume.get('auto_disabled',False))
  self.event('construct_done')
 def event(self,event,**extra):
  import resource
  row={'event':event,'time':time.time(),'env_id':self.env_id,'pid':os.getpid(),'rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**extra}
  with self.log.open('a') as f:f.write(json.dumps(row,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
 def obs(self,raw):return {k:v.numpy() for k,v in self.normalize(raw,self.standardizer).items()}
 def reset(self,*,seed=None,options=None):
  from acoserl.bimanual_autoserl import ActionOnly
  # Reset is an explicit episode boundary. Restore canonical frame0 only here.
  from acoserl.data import load_demo
  from acoserl.envs import restore_initial
  self.episode+=1;chosen=self.cfg.seed+self.episode*100003 if seed is None else int(seed)
  np.random.seed(chosen);self.torch.manual_seed(chosen);random.seed(chosen);super().reset(seed=chosen)
  if self.runner is not None:self.release_guard()
  root=Path(__file__).resolve().parents[2]
  if not hasattr(self,'demo'):
   self.demo=load_demo(root/'datasets/generated/two_arm_three_piece_assembly.hdf5','demo_0')
   self.w.reset(seed=chosen);restore_initial(self.w.env,self.demo)
   self.canonical_model=self.w.env.sim.model
   self.canonical_body_pos=self.canonical_model.body_pos.copy()
   self.canonical_body_quat=self.canonical_model.body_quat.copy()
  else:
   # Explicit NEW episode only: keep identical XML and its OSMesa context.
   # Official deterministic soft reset reinitializes controllers and grippers.
   from acoserl.envs import official_playback
   raw_env=self.w.env;raw_env.hard_reset=False;raw_env.deterministic_reset=True
   try:self.w.reset(seed=chosen)
   finally:raw_env.deterministic_reset=False
   raw_env.sim.reset()
   official_playback().reset_to(raw_env,{'states':self.demo['states'][0]})
   assert raw_env.sim.model is self.canonical_model
   assert np.array_equal(raw_env.sim.model.body_pos,self.canonical_body_pos)
   assert np.array_equal(raw_env.sim.model.body_quat,self.canonical_body_quat)
  assert np.array_equal(self.w.env.sim.get_state().flatten(),self.demo['states'][0]),'canonical reset state differs'
  self.w.protocol.reset();self.w.episode_steps=0
  self.guard_restore=[(self.w.env.sim,k,getattr(self.w.env.sim,k)) for k in ['set_state','set_state_from_flattened','reset']]+[(self.w.env,k,getattr(self.w.env,k)) for k in ['reset','reset_from_xml_string']]
  self.runner=ActionOnly(self.w.env);self.w.action_runner=self.runner
  self.runner.safety=self.bridge.manager.substep_safety
  self.prepared_episode=False;self.bridge.manager.reset();self.frame=0;self.dead=False;self.forced=False;self.stats['resets']+=1
  self.previous_eef_positions=np.asarray([p['position'] for p in self.bridge.manager.initial_snapshot['poses']])
  self.last=self.obs(self.w._process_obs(self.w.env._get_observations(force_update=True)))
  if not hasattr(self,'canonical_observation'):self.canonical_observation={k:v.copy() for k,v in self.last.items()}
  else:
   for k,v in self.last.items():assert np.array_equal(v,self.canonical_observation[k]),'canonical observation differs: '+k
  self.event('reset_ready',episode=self.episode,seed=chosen,npu_plugin_loaded='torch_npu' in sys.modules,reset_protocol='canonical_demo0_frame0_only')
  return self.last,{'env_id':self.env_id,'episode_id':self.episode,'pid':os.getpid(),'npu_plugin_loaded':False}
 def apply_capability(self,value):
  self.bridge.manager.set_global_capability(value)
 def force_stop(self,ids):
  if self.env_id in ids:self.forced=True
 def checkpoint_state(self):
  return {**self.diagnostic(),'rng_numpy':np.random.get_state(),'rng_torch':self.torch.get_rng_state().tolist(),'rng_python':random.getstate(),'rng_gym':self.np_random.bit_generator.state,
   'stage_autonomy':copy.deepcopy(self.bridge.manager.autonomy),
   'clean_full_episodes':self.bridge.manager.clean_full_episodes,
   'takeover_sequence':self.bridge.manager.takeover_sequence,
   'clean_successes':self.bridge.manager.clean_successes,'auto_disabled':self.bridge.manager.auto_disabled,
   'local_recovery_context':self.bridge.manager.local.checkpoint_state() if self.bridge.manager.local is not None else None,
   'v3_context':self.bridge.manager.v3_checkpoint_state(),
   'global_capability':copy.deepcopy(self.bridge.manager.global_capability),
   'supervisor_snapshot':{'mode':self.bridge.manager.mode,'indices':self.bridge.manager.indices(),'cooldown':self.bridge.manager.cooldown,'steps':self.bridge.manager.steps,'events':self.bridge.manager.events,'windows':[vars(w).copy() for w in self.bridge.manager.windows],'recovery_stage':self.bridge.manager.recovery_stage,'intervention_control_steps':self.bridge.manager.intervention_control_steps}}
 def collector_cut(self,reason):
  if not self.dead:
   self.dead=True;self.stats['episodes']+=1;self.stats['episode_length_sum']+=self.frame
   hist=self.stats['episode_length_histogram'];hist[str(self.frame)]=hist.get(str(self.frame),0)+1
   outcome=self.bridge.manager.episode_result(False,full_policy_episode=not self.prepared_episode)
   self.event('episode_end',episode=self.episode,length=self.frame,terminated=False,truncated=True,reason=reason)
   return dict(env_id=self.env_id,episode_id=self.episode,episode_outcome=outcome,collector_cut=True)
 def diagnostic(self):return {**copy.deepcopy(self.stats),'episode_id':self.episode,'frame':self.frame,'dead':self.dead,'npu_plugin_loaded':'torch_npu' in sys.modules}
 def step(self,action):
  from acoserl.autoserl_env import SafeStopBeforeStep
  assert not self.dead,'explicit reset required';self.stats['proposals']+=1;start=time.perf_counter();before=self.w.env.sim.data.time
  executed=False;reward=0.;term=False;trunc=False;info={};risk_state=self.last['observation.state'].tolist()
  try:
   if self.forced:raise SafeStopBeforeStep({'reason':'pressure_test_explicit_collector_cut','collector_cut':True})
   raw,reward,term,trunc,info=self.bridge.step(action);executed=True
   self.last=self.obs(raw);self.frame+=1;self.stats['real_steps']+=1
   mode=info['auto_intervention']['mode'];after=info['recovery_result']
   if mode.startswith(('Recover_','Guide_','Constrained_')):self.stats['intervention_steps']+=1
   if mode.startswith('Constrained_'):self.stats['interventions']+=1
   if mode.startswith(('Recover_','Guide_')) and info['auto_intervention'].get('recovery_step')==1:self.stats['interventions']+=1
   if after.get('handback'):self.stats['recoveries']+=1
  except SafeStopBeforeStep as error:
   trunc=True;info={'auto_intervention':error.decision,'supervisor_stop_reason':error.decision.get('reason'),'partial_physics_step':error.decision.get('partial_physics_step',False),'collector_cut':error.decision.get('collector_cut',False)}
  if term or trunc:
   self.dead=True;self.stats['episodes']+=1;self.stats['episode_length_sum']+=self.frame
   hist=self.stats['episode_length_histogram'];hist[str(self.frame)]=hist.get(str(self.frame),0)+1
   reason=info.get('supervisor_stop_reason')
   if reason:
    self.stats['safe_stops']+=1;self.stats['reasons'][reason]=self.stats['reasons'].get(reason,0)+1
   outcome=self.bridge.manager.episode_result(bool(info.get('success',False)),full_policy_episode=not self.prepared_episode)
   self.stats['assisted_successes']+=int(outcome['assisted_success'])
   self.stats['autonomous_successes']+=int(outcome['autonomous_success'])
   self.event('intervention_episode',episode=self.episode,**outcome,decisions=self.bridge.manager.decision_stats,v3_stats=copy.deepcopy(self.bridge.manager.v3_stats))
   self.event('episode_end',episode=self.episode,length=self.frame,terminated=bool(term),truncated=bool(trunc),reason=reason,stats=self.diagnostic())
  selected={k:info.get(k) for k in ['auto_intervention','recovery_result','supervisor_stop_reason','partial_physics_step','success','strict_task_success','intervention_action_normalized','task_quality']}
  selected['policy_action']=np.asarray(action).tolist()
  if self.preparing and executed:selected['diagnostic_preparation']=True
  if executed:
   selected.update(policy_action=np.asarray(action).tolist(),executed_action=info['executed_action_normalized'].numpy().tolist(),executed_action_physical=info['executed_action_physical'].tolist())
  from acoserl.bimanual_autoserl import snapshot
  supervision_state=snapshot(self.w.env);task_metrics=supervision_state['official_metrics']
  from acoserl.transition_semantics import boundary
  kind,bootstrap=boundary(term,trunc,bool(info.get('success',False)),info.get('supervisor_stop_reason'),executed,info.get('partial_physics_step',False))
  if info.get('collector_cut'):kind,bootstrap=6,True
  decision=info.get('auto_intervention',{})
  payload={'stage':decision.get('stage','unknown'),'takeover_id':decision.get('takeover_id',0),
   'action_source':('Policy' if decision.get('mode','Normal')=='Normal' else decision['mode']) if executed else 'Rejected',
   'complete_physics':executed,'boundary_kind':int(kind),'bootstrap_allowed':bootstrap,
   'env_id':self.env_id,'episode_id':self.episode,'frame':self.frame-1 if executed else self.frame,'executed':executed,
   'physics_time_advanced':float(self.w.env.sim.data.time-before),'step_seconds':time.perf_counter()-start,'task_metrics':task_metrics,'supervisor_objects':supervision_state['objects'],'supervisor_grasps':supervision_state['actual_grasps'],'eef_positions':[p['position'].tolist() for p in supervision_state['poses']],
   'eef_rotations':[p['rotation'].tolist() for p in supervision_state['poses']],
   'gripper_commands':[r.gripper['right'].current_action.tolist() for r in self.w.env.robots],
   'supervisor_contacts':self.bridge.manager.contacts.read(),'episode_outcome':getattr(self.bridge.manager,'last_episode_outcome',None) if term or trunc else None,**selected}
  if self.preparing and payload['action_source']=='Policy':payload['action_source']='DiagnosticPreparation'
  payload['prepared_episode']=self.prepared_episode
  from acoserl.action_provenance import describe
  from acoserl.transition_semantics import source_code
  if executed:
   roles=describe(12 if payload['action_source']=='DiagnosticPreparation' else source_code(decision.get('mode','Normal')),action,payload['executed_action'],
                  decision.get('active_arm'),bool((payload.get('task_quality') or {}).get('eligible')))
   payload['arm_control']=roles
  positions=np.asarray(payload['eef_positions'])
  motion=np.linalg.norm(positions-self.previous_eef_positions,axis=1)
  payload['eef_translation_m']=motion.tolist();payload['nonzero_physical_motion']=bool(executed and np.any(motion>1e-5))
  self.previous_eef_positions=positions.copy()
  self.event('step',**{k:v for k,v in payload.items() if k!='env_id'},reward=float(reward),terminated=bool(term),truncated=bool(trunc))
  if not executed or any(r.get('risk') for r in decision.get('policy_risks',[])):
   risk_row=dict(env_id=self.env_id,episode_id=self.episode,frame=payload['frame'],state_before=risk_state,policy_action=np.asarray(action).tolist(),executed_action=payload.get('executed_action'),risk=decision.get('policy_risks'),reason=payload.get('supervisor_stop_reason'),complete_physics=executed,partial=payload.get('partial_physics_step'),source=payload['action_source'])
   with (self.directory/f'env{self.env_id:02d}_risks.jsonl').open('a') as f:f.write(json.dumps(risk_row)+'\n')
  return self.last,float(reward),bool(term),bool(trunc),{'payload':json.dumps(payload,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))}
 def prepare_pregrasp(self,plans):
  """Integration-only real fixture actions, all returned to Online with provenance.

  Starts at canonical0; never changes physical state or Actor observation shape.
  This is not a successful policy episode and is forbidden in formal training.
  """
  plan=plans.get(self.env_id)
  if plan is None:return None
  assert plan['purpose']=='bounded_integration' and self.frame==0 and not self.dead
  from acoserl.bimanual_autoserl import snapshot
  from common import servo
  arm=plan['arm'];ai='AB'.index(arm);prefix=30 if ai==0 else 58
  m=self.bridge.manager;enabled=m.cfg['intervention_enabled'];m.cfg['intervention_enabled']=False
  self.preparing=True;self.prepared_episode=True;records=[]
  def execute(physical):
   old={k:v.copy() for k,v in self.last.items()}
   action=self.scaler.scale(self.torch.tensor(physical,dtype=self.torch.float32)).numpy()
   nxt,reward,term,trunc,info=self.step(action);payload=json.loads(info['payload'])
   assert payload['executed'] and not term and not trunc,payload
   records.append(dict(obs=old,action=payload['executed_action'],next_obs={k:v.copy() for k,v in nxt.items()},reward=reward,terminated=term,truncated=trunc,metadata=payload))
  try:
   for action in self.demo['actions'][:prefix]:execute(action)
   state=snapshot(self.w.env);targets=copy.deepcopy(state['poses'])
   grips=[1. if any(state['actual_grasps'][a].values()) else -1. for a in 'AB']
   tangent=m.positions[prefix+1,ai]-m.positions[prefix,ai]
   targets[ai]['position']+=.003*tangent/max(np.linalg.norm(tangent),1e-9)
   for _ in range(100):execute(servo(self.w.env,targets,grips,max_position_step=.006,position_gain=2))
   m.reset((prefix,prefix))
   # Actual stationary observations, not fabricated stagnation history.
   for _ in range(20):
    execute(servo(self.w.env,targets,grips,max_position_step=.006,position_gain=2))
    now=snapshot(self.w.env)
    for j in range(2):m.window_measure(j,now)
   for key in m.episode_touched:m.episode_touched[key]=True
   self.event('bounded_preparation_complete',arm=arm,actual_controls=len(records),canonical_prefix=prefix,policy_episode=False)
   return dict(env_id=self.env_id,records=records,observation={k:v.copy() for k,v in self.last.items()},hold_gate=m.passive_hold_entry(ai,snapshot(self.w.env)))
  finally:self.preparing=False;m.cfg['intervention_enabled']=enabled
 def release_guard(self):
  assert self.runner.physical_writes_attempted==0
  for obj,key,method in self.guard_restore:setattr(obj,key,method)
  self.guard_restore=[];self.runner=None;self.w.action_runner=None
 def close(self):
  if hasattr(self,'runner') and self.runner is not None:self.release_guard()
  if hasattr(self,'w'):self.w.close()

class FourByFour:
 def __init__(self,seed,normalization,directory,enabled=True,resume_workers=None,manager_config=None):
  self.groups=[];self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False);self.startup=self.directory/'startup.jsonl'
  try:
   for group in range(4):
    start=time.time();self.record('batch_start',group=group,ids=list(range(4*group,4*group+4)))
    factories=[WorkerFactory(i,seed+i*1009,normalization,str(self.directory),enabled,None if resume_workers is None else resume_workers[i],manager_config) for i in range(4*group,4*group+4)]
    v=gym.vector.AsyncVectorEnv(factories,shared_memory=True,context='spawn',autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    self.groups.append(v);obs,infos=v.reset()
    diagnostics=v.call('diagnostic');assert len(diagnostics)==4 and all(not d['npu_plugin_loaded'] and d['resets']>=1 for d in diagnostics)
    self.record('batch_ready',group=group,seconds=time.time()-start,workers=diagnostics)
   self.observation={k:np.concatenate([v.observations[k] for v in self.groups]) for k in IMAGES+['observation.state']}
   self.record('all_sixteen_ready',workers=self.diagnostics())
  except BaseException:self.close();raise
 def record(self,event,**extra):
  with self.startup.open('a') as f:f.write(json.dumps({'time':time.time(),'event':event,**extra},default=str)+'\n')
 def step(self,actions):
  assert np.shape(actions)==(16,14)
  dispatch=[]
  for g,v in enumerate(self.groups):dispatch.append(time.time());v.step_async(actions[g*4:g*4+4])
  result=[v.step_wait(timeout=300) for v in self.groups]
  self.last_dispatch=dispatch
  obs={k:np.concatenate([r[0][k] for r in result]) for k in self.observation}
  rewards=np.concatenate([r[1] for r in result]);terms=np.concatenate([r[2] for r in result]);truncs=np.concatenate([r[3] for r in result])
  payload=[json.loads(s) for r in result for s in r[4]['payload']]
  self.observation=obs;return obs,rewards,terms,truncs,payload
 def reset_done(self,mask):
  pending=[]
  for g,v in enumerate(self.groups):
   sub=np.asarray(mask[g*4:g*4+4],dtype=bool)
   if sub.any():v.reset_async(options={'reset_mask':sub});pending.append((g,v))
  for g,v in pending:
   obs,_=v.reset_wait(timeout=300)
   for k in self.observation:self.observation[k][g*4:g*4+4]=obs[k]
  return self.observation
 def prepare_pregrasp(self,plans):
  results=[d for v in self.groups for d in v.call('prepare_pregrasp',plans) if d is not None]
  for item in results:
   for key,value in item['observation'].items():self.observation[key][item['env_id']]=value
  self.record('bounded_diagnostic_preparation',workers=[{k:v for k,v in r.items() if k not in ('records','observation')} for r in results],actual_controls=sum(len(r['records']) for r in results))
  return results
 def apply_capability(self,value):
  for v in self.groups:v.call('apply_capability',value)
 def checkpoint_states(self):return [d for v in self.groups for d in v.call('checkpoint_state')]
 def cut_all(self,reason):
  return [d for v in self.groups for d in v.call('collector_cut',reason) if d is not None]
 def diagnostics(self):return [d for v in self.groups for d in v.call('diagnostic')]
 def force_stop(self,ids):
  for v in self.groups:v.call('force_stop',ids)
 def close(self):
  for v in self.groups:
   try:v.close(timeout=30)
   except Exception:v.close(terminate=True)
  self.groups=[]
