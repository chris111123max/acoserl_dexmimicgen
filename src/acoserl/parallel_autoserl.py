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
  from acoserl.single_demo_rlpd import normalize
  from resfit.rl_finetuning.utils.normalization import StateStandardizer,ActionScaler
  self.torch=torch;self.normalize=normalize;self.cfg=cfg;self.env_id=cfg.env_id;self.directory=Path(cfg.directory)
  self.directory.mkdir(parents=True,exist_ok=True);self.log=self.directory/f'env{self.env_id:02d}.jsonl'
  self.stats={'env_id':self.env_id,'pid':os.getpid(),'seed':cfg.seed,'episodes':0,'resets':0,'real_steps':0,'proposals':0,'interventions':0,'recoveries':0,'safe_stops':0,'reasons':{},'episode_length_sum':0,'episode_length_histogram':{}}
  self.event('construct_start');np.random.seed(cfg.seed)
  self.w=StrictAssemblyEnv(camera_size=84,env_id=self.env_id)
  s=cfg.normalization;self.standardizer=StateStandardizer(torch.tensor(s['state_mean']),torch.tensor(s['state_std_population']),min_std=.1)
  self.scaler=ActionScaler(torch.tensor(s['effective_action_min']),torch.tensor(s['effective_action_max']),action_scale=0,min_range_per_dim=.1)
  self.bridge=InterventionBridge(self.w,self.scaler,intervention_enabled=cfg.intervention_enabled)
  self.action_space=gym.spaces.Box(-1,1,(14,),np.float32);self.observation_space=spaces()
  self.episode=-1;self.frame=0;self.dead=True;self.forced=False;self.runner=None
  if cfg.resume:
   for key in self.stats:
    if key not in ['pid','env_id','seed'] and key in cfg.resume:self.stats[key]=copy.deepcopy(cfg.resume[key])
   self.episode=int(cfg.resume['episode_id'])
   self.bridge.manager.clean_successes=int(cfg.resume.get('clean_successes',0));self.bridge.manager.auto_disabled=bool(cfg.resume.get('auto_disabled',False))
  self.event('construct_done')
 def event(self,event,**extra):
  import resource
  row={'event':event,'time':time.time(),'env_id':self.env_id,'pid':os.getpid(),'rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**extra}
  with self.log.open('a') as f:f.write(json.dumps(row,default=str)+'\n')
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
  self.runner.safety=lambda c:'robot_force_limit' if c['max_robot_contact_force_N']>80 else 'dynamic_A_B_chain' if c['dynamic_A_B_coupled'] else None
  self.bridge.manager.reset();self.frame=0;self.dead=False;self.forced=False;self.stats['resets']+=1
  self.last=self.obs(self.w._process_obs(self.w.env._get_observations(force_update=True)))
  if not hasattr(self,'canonical_observation'):self.canonical_observation={k:v.copy() for k,v in self.last.items()}
  else:
   for k,v in self.last.items():assert np.array_equal(v,self.canonical_observation[k]),'canonical observation differs: '+k
  self.event('reset_ready',episode=self.episode,seed=chosen,npu_plugin_loaded='torch_npu' in sys.modules,reset_protocol='canonical_demo0_frame0_only')
  return self.last,{'env_id':self.env_id,'episode_id':self.episode,'pid':os.getpid(),'npu_plugin_loaded':False}
 def force_stop(self,ids):
  if self.env_id in ids:self.forced=True
 def checkpoint_state(self):
  return {**self.diagnostic(),'rng_numpy':np.random.get_state(),'rng_torch':self.torch.get_rng_state().tolist(),'rng_python':random.getstate(),'rng_gym':self.np_random.bit_generator.state,
   'clean_successes':self.bridge.manager.clean_successes,'auto_disabled':self.bridge.manager.auto_disabled,
   'supervisor_snapshot':{'mode':self.bridge.manager.mode,'indices':self.bridge.manager.indices(),'cooldown':self.bridge.manager.cooldown,'steps':self.bridge.manager.steps,'events':self.bridge.manager.events}}
 def collector_cut(self,reason):
  if not self.dead:
   self.dead=True;self.stats['episodes']+=1;self.stats['episode_length_sum']+=self.frame
   hist=self.stats['episode_length_histogram'];hist[str(self.frame)]=hist.get(str(self.frame),0)+1
   self.bridge.manager.episode_result(False,full_policy_episode=True)
   self.event('episode_end',episode=self.episode,length=self.frame,terminated=False,truncated=True,reason=reason)
 def diagnostic(self):return {**copy.deepcopy(self.stats),'episode_id':self.episode,'frame':self.frame,'dead':self.dead,'npu_plugin_loaded':'torch_npu' in sys.modules}
 def step(self,action):
  from acoserl.autoserl_env import SafeStopBeforeStep
  assert not self.dead,'explicit reset required';self.stats['proposals']+=1;start=time.perf_counter();before=self.w.env.sim.data.time
  executed=False;reward=0.;term=False;trunc=False;info={}
  try:
   if self.forced:raise SafeStopBeforeStep({'reason':'pressure_test_explicit_collector_cut'})
   raw,reward,term,trunc,info=self.bridge.step(action);executed=True
   self.last=self.obs(raw);self.frame+=1;self.stats['real_steps']+=1
   mode=info['auto_intervention']['mode'];after=info['recovery_result']
   if mode.startswith('Recover_') and info['auto_intervention'].get('recovery_step')==1:self.stats['interventions']+=1
   if after.get('handback'):self.stats['recoveries']+=1
  except SafeStopBeforeStep as error:
   trunc=True;info={'auto_intervention':error.decision,'supervisor_stop_reason':error.decision.get('reason'),'partial_physics_step':error.decision.get('partial_physics_step',False)}
  if term or trunc:
   self.dead=True;self.stats['episodes']+=1;self.stats['episode_length_sum']+=self.frame
   hist=self.stats['episode_length_histogram'];hist[str(self.frame)]=hist.get(str(self.frame),0)+1
   reason=info.get('supervisor_stop_reason')
   if reason:
    self.stats['safe_stops']+=1;self.stats['reasons'][reason]=self.stats['reasons'].get(reason,0)+1
   self.bridge.manager.episode_result(bool(term),full_policy_episode=True)
   self.event('episode_end',episode=self.episode,length=self.frame,terminated=bool(term),truncated=bool(trunc),reason=reason,stats=self.diagnostic())
  selected={k:info.get(k) for k in ['auto_intervention','recovery_result','supervisor_stop_reason','partial_physics_step']}
  if executed:
   selected.update(policy_action=np.asarray(action).tolist(),executed_action=info['executed_action_normalized'].numpy().tolist(),executed_action_physical=info['executed_action_physical'].tolist())
  from acoserl.bimanual_autoserl import snapshot
  task_metrics=snapshot(self.w.env)['official_metrics']
  payload={'env_id':self.env_id,'episode_id':self.episode,'frame':self.frame-1 if executed else self.frame,'executed':executed,
   'physics_time_advanced':float(self.w.env.sim.data.time-before),'step_seconds':time.perf_counter()-start,'task_metrics':task_metrics,**selected}
  self.event('step',**{k:v for k,v in payload.items() if k!='env_id'},reward=float(reward),terminated=bool(term),truncated=bool(trunc))
  return self.last,float(reward),bool(term),bool(trunc),{'payload':json.dumps(payload,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))}
 def release_guard(self):
  assert self.runner.physical_writes_attempted==0
  for obj,key,method in self.guard_restore:setattr(obj,key,method)
  self.guard_restore=[];self.runner=None;self.w.action_runner=None
 def close(self):
  if hasattr(self,'runner') and self.runner is not None:self.release_guard()
  if hasattr(self,'w'):self.w.close()

class FourByFour:
 def __init__(self,seed,normalization,directory,enabled=True,resume_workers=None):
  self.groups=[];self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False);self.startup=self.directory/'startup.jsonl'
  try:
   for group in range(4):
    start=time.time();self.record('batch_start',group=group,ids=list(range(4*group,4*group+4)))
    factories=[WorkerFactory(i,seed+i*1009,normalization,str(self.directory),enabled,None if resume_workers is None else resume_workers[i]) for i in range(4*group,4*group+4)]
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
 def checkpoint_states(self):return [d for v in self.groups for d in v.call('checkpoint_state')]
 def cut_all(self,reason):
  for v in self.groups:v.call('collector_cut',reason)
 def diagnostics(self):return [d for v in self.groups for d in v.call('diagnostic')]
 def force_stop(self,ids):
  for v in self.groups:v.call('force_stop',ids)
 def close(self):
  for v in self.groups:
   try:v.close(timeout=30)
   except Exception:v.close(terminate=True)
  self.groups=[]
