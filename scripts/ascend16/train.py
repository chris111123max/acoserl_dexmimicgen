"""One-method, NPU-only learner; sixteen CPU environments, isolated three-step rows."""
import argparse,json,time,random,traceback,os,signal,subprocess,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def write(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.partial');temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)
def log(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,default=str)+'\n')

def evaluate(agent,normalization,out,step,deadline,torch,device,episodes=20,manager_config=None,max_steps=None):
 import gymnasium as gym
 from acoserl.parallel_autoserl import WorkerFactory,IMAGES
 from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
 from acoserl.ascend_checkpoint import rng_state,restore_rng
 saved_rng=rng_state();records=[];interactions=0;start=time.perf_counter()
 vector=gym.vector.AsyncVectorEnv([WorkerFactory(16,9000000+step,normalization,str(out/'evaluation_workers'),False,manager_config=manager_config)],context='spawn',autoreset_mode=gym.vector.AutoresetMode.DISABLED)
 try:
  for episode in range(episodes):
   if time.time()>=deadline:break
   obs,_=vector.reset(seed=9000000+step+episode*1009);done=False;length=0;success=False;last={};seen_grasps={a:False for a in ('A','B')};seen_first=False
   while not done and time.time()<deadline:
    data={k:torch.from_numpy(v).to(device) for k,v in obs.items()}
    with torch.no_grad(),eval_mode(agent):action=agent.act(data,eval_mode=True,cpu=True).numpy()
    obs,r,t,tr,info=vector.step(action);last=json.loads(info['payload'][0]);length+=int(last['executed']);interactions+=int(last['executed']);success=bool(last.get('success',False));done=bool(t[0] or tr[0])
    for arm,obj in [('A','piece_1'),('B','piece_2')]:seen_grasps[arm]|=bool(last.get('supervisor_grasps',{}).get(arm,{}).get(obj,False))
    seen_first|=bool(last.get('task_metrics',{}).get('first_piece_assembled',False))
    if max_steps is not None and length>=max_steps:break
   records.append({'episode':episode,'length':length,'success':success,'completed':done,'observed_grasps':seen_grasps,'observed_first_assembly':seen_first,'task_metrics':last.get('task_metrics',{}),'stop_reason':last.get('supervisor_stop_reason'),'boundary_kind':last.get('boundary_kind')})
 finally:vector.close();restore_rng(saved_rng)
 result={'training_step':step,'intervention_enabled':False,'episodes':records,'requested_episodes':episodes,'complete':len(records)==episodes and all(r['completed'] for r in records),
  'success_rate':sum(r['success'] for r in records)/len(records) if records else None,'fixed_canonical_initial_state':True,'independent_initial_states':False,'hard_safety_manager_config':manager_config,'actual_evaluation_interactions':interactions,'seconds':time.perf_counter()-start}
 write(out/f'evaluation_{step:07d}.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['stress','formal','integration'],required=True);p.add_argument('--tag',required=True);p.add_argument('--utd',type=int,choices=[1,2,4]);p.add_argument('--resume',type=Path);p.add_argument('--stress-steps',type=int,default=64);p.add_argument('--gate',type=Path,default=ROOT/'docs/ascend16/deployment_gate.json');p.add_argument('--visual',choices=['minivit','dinov2_vits14','resnet18_imagenet','squeezenet1_1_imagenet','resnet18_spatial84'],default='minivit');p.add_argument('--intervention-method',choices=['plain_rlpd','independent_window','window_recovery','full_acoserl'],default='full_acoserl');p.add_argument('--seed',type=int);p.add_argument('--integration-steps',type=int,default=256);p.add_argument('--integration-seconds',type=int,default=1200);p.add_argument('--demo-original-quota',type=int,choices=[64,96,128]);p.add_argument('--demo-positive-quota',type=int,choices=[0,2,4]);p.add_argument('--demo-admission',choices=['official','safe_handback']);p.add_argument('--integration-pregrasp',action='store_true');p.add_argument('--supervisor-version',choices=['v2','v3','baseline'],default='baseline');a=p.parse_args()
 assert not a.integration_pregrasp or (a.mode=='integration' and a.resume is None and a.supervisor_version=='baseline')
 assert a.tag.replace('_','').replace('-','').isalnum() and '/' not in a.tag
 stop_requested={'value':False}
 signal.signal(signal.SIGTERM,lambda *_:stop_requested.update(value=True))
 signal.signal(signal.SIGINT,lambda *_:stop_requested.update(value=True))
 config=json.loads((ROOT/'configs/ascend16/formal.json').read_text())
 config['purpose']=a.mode
 if a.seed is not None:
  assert a.mode=='integration','seed override is limited to independent bounded validation'
  config['seed']=a.seed
 config['visual_encoder']=a.visual
 from acoserl.intervention_manager import method_config
 manager_config=method_config(a.intervention_method)
 if a.supervisor_version=='baseline':manager_config.update(json.loads((ROOT/'configs/bimanual_autoserl/baseline.json').read_text()))
 if a.demo_original_quota is not None:manager_config['original_quota']=a.demo_original_quota
 if a.demo_positive_quota is not None:manager_config['positive_quota']=a.demo_positive_quota
 if a.demo_admission is not None:manager_config['admission_mode']=a.demo_admission
 if not manager_config['dynamic_demo_enabled']:manager_config['original_quota']=128
 manager_config['original_demo_fraction']=manager_config['original_quota']/128
 config['manager_config']=manager_config
 manager_config['v3_enabled']=a.supervisor_version in ('v3','baseline')
 config['integration_diagnostic_pregrasp']=a.integration_pregrasp
 config['collector_protocol']='autoserl_baseline_v1' if a.supervisor_version=='baseline' else 'acoserl_'+a.supervisor_version
 if a.visual!='minivit':config['initial_weights']='verified frozen public pretrained visual backbone; fresh random Actor/Critic/adapters; empty online replay; no smoke/old RL checkpoint'
 config['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if a.mode=='formal':
  from acoserl.baseline_gate import require_baseline
  assert a.supervisor_version=='baseline', 'formal training requires current single-task baseline acceptance'
  gate=require_baseline(a.gate,ROOT)
  config['utd']=gate['selected_utd'];assert a.utd in [None,config['utd']]
  if gate.get('selected_visual'):assert gate['selected_visual']==a.visual,'encoder disagrees with acceptance gate'
  if a.visual!='minivit':assert gate.get('selected_visual')==a.visual and gate.get('pretrained_validation_passed'), 'pretrained-specific acceptance gate required'
 elif a.mode=='integration':
  assert 16<=a.integration_steps<=2048 and 60<=a.integration_seconds<=1800
  assert a.utd in (None,1)
  config.update(utd=1,warmup_real_steps=0,online_capacity=2048,maximum_actual_interactions=a.integration_steps,wall_limit_seconds=a.integration_seconds)
 else:
  assert a.utd in [1,2,4] and 16<=a.stress_steps<=128
  config.update(utd=a.utd,warmup_real_steps=0,online_capacity=256,maximum_actual_interactions=a.stress_steps+16,wall_limit_seconds=1200)
 out=ROOT/'outputs/ascend16'/a.tag;out.mkdir(exist_ok=False)
 write(out/'config.json',config);started=time.time();deadline=started+config['wall_limit_seconds'];pool=None;telemetry=None
 result={'status':'initializing','config':config};c={'actual_interactions':0,'valid_real_steps':0,'partial_action_attempts':0,'warmup_real_steps':0,'critic_updates':0,'actor_updates':0,'proposals':0,'evaluation_interactions':0,'evaluation_seconds':0.}
 try:
  import torch,torch_npu,platform,mujoco
  from dataclasses import asdict
  from acoserl.ascend_data import offline,replay,raw_row
  from acoserl.isolated_nstep import IsolatedNStep
  from acoserl.dynamic_demo import DynamicDemo
  from acoserl.global_capability import GlobalCapability
  from acoserl.parallel_autoserl import FourByFour
  from acoserl.ascend_monitor import Telemetry
  from acoserl.ascend_checkpoint import save,load,restore_rng,verify_roundtrip
  from resfit.rl_finetuning.config.rlpd import QAgentConfig
  from acoserl.pretrained_visual import make_agent,manifest,digest
  from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
  from acoserl.single_demo_rlpd import IMAGES
  from acoserl.bimanual_autoserl import DEFAULT as supervisor_config
  config['runtime_versions']=dict(python=platform.python_version(),torch=torch.__version__,torch_npu=torch_npu.__version__,mujoco=mujoco.__version__)
  config['task_metric_counts_are_legacy_step_occurrences']=True
  config['supervisor_config']=supervisor_config
  config['recovery_points']=json.loads((ROOT/'configs/bimanual_autoserl/recovery_points.json').read_text())
  config['reference_sha256']=hashlib.sha256((ROOT/config['recovery_points']['reference_path']).read_bytes()).hexdigest()
  config['acceptance_gate']=str(a.gate)
  qualification=ROOT/('configs/bimanual_autoserl/qualification_'+a.supervisor_version+'.json')
  config['qualification']=json.loads(qualification.read_text()) if qualification.exists() else None
  from acoserl.v3_gate import CONTROL_FILES
  config['controller_hashes']={n:hashlib.sha256((ROOT/'src/acoserl'/n).read_bytes()).hexdigest() for n in CONTROL_FILES}
  assert torch.npu.is_available();device=torch.device('npu:0');torch.npu.set_device(device);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.config.allow_internal_format=False
  torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.npu.manual_seed_all(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
  telemetry=Telemetry(out/'npu_telemetry.jsonl')
  off,std,scale,norm=offline(config['batch_size']//2,config['gamma']);on=replay(config['online_capacity'],config['batch_size']//2)
  dynamic=DynamicDemo(manager_config['dynamic_capacity'],manager_config['original_demo_fraction'],3,config['gamma'],admission_mode=manager_config['admission_mode'],original_quota=manager_config['original_quota'],positive_quota=manager_config['positive_quota'],sampling_seed=config['seed'])
  agent=make_agent((3,84,84),(18,),14,IMAGES,QAgentConfig(device=str(device)),a.visual)
  visual_initial=manifest(agent);config['visual_manifest']=visual_initial
  write(out/'initialization.json',{'fresh_rl':a.resume is None,'fresh_online_replay':len(on)==0,'online_rows':len(on),'offline_rows':len(off),'actor_hash':digest(agent.actor),'critic_hash':digest(agent.critic),'visual':visual_initial})
  visual_snapshot={k:p.detach().cpu().clone() for k,p in agent.named_parameters() if p.requires_grad} if a.mode in ('stress','integration') else None
  assert all(t.device.type=='npu' for t in agent.parameters());assert agent.cfg.critic.num_q==10
  config['agent_config']=asdict(agent.cfg);config['normalization_frames']=224;write(out/'config.json',config);write(out/'normalization.json',norm)
  assembler=IsolatedNStep(3,config['gamma']);restored=None
  if a.resume:
   restored,assembler=load(a.resume,agent,on,off,norm,dynamic);assert restored['config']['purpose']==a.mode and restored['config']['utd']==config['utd']
   assert restored['config'].get('visual_encoder','minivit')==a.visual,'incompatible visual encoder on resume'
   assert restored['config'].get('qualification')==config['qualification'] and restored['config'].get('controller_hashes')==config['controller_hashes'],'incompatible qualification/controller code'
   assert restored['config'].get('manager_config')==config['manager_config'],'incompatible intervention/data protocol'
   assert restored['config'].get('supervisor_config')==config['supervisor_config'] and restored['config'].get('reference_sha256')==config['reference_sha256'],'incompatible recovery protocol/reference on resume'
   c=restored['counters'];write(out/'resume_verified.json',{'parameters_exact':True,'optimizers_loaded':True,'online_rows':len(on),'offline_rows':len(off),'pending_tails_legally_truncated':True,'source':str(a.resume)})
  pool=FourByFour(config['seed'],norm,out/'workers',resume_workers=None if restored is None else restored['workers'],manager_config=manager_config)
  if restored:restore_rng(restored['rng'])
  capability=GlobalCapability(c.get('global_capability'));pool.apply_capability(capability.snapshot())
  training_start=time.perf_counter();initial_actual=c['actual_interactions'];initial_valid=c['valid_real_steps'];initial_updates=c['critic_updates']
  next_checkpoint=((c['actual_interactions']//25000)+1)*25000;next_eval=((c['actual_interactions']//50000)+1)*50000
  budget=0;rounds=0;stage_counts={};timing={'inference':0.,'step':0.,'reset':0.,'sample':0.,'transfer':0.,'update':0.};stop_reason=None
  from acoserl.collection_health import CollectionHealth
  health=CollectionHealth(c.get('collection_health'));next_diagnostic=next((x for x in (2000,5000,10000,25000) if x>c['actual_interactions']),None)
  if a.integration_pregrasp:
   prepared=pool.prepare_pregrasp({0:dict(purpose='bounded_integration',arm='A'),1:dict(purpose='bounded_integration',arm='B')})
   for fixture in prepared:
    for item in fixture['records']:
     payload=item['metadata'];e=payload['env_id']
     row=raw_row({k:torch.from_numpy(v) for k,v in item['obs'].items()},item['action'],{k:torch.from_numpy(v) for k,v in item['next_obs'].items()},item['reward'],False,False,e,payload['episode_id'],payload['frame'],metadata=payload)
     for assembled in assembler.append(row):on.add(assembled)
     dynamic.observe(row,safe=True);health.observe(payload);capability.observe(payload,actor_controlled=False)
     c['actual_interactions']+=1;c['valid_real_steps']+=1;c['proposals']+=1;budget+=config['utd']
   write(out/'diagnostic_preparation.json',dict(actual_controls=sum(len(f['records']) for f in prepared),workers=[{k:v for k,v in f.items() if k not in ('records','observation')} for f in prepared],all_real_actions_in_online=True,not_policy_episodes=True))
  result['status']='running';write(out/'summary.json',result)
  while time.time()<deadline and not stop_requested['value']:
   if a.mode=='integration' and c['valid_real_steps']-initial_valid+16>a.integration_steps:stop_reason='bounded_integration_complete';break
   if a.mode=='stress' and c['valid_real_steps']-initial_valid>=a.stress_steps:stop_reason='bounded_stress_complete';break
   if c['actual_interactions']+16>config['maximum_actual_interactions'] and a.mode=='formal':stop_reason='maximum_actual_interactions';break
   old=pool.observation;warm=c['valid_real_steps']<config['warmup_real_steps'];start=time.perf_counter()
   if warm:actions=np.random.uniform(-1,1,(16,14)).astype(np.float32)
   else:
    data={k:torch.from_numpy(v).to(device) for k,v in old.items()}
    with torch.no_grad(),eval_mode(agent):actions=agent.act(data,eval_mode=False,stddev=config['stddev'],cpu=True).numpy()
   timing['inference']+=time.perf_counter()-start;assert np.isfinite(actions).all()
   start=time.perf_counter();nxt,reward,term,trunc,info=pool.step(actions);timing['step']+=time.perf_counter()-start
   real=0;partial=0
   for e,payload in enumerate(info):
    assert payload['env_id']==e
    if payload['executed']:
     assert float(reward[e])==float(bool(payload.get('success',False))), 'sparse reward/safe strict success mismatch'
     real+=1;row=raw_row({k:torch.from_numpy(v[e]) for k,v in old.items()},payload['executed_action'],{k:torch.from_numpy(v[e]) for k,v in nxt.items()},reward[e],term[e],trunc[e],e,payload['episode_id'],payload['frame'],metadata=payload)
     for item in assembler.append(row):on.add(item)
     if dynamic is not None and manager_config['dynamic_demo_enabled']:dynamic.observe(row,handback=bool((payload.get('recovery_result') or {}).get('safe_handback')),safe=not bool(payload.get('supervisor_stop_reason')))
     for name,yes in payload.get('task_metrics',{}).items():
      if isinstance(yes,bool) and yes:stage_counts[name]=stage_counts.get(name,0)+1
    else:
     if payload.get('partial_physics_step'):partial+=1
     for item in assembler.close(e,bootstrap=bool(payload['bootstrap_allowed']),boundary_kind=payload['boundary_kind']):on.add(item)
     if dynamic is not None:dynamic.close_episode(e,bootstrap=bool(payload['bootstrap_allowed']),kind=payload['boundary_kind'])
   for e,payload in enumerate(info):
    capability.observe(payload,actor_controlled=not warm);health.observe(payload)
    if term[e] or trunc[e]:health.episode(payload)
   health.set_dynamic_demo_rows(len(dynamic.buffer) if dynamic is not None else 0)
   c['collection_health']=health.snapshot()
   c['global_capability']=capability.snapshot();pool.apply_capability(capability.snapshot())
   c['actual_interactions']+=real+partial;c['valid_real_steps']+=real;c['partial_action_attempts']+=partial;c['proposals']+=16
   if warm:c['warmup_real_steps']+=real
   else:budget+=real*config['utd']
   start=time.perf_counter();pool.reset_done(term|trunc);timing['reset']+=time.perf_counter()-start
   if a.mode=='stress' and rounds==0:
    # Exercise every stream's no-action cut across the next round, independently.
    pool.force_stop(list(range(0,16,4)))
   metrics={}
   while budget>(1 if a.mode=='integration' else 0) and len(on) and time.time()<deadline:
    start=time.perf_counter();batch=torch.cat([on.sample(config['batch_size']//2).clone(),(dynamic.sample_demo(off,config['batch_size']//2) if dynamic is not None else off.sample(config['batch_size']//2).clone())],0);timing['sample']+=time.perf_counter()-start
    start=time.perf_counter();batch=batch.to(device);torch.npu.synchronize();timing['transfer']+=time.perf_counter()-start
    update_actor=(c['critic_updates']+1)%config['utd']==0
    start=time.perf_counter();values=agent.update(batch,config['stddev'],update_actor,bc_batch=None,ref_agent=agent);torch.npu.synchronize();timing['update']+=time.perf_counter()-start
    metrics={k:float(v) for k,v in values.items() if not k.startswith('_')};assert all(np.isfinite(v) for v in metrics.values()),metrics
    c['critic_updates']+=1;c['actor_updates']+=int(update_actor);budget-=1
    log(out/'updates.jsonl',{'actual_interactions':c['actual_interactions'],'critic_update':c['critic_updates'],'actor_updated':update_actor,**metrics})
   rounds+=1;elapsed=time.perf_counter()-training_start
   log(out/'learning_curve.jsonl',{'time':time.time(),'round':rounds,**c,'online_rows':len(on),'pending_nstep_rows':assembler.pending(),'actual_steps_s':(c['actual_interactions']-initial_actual)/elapsed,'updates_s':(c['critic_updates']-initial_updates)/elapsed,'timing':timing.copy(),'dynamic_demo_rows':len(dynamic.buffer) if dynamic is not None else 0,'demo_sampling':dynamic.last_sample.copy() if dynamic is not None else {},'dynamic_demo_stats':dynamic.stats.copy() if dynamic is not None else {},'stage_counts':stage_counts.copy(),'dispatch_span_seconds':max(pool.last_dispatch)-min(pool.last_dispatch),**metrics})
   print('ROUND',rounds,'actual',c['actual_interactions'],'updates',c['critic_updates'],'real_this_round',real,flush=True)
   if a.mode=='formal' and next_diagnostic is not None and c['actual_interactions']>=next_diagnostic:
    write(out/f'early_diagnostic_{next_diagnostic:07d}.json',dict(counters=c,health=health.snapshot(),stall=health.stall(),workers=pool.diagnostics()))
    next_diagnostic=next((x for x in (2000,5000,10000,25000) if x>c['actual_interactions']),None)
   if a.mode=='formal' and health.stall()['stop']:stop_reason='collection_deadlock_intervention_or_task_progress';break
   if a.mode=='formal' and c['actual_interactions']>=next_checkpoint:
    save(out/'checkpoints'/f'step_{c["actual_interactions"]:07d}',agent,on,off,assembler,pool,c,config,norm,dynamic);next_checkpoint+=25000
   if a.mode=='formal' and c['actual_interactions']>=next_eval:
    evaluation=evaluate(agent,norm,out,c['actual_interactions'],deadline,torch,device,config['evaluation_episodes'],manager_config=manager_config);c['evaluation_interactions']+=evaluation['actual_evaluation_interactions'];c['evaluation_seconds']+=evaluation['seconds'];next_eval+=50000
   if a.mode=='stress' and rounds>=40:stop_reason='stress_no_progress_limit';break
  if stop_reason is None:stop_reason='operator_stop' if stop_requested['value'] else 'wall_clock_limit'
  for e in list(assembler.queues):
   for item in assembler.truncate(e):on.add(item)
  # Flush legitimate final tails, then complete any bounded stress update debt.
  while a.mode=='stress' and budget and len(on) and time.time()<deadline:
   batch=torch.cat([on.sample(config['batch_size']//2).clone(),(dynamic.sample_demo(off,config['batch_size']//2) if dynamic is not None else off.sample(config['batch_size']//2).clone())],0).to(device)
   actor=(c['critic_updates']+1)%config['utd']==0;values=agent.update(batch,.1,actor,bc_batch=None,ref_agent=agent);torch.npu.synchronize();assert all(np.isfinite(float(v)) for k,v in values.items() if not k.startswith('_'))
   c['critic_updates']+=1;c['actor_updates']+=int(actor);budget-=1
  if a.mode=='integration':
   evaluation=evaluate(agent,norm,out,c['actual_interactions'],deadline,torch,device,episodes=1,manager_config=manager_config,max_steps=32)
   c['evaluation_interactions']+=evaluation['actual_evaluation_interactions'];c['evaluation_seconds']+=evaluation['seconds']
   assert c['valid_real_steps']-initial_valid+c['evaluation_interactions']<=2048,'integration physics budget including evaluation'
  for payload in pool.cut_all(stop_reason):capability.observe(payload,actor_controlled=True)
  c['global_capability']=capability.snapshot();pool.apply_capability(capability.snapshot())
  if dynamic is not None:
   for e in set(dynamic.pending)|set(dynamic.stream.queues):dynamic.close_episode(e,True,6)
  health.set_dynamic_demo_rows(len(dynamic.buffer) if dynamic is not None else 0)
  c['collection_health']=health.snapshot()
  visual_final=manifest(agent);assert visual_initial['backbone_hashes']==visual_final['backbone_hashes'],'frozen pretrained weights changed'
  visual_audit={'frozen_backbones_unchanged':True,'backbones_eval':all(not e.backbone.training for e in agent.encoders if hasattr(e,'backbone')),'adapter_grad_norms':[float(e.adapter.weight.grad.norm()) for e in agent.encoders if hasattr(e,'adapter') and e.adapter.weight.grad is not None]}
  if visual_snapshot is not None:
   changes={k:bool(torch.any(p.detach().cpu()!=visual_snapshot[k])) for k,p in agent.named_parameters() if k in visual_snapshot}
   visual_audit['changed_tensors']={prefix:sum(v for k,v in changes.items() if k.startswith(prefix)) for prefix in ['actor.','critic.','encoders.']}
   assert all(visual_audit['changed_tensors'].values())
  visual_audit['all_parameter_gradients_finite']=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in agent.parameters())
  assert visual_audit['all_parameter_gradients_finite']
  write(out/'visual_audit.json',visual_audit)
  elapsed=time.perf_counter()-training_start
  restoration_checks=None
  if a.mode=='integration':
   c['pending_update_budget']=budget
   pre=save(out/'checkpoints'/'pre_restore',agent,on,off,assembler,pool,c,config,norm,dynamic)
   restored_state,assembler,restoration_checks=verify_roundtrip(pre,agent,on,off,norm,dynamic)
   if budget:
    assert budget==1 and time.time()<deadline,'integration continuation budget/deadline'
    batch=torch.cat([on.sample(128).clone(),dynamic.sample_demo(off,128) if dynamic is not None else off.sample(128).clone()],0).to(device)
    values=agent.update(batch,config['stddev'],True,bc_batch=None,ref_agent=agent);torch.npu.synchronize()
    assert all(np.isfinite(float(v)) for k,v in values.items() if not k.startswith('_'))
    c['critic_updates']+=1;c['actor_updates']+=1;budget-=1
    restoration_checks['optimizer_continuation_update']=True
   c['pending_update_budget']=budget
   log(out/'updates.jsonl',{'phase':'post_restore_continuation','critic_update':c['critic_updates'],'actor_updated':True,**{k:float(v) for k,v in values.items() if not k.startswith('_')}})
   write(out/'checkpoint_reload.json',restoration_checks)
  final=save(out/'checkpoints'/'final',agent,on,off,assembler,pool,c,config,norm,dynamic)
  elapsed=time.perf_counter()-training_start
  result.update(restoration_checks=restoration_checks,dynamic_demo_rows=len(dynamic.buffer) if dynamic is not None else 0,dynamic_demo_stats=dynamic.stats if dynamic is not None else {},status='completed',stop_reason=stop_reason,counters=c,workers=pool.diagnostics(),online_rows=len(on),offline_rows=len(off),rounds=rounds,
   wall_seconds=time.time()-started,training_seconds=elapsed,timing=timing,valid_steps_s=(c['valid_real_steps']-initial_valid)/elapsed,actual_steps_s=(c['actual_interactions']-initial_actual)/elapsed,
   updates_s=(c['critic_updates']-initial_updates)/elapsed,stage_counts=stage_counts,checkpoint=str(final),peak_npu_allocated_bytes=torch.npu.max_memory_allocated(),
   utd_effective=(c['critic_updates']-initial_updates)/max(1,c['valid_real_steps']-initial_valid-(c['warmup_real_steps'] if restored is None else 0)),
   all_losses_and_gradient_norms_finite=True,all_workers_cpu_only=True,all_actions_stored_are_executed=True)
  if a.mode=='stress':
   result['passed']=stop_reason=='bounded_stress_complete' and budget==0 and c['critic_updates']-initial_updates>0
   assert result['passed'],result
 except BaseException:
  result.update(status='failed',error=traceback.format_exc(),counters=c);raise
 finally:
  if pool is not None:
   try:result.setdefault('workers',pool.diagnostics())
   except Exception as error:result['worker_diagnostic_error']=repr(error)
   try:pool.close()
   except Exception as error:result['worker_close_error']=repr(error)
  if telemetry is not None:telemetry.close()
  write(out/'summary.json',result)
if __name__=='__main__':main()
