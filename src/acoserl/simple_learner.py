"""Simple dual-arm AutoSERL using the existing ResFiT learner and checkpoints.

One 14D Actor, 10Q, sparse strict success, UTD=1, online/demo 1:1. This
entry point never instantiates ReviewedRuntime, V2/V3 or ForwardGraspGuide.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback
import numpy as np
from acoserl.qualification import ROOT, external_path, source_manifest, dependency_manifest


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    tmp.replace(path)


def log(path, value):
    with Path(path).open('a') as f:f.write(json.dumps(value,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')


def evaluate(agent, norm, out, c, manager_config, deadline, device):
    import gymnasium as gym
    import torch
    from acoserl.simple_collection import SimpleFactory as WorkerFactory
    from acoserl.ascend_checkpoint import rng_state,restore_rng
    from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
    saved=rng_state();rows=[];count=0;start=time.perf_counter()
    env=gym.vector.AsyncVectorEnv([WorkerFactory(16,9000000+c['valid_real_steps'],norm,
        str(out/'evaluation_workers'),False)],
        context='spawn',autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    try:
        for ep in range(20):
            if time.time()>=deadline:break
            obs,_=env.reset(seed=9000000+ep);done=False;steps=0;last={}
            while not done and time.time()<deadline:
                data={k:torch.from_numpy(v).to(device) for k,v in obs.items()}
                with torch.no_grad(),eval_mode(agent):action=agent.act(data,eval_mode=True,cpu=True).numpy()
                obs,r,t,tr,info=env.step(action);last=json.loads(info['payload'][0])
                steps+=int(last['executed']);count+=int(last['executed'])+int(last.get('partial_physics_step',False))
                done=bool(t[0] or tr[0])
            rows.append(dict(episode=ep,complete=done,length=steps,success=bool(last.get('success')),
                             stop_reason=last.get('supervisor_stop_reason'),metrics=last.get('task_metrics')))
    finally:env.close();restore_rng(saved)
    result=dict(training_step=c['valid_real_steps'],intervention_enabled=False,
                safety_projection_enabled=False,independent_initial_states=False,
                episodes=rows,actual_evaluation_interactions=count,seconds=time.perf_counter()-start)
    write(out/f"evaluation_{c['valid_real_steps']:07d}.json",result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('integration','formal'),required=True)
    p.add_argument('--tag',required=True);p.add_argument('--output-root',type=Path)
    p.add_argument('--resume',type=Path);p.add_argument('--seed',type=int)
    p.add_argument('--method',choices=('autoserl','rlpd'),default='autoserl')
    p.add_argument('--integration-steps',type=int,default=2048)
    p.add_argument('--integration-seconds',type=int,default=3600)
    p.add_argument('--utd',type=int,choices=(1,),default=1)
    p.add_argument('--visual',choices=('resnet18_spatial84','minivit'),default='resnet18_spatial84')
    p.add_argument('--checkpoint-seconds',type=int,default=3600)
    p.add_argument('--checkpoint-mirror-root',type=Path)
    p.add_argument('--no-grasp-budget',type=int,default=10000)
    a=p.parse_args()
    if not a.tag.replace('_','').replace('-','').isalnum():p.error('invalid tag')
    if not 16<=a.integration_steps<=2048 or not 60<=a.integration_seconds<=3600:
        p.error('invalid bounded integration budget')
    if a.checkpoint_seconds<60 or a.no_grasp_budget<0:p.error('invalid checkpoint/progress budget')
    cfg=json.loads((ROOT/'configs/ascend16/formal.json').read_text())
    manifest,dependency=source_manifest(),dependency_manifest()
    enabled=a.method=='autoserl'
    manager=dict(runtime='stage_aware_bimanual_recovery_v1',method=a.method,
                 dynamic_demo_enabled=enabled,dynamic_capacity=5000,
                 original_quota=96 if enabled else 128,positive_quota=2,
                 no_passive_hold=True,no_forward_grasp_extension=True,
                 guide_stall_policy='stage_anchor_and_20_step_target_error',
                 recovery_anchors='S0_P0_P1',
                 guide_exit='three_step_target_or_converged_after_real_progress',
                 sliding_window_clock='completed_policy_controls_only',
                 post_guide_policy_steps=32,
                 dynamic_demo_admission='verified_per_arm_handoff_v2')
    cfg.update(method='ResFiT RLPD + independent AutoSERL baseline',purpose=a.mode,collector_protocol='simple_bimanual_collector_v1',utd=1,
               manager_config=manager,visual_encoder=a.visual,
               checkpoint_seconds=a.checkpoint_seconds,no_grasp_budget=a.no_grasp_budget,
               intervention_enabled=enabled,passive_hold_entry_gate_enabled=False,
               runtime_safety='ActionOnly: live workspace/collision/force plus private substep preview',
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               source_sha256=manifest,dependency=dependency)
    if a.seed is not None:cfg['seed']=a.seed
    if a.mode=='integration':
        cfg.update(warmup_real_steps=0,online_capacity=2048,
                   maximum_actual_interactions=a.integration_steps,wall_limit_seconds=a.integration_seconds)
    root=external_path(a.output_root or ROOT.parent/'training_runs/acoserl_dexmimicgen/simple'/a.mode)
    out=root/a.tag;out.mkdir(parents=True,exist_ok=False)
    from acoserl.run_observability import RunStore,task_progress_stop
    mirror=external_path(a.checkpoint_mirror_root) if a.checkpoint_mirror_root else None
    store=RunStore(out,cfg['source_commit'],mirror)
    stop={'requested':False}
    signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True))
    signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True))
    write(out/'config.json',cfg)
    # Process-scoped lock, not a PID-file assertion. Never kill other jobs.
    import fcntl
    lock=open('/tmp/acoserl_dexmimicgen_formal.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();deadline=start+cfg['wall_limit_seconds'];pool=None
    c=dict(actual_interactions=0,valid_real_steps=0,partial_action_attempts=0,warmup_real_steps=0,
           critic_updates=0,actor_updates=0,proposals=0,evaluation_interactions=0,evaluation_seconds=0.)
    summary=dict(status='initializing',formal_training_mode=a.mode=='formal',runtime=manager['runtime'],
                 physical_qualification_claimed=False)
    try:
        import torch,torch_npu
        from acoserl.ascend_data import offline,replay
        from acoserl.simple_replay import SimpleDemo,make_row
        from acoserl.isolated_nstep import IsolatedNStep
        from acoserl.simple_collection import SimpleCapability as GlobalCapability
        from acoserl.simple_collection import SimplePool as FourByFour
        from acoserl.collection_health import CollectionHealth
        from acoserl.ascend_checkpoint import save,load,restore_rng,verify_roundtrip
        from acoserl.pretrained_visual import make_agent,manifest as visual_manifest
        from resfit.rl_finetuning.config.rlpd import QAgentConfig
        from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
        from acoserl.single_demo_rlpd import IMAGES
        if not torch.npu.is_available():raise RuntimeError('Ascend NPU unavailable')
        device=torch.device('npu:0');torch.npu.set_device(device)
        torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.config.allow_internal_format=False
        torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.npu.manual_seed_all(cfg['seed'])
        np.random.seed(cfg['seed']);random.seed(cfg['seed'])
        off,std,scale,norm=offline(128,cfg['gamma']);on=replay(cfg['online_capacity'],128)
        dynamic=SimpleDemo(manager['dynamic_capacity'],cfg['gamma'],manager['original_quota'],
                           manager['positive_quota'],cfg['seed'])
        agent=make_agent((3,84,84),(18,),14,IMAGES,QAgentConfig(device=str(device)),a.visual)
        assert agent.cfg.critic.num_q==10
        initial_visual=visual_manifest(agent)
        cfg['visual_manifest']=initial_visual;write(out/'config.json',cfg);write(out/'normalization.json',norm)
        assembler=IsolatedNStep(3,cfg['gamma']);restored=None
        if a.resume:
            saved=torch.load(a.resume/'collector.pt',weights_only=False,map_location='cpu')
            for key in ('collector_protocol','source_sha256','dependency','manager_config','visual_encoder'):
                if saved['config'].get(key)!=cfg[key]:raise ValueError('resume incompatible: '+key)
            restored,assembler=load(a.resume,agent,on,off,norm,dynamic);c=restored['counters']
        pool=FourByFour(cfg['seed'],norm,out/'workers',enabled=enabled,
                         resume_workers=None if restored is None else restored['workers'])
        if restored is not None:restore_rng(restored['rng'])
        capability=GlobalCapability(c.get('global_capability'));sent=capability.snapshot();pool.apply_capability(sent)
        health=CollectionHealth(c.get('collection_health'))
        start_valid=c['valid_real_steps'];start_actual=c['actual_interactions'];budget=int(c.get('pending_update_budget',0))
        next_save=(c['valid_real_steps']//25000+1)*25000;next_eval=(c['valid_real_steps']//50000+1)*50000
        rounds=0;empty_rounds=0;reason='wall_clock_limit'
        last_save=time.monotonic();last_status=0.;rate_steps=c['valid_real_steps'];rate_time=time.monotonic()
        timing={k:0. for k in ('inference','step','reset','sample_transfer','update')}
        while time.time()<deadline and not stop['requested']:
            used=c['valid_real_steps']-start_valid
            if a.mode=='integration' and used+16>a.integration_steps:reason='bounded_integration_complete';break
            if a.mode=='formal' and c['actual_interactions']+16>cfg['maximum_actual_interactions']:reason='maximum_actual_interactions';break
            old=pool.observation;warm=c['valid_real_steps']<cfg['warmup_real_steps'];t=time.perf_counter()
            if warm:actions=np.random.uniform(-1,1,(16,14)).astype(np.float32)
            else:
                data={k:torch.from_numpy(v).to(device) for k,v in old.items()}
                with torch.no_grad(),eval_mode(agent):actions=agent.act(data,eval_mode=False,stddev=cfg['stddev'],cpu=True).numpy()
            timing['inference']+=time.perf_counter()-t
            t=time.perf_counter();nxt,reward,term,trunc,info=pool.step(actions);timing['step']+=time.perf_counter()-t
            real=partial=0
            for e,payload in enumerate(info):
                if payload['env_id']!=e:raise AssertionError('worker/transition index mismatch')
                if payload['executed']:
                    if float(reward[e])!=float(bool(payload.get('success',False))):
                        raise AssertionError('strict sparse reward mismatch')
                    real+=1
                    row=make_row({k:torch.from_numpy(v[e]) for k,v in old.items()},payload['executed_action'],
                        {k:torch.from_numpy(v[e]) for k,v in nxt.items()},reward[e],term[e],trunc[e],payload)
                    for item in assembler.append(row):on.add(item)
                    if manager['dynamic_demo_enabled']:
                        recovery_result = payload.get('recovery_result') or {}
                        dynamic.observe(row,
                            decision=payload.get('auto_intervention') or {},
                            handbacks=recovery_result.get('handbacks', []),
                            control_events=recovery_result.get('control_events', []),
                            safe=not bool(payload.get('supervisor_stop_reason')))
                else:
                    partial+=int(payload.get('partial_physics_step',False))
                    for item in assembler.close(e,bootstrap=bool(payload['bootstrap_allowed']),boundary_kind=payload['boundary_kind']):on.add(item)
                    dynamic.close_episode(e,bool(payload['bootstrap_allowed']),payload['boundary_kind'])
                capability.observe(payload,actor_controlled=not warm);health.observe(payload)
                if term[e] or trunc[e]:health.episode(payload)
            c['actual_interactions']+=real+partial;c['valid_real_steps']+=real
            c['partial_action_attempts']+=partial;c['proposals']+=16
            if warm:c['warmup_real_steps']+=real
            else:budget+=real
            health.set_dynamic_demo_rows(len(dynamic.buffer))
            current=capability.snapshot()
            if current!=sent:pool.apply_capability(current);sent=current
            t=time.perf_counter();pool.reset_done(term|trunc);timing['reset']+=time.perf_counter()-t
            metrics={}
            # Reserve one genuine update for post-reload continuation in integration.
            while budget>(1 if a.mode=='integration' else 0) and len(on) and time.time()<deadline:
                t=time.perf_counter()
                batch=torch.cat([on.sample(128).clone(),dynamic.sample_demo(off,128)],0).to(device)
                timing['sample_transfer']+=time.perf_counter()-t;t=time.perf_counter()
                values=agent.update(batch,cfg['stddev'],True,bc_batch=None,ref_agent=agent)
                metrics={k:float(v) for k,v in values.items() if not k.startswith('_')}
                if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('nonfinite loss/metric')
                timing['update']+=time.perf_counter()-t
                c['critic_updates']+=1;c['actor_updates']+=1;budget-=1
            rounds+=1;empty_rounds=empty_rounds+1 if real==0 else 0
            c.update(global_capability=capability.snapshot(),collection_health=health.snapshot(),pending_update_budget=budget)
            log(out/'learning_curve.jsonl',dict(round=rounds,**c,online_rows=len(on),
                dynamic_demo_rows=len(dynamic.buffer),dynamic_demo_stats=dynamic.stats,
                demo_sampling=dynamic.last_sample,timing=timing,metrics=metrics))
            print('ROUND',rounds,'valid',c['valid_real_steps'],'updates',c['critic_updates'],'dynamic',len(dynamic.buffer),flush=True)
            if empty_rounds>=32:reason='no_live_transition_progress';break
            now=time.monotonic()
            if now-last_status>=60.:
                c['rolling_valid_steps_s']=(c['valid_real_steps']-rate_steps)/max(now-rate_time,1e-6)
                store.heartbeat(c,health.snapshot(),timing,time.time()-start,len(dynamic.buffer),dynamic.stats)
                rate_steps=c['valid_real_steps'];rate_time=now;last_status=now
            if a.mode=='formal' and task_progress_stop(health.snapshot(),a.no_grasp_budget):
                reason='no_stable_grasp_despite_training';break
            if a.mode=='formal' and health.stall()['stop']:reason='collection_health_stop';break
            if a.mode=='formal' and len(on) and (c['valid_real_steps']>=next_save or now-last_save>=a.checkpoint_seconds):
                cp=save(out/'checkpoints'/f"step_{c['valid_real_steps']:07d}_p{c['proposals']:08d}",agent,on,off,assembler,pool,c,cfg,norm,dynamic)
                store.checkpoint(cp);last_save=time.monotonic()
                while next_save<=c['valid_real_steps']:next_save+=25000
            if a.mode=='formal' and c['valid_real_steps']>=next_eval:
                ev=evaluate(agent,norm,out,c,manager,deadline,device)
                c['evaluation_interactions']+=ev['actual_evaluation_interactions'];c['evaluation_seconds']+=ev['seconds'];next_eval+=50000
        if stop['requested']:reason='operator_stop'
        for e in list(assembler.queues):
            for row in assembler.truncate(e):on.add(row)
        for e in set(dynamic.last)|set(dynamic.stream.queues):dynamic.close_episode(e,True,6)
        for payload in pool.cut_all(reason):capability.observe(payload,actor_controlled=True)
        c.update(global_capability=capability.snapshot(),collection_health=health.snapshot(),pending_update_budget=budget)
        reload_result=None
        if a.mode=='integration' and len(on):
            pre=save(out/'checkpoints'/'pre_restore',agent,on,off,assembler,pool,c,cfg,norm,dynamic)
            _,assembler,reload_result=verify_roundtrip(pre,agent,on,off,norm,dynamic)
            if budget and time.time()<deadline:
                batch=torch.cat([on.sample(128).clone(),dynamic.sample_demo(off,128)],0).to(device)
                values=agent.update(batch,cfg['stddev'],True,bc_batch=None,ref_agent=agent)
                if not all(np.isfinite(float(v)) for k,v in values.items() if not k.startswith('_')):raise FloatingPointError('post-restore update')
                c['critic_updates']+=1;c['actor_updates']+=1;budget-=1;reload_result['optimizer_continuation_update']=True
            else:reload_result['optimizer_continuation_update']=False
            write(out/'checkpoint_reload.json',reload_result)
        c['pending_update_budget']=budget
        final=save(out/'checkpoints'/'final',agent,on,off,assembler,pool,c,cfg,norm,dynamic) if len(on) else None
        if final is not None:store.checkpoint(final)
        if visual_manifest(agent)['backbone_hashes']!=initial_visual['backbone_hashes']:raise AssertionError('frozen visual backbone changed')
        if not all(x.grad is None or bool(torch.isfinite(x.grad).all()) for x in agent.parameters()):raise FloatingPointError('nonfinite gradient')
        summary.update(status='completed',stop_reason=reason,counters=c,workers=pool.diagnostics(),
            online_rows=len(on),offline_rows=len(off),dynamic_demo_rows=len(dynamic.buffer),dynamic_demo_stats=dynamic.stats,
            checkpoint=str(final) if final else None,restoration_checks=reload_result,timing=timing,
            valid_steps_s=(c['valid_real_steps']-start_valid)/max(time.time()-start,1e-6),
            checkpoint_source_sha256=manifest,wall_seconds=time.time()-start)
    except BaseException:
        summary.update(status='failed',error=traceback.format_exc(),counters=c)
        if pool is not None and 'on' in locals() and len(on) and 'agent' in locals():
            try:
                emergency=save(out/'checkpoints'/f"failure_{c['valid_real_steps']:07d}",
                    agent,on,off,assembler,pool,c,cfg,norm,dynamic)
                store.checkpoint(emergency)
                summary['emergency_checkpoint']=str(emergency)
            except Exception:
                summary['emergency_checkpoint_error']=traceback.format_exc()
        raise
    finally:
        try:
            if pool is not None:pool.close()
        except Exception:
            summary['worker_close_error']=traceback.format_exc()
        finally:
            try:
                summary['storage_manifest']=store.record
                summary['latest_complete_checkpoint']=store.last_checkpoint
                write(out/'summary.json',summary)
                write(out/'live_status.json',summary)
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__ == '__main__':
    main()
