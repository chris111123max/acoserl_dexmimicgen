"""Existing ResFiT learner/collectors with external test output and frozen gates.

No new RL algorithm: one 14D Actor, 10Q, sparse success, UTD=1, online/demo 1:1.
Only orchestration, audit, interruption and output ownership are centralized.
"""
from __future__ import annotations
import argparse
import hashlib
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


def require_gate(path, manifest, dependency):
    gate=json.loads(external_path(path).read_text())
    required=('workspace_projection','guide_A_ownership','guide_B_ownership',
              'guide_A_handback','guide_B_handback','static_hold_requalified',
              'buffer_integrity','nstep_boundaries','checkpoint_continue',
              'npu_seed_20261032','npu_seed_20261033','strict_demo_replay')
    if gate.get('source_sha256')!=manifest or gate.get('dependency')!=dependency:
        raise ValueError('acceptance source/dependency fingerprint mismatch')
    if gate.get('protocol')!='reviewed_baseline_acceptance_v1' or not gate.get('formal_training_approved'):
        raise ValueError('formal training not approved')
    for name in required:
        item=gate.get('checks',{}).get(name,{})
        if item.get('status')!='PASS':raise ValueError('acceptance missing: '+name)
        evidence=external_path(item['path'])
        if hashlib.sha256(evidence.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('acceptance evidence changed: '+name)
    return gate


def evaluate(agent, norm, out, c, manager_config, deadline, device):
    import gymnasium as gym
    import torch
    from acoserl.parallel_autoserl import WorkerFactory
    from acoserl.ascend_checkpoint import rng_state,restore_rng
    from resfit.rl_finetuning.off_policy.common_utils.utils import eval_mode
    saved=rng_state();rows=[];count=0;start=time.perf_counter()
    env=gym.vector.AsyncVectorEnv([WorkerFactory(16,9000000+c['valid_real_steps'],norm,
        str(out/'evaluation_workers'),False,manager_config=manager_config)],
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
    p.add_argument('--gate',type=Path);p.add_argument('--static-hold-certificate',type=Path)
    p.add_argument('--resume',type=Path);p.add_argument('--seed',type=int)
    p.add_argument('--integration-steps',type=int,default=1024)
    p.add_argument('--integration-seconds',type=int,default=1800)
    p.add_argument('--utd',type=int,choices=(1,),default=1)
    p.add_argument('--visual',choices=('resnet18_spatial84','minivit'),default='resnet18_spatial84')
    p.add_argument('--supervisor-version',choices=('baseline',),default='baseline')
    p.add_argument('--intervention-method',choices=('full_acoserl','plain_rlpd'),default='full_acoserl')
    p.add_argument('--demo-admission',choices=('official','safe_handback'),default='official')
    p.add_argument('--demo-original-quota',type=int,choices=(64,96,128),default=96)
    p.add_argument('--demo-positive-quota',type=int,choices=(0,2,4),default=2)
    a=p.parse_args()
    if not a.tag.replace('_','').replace('-','').isalnum():p.error('invalid tag')
    if not 16<=a.integration_steps<=2048 or not 60<=a.integration_seconds<=3600:
        p.error('invalid bounded integration budget')
    cfg=json.loads((ROOT/'configs/ascend16/formal.json').read_text())
    manifest,dependency=source_manifest(),dependency_manifest()
    if a.mode=='formal':
        if a.gate is None or a.static_hold_certificate is None:p.error('formal requires external acceptance and static-hold certificate')
        require_gate(a.gate,manifest,dependency)
        hold=json.loads(external_path(a.static_hold_certificate).read_text())
        if (hold.get('protocol')!='static_hold_requalification_v1' or not hold.get('passed') or
                hold.get('source_sha256')!=manifest or hold.get('dependency')!=dependency):
            raise ValueError('static-hold certificate is missing or stale')
        for receipt in hold.get('receipts',[]):
            path=external_path(receipt['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['sha256']:
                raise ValueError('static-hold evidence changed')
        if min(hold.get('counts',{}).get(arm,0) for arm in ('A','B'))<3:
            raise ValueError('static-hold physical evidence incomplete')
    from acoserl.intervention_manager import method_config
    manager=method_config(a.intervention_method)
    manager.update(json.loads((ROOT/'configs/bimanual_autoserl/baseline.json').read_text()))
    manager.update(admission_mode=a.demo_admission,original_quota=a.demo_original_quota,
                   positive_quota=a.demo_positive_quota,original_demo_fraction=a.demo_original_quota/128)
    if a.intervention_method=='plain_rlpd':manager['dynamic_demo_enabled']=False;manager['original_quota']=128
    if a.static_hold_certificate:manager['static_hold_certificate']=str(external_path(a.static_hold_certificate))
    cfg.update(purpose=a.mode,collector_protocol='reviewed_baseline_v1',utd=1,
               manager_config=manager,visual_encoder=a.visual,
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               source_sha256=manifest,dependency=dependency)
    if a.seed is not None:cfg['seed']=a.seed
    if a.mode=='integration':
        cfg.update(warmup_real_steps=0,online_capacity=2048,
                   maximum_actual_interactions=a.integration_steps,wall_limit_seconds=a.integration_seconds)
    root=a.output_root or (ROOT.parents[1]/'testing'/'runs' if a.mode=='integration' else ROOT/'outputs/ascend16')
    root=external_path(root) if a.mode=='integration' else root.expanduser().resolve()
    out=root/a.tag;out.mkdir(parents=True,exist_ok=False)
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
    summary=dict(status='initializing',formal_training_approved=a.mode=='formal')
    try:
        import torch,torch_npu
        from acoserl.ascend_data import offline,replay,raw_row
        from acoserl.isolated_nstep import IsolatedNStep
        from acoserl.dynamic_demo import DynamicDemo
        from acoserl.global_capability import GlobalCapability
        from acoserl.parallel_autoserl import FourByFour
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
        dynamic=DynamicDemo(manager['dynamic_capacity'],manager['original_demo_fraction'],3,cfg['gamma'],
            admission_mode=manager['admission_mode'],original_quota=manager['original_quota'],
            positive_quota=manager['positive_quota'],sampling_seed=cfg['seed'])
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
        pool=FourByFour(cfg['seed'],norm,out/'workers',resume_workers=None if restored is None else restored['workers'],manager_config=manager)
        if restored is not None:restore_rng(restored['rng'])
        capability=GlobalCapability(c.get('global_capability'));pool.apply_capability(capability.snapshot())
        health=CollectionHealth(c.get('collection_health'))
        start_valid=c['valid_real_steps'];start_actual=c['actual_interactions'];budget=int(c.get('pending_update_budget',0))
        next_save=(c['valid_real_steps']//25000+1)*25000;next_eval=(c['valid_real_steps']//50000+1)*50000
        rounds=0;empty_rounds=0;reason='wall_clock_limit'
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
                    row=raw_row({k:torch.from_numpy(v[e]) for k,v in old.items()},payload['executed_action'],
                        {k:torch.from_numpy(v[e]) for k,v in nxt.items()},reward[e],term[e],trunc[e],
                        e,payload['episode_id'],payload['frame'],metadata=payload)
                    for item in assembler.append(row):on.add(item)
                    if manager['dynamic_demo_enabled']:
                        dynamic.observe(row,handback=bool((payload.get('recovery_result') or {}).get('safe_handback')),
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
            health.set_dynamic_demo_rows(len(dynamic.buffer));pool.apply_capability(capability.snapshot())
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
            if a.mode=='formal' and health.stall()['stop']:reason='collection_health_stop';break
            if a.mode=='formal' and c['valid_real_steps']>=next_save:
                save(out/'checkpoints'/f"step_{c['valid_real_steps']:07d}",agent,on,off,assembler,pool,c,cfg,norm,dynamic);next_save+=25000
            if a.mode=='formal' and c['valid_real_steps']>=next_eval:
                ev=evaluate(agent,norm,out,c,manager,deadline,device)
                c['evaluation_interactions']+=ev['actual_evaluation_interactions'];c['evaluation_seconds']+=ev['seconds'];next_eval+=50000
        if stop['requested']:reason='operator_stop'
        for e in list(assembler.queues):
            for row in assembler.truncate(e):on.add(row)
        for e in set(dynamic.pending)|set(dynamic.stream.queues):dynamic.close_episode(e,True,6)
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
        if visual_manifest(agent)['backbone_hashes']!=initial_visual['backbone_hashes']:raise AssertionError('frozen visual backbone changed')
        if not all(x.grad is None or bool(torch.isfinite(x.grad).all()) for x in agent.parameters()):raise FloatingPointError('nonfinite gradient')
        summary.update(status='completed',stop_reason=reason,counters=c,workers=pool.diagnostics(),
            online_rows=len(on),offline_rows=len(off),dynamic_demo_rows=len(dynamic.buffer),dynamic_demo_stats=dynamic.stats,
            checkpoint=str(final) if final else None,restoration_checks=reload_result,timing=timing,
            valid_steps_s=(c['valid_real_steps']-start_valid)/max(time.time()-start,1e-6),
            checkpoint_source_sha256=manifest,wall_seconds=time.time()-start)
    except BaseException:
        summary.update(status='failed',error=traceback.format_exc(),counters=c)
        raise
    finally:
        try:
            if pool is not None:pool.close()
        except Exception:
            summary['worker_close_error']=traceback.format_exc()
        finally:
            try:write(out/'summary.json',summary)
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
