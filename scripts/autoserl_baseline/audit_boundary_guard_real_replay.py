#!/usr/bin/env python3
"""CPU-only real-MuJoCo replay/MDP data audit from prior 16-worker Policy proposals.

Runs the PRODUCTION SimpleWorker and complete real RGB observation pipeline
from canonical episode-0 initial state. Replays only policy proposals recorded
before the first episode change, using the current BoundaryGuard controller.
Feeds resulting genuinely executed actions into the actual make_row,
IsolatedNStep, Online Replay and SimpleDemo APIs of the production learner.
Never performs weight updates, invokes Torch-NPU, alters robot qpos mid-episode,
or writes production files. Uses fresh external output directory.

This reproduces the recorded proposals, not a guarantee of matching old
post-intervention physics; stopped or diverged trajectories are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from acoserl.ascend_data import replay
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.simple_collection import SimpleFactory, SimpleWorker
from acoserl.simple_replay import SimpleDemo, make_row
from acoserl.single_demo_rlpd import IMAGES


def fail(message):
    raise AssertionError(message)


def digest(tensor):
    value = torch.as_tensor(tensor).detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def load_proposals(path):
    if not path.is_file():
        raise FileNotFoundError(f"prior worker log unavailable: {path}")
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    entries = [r for r in rows if r.get('event') == 'step'
               and int(r.get('episode_id', -1)) == 0]
    if not entries:
        raise ValueError(f"no episode-0 proposals in {path}")
    for i, row in enumerate(entries):
        proposal = np.asarray(row.get('policy_action'), dtype=np.float32)
        if proposal.shape != (14,) or not np.isfinite(proposal).all():
            raise ValueError(f"missing or invalid recorded Policy proposal in {path} line {i}")
    return entries


def obs_tensors(observation):
    result = {key: torch.as_tensor(value).clone() for key, value in observation.items()}
    for key in IMAGES:
        if key not in result or result[key].shape != (3,84,84) or result[key].dtype != torch.uint8:
            raise AssertionError(f"real RGB image missing or malformed: {key}")
    if result['observation.state'].shape != (18,):
        raise AssertionError("real normalized robot state missing")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, required=True,
                   help='Previously audited 512-attempt integration run directory')
    p.add_argument('--output', type=Path, required=True,
                   help='New path outside the repository. Must not exist.')
    p.add_argument('--env-ids', type=int, nargs='+', default=list(range(16)))
    p.add_argument('--max-proposals', type=int, default=64)
    args=p.parse_args()
    reference=args.reference_run.resolve(strict=True)
    if args.output.exists():
        p.error(f"Refusing to overwrite existing audit output: {args.output}")
    if (not 1<=len(set(args.env_ids))<=16 or
            any(not 0<=v<16 for v in args.env_ids) or
            not 1<=args.max_proposals<=128):
        p.error('Invalid bounded env ID or proposal count')
    cfg=json.loads((reference/'config.json').read_text())
    norm=json.loads((reference/'normalization.json').read_text())
    if cfg.get('manager_config',{}).get('runtime') != 'stage_aware_bimanual_recovery_v1':
        p.error('Expected prior v1 integration source, not a new or mixed protocol')
    torch.set_num_threads(1)
    output=args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'workers').mkdir()
    gamma=float(cfg['gamma'])
    online=replay(2048,None)
    assembler=IsolatedNStep(3,gamma)
    dynamic=SimpleDemo(capacity=5000,gamma=gamma,seed=int(cfg['seed']))
    accepted={}
    attempted=[]
    per_worker=[]
    guard_keys=set()
    guard_evidence=[]
    preserved_pending=0

    for env in args.env_ids:
        proposals=load_proposals(reference/'workers'/f'env{env:02d}.jsonl')[:args.max_proposals]
        worker=SimpleWorker(SimpleFactory(
            env_id=env, seed=int(cfg['seed'])+env*1009, normalization=norm,
            directory=str(output/'workers'), intervention_enabled=True))
        report=dict(env_id=env, recorded=len(proposals), completed=0, guard=0,
                    partial=0, refused=0, first_guard_frame=None,
                    replay_pre_guard_max_eef_error_m=0., guard_discarded_candidates=0,
                    safety_stops=[])
        try:
            previous,_=worker.reset()
            previous=obs_tensors(previous)
            first_guard=False
            for entry in proposals:
                if worker.dead:
                    break
                proposal=np.asarray(entry['policy_action'],dtype=np.float32)
                nxt,reward,term,trunc,info=worker.step(proposal)
                payload=json.loads(info['payload'])
                if int(payload['env_id'])!=env or int(payload['episode_id'])!=0:
                    fail('worker episode/provenance mismatch')
                complete=bool(payload.get('executed'))
                owners=payload.get('auto_intervention',{}).get('per_arm_control', [])
                is_guard='BoundaryGuard' in owners
                key=(env, int(payload['episode_id']), int(payload['frame']))
                attempted.append(dict(env=env,episode=0,frame=int(payload['frame']),
                                      complete=complete,owners=owners,
                                      source=payload.get('action_source'),
                                      partial=bool(payload.get('partial_physics_step',False))))
                if not complete:
                    report['partial']+=int(payload.get('partial_physics_step',False))
                    report['refused']+=int(not payload.get('partial_physics_step',False))
                    report['safety_stops'].append(dict(
                        frame=int(payload['frame']),
                        reason=payload.get('supervisor_stop_reason')))
                    for item in assembler.close(env,
                            bootstrap=bool(payload['bootstrap_allowed']),
                            boundary_kind=int(payload['boundary_kind'])):
                        online.add(item)
                    dynamic.close_episode(env, bool(payload['bootstrap_allowed']),
                                          int(payload['boundary_kind']))
                    break
                next_obs=obs_tensors(nxt)
                if key in accepted:
                    fail(f'duplicate real completed control {key}')
                if not first_guard and not is_guard:
                    if not entry.get('executed'):
                        fail(f'old episode differs before first Guard {key}')
                    prev=np.asarray(entry['eef_positions'],dtype=np.float64)
                    now=np.asarray(payload['eef_positions'],dtype=np.float64)
                    e=float(np.max(np.abs(prev-now)))
                    report['replay_pre_guard_max_eef_error_m']=max(
                        report['replay_pre_guard_max_eef_error_m'],e)
                    if e>1e-5:
                        fail(f'canonical old/new prefix diverged before Guard {key}: {e} m')
                if is_guard and not first_guard:
                    report['first_guard_frame']=int(payload['frame'])
                    first_guard=True
                action=torch.as_tensor(payload['executed_action'],dtype=torch.float32)
                if action.shape != (14,) or not torch.isfinite(action).all():
                    fail(f'nonfinite completed executed action {key}')
                original_physical=np.asarray(payload['executed_action_physical'],dtype=np.float32)
                reconstructed_physical=worker.scaler.unscale(action).detach().cpu().numpy()
                if original_physical.shape != (14,) or not np.allclose(
                        original_physical,reconstructed_physical,atol=2e-6,rtol=0):
                    fail(f'executed normalized vs physical OSC action mismatch {key}')
                row=make_row(previous,action,next_obs,reward,term,trunc,payload)
                if not bool(row['complete_physics']):
                    fail(f'incomplete physics produced a Replay row: {key}')
                if not torch.equal(row['action'],action):
                    fail(f'actual action mismatch in raw Replay {key}')
                expected_source=9 if owners==['BoundaryGuard','Guide'] else (
                    9 if owners and owners[0]=='BoundaryGuard' else
                    10 if len(owners)>1 and owners[1]=='BoundaryGuard' else
                    13 if 'Guide' in owners else 0)
                if is_guard:
                    guard_keys.add(key)
                    report['guard']+=1
                    if int(row['action_source']) != expected_source:
                        fail(f'safety source code mismatch {key}')
                    if bool(row['expert_action_mask'].any()) or bool(row['quality_validated']):
                        fail(f'safety action falsely credited as expert {key}')
                    for ai,source in enumerate(owners):
                        if source=='BoundaryGuard':
                            if int(row['pose_sources'][ai])!=4:
                                fail(f'missing safety pose source {key}')
                            if bool(row['policy_dimension_mask'][7*ai:7*ai+6].any()):
                                fail(f'safety-owned pose credited to Policy {key}')
                before_candidates=len(dynamic.pending.get(env,()))
                for item in assembler.append(row):
                    online.add(item)
                result=payload.get('recovery_result') or {}
                dynamic.observe(row, decision=payload.get('auto_intervention') or {},
                    handbacks=result.get('handbacks', []),
                    control_events=result.get('control_events', []),
                    safe=not bool(payload.get('supervisor_stop_reason')))
                if is_guard:
                    after_candidates=len(dynamic.pending.get(env,()))
                    if after_candidates:
                        fail(f'safety event left uncertified expert candidates {key}')
                    preserved_pending+=before_candidates
                    guard_evidence.append(dict(
                        env_id=env,episode_id=0,frame=int(payload['frame']),
                        per_arm_control=owners,
                        action_source=payload.get('action_source'),
                        pre_rgb_sha256={k:digest(previous[k]) for k in IMAGES},
                        post_rgb_sha256={k:digest(next_obs[k]) for k in IMAGES},
                        executed_action_normalized=action.tolist(),
                        executed_action_physical=original_physical.tolist(),
                        physics_execution=payload.get('auto_intervention',{}).get('physics_execution'),
                        boundary_safety=payload.get('auto_intervention',{}).get('boundary_safety'),
                        pending_guide_candidates_before=before_candidates,
                        pending_guide_candidates_after=after_candidates,
                        replay_frame_key=list(key)))
                accepted[key]=dict(
                    source=int(row['action_source']),
                    normalized_action=action.clone(),
                    obs={k:v.clone() for k,v in previous.items()},
                    next_obs={k:v.clone() for k,v in next_obs.items()},
                    owners=owners)
                report['completed']+=1
                previous=next_obs
                if term or trunc:
                    break
            for item in assembler.truncate(env):
                online.add(item)
            dynamic.close_episode(env,bootstrap=True,kind=6)
        finally:
            worker.close()
        per_worker.append(report)
        print(f"ENV {env:02d}: completed={report['completed']} guard={report['guard']}",
              flush=True)

    stored=online[:]
    stored_guard=set()
    pixel_checked=0
    keys=set()
    for item in stored.unbind(0):
        key=tuple(int(item[k]) for k in ('env_id','episode','frame'))
        if key in keys:
            fail(f'duplicate Online Replay stored key {key}')
        keys.add(key)
        evidence=accepted.get(key)
        if evidence is None:
            fail(f'Online Replay contains an unexecuted or unknown action {key}')
        if not torch.equal(item['action'],evidence['normalized_action']):
            fail(f'Online Replay changed real action {key}')
        n=int(item['steps_to_next_obs'])
        final_key=(key[0],key[1],key[2]+n-1)
        end=accepted.get(final_key)
        if end is None:
            fail(f'n-step next observation points beyond complete physics {key}')
        for image in IMAGES:
            if not torch.equal(item['obs',image],evidence['obs'][image]):
                fail(f'Online Replay before-image mismatch {key} {image}')
            if not torch.equal(item['next','obs',image],end['next_obs'][image]):
                fail(f'Online Replay next-image mismatch {key} {image}')
            pixel_checked+=2
        if key in guard_keys:
            stored_guard.add(key)
            if int(item['action_source']) not in (9,10,11):
                fail(f'Guard action source lost by n-step {key}')
            if bool(item['quality_validated']) or bool(item['expert_action_mask'].any()):
                fail(f'Guard action became expert in Online Replay {key}')
    if guard_keys != stored_guard:
        fail(f'missing actual Guard rows in Online Replay: {sorted(guard_keys-stored_guard)}')
    if set(accepted) != keys:
        fail(f'missing complete rows in Online Replay: {sorted(set(accepted)-keys)}')
    demo_keys=set()
    if len(dynamic.buffer):
        for item in dynamic.buffer[:].unbind(0):
            key=tuple(int(item[k]) for k in ('env_id','episode','frame'))
            demo_keys.add(key)
            if key in guard_keys or int(item['action_source']) in (9,10,11):
                fail(f'Guard contamination in dynamic Demo {key}')
            if not bool(item['quality_handoff_confirmed']):
                fail(f'Unverified handback in dynamic Demo {key}')
    verified=bool(guard_keys and preserved_pending>0)
    report=dict(status='PASS' if verified else
                'PARTIAL_NO_REAL_GUARD' if not guard_keys else
                'PARTIAL_NO_PENDING_CANDIDATE_DISCARD_EVIDENCE',
        real_completed=len(accepted),online_rows=len(online),
        real_guard_completed=len(guard_keys),guard_rows_stored=len(stored_guard),
        dynamic_demo_rows=len(dynamic.buffer),
        dynamic_demo_guard_rows=len(demo_keys & guard_keys),
        uncertified_candidate_discarded_at_guard=preserved_pending,
        pixel_exact_comparisons=pixel_checked,
        attempted=len(attempted),
        observed_safety_refusals=sum(not p['complete'] for p in attempted),
        workers=per_worker,
        dynamic_stats=dynamic.stats,
        claim='Production real MuJoCo observations and actual completed replay path; '
              'does not prove long-term risk clearance or NPU training.')
    (output/'audit_result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (output/'attempted_provenance.json').write_text(json.dumps(attempted,indent=2)+"\n")
    (output/'real_guard_rgb_and_action_evidence.json').write_text(
        json.dumps(guard_evidence,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ('workers','dynamic_stats')},
                     indent=2,ensure_ascii=False))
    if not verified:
        raise RuntimeError("Real Replay/Demo audit has incomplete observed guard or "
                           "pending-candidate evidence; report is not PASS")


if __name__=='__main__':
    main()
