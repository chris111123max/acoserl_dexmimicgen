"""Existing canonical reset and 4x4 collector, with the independent supervisor.

No alternate simulator, state teleportation or duplicate learner is introduced.
Only the supervisor and per-arm provenance differ from the retained collectors.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import multiprocessing
import os
import random
import sys
import time
import numpy as np
import gymnasium as gym
from acoserl.parallel_autoserl import AutoSERLWorker, FourByFour, SpaceOnly, spaces, IMAGES


@dataclass
class SimpleFactory:
    env_id: int
    seed: int
    normalization: dict
    directory: str
    intervention_enabled: bool = True
    resume: dict | None = None

    def __call__(self):
        if multiprocessing.current_process().name == 'MainProcess':
            return SpaceOnly()
        return SimpleWorker(self)


class SimpleWorker(AutoSERLWorker):
    """Reuse observation/reset/I/O plumbing, not the enhanced control chain."""
    def __init__(self, cfg):
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        os.environ.update(LP_NUM_THREADS='1', MESA_GLTHREAD='false', ASCEND_RT_VISIBLE_DEVICES='',
                          MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
        import torch
        torch.set_num_threads(1); torch.manual_seed(cfg.seed); random.seed(cfg.seed)
        if 'torch_npu' in sys.modules:
            raise RuntimeError('CPU collector imported NPU plugin')
        from acoserl.autoserl_env import StrictAssemblyEnv, InterventionBridge
        from acoserl.simple_autoserl import SimpleBimanualAutoSERL
        from acoserl.single_demo_rlpd import normalize
        from resfit.rl_finetuning.utils.normalization import StateStandardizer, ActionScaler

        class SimpleBridge(InterventionBridge):
            def __init__(self, env, scaler, enabled):
                self.env, self.scaler = env, scaler
                self.manager = SimpleBimanualAutoSERL(env.env, intervention_enabled=enabled)
                self.manager.action_bounds = (scaler.limits.min.numpy().copy(), scaler.limits.max.numpy().copy())

        self.torch, self.normalize, self.cfg = torch, normalize, cfg
        self.env_id, self.directory = cfg.env_id, Path(cfg.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = self.directory/f'env{self.env_id:02d}.jsonl'
        self.stats = dict(env_id=self.env_id, pid=os.getpid(), seed=cfg.seed, episodes=0, resets=0,
                          real_steps=0, proposals=0, interventions=0, recoveries=0, safe_stops=0,
                          reasons={}, episode_length_sum=0, episode_length_histogram={},
                          assisted_successes=0, autonomous_successes=0, intervention_steps=0,
                          guide_events=0, recovery_events=0, handoff_disagreements=0)
        np.random.seed(cfg.seed)
        self.event('construct_start', runtime='privileged_handoff_v1_event_termination_10')
        self.w = StrictAssemblyEnv(camera_size=84, env_id=self.env_id)
        n = cfg.normalization
        self.standardizer = StateStandardizer(torch.tensor(n['state_mean']), torch.tensor(n['state_std_population']), min_std=.1)
        self.scaler = ActionScaler(torch.tensor(n['effective_action_min']), torch.tensor(n['effective_action_max']), action_scale=0, min_range_per_dim=.1)
        self.bridge = SimpleBridge(self.w, self.scaler, cfg.intervention_enabled)
        self.action_space = gym.spaces.Box(-1, 1, (14,), np.float32)
        self.observation_space = spaces()
        self.preparing = self.prepared_episode = False
        self.episode, self.frame, self.dead, self.forced, self.runner = -1, 0, True, False, None
        if cfg.resume:
            if cfg.resume.get('protocol') != SimpleBimanualAutoSERL.protocol:
                raise ValueError('worker checkpoint protocol mismatch')
            for key in self.stats:
                if key not in ('pid', 'env_id', 'seed') and key in cfg.resume:
                    self.stats[key] = copy.deepcopy(cfg.resume[key])
            self.episode = int(cfg.resume['episode_id'])
            self.bridge.manager.sequence = int(cfg.resume['sequence'])
            self.bridge.manager.auto_disabled = bool(cfg.resume['auto_disabled'])
        self.event('construct_done')

    def checkpoint_state(self):
        m = self.bridge.manager
        return dict(self.diagnostic(), protocol=m.protocol, sequence=m.sequence, auto_disabled=m.auto_disabled,
                    controller_audit=[dict(mode=a.mode, start=a.start, floor=a.floor, takeover=a.takeover) for a in m.arms],
                    resume_boundary='new canonical episode; no mid-physics state restore')

    def prepare_pregrasp(self, plans):
        raise RuntimeError('simple baseline uses external real-action fixtures, never collector preparation')

    def step(self, action):
        from acoserl.autoserl_env import SafeStopBeforeStep
        from acoserl.bimanual_core import snapshot
        from acoserl.transition_semantics import boundary
        if self.dead:
            raise RuntimeError('explicit reset required')
        self.stats['proposals'] += 1
        before = float(self.w.env.sim.data.time); started = time.perf_counter()
        executed = False; reward = 0.; term = trunc = False; info = {}; outcome = None
        try:
            if self.forced:
                raise SafeStopBeforeStep(dict(reason='collector_cut', collector_cut=True))
            raw, reward, term, trunc, info = self.bridge.step(action)
            executed = True; self.last = self.obs(raw); self.frame += 1; self.stats['real_steps'] += 1
            d, a = info['auto_intervention'], info['recovery_result']
            self.stats['intervention_steps'] += int(d['mode'] != 'Normal')
            self.stats['recoveries'] += len(a.get('handbacks', []))
        except SafeStopBeforeStep as error:
            trunc = True
            info = dict(auto_intervention=error.decision, supervisor_stop_reason=error.decision.get('reason'),
                        partial_physics_step=error.decision.get('partial_physics_step', False))
        m = self.bridge.manager
        if term or trunc:
            self.dead = True; self.stats['episodes'] += 1; self.stats['episode_length_sum'] += self.frame
            hist = self.stats['episode_length_histogram']; hist[str(self.frame)] = hist.get(str(self.frame), 0)+1
            reason = info.get('supervisor_stop_reason')
            if reason:
                self.stats['safe_stops'] += 1
                self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0)+1
            outcome = m.episode_result(bool(info.get('success', False)))
            self.stats['assisted_successes'] += int(outcome['assisted_success'])
            self.stats['autonomous_successes'] += int(outcome['autonomous_success'])
            self.stats['guide_events'] += outcome['guide_events']
            self.stats['recovery_events'] += outcome['recovery_events']
            self.stats['handoff_disagreements'] += outcome['handoff_disagreement_steps']
            self.stats['interventions'] += outcome['intervention_count']
            self.event('episode_end', episode=self.episode, length=self.frame, reason=reason, outcome=outcome)
        state = snapshot(self.w.env)
        kind, bootstrap = boundary(term, trunc, bool(info.get('success', False)), info.get('supervisor_stop_reason'),
                                   executed, bool(info.get('partial_physics_step', False)))
        decision = info.get('auto_intervention') or {}
        controls = decision.get('per_arm_control', ('Policy', 'Policy'))
        safety = [ai for ai, source in enumerate(controls) if source == 'BoundaryGuard']
        executed_source = ('Constrained_Joint' if len(safety) == 2 else
                           'Constrained_A' if safety == [0] else
                           'Constrained_B' if safety == [1] else
                           'Policy' if decision.get('mode') == 'Normal' else
                           'Handoff_AB' if decision.get('mode') == 'Handoff_AB' else 'Guide_AB')
        positions = np.asarray([p['position'] for p in state['poses']])
        motion = np.linalg.norm(positions-self.previous_eef_positions, axis=1)
        self.previous_eef_positions = positions.copy()
        payload = dict(env_id=self.env_id, episode_id=self.episode, frame=self.frame-1 if executed else self.frame,
            executed=executed, complete_physics=executed, boundary_kind=int(kind), bootstrap_allowed=bootstrap,
            partial_physics_step=bool(info.get('partial_physics_step', False)),
            physics_time_advanced=float(self.w.env.sim.data.time)-before, step_seconds=time.perf_counter()-started,
            policy_action=np.asarray(action).tolist(), auto_intervention=decision,
            takeover_id=decision.get('takeover_id', 0), stage=decision.get('stage', 'unknown'),
            action_source=executed_source if executed else 'Rejected',
            recovery_result=info.get('recovery_result'), task_quality=info.get('task_quality', {}),
            supervisor_stop_reason=info.get('supervisor_stop_reason'), success=bool(info.get('success', False)),
            episode_outcome=outcome, task_metrics=state['official_metrics'], supervisor_grasps=state['actual_grasps'],
            supervisor_objects=state['objects'], eef_positions=positions.tolist(),
            eef_rotations=[p['rotation'].tolist() for p in state['poses']],
            nonzero_physical_motion=bool(executed and np.any(motion > 1e-5)), prepared_episode=False)
        if executed:
            payload.update(executed_action=info['executed_action_normalized'].numpy().tolist(),
                           executed_action_physical=info['executed_action_physical'].tolist(),
                           intervention_action_normalized=info['intervention_action_normalized'].tolist())
        self.event('step', **{k: v for k, v in payload.items() if k != 'env_id'}, reward=float(reward))
        return self.last, float(reward), bool(term), bool(trunc), {'payload': json.dumps(
            payload, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))}


class SimplePool(FourByFour):
    """Same asynchronous groups and normalization as the retained learner."""
    def __init__(self, seed, normalization, directory, enabled=True, resume_workers=None, **_):
        self.groups = []; self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.startup = self.directory/'startup.jsonl'
        try:
            for g in range(4):
                factories = [SimpleFactory(i, seed+i*1009, normalization, str(self.directory), enabled,
                                           None if resume_workers is None else resume_workers[i]) for i in range(4*g, 4*g+4)]
                v = gym.vector.AsyncVectorEnv(factories, shared_memory=True, context='spawn',
                                             autoreset_mode=gym.vector.AutoresetMode.DISABLED)
                self.groups.append(v); v.reset()
                self.record('batch_ready', group=g, workers=v.call('diagnostic'))
            self.observation = {k: np.concatenate([v.observations[k] for v in self.groups]) for k in IMAGES+['observation.state']}
            self.record('all_sixteen_ready', workers=self.diagnostics())
        except BaseException:
            self.close(); raise


class SimpleCapability:
    def __init__(self, state=None):
        self.disabled = bool((state or {}).get('disable', False))

    def observe(self, payload, **_):
        outcome = payload.get('episode_outcome') or {}
        if outcome.get('protocol') == 'privileged_handoff_v1_event_termination_10' and outcome.get('disable_requested'):
            self.disabled = True

    def snapshot(self):
        return {'disable': self.disabled}
