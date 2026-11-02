"""Simulator-only dual-arm AutoSERL baseline; no V2/V3 control inheritance.

Reference: autoserl/AutoSERL auto_intervention_wrapper. Each arm owns its
window, nearest-reference history, Guide and P0 -> P1 replay. Adaptations for
this grasping task (explicit gripper actions, event barriers, bounded recovery
and observed handback) are documented, not claimed to be upstream features.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('A', 'B')
EXPERT = frozenset(('Guide', 'ReturnP0', 'Replay', 'Settle'))
# Non-safety control limits, kept separate from 350-step recovery and hard
# physical protections. An abandoned Guide is NOT an expert handback.
GUIDE_MAX_CONTROLS = 80
GUIDE_PROGRESS_WINDOW = 20
GUIDE_MIN_POSITION_PROGRESS_M = .001
GUIDE_MIN_ROTATION_PROGRESS_RAD = .01
POLICY_AFTER_GUIDE_STEPS = 32
# Bounded task-adaptation settling; never relax the 10-step grasp criterion.
SIMPLE_SETTLE_MAX_CONTROLS = 80


@dataclass
class ArmState:
    """No state shared with the other arm, except the final safety check."""
    p0: int
    p1: int
    start: int = 0
    floor: int = 0
    mode: str = 'Policy'
    target: int | None = None
    replay: int | None = None
    takeover: int = 0
    age: int = 0
    stable: int = 0
    cooldown: int = 0
    handoff_holding: bool = False
    nearest: deque = field(default_factory=lambda: deque(maxlen=20))
    guide_errors: deque = field(default_factory=lambda: deque(maxlen=GUIDE_PROGRESS_WINDOW))
    relative: deque = field(default_factory=lambda: deque(maxlen=10))

    @property
    def end(self):
        # Upstream uses P1-P0+1 entries, starting at [0, ..., L-1].
        return self.start + self.p1 - self.p0

    def reset_intervention(self, holding=False):
        self.handoff_holding = bool(holding)
        self.mode = 'Handoff'
        self.age = self.stable = 0
        self.nearest.clear()
        self.guide_errors.clear()


def forward_direction(positions, nearest, current, target):
    i = min(int(nearest), len(positions)-2)
    tangent = positions[i+1] - positions[i]
    toward = target - current
    norm = np.linalg.norm(tangent) * np.linalg.norm(toward)
    # Upstream divides by zero on stationary samples. Do not invent a direction.
    return bool(norm > 1e-12 and float(tangent @ toward) >= 0.)


def relative_pose(state, ai, obj):
    pose, item = state['poses'][ai], state['objects'][obj]
    rotation = np.asarray(pose['rotation'])
    return (rotation.T @ (np.asarray(item['position'])-pose['position']),
            rotation.T @ Rotation.from_quat(np.roll(item['quaternion_wxyz'], -1)).as_matrix())


class SimpleBimanualAutoSERL:
    """One explicit control path; Policy/Guide/Recovery compose before physics.

    The legacy project is imported only for observations and OSC conversion.
    None of its intervention eligibility, QP or certificate code is called.
    Actual force/workspace/collision checks and ActionOnly substeps remain on.
    """
    protocol = 'simple_bimanual_autoserl_v1'

    def __init__(self, env, config=None, intervention_enabled=True):
        from acoserl.bimanual_core import DEFAULT, Contacts
        self.env = env
        self.cfg = copy.deepcopy(DEFAULT)
        if config:
            self.cfg.update({k: v for k, v in config.items() if k != 'manager'})
        self.cfg['intervention_enabled'] = bool(intervention_enabled)
        self.points = json.loads((ROOT/'configs/bimanual_autoserl/recovery_points.json').read_text())
        with np.load(ROOT/self.points['reference_path']) as trace:
            self.trace = {k: trace[k].copy() for k in trace.files}
        self.positions = self.trace['positions']
        self.rotations = self.trace['rotations']
        self.n_states = len(self.positions)
        if self.positions.shape[1:] != (2, 3) or len(self.trace['actions'])+1 != self.n_states:
            raise ValueError('inconsistent dual-arm reference')
        for arm in ARMS:
            p = self.points['arms'][arm]
            if not 0 <= p['p0'] < p['first_grasp'] <= p['p1'] < self.n_states:
                raise ValueError('invalid P0/P1 grasp interval')
        self.contacts = Contacts(env)
        self.auto_disabled = False
        self.sequence = 0
        self.reset()

    def reset(self, initial_progress=(0, 0)):
        from acoserl.bimanual_core import snapshot
        self.arms = [ArmState(self.points['arms'][a]['p0'], self.points['arms'][a]['p1'], int(i))
                     for a, i in zip(ARMS, initial_progress)]
        self.completed = [False, False]
        self.grasp_streak = [0, 0]
        self.mode = 'Normal'
        self.stop_reason = None
        self.last_decision = None
        self.steps = self.intervention_control_steps = self.episode_interventions = 0
        self.handbacks = [0, 0]
        self.initial_snapshot = snapshot(self.env)
        self._issued = None
        self._segment_signature = None
        self._segment = 0
        self._touched = set()
        self._stable_grasp = set()
        self._assembly_anchor = None
        self._assembly_rest = deque(maxlen=3)
        self._error_before = 0.
        self._issued_targets = [None, None]
        self.events = []
        self._event_cursor = 0

    def indices(self):
        return [min(x.end, self.n_states-1) for x in self.arms]

    def pose(self, ai, index):
        return dict(position=self.positions[index, ai].copy(), rotation=self.rotations[index, ai].copy())

    def stop(self, reason):
        self.mode, self.stop_reason = 'Safe_Stop', str(reason)

    def _refuse(self, policy, reason):
        self.stop(reason)
        self.last_decision = dict(mode='Safe_Stop', execute=False, reason=reason,
                                  policy_action=np.asarray(policy).tolist(), indices=self.indices())
        self._issued = None
        return None, self.last_decision

    def _stage(self, state, ai):
        if self.completed[ai]:
            return 'release_completion'
        held = state['actual_grasps'][ARMS[ai]][self.points['arms'][ARMS[ai]]['object']]
        return ('independent_carry' if self.grasp_streak[ai] >= 10 else 'grasp_stabilization') if held else 'approach_pregrasp'

    def _active_contact(self, contact, ai):
        return any(ARMS[ai] in (e['a'], e['b']) for e in contact['edges'])

    def _assembly_contact(self, state, contact, ai):
        obj = self.points['arms'][ARMS[ai]]['object']
        return any(obj in (e['a'], e['b']) and
                   bool(({e['a'], e['b']}-{obj}) & {'base', 'piece_1', 'piece_2'})
                   for e in contact['edges'])

    def physical_danger(self, state, contact):
        if not np.isfinite(state['qpos']).all() or not np.isfinite(state['qvel']).all():
            return 'nonfinite_physics'
        if contact['max_robot_contact_force_N'] > self.cfg['robot_force_stop_N']:
            return 'robot_force_limit'
        for ai, arm in enumerate(ARMS):
            p = np.asarray(state['poses'][ai]['position'])
            lo = self.positions[:, ai].min(0)-self.cfg['free_workspace_margin_m']
            hi = self.positions[:, ai].max(0)+self.cfg['free_workspace_margin_m']
            if not np.isfinite(p).all() or np.any(p < lo) or np.any(p > hi):
                return 'workspace_exit_'+arm
            if any(h for obj, h in state['actual_grasps'][arm].items()
                   if obj != self.points['arms'][arm]['object']):
                return 'unexpected_grasp_'+arm
        for edge in contact['edges']:
            pair = {edge['a'], edge['b']}
            if pair in ({'A', 'B'}, {'A', 'static'}, {'B', 'static'}):
                return 'robot_environment_collision'
            # Task-object contact itself is not a collision certificate.
            # Normal Actor assembly remains allowed under measured force and
            # workspace limits; interventions are restricted below.
        # Policy is allowed to perform the actual assembly. A correction must
        # never rewind through an unknown mechanical coupling or assembly.
        issued = self._issued or ('Policy', 'Policy')
        if any(s in EXPERT for s in issued):
            if contact['dynamic_A_B_coupled']:
                return 'coupled_intervention_refused'
            for ai, source in enumerate(issued):
                if source in EXPERT:
                    own = self.points['arms'][ARMS[ai]]['object']
                    non_target = any(ARMS[ai] in (e['a'], e['b']) and
                                     bool(({e['a'], e['b']}-{ARMS[ai]}) &
                                          ({'base', 'piece_1', 'piece_2'}-{own}))
                                     for e in contact['edges'])
                    if non_target or self._assembly_contact(state, contact, ai):
                        return 'assembly_intervention_refused_'+ARMS[ai]
        if self._assembly_anchor is not None:
            from acoserl.object_geometry import object_error
            error = object_error(state['objects']['piece_1'], self._assembly_anchor)
            if (not state['official_metrics']['first_piece_assembled'] or
                    error['position_m'] > self.cfg['assembled_object_drift_m'] or
                    error['rotation_rad'] > self.cfg['assembled_object_rotation_drift_rad']):
                return 'completed_piece1_progress_lost'
        return None

    def substep_safety(self, contact):
        from acoserl.bimanual_core import snapshot
        return self.physical_danger(snapshot(self.env), contact)

    def _domain(self, state, contact, ai):
        """Task events cap reference search, not a hypothetical open-jaw gap."""
        arm = ARMS[ai]
        p = self.points['arms'][arm]
        if self.completed[ai] or self._assembly_contact(state, contact, ai):
            return None
        if state['actual_grasps'][arm][p['object']]:
            barrier = self.points['assembly'][ai]['first_contact']
            return int(p['first_grasp']), int(barrier)-1
        # Sliding correction stays before first contact; P0/P1 replay supplies
        # the gripper-equipped transition into stable grasp.
        return 0, int(p['first_contact'])-1

    def _measure(self, state, ai, domain):
        low, high = domain
        arm = self.arms[ai]
        pos = state['poses'][ai]['position']
        nearest = low+int(np.argmin(np.linalg.norm(self.positions[low:high+1, ai]-pos, axis=1)))
        begin = max(low, min(arm.start, high))
        end = min(high, max(begin, arm.end))
        target = begin+int(np.argmin(np.linalg.norm(self.positions[begin:end+1, ai]-pos, axis=1)))
        distance = float(np.linalg.norm(self.positions[target, ai]-pos))
        stagnant = (len(arm.nearest) == arm.nearest.maxlen and
                    np.linalg.norm(self.positions[arm.nearest[-1], ai]-self.positions[arm.nearest[0], ai])
                    < self.cfg['stagnation_motion_m'])
        return nearest, target, distance, stagnant

    def _can_return(self, state, contact, ai):
        arm, p = ARMS[ai], self.points['arms'][ARMS[ai]]
        if (any(state['actual_grasps'][arm].values()) or self._active_contact(contact, ai) or
                contact['dynamic_A_B_coupled']):
            return False
        from acoserl.object_geometry import object_error
        oi = ('base', 'piece_1', 'piece_2').index(p['object'])
        ref = dict(position=self.trace['object_positions'][p['p0'], oi],
                   quaternion_wxyz=self.trace['object_quaternions'][p['p0'], oi])
        err = object_error(state['objects'][p['object']], ref)
        return err['position_m'] <= .003 and err['rotation_rad'] <= .035

    @staticmethod
    def _guide_no_progress(arm):
        """Detect a plateau at a fixed Guide target, not merely a fixed nearest index.

        A nearest reference index can remain fixed while the robot is still
        meaningfully translating or rotating toward that target.
        """
        if len(arm.guide_errors) != GUIDE_PROGRESS_WINDOW:
            return False
        start_pos, start_rot = arm.guide_errors[0]
        end_pos, end_rot = arm.guide_errors[-1]
        return (start_pos - end_pos < GUIDE_MIN_POSITION_PROGRESS_M and
                start_rot - end_rot < GUIDE_MIN_ROTATION_PROGRESS_RAD)

    def _release_guide_to_policy(self, ai, reason):
        """Return one arm to Policy without falsely declaring a safe handback.

        No command is synthesized, no physical gate is bypassed and the active
        Guide's unsuccessful actions are not treated as a completed recovery.
        """
        arm = self.arms[ai]
        if arm.mode != 'Guide':
            raise RuntimeError('only an active Guide may be released here')
        prior_takeover = arm.takeover
        arm.mode, arm.target = 'Policy', None
        arm.age = arm.stable = 0
        arm.cooldown = max(arm.cooldown, POLICY_AFTER_GUIDE_STEPS)
        arm.nearest.clear()
        arm.guide_errors.clear()
        self.events.append(dict(event='guide_released_to_policy', arm=ARMS[ai],
            reason=reason, takeover=prior_takeover,
            policy_budget=POLICY_AFTER_GUIDE_STEPS))

    def _start(self, ai, mode, target):
        arm = self.arms[ai]
        self.sequence += 1
        arm.mode, arm.target, arm.takeover = mode, int(target), self.sequence
        arm.age = arm.stable = 0
        arm.relative.clear()
        arm.nearest.clear()
        arm.guide_errors.clear()
        self.episode_interventions += 1
        self.events.append(dict(event='start', arm=ARMS[ai], kind=mode, target=int(target), takeover=arm.takeover))

    def choose(self, policy_action):
        from acoserl.bimanual_core import snapshot, servo
        policy = np.asarray(policy_action, dtype=float)
        if policy.shape != (14,) or not np.isfinite(policy).all():
            return self._refuse(policy, 'invalid_policy_action')
        state, contact = snapshot(self.env), self.contacts.read()
        if self.stop_reason:
            return self._refuse(policy, self.stop_reason)
        danger = self.physical_danger(state, contact)
        if danger:
            return self._refuse(policy, danger)
        self.steps += 1
        self._event_cursor = len(self.events)
        enabled = self.cfg['intervention_enabled'] and not self.auto_disabled
        action = policy.copy()
        sources = ['Policy', 'Policy']
        targets = copy.deepcopy(state['poses'])
        grips = [float(policy[6]), float(policy[13])]
        measurements = []
        self._issued_targets = [None, None]
        for ai, arm in enumerate(self.arms):
            domain = self._domain(state, contact, ai)
            measurement = self._measure(state, ai, domain) if domain else None
            measurements.append(measurement)
            if not enabled:
                continue
            # A Guide that no longer advances along the reference may safely
            # switch to P0 recovery. Qualify against both the reference and
            # the actual target-error trend, then retain the original
            # empty-gripper/contact/object-pose recovery requirements.
            if arm.mode == 'Guide' and domain and not contact['dynamic_A_B_coupled']:
                nearest, _, _, reference_stalled = measurement
                if reference_stalled and self._guide_no_progress(arm):
                    if arm.p0 <= nearest <= arm.p1 and self._can_return(state, contact, ai):
                        previous_takeover = arm.takeover
                        self._start(ai, 'ReturnP0', arm.p0)
                        self.events.append(dict(event='guide_stall_to_p0', arm=ARMS[ai],
                            from_takeover=previous_takeover, to_takeover=arm.takeover,
                            nearest_index=int(nearest)))
                    else:
                        self._release_guide_to_policy(ai, 'stalled_no_qualified_p0')
            if arm.mode == 'Policy' and domain and not contact['dynamic_A_B_coupled']:
                nearest, target, distance, stagnant = measurement
                # Recovery has priority over ordinary window correction.
                if (arm.cooldown == 0 and arm.p0 <= nearest <= arm.p1 and
                        stagnant and self._can_return(state, contact, ai)):
                    self._start(ai, 'ReturnP0', arm.p0)
                elif (arm.cooldown == 0 and distance > self.cfg['trigger_position_m'] and
                      forward_direction(self.positions[:, ai], nearest, state['poses'][ai]['position'],
                                        self.positions[target, ai])):
                    self._start(ai, 'Guide', target)
            if arm.mode in EXPERT:
                incompatible = (arm.mode == 'Guide' and domain is not None and
                                not domain[0] <= arm.target <= domain[1])
                if domain is None or incompatible:
                    # Do not plan a correction through an assembly event.
                    arm.reset_intervention(state['actual_grasps'][ARMS[ai]][self.points['arms'][ARMS[ai]]['object']])
                else:
                    sources[ai] = arm.mode
                    self._issued_targets[ai] = (arm.target if arm.mode == 'Guide' else arm.p0
                        if arm.mode == 'ReturnP0' else arm.p1 if arm.mode == 'Settle' else arm.replay+1)
                    if arm.mode == 'Guide':
                        targets[ai] = self.pose(ai, arm.target)
                    elif arm.mode == 'ReturnP0':
                        if not self._can_return(state, contact, ai):
                            return self._refuse(policy, 'recovery_pregrasp_domain_changed_'+ARMS[ai])
                        targets[ai] = self.pose(ai, arm.p0); grips[ai] = -1.
                    elif arm.mode == 'Settle':
                        targets[ai] = self.pose(ai, arm.p1); grips[ai] = 1.
                    else:
                        # Reference indices designate states: replay actions
                        # [P0, P1) to reach state P1, never an off-by-one replay.
                        error = np.linalg.norm(state['poses'][ai]['position']-self.positions[arm.replay, ai])
                        if error > .02:
                            return self._refuse(policy, 'recovery_replay_tracking_'+ARMS[ai])
                        action[7*ai:7*ai+7] = self.trace['actions'][arm.replay, 7*ai:7*ai+7]
        if any(s in ('Guide', 'ReturnP0', 'Settle') for s in sources):
            correction = servo(self.env, targets, grips,
                max_position_step=self.cfg['position_command_cap_m'],
                max_rotation_step=self.cfg['rotation_command_cap_rad'],
                position_gain=self.cfg['position_gain'], rotation_gain=self.cfg['rotation_gain'])
            for ai, source in enumerate(sources):
                if source in ('Guide', 'ReturnP0', 'Settle'):
                    action[7*ai:7*ai+6] = correction[7*ai:7*ai+6]
                    if source != 'Guide':
                        action[7*ai+6] = grips[ai]
        lo, hi = getattr(self, 'action_bounds', (-np.ones(14), np.ones(14)))
        # Bound expert outputs, never silently alter an independent Policy arm.
        for ai, source in enumerate(sources):
            sl = slice(7*ai, 7*ai+7)
            if source in EXPERT:
                action[sl] = np.clip(action[sl], np.maximum(lo[sl], -1), np.minimum(hi[sl], 1))
            elif np.any(action[sl] < lo[sl]-1e-6) or np.any(action[sl] > hi[sl]+1e-6):
                return self._refuse(policy, 'policy_action_out_of_range')
        if not np.isfinite(action).all() or np.any(abs(action) > 1.000001):
            return self._refuse(policy, 'invalid_composed_action')
        distances = [np.linalg.norm(state['poses'][i]['position']-self.positions[j, i])
                     for i, j in enumerate(self._issued_targets) if j is not None]
        self._error_before = float(np.mean(distances)) if distances else 0.
        self._issued = tuple(sources)
        danger = self.physical_danger(state, contact)
        if danger:
            return self._refuse(policy, danger)
        signature = tuple((s, a.takeover if s in EXPERT else 0) for s, a in zip(sources, self.arms))
        if signature != self._segment_signature:
            self._segment += 1; self._segment_signature = signature
        expert = [s in EXPERT for s in sources]
        mode = 'Normal' if not any(expert) else 'Guide_AB'
        self.mode = mode
        self.last_decision = dict(mode=mode, execute=True, policy_action=policy.tolist(),
            executed_action=action.tolist(), intervention_action_proposed=action.tolist(),
            indices=self.indices(), takeover_id=self._segment, per_arm_control=sources,
            arm_takeover_ids=[a.takeover for a in self.arms], simple_protocol=self.protocol,
            measurements=measurements, recovery_step=1 if any(s in EXPERT and a.age == 0 for s, a in zip(sources, self.arms)) else 2,
            stage=max((self._stage(state, i) for i in range(2)), key=lambda x: x != 'approach_pregrasp'))
        return action, self.last_decision

    def after_step(self):
        from acoserl.bimanual_core import snapshot, errors
        if self._issued is None:
            raise RuntimeError('after_step requires a complete, accepted action')
        state, contact = snapshot(self.env), self.contacts.read()
        danger = self.physical_danger(state, contact)
        if danger:
            self.stop(danger)
        issued, handbacks = self._issued, []
        settle_diagnostics = {}
        self.intervention_control_steps += int(any(s in EXPERT for s in issued))
        for ai, arm in enumerate(self.arms):
            p = self.points['arms'][ARMS[ai]]
            held = bool(state['actual_grasps'][ARMS[ai]][p['object']])
            self.grasp_streak[ai] = self.grasp_streak[ai]+1 if held else 0
            completed = state['official_metrics'].get('first_piece_assembled' if ai == 0 else 'task', False)
            self.completed[ai] |= bool(completed and not held)
            if held:
                arm.relative.append(relative_pose(state, ai, p['object']))
            else:
                arm.relative.clear()
            stable = len(arm.relative) == arm.relative.maxlen
            relative_translation_span_m = None
            relative_rotation_span_rad = None
            if stable:
                x0, r0 = arm.relative[0]
                relative_translation_span_m = float(max(
                    np.linalg.norm(x-x0) for x, _ in arm.relative))
                relative_rotation_span_rad = float(max(
                    Rotation.from_matrix(r@r0.T).magnitude() for _, r in arm.relative))
                stable = (relative_translation_span_m < self.cfg['relative_grasp_motion_m'] and
                          relative_rotation_span_rad <= .035)
            if stable:
                self._stable_grasp.add(ai)
            domain = self._domain(state, contact, ai)
            if domain:
                nearest = self._measure(state, ai, domain)[0]
                arm.nearest.append(nearest)
                arm.floor = max(arm.floor, domain[0])
            if issued[ai] not in ('Guide', 'ReturnP0'):
                arm.start = min(self.n_states-1, max(arm.floor, arm.start+1))
            arm.cooldown = max(0, arm.cooldown-1)
            if issued[ai] in EXPERT:
                self._touched.add(ai); arm.age += 1
            if self.stop_reason:
                continue
            if issued[ai] == 'Guide':
                e = errors(state['poses'][ai], self.pose(ai, arm.target))
                arm.guide_errors.append((float(e['position_m']), float(e['rotation_rad'])))
                arm.stable = arm.stable+1 if e['position_m'] < .005 and e['rotation_rad'] < .02 else 0
                if arm.stable >= 3:
                    self.events.append(dict(event='guide_target_reached', arm=ARMS[ai],
                        takeover=arm.takeover, guide_controls=arm.age))
                    arm.reset_intervention(held)
            elif issued[ai] == 'ReturnP0':
                e = errors(state['poses'][ai], self.pose(ai, arm.p0))
                arm.stable = arm.stable+1 if e['position_m'] < .001 and e['rotation_rad'] < .02 else 0
                if arm.stable >= 3:
                    arm.mode, arm.replay, arm.stable = 'Replay', arm.p0, 0
            elif issued[ai] == 'Replay':
                arm.replay += 1
                if arm.replay >= arm.p1:
                    arm.mode = 'Settle'; arm.stable = 0
            elif issued[ai] == 'Settle':
                arm.stable += 1
                e = errors(state['poses'][ai], self.pose(ai, arm.p1))
                command = np.asarray(self.env.robots[ai].gripper['right'].current_action)
                closed = command.shape == (2,) and np.allclose(command, [-1., 1.])
                ready = bool(held and stable and closed and
                             e['position_m'] <= .005 and e['rotation_rad'] <= .02)
                settle_diagnostics[ARMS[ai]] = dict(
                    controls=int(arm.stable), limit=SIMPLE_SETTLE_MAX_CONTROLS,
                    held=bool(held), gripper_closed=bool(closed),
                    relative_window_steps=len(arm.relative),
                    required_window_steps=arm.relative.maxlen,
                    relative_translation_span_m=relative_translation_span_m,
                    relative_translation_limit_m=float(self.cfg['relative_grasp_motion_m']),
                    relative_rotation_span_rad=relative_rotation_span_rad,
                    relative_rotation_limit_rad=.035,
                    p1_position_error_m=float(e['position_m']),
                    p1_rotation_error_rad=float(e['rotation_rad']),
                    ready=ready)
                if ready:
                    arm.floor = max(arm.floor, arm.p1)
                    arm.start = max(arm.start, arm.p1)
                    arm.reset_intervention(True)
                elif arm.stable >= SIMPLE_SETTLE_MAX_CONTROLS:
                    # The overall max_recovery_steps check below remains
                    # authoritative. Never report an unstable grasp as a
                    # successful recovery merely because time elapsed.
                    self.stop('recovery_grasp_not_established_'+ARMS[ai])
            elif arm.mode == 'Handoff':
                if arm.handoff_holding and not held:
                    self.stop('handoff_grasp_lost_'+ARMS[ai]);continue
                handbacks.append(dict(arm=ARMS[ai], takeover_id=arm.takeover))
                self.handbacks[ai] += 1
                arm.mode = 'Policy'
                arm.cooldown = max(arm.cooldown, POLICY_AFTER_GUIDE_STEPS)
                self.events.append(dict(event='policy_handback_confirmed', arm=ARMS[ai],
                    takeover=arm.takeover, policy_budget=arm.cooldown))
            if self.stop_reason is None and arm.mode == 'Guide' and arm.age >= GUIDE_MAX_CONTROLS:
                # A failed Guide is not a failed P0/P1 recovery. Release only
                # this arm to normal Policy for a bounded exploration window.
                self._release_guide_to_policy(ai, 'guide_max_controls')
            elif (self.stop_reason is None and arm.mode in ('ReturnP0', 'Replay', 'Settle')
                  and arm.age >= self.cfg['max_recovery_steps']):
                self.stop('recovery_timeout_'+ARMS[ai])
        if self.completed[0] and state['official_metrics']['first_piece_assembled'] and self._assembly_anchor is None:
            self._assembly_rest.append(copy.deepcopy(state['objects']['piece_1']))
            if len(self._assembly_rest) == self._assembly_rest.maxlen:
                from acoserl.object_geometry import object_error
                deviations = [object_error(v, self._assembly_rest[0]) for v in self._assembly_rest]
                if all(v['position_m'] <= .0005 and v['rotation_rad'] <= .005 for v in deviations):
                    self._assembly_anchor = copy.deepcopy(state['objects']['piece_1'])
        elif not state['official_metrics']['first_piece_assembled']:
            self._assembly_rest.clear()
        self._issued = None
        if self.stop_reason:
            for item in handbacks:
                self.handbacks[ARMS.index(item['arm'])] -= 1
            handbacks = []
        distances = [np.linalg.norm(state['poses'][i]['position']-self.positions[j, i])
                     for i, j in enumerate(self._issued_targets) if j is not None]
        error_after = float(np.mean(distances)) if distances else 0.
        return dict(handback=bool(handbacks), safe_handback=bool(handbacks), handbacks=handbacks,
            takeover_id=self._segment, indices=self.indices(),
            task_quality=dict(eligible=not self.stop_reason and any(s in EXPERT for s in issued),
                              error_before_m=self._error_before, error_after_m=error_after, handoff_confirmed=bool(handbacks)),
            per_arm_modes=[a.mode for a in self.arms],
            per_arm_cooldown=[a.cooldown for a in self.arms],
            control_events=copy.deepcopy(self.events[self._event_cursor:]),
            stable_grasps=[ARMS[i] for i in sorted(self._stable_grasp)],
            settle_diagnostics=settle_diagnostics)

    def set_global_capability(self, value):
        self.auto_disabled = bool(self.auto_disabled or value.get('disable', False))

    def episode_result(self, success, **_):
        request = bool(success and self.intervention_control_steps < self.cfg['termination_intervention_steps'])
        self.auto_disabled |= request
        return dict(stable_full_success=bool(success),
                    assisted_success=bool(success and self.intervention_control_steps),
                    autonomous_success=bool(success and not self.intervention_control_steps),
                    intervention_control_steps=self.intervention_control_steps,
                    intervention_events=self.episode_interventions,
                    handbacks=sum(self.handbacks), disable_requested=request,
                    stable_grasp_arms=[ARMS[i] for i in sorted(self._stable_grasp)],
                    protocol=self.protocol)
