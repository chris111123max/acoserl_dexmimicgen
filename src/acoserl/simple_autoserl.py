"""Stage-aware dual-arm intervention and recovery for real MuJoCo rollouts.

Borrow the sliding-window, convergence-based Guide exit, and stagnation
recovery ideas from AutoSERL. This is a dual-arm training method, not an exact
reproduction of the original. Every anchor move is a real, safety-supervised
control sequence; no mid-episode physics state setters are used.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
import numpy as np
from scipy.spatial.transform import Rotation
from acoserl.privileged_handoff import PrivilegedHandoff

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('A', 'B')
RETURN_MODES = frozenset(('ReturnS0', 'ReturnP0', 'ReturnP1'))
EXPERT = frozenset(('Guide', 'ReturnS0', 'ReturnP0', 'ReturnP1', 'Replay', 'Settle'))
# Never credit a boundary-safety override as an Actor or expert action.
from acoserl.boundary_guard import SOURCE as BOUNDARY_GUARD, evaluate as boundary_risk, world_policy_displacement
SAFETY_ONLY = frozenset((BOUNDARY_GUARD,))
# Non-safety control limits, kept separate from 350-step recovery and hard
# physical protections. An abandoned Guide is NOT an expert handback.
GUIDE_MAX_CONTROLS = 80
GUIDE_PROGRESS_WINDOW = 20
GUIDE_MIN_POSITION_PROGRESS_M = .001
GUIDE_MIN_ROTATION_PROGRESS_RAD = .01
POLICY_AFTER_GUIDE_STEPS = 32
# AutoSERL-style alternative Guide exit: only after real progress AND
# convergence inside the permitted window, not merely 80 elapsed controls.
GUIDE_MOTION_WINDOW = 3
GUIDE_EXIT_PROGRESS_M = .005
GUIDE_CONVERGED_POSITION_M = .001
GUIDE_CONVERGED_ROTATION_RAD = .01
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
    boundary_guard_streak: int = 0
    boundary_guard_total: int = 0
    handoff_holding: bool = False
    recovery_anchor: int | None = None
    recovery_goal: int | None = None
    guide_origin: np.ndarray | None = None
    nearest: deque = field(default_factory=lambda: deque(maxlen=20))
    guide_errors: deque = field(default_factory=lambda: deque(maxlen=GUIDE_PROGRESS_WINDOW))
    guide_motion: deque = field(default_factory=lambda: deque(maxlen=GUIDE_MOTION_WINDOW))
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
        self.guide_motion.clear()


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
    protocol = 'privileged_handoff_v1_event_termination_10'

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
            p.setdefault('s0', 0)
            if not 0 <= p['s0'] < p['p0'] < p['first_grasp'] <= p['p1'] < self.n_states:
                raise ValueError('invalid S0/P0/P1 recovery anchors')
        self.contacts = Contacts(env)
        self.gripper_expert = PrivilegedHandoff(self.points, self.trace, self.cfg)
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
        self.guide_events = self.recovery_events = self.handoff_disagreements = 0
        self._handoff_pending = None
        self.gripper_expert.reset()
        self.handbacks = [0, 0]
        self.boundary_guard_steps = 0
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
        self._boundary_risks = [None, None]
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
        if str(reason).startswith('boundary_guard_'):
            # Keep refusal evidence even though no complete physical action
            # was executed and no Replay transition will be produced.
            self.last_decision.update(
                boundary_safety=copy.deepcopy(getattr(self, '_boundary_risks', [None, None])),
                per_arm_cooldown=[int(a.cooldown) for a in self.arms],
                per_arm_control=['Rejected', 'Rejected'],
                simple_protocol=self.protocol)
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
        """Existing strict empty-gripper and P0 object-pose eligibility."""
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

    def _grasp_stable_now(self, ai):
        """Current 10-step relative stability, never a historical success flag."""
        arm = self.arms[ai]
        if len(arm.relative) < arm.relative.maxlen or self.grasp_streak[ai] < 10:
            return False
        x0, r0 = arm.relative[0]
        return all(np.linalg.norm(x-x0) < self.cfg['relative_grasp_motion_m']
                   and Rotation.from_matrix(r@r0.T).magnitude() <= .035
                   for x, r in arm.relative)

    def _phase_name(self, state, contact, ai, nearest):
        arm, p = self.arms[ai], self.points['arms'][ARMS[ai]]
        if (self.completed[ai] or self._assembly_contact(state, contact, ai)
                or (ai == 0 and state['official_metrics'].get('first_piece_assembled', False))
                or (ai == 1 and state['official_metrics'].get('task', False))):
            return 'assembly_contact'
        if state['actual_grasps'][ARMS[ai]][p['object']]:
            return 'carry' if self._grasp_stable_now(ai) else 'grasp_unstable'
        return 'approach' if nearest < arm.p0 else 'grasp'

    def _phase_anchor(self, state, contact, ai, nearest):
        """Pick an anchor from observed task state, not only the sliding index."""
        phase = self._phase_name(state, contact, ai, nearest)
        arm, p = self.arms[ai], self.points['arms'][ARMS[ai]]
        if phase == 'grasp':
            return 'ReturnP0', arm.p0
        return None  # Only failed initial grasp can return to P0; never rewind carry/assembly.

    def _eligible_anchor(self, state, contact, ai, mode, anchor):
        """Qualification does not turn an anchor into a teleport or a path proof."""
        arm, p = ARMS[ai], self.points['arms'][ARMS[ai]]
        if self.completed[ai] or contact['dynamic_A_B_coupled'] or self._assembly_contact(state, contact, ai):
            return False
        if mode in ('ReturnS0', 'ReturnP0'):
            if not self._can_return(state, contact, ai):
                return False
            if mode == 'ReturnS0':
                # Entire pregrasp prefix is reusable only while the object
                # still occupies its measured pregrasp pick pose.
                from acoserl.object_geometry import object_error
                oi = ('base', 'piece_1', 'piece_2').index(p['object'])
                ref = dict(position=self.trace['object_positions'][anchor, oi],
                           quaternion_wxyz=self.trace['object_quaternions'][anchor, oi])
                err = object_error(state['objects'][p['object']], ref)
                return err['position_m'] <= .003 and err['rotation_rad'] <= .035
            return True
        if mode != 'ReturnP1':
            return False
        if not state['actual_grasps'][arm][p['object']] or not self._grasp_stable_now(ai):
            return False
        # Carry recovery preserves the actual grasp; permit contact only
        # between the active arm and its own held object.
        if any(arm in (edge['a'], edge['b']) and
               ({edge['a'], edge['b']} - {arm}) != {p['object']}
               for edge in contact['edges']):
            return False
        from acoserl.object_geometry import object_error
        oi = ('base', 'piece_1', 'piece_2').index(p['object'])
        ref = dict(position=self.trace['object_positions'][anchor, oi],
                   quaternion_wxyz=self.trace['object_quaternions'][anchor, oi])
        err = object_error(state['objects'][p['object']], ref)
        return (err['position_m'] <= self.cfg['object_window_position_m']
                and err['rotation_rad'] <= self.cfg['object_window_rotation_rad'])

    def _qualified_stage_recovery(self, state, contact, ai, nearest):
        plan = self._phase_anchor(state, contact, ai, nearest)
        if plan is None:
            return None
        mode, anchor = plan
        return plan if self._eligible_anchor(state, contact, ai, mode, anchor) else None

    @staticmethod
    def _guide_converged_after_progress(arm, current, cfg, pose_error):
        """Borrow AutoSERL's convergence exit; do not accept static no-motion."""
        if arm.guide_origin is None or len(arm.guide_motion) < arm.guide_motion.maxlen:
            return False
        if np.linalg.norm(current['position'] - arm.guide_origin) <= GUIDE_EXIT_PROGRESS_M:
            return False
        first_p, first_r = arm.guide_motion[0]
        if not all(np.linalg.norm(p-first_p) < GUIDE_CONVERGED_POSITION_M
                   and Rotation.from_matrix(r@first_r.T).magnitude() < GUIDE_CONVERGED_ROTATION_RAD
                   for p, r in arm.guide_motion):
            return False
        return (pose_error['position_m'] < cfg['trigger_position_m']
                and pose_error['rotation_rad'] < cfg['exit_rotation_rad'])

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
        arm.guide_motion.clear()
        arm.guide_origin = None
        self.events.append(dict(event='guide_released_to_policy', arm=ARMS[ai],
            reason=reason, takeover=prior_takeover,
            policy_budget=POLICY_AFTER_GUIDE_STEPS))

    def _start(self, ai, mode, target, state=None):
        arm = self.arms[ai]
        self.sequence += 1
        arm.mode, arm.target, arm.takeover = mode, int(target), self.sequence
        arm.age = arm.stable = 0
        arm.recovery_anchor = int(target) if mode in RETURN_MODES else None
        arm.recovery_goal = arm.p1 if mode in RETURN_MODES else None
        if mode != 'ReturnP1':
            arm.relative.clear()
        arm.nearest.clear()
        arm.guide_errors.clear()
        arm.guide_motion.clear()
        arm.guide_origin = (np.asarray(state['poses'][ai]['position']).copy()
                            if mode == 'Guide' and state is not None else None)
        self.episode_interventions += 1
        self.events.append(dict(event='start', arm=ARMS[ai], kind=mode,
            anchor=arm.recovery_anchor, goal=arm.recovery_goal,
            target=int(target), takeover=arm.takeover))

    def _boundary_safety_candidate(self, ai, state, physical_policy):
        """Cooldown-only inward proposal; ActionOnly preview is authoritative.

        The proposal is never called a Guide/Recovery. Refuse an unverified
        risk rather than silently revert to the unsafe outward Policy action.
        """
        from acoserl.boundary_guard import MAX_CONSECUTIVE_CONTROLS
        arm = self.arms[ai]
        if (not self.cfg.get('boundary_guard_enabled', False)
                or arm.mode != 'Policy' or arm.cooldown <= 0
                or not self.cfg['intervention_enabled'] or self.auto_disabled):
            return None
        pos = np.asarray(state['poses'][ai]['position'], dtype=float)
        lo = self.positions[:, ai].min(0) - self.cfg['free_workspace_margin_m']
        hi = self.positions[:, ai].max(0) + self.cfg['free_workspace_margin_m']
        robot = self.env.robots[ai]
        name = self.env.sim.model.site_id2name(robot.eef_site_id['right'])
        vel = np.asarray(self.env.sim.data.get_site_xvelp(name), dtype=float)
        world_delta = world_policy_displacement(robot, physical_policy[7*ai:7*ai+7])
        plan = boundary_risk(
            pos, lo, hi, world_delta, vel,
            enter_distance_m=self.cfg.get('boundary_guard_entry_m', .025),
            retreat_distance_m=self.cfg.get('boundary_guard_retreat_m', .035))
        if plan is None:
            return None
        if arm.boundary_guard_streak >= self.cfg.get(
                'boundary_guard_max_consecutive', MAX_CONSECUTIVE_CONTROLS):
            raise ValueError('consecutive_boundary_guard_limit_exceeded')
        return plan

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
        self._handoff_pending = None
        enabled = self.cfg['intervention_enabled'] and not self.auto_disabled
        action = policy.copy()
        sources = ['Policy', 'Policy']
        targets = copy.deepcopy(state['poses'])
        grips = [float(policy[6]), float(policy[13])]
        measurements = []
        self._issued_targets = [None, None]
        self._boundary_risks = [None, None]
        for ai, arm in enumerate(self.arms):
            domain = self._domain(state, contact, ai)
            measurement = self._measure(state, ai, domain) if domain else None
            measurements.append(measurement)
            if not enabled:
                continue
            # Boundary safety precedes the ordinary cooldown gate. It is
            # stateless with respect to Expert handback and never captures
            # another arm or a gripper. The original physics preview remains.
            plan = None
            if arm.mode == 'Policy' and arm.cooldown:
                try:
                    plan = self._boundary_safety_candidate(ai, state, policy)
                except (ValueError, KeyError, AttributeError) as error:
                    return self._refuse(policy, 'boundary_guard_unverified_'+ARMS[ai]+':'+str(error))
            if plan is not None:
                if (domain is None or contact['dynamic_A_B_coupled']
                        or self._active_contact(contact, ai)
                        or self._assembly_contact(state, contact, ai)):
                    return self._refuse(policy, 'boundary_guard_unqualified_contact_'+ARMS[ai])
                sources[ai] = BOUNDARY_GUARD
                targets[ai]['position'] = plan['target_world_position'].copy()
                self._boundary_risks[ai] = dict(
                    axes=plan['axes'],
                    target_world_position=plan['target_world_position'].tolist(),
                    requested_world_delta=plan['requested_world_delta'].tolist(),
                    measured_world_velocity=plan['measured_world_velocity'].tolist(),
                    entering_distance_m=plan['entering_distance_m'],
                    retreat_distance_m=plan['retreat_distance_m'],
                    cooldown_before=int(arm.cooldown),
                    consecutive_before=int(arm.boundary_guard_streak))
                self.events.append(dict(
                    event='boundary_guard_issued', arm=ARMS[ai],
                    cooldown_remaining=int(arm.cooldown),
                    axes=copy.deepcopy(plan['axes'])))
                continue
            # A stalled Guide can recover using S0, P0 or P1 according to the
            # actual arm/object stage. Never rewind across assembly contact.
            if arm.mode == 'Guide' and domain and not contact['dynamic_A_B_coupled']:
                nearest, _, _, reference_stalled = measurement
                if reference_stalled and self._guide_no_progress(arm):
                    plan = self._qualified_stage_recovery(state, contact, ai, nearest)
                    if plan:
                        previous_takeover = arm.takeover
                        kind, anchor = plan
                        self._start(ai, kind, anchor, state=state)
                        self.events.append(dict(event='guide_stall_to_stage_recovery',
                            arm=ARMS[ai], kind=kind, anchor=int(anchor),
                            from_takeover=previous_takeover, to_takeover=arm.takeover,
                            nearest_index=int(nearest)))
                    else:
                        self._release_guide_to_policy(ai, 'stalled_no_qualified_anchor')
            if arm.mode == 'Policy' and domain and not contact['dynamic_A_B_coupled']:
                nearest, target, distance, stagnant = measurement
                plan = (self._qualified_stage_recovery(state, contact, ai, nearest)
                        if arm.cooldown == 0 and stagnant else None)
                # Stage recovery has priority over ordinary Guide correction.
                if plan:
                    kind, anchor = plan
                    self._start(ai, kind, anchor, state=state)
                elif (arm.cooldown == 0 and distance > self.cfg['trigger_position_m'] and
                      forward_direction(self.positions[:, ai], nearest, state['poses'][ai]['position'],
                                        self.positions[target, ai])):
                    self._start(ai, 'Guide', target, state=state)
            if arm.mode in EXPERT:
                incompatible = (arm.mode == 'Guide' and domain is not None and
                                not domain[0] <= arm.target <= domain[1])
                if domain is None or incompatible:
                    # A domain change is not a completed recovery or a
                    # verified expert handback. Never label it as one.
                    if arm.mode == 'Guide':
                        self._release_guide_to_policy(ai, 'guide_domain_changed')
                    else:
                        return self._refuse(policy, 'recovery_domain_changed_'+ARMS[ai])
                else:
                    sources[ai] = arm.mode
                    self._issued_targets[ai] = (arm.target if arm.mode == 'Guide'
                        else arm.recovery_anchor if arm.mode in RETURN_MODES
                        else arm.p1 if arm.mode == 'Settle' else arm.replay+1)
                    if arm.mode == 'Guide':
                        targets[ai] = self.pose(ai, arm.target)
                    elif arm.mode in RETURN_MODES:
                        if not self._eligible_anchor(state, contact, ai, arm.mode, arm.recovery_anchor):
                            return self._refuse(policy, 'recovery_anchor_domain_changed_'+ARMS[ai])
                        targets[ai] = self.pose(ai, arm.recovery_anchor)
                        grips[ai] = 1. if arm.mode == 'ReturnP1' else -1.
                    elif arm.mode == 'Settle':
                        targets[ai] = self.pose(ai, arm.p1); grips[ai] = 1.
                    else:
                        # Reference indices designate states. Replay real
                        # actions [S0 or P0, P1) to reach state P1.
                        error = np.linalg.norm(state['poses'][ai]['position']-self.positions[arm.replay, ai])
                        if error > .02:
                            return self._refuse(policy, 'recovery_replay_tracking_'+ARMS[ai])
                        action[7*ai:7*ai+7] = self.trace['actions'][arm.replay, 7*ai:7*ai+7]
        if any(s in ((EXPERT | SAFETY_ONLY) - {'Replay'}) for s in sources):
            correction = servo(self.env, targets, grips,
                max_position_step=self.cfg['position_command_cap_m'],
                max_rotation_step=self.cfg['rotation_command_cap_rad'],
                position_gain=self.cfg['position_gain'], rotation_gain=self.cfg['rotation_gain'])
            for ai, source in enumerate(sources):
                if source in ((EXPERT | SAFETY_ONLY) - {'Replay'}):
                    action[7*ai:7*ai+6] = correction[7*ai:7*ai+6]
                    if source in RETURN_MODES or source == 'Settle':
                        action[7*ai+6] = grips[ai]
        # The Handoff expert owns ONLY the 1D gripper; Guide still owns only 6D.
        # Recovery/BoundaryGuard owns its own action and must not be overwritten.
        grip_owners = ['Policy', 'Policy']
        grip_mask = [False, False]
        grip_targets = [None, None]
        grip_phases = list(self.gripper_expert.phases)
        grip_events = []
        if enabled:
            skip = tuple(src in (RETURN_MODES | {'Replay', 'Settle'} | SAFETY_ONLY)
                         for src in sources)
            grippers = tuple(self.env.robots[i].gripper['right'].current_action
                             for i in range(2))
            # Compare the ORIGINAL Actor proposal, not Guide's posed movement.
            grip_action, grip_mask, grip_targets, grip_phases, grip_events = (
                self.gripper_expert.propose(state, policy, skip, grippers))
            for ai in range(2):
                if skip[ai]:
                    grip_owners[ai] = 'Recovery' if sources[ai] in EXPERT else 'Safety'
                elif grip_mask[ai]:
                    action[7 * ai + 6] = grip_action[7 * ai + 6]
                    grip_owners[ai] = 'Handoff'
        else:
            grip_owners = ['Policy', 'Policy']
        lo, hi = getattr(self, 'action_bounds', (-np.ones(14), np.ones(14)))
        # Bound expert outputs, never silently alter an independent Policy arm.
        for ai, source in enumerate(sources):
            sl = slice(7*ai, 7*ai+7)
            if source in SAFETY_ONLY:
                # Clipping an inward servo action does not prove physical
                # feasibility. Reject instead of outputting an unverified
                # substitute. The existing substep preview still decides.
                if (np.any(action[sl] < np.maximum(lo[sl], -1)-1e-6)
                        or np.any(action[sl] > np.minimum(hi[sl], 1)+1e-6)):
                    return self._refuse(policy, 'boundary_guard_outside_action_domain_'+ARMS[ai])
            elif source in EXPERT:
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
        signature = tuple((s, a.takeover if s in EXPERT else 0,
                           grip_owners[i] == 'Handoff')
                          for i, (s, a) in enumerate(zip(sources, self.arms)))
        if signature != self._segment_signature:
            self._segment += 1; self._segment_signature = signature
        expert = [s in EXPERT for s in sources]
        mode = ('Guide_AB' if any(expert) else
                'BoundarySafety' if any(s in SAFETY_ONLY for s in sources) else
                'Handoff_AB' if any(grip_mask) else 'Normal')
        self.mode = mode
        self.last_decision = dict(mode=mode, execute=True, policy_action=policy.tolist(),
            executed_action=action.tolist(), intervention_action_proposed=action.tolist(),
            indices=self.indices(), takeover_id=self._segment, per_arm_control=sources,
            gripper_control=grip_owners, handoff_disagreement=grip_mask,
            privileged_gripper_target=grip_targets,
            privileged_gripper_phase_before=list(self.gripper_expert.phases),
            privileged_gripper_events=grip_events,
            arm_takeover_ids=[a.takeover for a in self.arms], simple_protocol=self.protocol,
            per_arm_phase=[self._phase_name(state, contact, i, m[0]) if m is not None
                           else 'assembly_contact' for i, m in enumerate(measurements)],
            recovery_anchor_indices=[a.recovery_anchor for a in self.arms],
            measurements=measurements, boundary_safety=copy.deepcopy(self._boundary_risks),
            recovery_step=1 if any(s in EXPERT and a.age == 0 for s, a in zip(sources, self.arms)) else 2,
            stage=max((self._stage(state, i) for i in range(2)), key=lambda x: x != 'approach_pregrasp'))
        self._handoff_pending = grip_phases if enabled else None
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
        # Boundary safety is non-autonomous too: never label its episode
        # as clean Policy success or prematurely disable the supervisor.
        self.intervention_control_steps += int(any(s in EXPERT | SAFETY_ONLY for s in issued)
                                               or any(self.last_decision.get('handoff_disagreement', ())))
        # Count unique Guide/Recovery starts once, not every control they occupy.
        for e in self.events[self._event_cursor:]:
            if e.get('event') == 'start':
                if e.get('kind') == 'Guide':
                    self.guide_events += 1
                elif e.get('kind') in RETURN_MODES:
                    self.recovery_events += 1
        self.handoff_disagreements += sum(self.last_decision.get('handoff_disagreement', ()))
        # A P0 recovery may have opened a previously latched unsuccessful
        # closure. Re-arm its close event only at the real recovery boundary.
        for e in self.events[self._event_cursor:]:
            if e.get('event') == 'start' and e.get('kind') == 'ReturnP0':
                ai = ARMS.index(e['arm'])
                self.gripper_expert.reset_arm(ai)
                if self._handoff_pending is not None:
                    self._handoff_pending[ai] = 'pregrasp'
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
            # The sliding reference window advances ONLY on a completed
            # Policy control cycle. Guide, Return, Replay and even a long
            # Settle are expert operations, not evidence of Actor progress.
            if issued[ai] == 'Policy':
                arm.start = min(self.n_states-1, max(arm.floor, arm.start+1))
            # Exactly 32 completed Policy cycles; safety override and expert
            # steps neither spend cooldown nor advance the Policy window.
            if issued[ai] == 'Policy':
                arm.cooldown = max(0, arm.cooldown-1)
            if issued[ai] == BOUNDARY_GUARD:
                arm.boundary_guard_streak += 1
                arm.boundary_guard_total += 1
                self.boundary_guard_steps += 1
                self.events.append(dict(event='boundary_guard_completed',
                    arm=ARMS[ai], cooldown_remaining=int(arm.cooldown),
                    consecutive=int(arm.boundary_guard_streak)))
            elif arm.boundary_guard_streak:
                self.events.append(dict(event='boundary_guard_cleared',
                    arm=ARMS[ai], consecutive=int(arm.boundary_guard_streak)))
                arm.boundary_guard_streak = 0
            if issued[ai] in EXPERT:
                self._touched.add(ai); arm.age += 1
            if self.stop_reason:
                continue
            if issued[ai] == 'Guide':
                e = errors(state['poses'][ai], self.pose(ai, arm.target))
                arm.guide_errors.append((float(e['position_m']), float(e['rotation_rad'])))
                arm.guide_motion.append((np.asarray(state['poses'][ai]['position']).copy(),
                                         np.asarray(state['poses'][ai]['rotation']).copy()))
                in_target = (e['position_m'] < self.cfg['exit_position_m']
                             and e['rotation_rad'] < self.cfg['exit_rotation_rad'])
                converged_progress = self._guide_converged_after_progress(
                    arm, state['poses'][ai], self.cfg, e)
                arm.stable = arm.stable+1 if (in_target or converged_progress) else 0
                if arm.stable >= self.cfg['stable_exit_steps']:
                    self.events.append(dict(event='guide_exit_to_handoff', arm=ARMS[ai],
                        reason='target' if in_target else 'converged_after_progress',
                        takeover=arm.takeover, guide_controls=arm.age))
                    arm.reset_intervention(held)
            elif issued[ai] in RETURN_MODES:
                e = errors(state['poses'][ai], self.pose(ai, arm.recovery_anchor))
                position_tolerance = (.005 if issued[ai] == 'ReturnP1'
                                      else self.cfg['p0_alignment_m'])
                aligned = (e['position_m'] < position_tolerance
                           and e['rotation_rad'] < self.cfg['exit_rotation_rad'])
                arm.stable = arm.stable+1 if aligned else 0
                if arm.stable >= self.cfg['stable_exit_steps']:
                    if issued[ai] == 'ReturnP1':
                        arm.mode = 'Settle'
                    else:
                        arm.mode, arm.replay = 'Replay', arm.recovery_anchor
                    arm.stable = 0
            elif issued[ai] == 'Replay':
                arm.replay += 1
                if arm.replay >= arm.recovery_goal:
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
                    # Recovery restored real physics to P1. Rebase the
                    # per-arm window there; pre-recovery clock drift must
                    # not pretend the Actor already traversed the carry path.
                    arm.start = arm.p1
                    self.events.append(dict(event='stage_recovery_complete',
                        arm=ARMS[ai], anchor=arm.recovery_anchor,
                        goal=arm.p1, takeover=arm.takeover))
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
            elif (self.stop_reason is None and (arm.mode in RETURN_MODES or
                   arm.mode in ('Replay', 'Settle'))
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
        # Commit proposed gripper phase changes only after complete MuJoCo control.
        if self._handoff_pending is not None:
            stable = [self.grasp_streak[i] if self._grasp_stable_now(i) else 0
                      for i in range(2)]
            grip_changes = self.gripper_expert.commit(self._handoff_pending, state, stable)
            self.events.extend(dict(event='privileged_gripper_phase', **x)
                               for x in grip_changes)
            self._handoff_pending = None
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
            task_quality=dict(eligible=not self.stop_reason and any(s in EXPERT for s in issued) and not any(s in SAFETY_ONLY for s in issued),
                              error_before_m=self._error_before, error_after_m=error_after, handoff_confirmed=bool(handbacks)),
            per_arm_modes=[a.mode for a in self.arms],
            per_arm_cooldown=[a.cooldown for a in self.arms],
            boundary_guard_streak=[a.boundary_guard_streak for a in self.arms],
            boundary_guard_total=[a.boundary_guard_total for a in self.arms],
            control_events=copy.deepcopy(self.events[self._event_cursor:]),
            stable_grasps=[ARMS[i] for i in sorted(self._stable_grasp)],
            settle_diagnostics=settle_diagnostics)

    def set_global_capability(self, value):
        self.auto_disabled = bool(self.auto_disabled or value.get('disable', False))

    def episode_result(self, success, **_):
        total = self.guide_events + self.recovery_events + self.handoff_disagreements
        # A safety-projected episode is not evidence of autonomous readiness.
        # Keep boundary guards out of the requested 3-way count, but refuse
        # to unlock on an episode that depended on an uncounted safety action.
        request = bool(success and self.boundary_guard_steps == 0 and
                       total < self.cfg['termination_intervention_steps'])
        self.auto_disabled |= request
        return dict(stable_full_success=bool(success),
                    assisted_success=bool(success and self.intervention_control_steps),
                    autonomous_success=bool(success and not self.intervention_control_steps),
                    intervention_control_steps=self.intervention_control_steps,
                    intervention_events=self.episode_interventions,
                    guide_events=self.guide_events, recovery_events=self.recovery_events,
                    handoff_disagreement_steps=self.handoff_disagreements,
                    intervention_count=total, termination_threshold=self.cfg['termination_intervention_steps'],
                    handbacks=sum(self.handbacks), boundary_guard_steps=self.boundary_guard_steps,
            disable_requested=request,
                    stable_grasp_arms=[ARMS[i] for i in sorted(self._stable_grasp)],
                    protocol=self.protocol)
