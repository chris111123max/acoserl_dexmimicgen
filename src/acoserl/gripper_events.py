"""Two event gripper control with verified grasp and runtime TCP-relative references.

All references derive from the read-only demo and executed robot TCP poses.
An attempted close is NOT a successful grasp. The existing supervisor verifies
10 consecutive genuine stable grasp controls before the latch is confirmed.
The simulator-only verification must be replaced by observable feedback on a
real robot; it is deliberately supplied by the caller, not inferred here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GripProposal:
    command: float
    next_phase: str
    overridden: bool
    event: str | None


def compose_relative(anchor_actual, anchor_demo, target_demo):
    """Apply a demo-relative rigid transform using SE(3), including rotation."""
    actual_r = np.asarray(anchor_actual['rotation'], dtype=float)
    demo_r = np.asarray(anchor_demo['rotation'], dtype=float)
    target_r = np.asarray(target_demo['rotation'], dtype=float)
    offset = demo_r.T @ (np.asarray(target_demo['position']) -
                         np.asarray(anchor_demo['position']))
    return {'position': np.asarray(anchor_actual['position']) + actual_r @ offset,
            'rotation': actual_r @ (demo_r.T @ target_r)}


class DemoGripperEvents:
    """Open -> closing -> confirmed holding -> released, independently per arm.

    'closing' is an internal validation phase, not a third gripper event.
    Only a completed grasp and a qualified release are counted as events.
    """

    def __init__(self, points, positions, rotations, cfg):
        self.points = points
        # Immutable original demo: runtime transformed Guide references MUST
        # never change the event offsets or offline demonstration.
        self.positions = np.asarray(positions, dtype=float).copy()
        self.rotations = np.asarray(rotations, dtype=float).copy()
        self.close_position_m = float(cfg.get('gripper_close_position_m', .02))
        self.close_rotation_rad = float(cfg.get('gripper_close_rotation_rad', .15))
        self.release_position_m = float(cfg.get('gripper_release_position_m', .005))
        self.release_rotation_rad = float(cfg.get('gripper_release_rotation_rad', .02))
        self.quiet_translation_m = float(cfg.get('gripper_release_max_motion_step_m', .0005))
        self.quiet_rotation_rad = float(cfg.get('gripper_release_max_rotation_step_rad', .01))
        self.quiet_steps = int(cfg.get('gripper_release_stable_steps', 3))
        self.grasp_confirm_max_steps = int(cfg.get('gripper_grasp_confirm_max_steps', 80))
        # With randomized objects there is no trustworthy world-frame pick
        # target. Actor is allowed to choose a NEW grasp TCP using its images.
        # No privileged object pose is read to qualify this candidate.
        self.actor_anchor_grasp = not bool(cfg.get('fixed_initial_state', True))
        if (self.quiet_steps < 1 or self.grasp_confirm_max_steps < 10 or
                min(self.close_position_m, self.release_position_m) <= 0 or
                min(self.close_rotation_rad, self.release_rotation_rad) <= 0 or
                min(self.quiet_translation_m, self.quiet_rotation_rad) < 0):
            raise ValueError('invalid gripper event tolerances')
        self.reset()

    @staticmethod
    def _clone_pose(pose):
        return {'position': np.asarray(pose['position'], dtype=float).copy(),
                'rotation': np.asarray(pose['rotation'], dtype=float).copy()}

    def reset(self, initial_poses=None):
        self.phases = ['open', 'open']
        self.grasp_attempt_anchor = [None, None]
        self.grasp_anchor = [None, None]  # populated only after verified grasp
        self.grasp_attempt_steps = [0, 0]
        self.release_streak = [0, 0]
        self.first_release_anchor = None
        self.previous_poses = ([self._clone_pose(p) for p in initial_poses]
                               if initial_poses is not None else [None, None])

    def demo_pose(self, ai, index):
        return {'position': self.positions[index, ai].copy(),
                'rotation': self.rotations[index, ai].copy()}

    def relative_reference(self, ai, index):
        """Post-grasp Guide target tied to the verified real grasp TCP."""
        if self.grasp_anchor[ai] is None:
            return None
        p = self.points['arms']['AB'[ai]]
        return compose_relative(self.grasp_anchor[ai],
                                self.demo_pose(ai, int(p['first_close_action'])),
                                self.demo_pose(ai, index))

    def release_reference(self, ai):
        """A relative to its grasp, B relative to A's executed release.

        This is not proof that an independently moved base or object is at the
        desired place. An independent target displacement requires perception.
        """
        if ai == 0:
            index = int(self.points['arms']['A']['first_release_action'])
            return self.relative_reference(0, index)
        if self.first_release_anchor is None or self.grasp_anchor[1] is None:
            return None
        a = self.points['arms']['A']
        b = self.points['arms']['B']
        return compose_relative(self.first_release_anchor,
                                self.demo_pose(0, int(a['first_release_action'])),
                                self.demo_pose(1, int(b['first_release_action'])))

    def aligned(self, ai, pose, kind):
        if kind == 'grasp':
            p = self.points['arms']['AB'[ai]]
            target = self.demo_pose(ai, int(p['first_close_action']))
            p_limit, r_limit = self.close_position_m, self.close_rotation_rad
        elif kind == 'release':
            target = self.release_reference(ai)
            p_limit, r_limit = self.release_position_m, self.release_rotation_rad
        else:
            raise ValueError('unknown gripper event')
        if target is None:
            return False
        dp = float(np.linalg.norm(np.asarray(pose['position']) - target['position']))
        dr = float(Rotation.from_matrix(np.asarray(pose['rotation']) @
                                        target['rotation'].T).magnitude())
        return bool(np.isfinite([dp, dr]).all() and dp <= p_limit and dr <= r_limit)

    def release_qualified(self, ai, pose):
        return bool(self.phases[ai] == 'holding' and
                    (ai == 0 or self.phases[0] == 'released') and
                    self.release_streak[ai] >= self.quiet_steps and
                    self.aligned(ai, pose, 'release'))

    def propose(self, ai, pose, requested, qualified_close=False):
        """Binary command selection; no mixing and no speculative commit."""
        requested = float(requested)
        phase = self.phases[ai]
        next_phase, event = phase, None
        if phase == 'open':
            # Existing demonstrated Guide may initiate close at qualified
            # pick. In a random-start scene the visual Actor can initiate at
            # an independent TCP; the *physical* grasp must still be verified.
            close_allowed = qualified_close or (requested >= .95 and
                            (self.actor_anchor_grasp or self.aligned(ai, pose, 'grasp')))
            if close_allowed:
                next_phase, event, desired = 'closing', 'close_attempt', 1.
            else:
                desired = -1.
        elif phase == 'closing':
            desired = 1.
        elif phase == 'holding':
            if self.release_qualified(ai, pose):
                next_phase, event, desired = 'released', 'release', -1.
            else:
                desired = 1.
        elif phase == 'released':
            desired = -1.
        else:
            raise ValueError('invalid gripper phase')
        matched = (requested >= .95 if desired > 0 else requested <= -.95)
        return GripProposal(requested if matched else desired,
                            next_phase, not matched, event)

    def complete_step(self, next_phases, observed_poses, pre_action_poses=None,
                      grasp_confirmed=(False, False)):
        """Commit only after one fully executed physical control step.

        grasp_confirmed is supplied from the existing supervisor's live
        10-step grasp relation and relative-motion qualification.
        """
        if pre_action_poses is None:
            pre_action_poses = self.previous_poses
        events, timeouts = [], []
        for ai in range(2):
            before, proposed = self.phases[ai], next_phases[ai]
            if before == 'open' and proposed == 'closing':
                if pre_action_poses[ai] is None:
                    raise ValueError('missing attempted-grasp TCP anchor')
                self.grasp_attempt_anchor[ai] = self._clone_pose(pre_action_poses[ai])
                self.grasp_attempt_steps[ai] = 0
                self.phases[ai] = 'closing'
                events.append((ai, 'close_started'))
            elif before == 'holding' and proposed == 'released':
                if ai == 0:
                    if pre_action_poses[ai] is None:
                        raise ValueError('missing first-release TCP anchor')
                    self.first_release_anchor = self._clone_pose(pre_action_poses[ai])
                self.phases[ai] = 'released'
                events.append((ai, 'release'))
            elif before != proposed:
                raise ValueError(f'forbidden gripper transition: {before} -> {proposed}')
            if self.phases[ai] == 'closing':
                self.grasp_attempt_steps[ai] += 1
                if grasp_confirmed[ai]:
                    if self.grasp_attempt_anchor[ai] is None:
                        raise ValueError('stable grasp without a close anchor')
                    self.grasp_anchor[ai] = self._clone_pose(self.grasp_attempt_anchor[ai])
                    self.phases[ai] = 'holding'
                    events.append((ai, 'grasp_confirmed'))
                elif self.grasp_attempt_steps[ai] >= self.grasp_confirm_max_steps:
                    timeouts.append(ai)
            now = self._clone_pose(observed_poses[ai])
            earlier = self.previous_poses[ai]
            quiet = False
            if earlier is not None:
                dp = np.linalg.norm(now['position'] - earlier['position'])
                dr = Rotation.from_matrix(now['rotation'] @ earlier['rotation'].T).magnitude()
                quiet = bool(np.isfinite([dp, dr]).all() and
                             dp <= self.quiet_translation_m and dr <= self.quiet_rotation_rad)
            if self.phases[ai] == 'holding' and quiet and self.aligned(ai, now, 'release'):
                self.release_streak[ai] += 1
            else:
                self.release_streak[ai] = 0
            self.previous_poses[ai] = now
        return {'events': events, 'grasp_timeouts': timeouts}
