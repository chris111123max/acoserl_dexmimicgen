"""Two per-arm demo gripper events, with pose references in *relative* frames.

Only robot TCP pose/orientation and prior executed gripper events enter decisions.
No simulator object pose, grasp oracle, contact, or task-success flag is read.
The first release target follows the real grasp TCP; the second follows the
first arm's executed release TCP, rather than using fixed demo world positions.
These are *candidate* release poses, not proof of physical assembly.
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
    """SE(3) target = actual_anchor * inverse(demo_anchor) * demo_target.

    Homogeneous pose arithmetic correctly rotates translation offsets; simple
    world-coordinate subtraction would not support rotated scenes.
    """
    r_actual = np.asarray(anchor_actual['rotation'], dtype=float)
    r_anchor = np.asarray(anchor_demo['rotation'], dtype=float)
    r_target = np.asarray(target_demo['rotation'], dtype=float)
    delta_position = r_anchor.T @ (
        np.asarray(target_demo['position']) - np.asarray(anchor_demo['position']))
    return dict(position=np.asarray(anchor_actual['position']) + r_actual @ delta_position,
                rotation=r_actual @ (r_anchor.T @ r_target))


class DemoGripperEvents:
    """Exactly one grasp and one release event per arm, reset per episode.

    The *original* demo is read-only. Grasp/release position references are
    dynamically re-expressed relative to executed TCP anchor events.
    """

    def __init__(self, points, positions, rotations, cfg):
        self.points = points
        self.positions = np.asarray(positions)
        self.rotations = np.asarray(rotations)
        self.close_position_m = float(cfg.get('gripper_close_position_m', .02))
        self.close_rotation_rad = float(cfg.get('gripper_close_rotation_rad', .15))
        self.release_position_m = float(cfg.get('gripper_release_position_m', .005))
        self.release_rotation_rad = float(cfg.get('gripper_release_rotation_rad', .02))
        self.quiet_translation_m = float(cfg.get('gripper_release_max_motion_step_m', .0005))
        self.quiet_rotation_rad = float(cfg.get('gripper_release_max_rotation_step_rad', .01))
        self.quiet_steps = int(cfg.get('gripper_release_stable_steps', 3))
        if (self.quiet_steps < 1 or min(self.close_position_m, self.release_position_m) <= 0 or
                min(self.close_rotation_rad, self.release_rotation_rad) <= 0 or
                min(self.quiet_translation_m, self.quiet_rotation_rad) < 0):
            raise ValueError('invalid gripper event tolerances')
        self.reset()

    def reset(self, initial_poses=None):
        self.phases = ['open', 'open']
        self.release_streak = [0, 0]
        self.grasp_anchor = [None, None]       # actual TCP immediately BEFORE grasp command
        self.first_release_anchor = None      # actual TCP immediately BEFORE first release
        self.previous_poses = ([self._clone_pose(p) for p in initial_poses]
                               if initial_poses is not None else [None, None])

    @staticmethod
    def _clone_pose(pose):
        return dict(position=np.asarray(pose['position'], dtype=float).copy(),
                    rotation=np.asarray(pose['rotation'], dtype=float).copy())

    def demo_pose(self, ai, index):
        return dict(position=self.positions[index, ai],
                    rotation=self.rotations[index, ai])

    def release_reference(self, ai):
        """A: TCP displacement from its grasp. B: target relative to A release.

        This compensates a *coherent* scene transform observed through TCP
        events, but CANNOT identify an independently displaced base/piece.
        That requires scene observations beyond robot proprioception.
        """
        if ai == 0:
            if self.grasp_anchor[0] is None:
                return None
            a = self.points['arms']['A']
            return compose_relative(
                self.grasp_anchor[0],
                self.demo_pose(0, int(a['first_close_action'])),
                self.demo_pose(0, int(a['first_release_action'])))
        if self.first_release_anchor is None:
            return None
        a = self.points['arms']['A']
        b = self.points['arms']['B']
        return compose_relative(
            self.first_release_anchor,
            self.demo_pose(0, int(a['first_release_action'])),
            self.demo_pose(1, int(b['first_release_action'])))

    def aligned(self, ai, pose, kind):
        if kind == 'grasp':
            # Initial grasp entry is owned by the existing qualified Guide or
            # the Actor at the recorded demo pick. No object truth is injected.
            index = int(self.points['arms']['AB'[ai]]['first_close_action'])
            reference = self.demo_pose(ai, index)
            p_limit, r_limit = self.close_position_m, self.close_rotation_rad
        elif kind == 'release':
            reference = self.release_reference(ai)
            if reference is None:
                return False
            p_limit, r_limit = self.release_position_m, self.release_rotation_rad
        else:
            raise ValueError(f'unknown event kind: {kind}')
        dp = np.linalg.norm(np.asarray(pose['position']) - reference['position'])
        dr = Rotation.from_matrix(np.asarray(pose['rotation']) @
                                  reference['rotation'].T).magnitude()
        return bool(np.isfinite([dp, dr]).all() and dp <= p_limit and dr <= r_limit)

    def release_qualified(self, ai, pose):
        return bool(self.phases[ai] == 'holding' and
                    (ai == 0 or self.phases[0] == 'released') and
                    self.release_streak[ai] >= self.quiet_steps and
                    self.aligned(ai, pose, 'release'))

    def propose(self, ai, pose, requested, qualified_close=False):
        """Choose exact Actor or Guide gripper action; no numerical blending."""
        requested = float(requested)
        phase = self.phases[ai]
        event = None
        next_phase = phase
        if phase == 'open':
            if qualified_close or (requested >= .95 and self.aligned(ai, pose, 'grasp')):
                next_phase = 'holding'
                event = 'grasp'
                desired = 1.
            else:
                desired = -1.
        elif phase == 'holding':
            if self.release_qualified(ai, pose):
                next_phase = 'released'
                event = 'release'
                desired = -1.
            else:
                desired = 1.
        else:
            desired = -1.
        matched = (requested >= .95 if desired > 0 else requested <= -.95)
        return GripProposal(requested if matched else desired,
                            next_phase, not matched, event)

    def complete_step(self, next_phases, observed_poses, pre_action_poses=None):
        """Commit events ONLY after full physics execution, never on proposal.

        Anchors use the actual robot TCP from before the event command, so
        they correspond to the demo's pre-action frame (obs[t] -> action[t]).
        """
        before_phases = self.phases[:]
        if pre_action_poses is None:
            pre_action_poses = self.previous_poses
        for ai in range(2):
            if before_phases[ai] == 'open' and next_phases[ai] == 'holding':
                if pre_action_poses[ai] is None:
                    raise ValueError('grasp anchor missing for committed close')
                self.grasp_anchor[ai] = self._clone_pose(pre_action_poses[ai])
            if ai == 0 and before_phases[ai] == 'holding' and next_phases[ai] == 'released':
                if pre_action_poses[ai] is None:
                    raise ValueError('release anchor missing for committed first release')
                self.first_release_anchor = self._clone_pose(pre_action_poses[ai])
        self.phases = list(next_phases)
        for ai in range(2):
            now = self._clone_pose(observed_poses[ai])
            before = self.previous_poses[ai]
            quiet = False
            if before is not None:
                dp = np.linalg.norm(now['position'] - before['position'])
                dr = Rotation.from_matrix(now['rotation'] @ before['rotation'].T).magnitude()
                quiet = bool(dp <= self.quiet_translation_m and dr <= self.quiet_rotation_rad)
            if self.phases[ai] == 'holding' and quiet and self.aligned(ai, now, 'release'):
                self.release_streak[ai] += 1
            else:
                self.release_streak[ai] = 0
            self.previous_poses[ai] = now
