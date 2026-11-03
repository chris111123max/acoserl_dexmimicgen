"""Conservative per-arm post-grasp rebase for the existing Motion Guide.

Do not rewrite offline demo data. The pre-grasp reference is trusted only
when the existing canonical-reset setting is enabled. In randomized scenes,
visual Actor selects the candidate grasp TCP; a *verified stable* grasp locks
the new post-grasp frame. No object pose is used for computing that transform.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class RelativeGuideReference:
    """Active pose arrays backing existing Guide, Recovery and window code."""

    def __init__(self, positions, rotations, points, gripper_events, cfg):
        self.demo_positions = np.asarray(positions, dtype=float).copy()
        self.demo_rotations = np.asarray(rotations, dtype=float).copy()
        self.points = points
        self.events = gripper_events
        self.fixed_initial_state = bool(cfg.get('fixed_initial_state', True))
        self.margin = float(cfg['free_workspace_margin_m'])
        self.release_compat_position_m = float(cfg.get('gripper_release_position_m', .005))
        self.release_compat_rotation_rad = float(cfg.get('gripper_release_rotation_rad', .02))
        # Preserve array identity because simple_autoserl.py uses these arrays
        # in pose(), window measurement, direction and fallback guards.
        self.positions = self.demo_positions.copy()
        self.rotations = self.demo_rotations.copy()
        self.ready = [False, False]
        self.rejected = [None, None]

    def reset(self):
        self.positions[:] = self.demo_positions
        self.rotations[:] = self.demo_rotations
        self.ready[:] = [False, False]
        self.rejected[:] = [None, None]

    def workspace_bounds(self, ai):
        """Use original physical workspace; rebase cannot expand safety limits."""
        return (self.demo_positions[:, ai].min(axis=0) - self.margin,
                self.demo_positions[:, ai].max(axis=0) + self.margin)

    def activate(self, ai):
        """Rebase only after `gripper_events` committed verified holding."""
        if self.ready[ai]:
            return True
        if self.events.phases[ai] != 'holding' or self.events.grasp_anchor[ai] is None:
            return False
        p = self.points['arms']['AB'[ai]]
        start = int(p['first_grasp'])
        new_positions = []
        new_rotations = []
        for index in range(start, len(self.positions)):
            reference = self.events.relative_reference(ai, index)
            new_positions.append(reference['position'])
            new_rotations.append(reference['rotation'])
        new_positions = np.asarray(new_positions)
        new_rotations = np.asarray(new_rotations)
        lo, hi = self.workspace_bounds(ai)
        if (not np.isfinite(new_positions).all() or
                not np.isfinite(new_rotations).all() or
                np.any(new_positions < lo) or np.any(new_positions > hi)):
            self.rejected[ai] = 'relative_reference_outside_original_workspace'
            return False
        self.positions[start:, ai] = new_positions
        self.rotations[start:, ai] = new_rotations
        self.ready[ai] = True
        self.rejected[ai] = None
        return True

    def release_consistent(self, ai):
        """Never steer B to an endpoint that contradicts A-anchored release.

        A's carried trajectory is exactly tied to its release reference. B's
        carry trajectory is tied to B grasp, but its final assembly release is
        tied to actual A release. If those frames disagree, disable B motion
        correction and permit the visual Actor to finish alignment instead.
        """
        if ai != 1 or self.events.phases[0] != 'released':
            return True
        target = self.events.release_reference(1)
        if not self.ready[ai] or target is None:
            return False
        end_index = int(self.points['arms']['B']['first_release_action'])
        dp = np.linalg.norm(self.positions[end_index, ai] - target['position'])
        dr = Rotation.from_matrix(self.rotations[end_index, ai] @
                                  target['rotation'].T).magnitude()
        return bool(np.isfinite([dp, dr]).all() and
                    dp <= self.release_compat_position_m and
                    dr <= self.release_compat_rotation_rad)

    def can_guide(self, ai):
        if self.rejected[ai] is not None:
            return False
        # In randomized starts no canonical object reference is available.
        # Before a confirmed Actor-selected grasp, do not blindly replay the
        # old P0/P1 reference, because its absolute pick pose may be wrong.
        if not self.ready[ai] and not self.fixed_initial_state:
            return False
        if self.ready[ai] and not self.release_consistent(ai):
            return False
        return True
