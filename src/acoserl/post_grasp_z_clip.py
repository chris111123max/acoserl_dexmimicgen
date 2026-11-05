"""World-Z-only Policy safety projection during verified early post-grasp carry.

The OSC translation action is normalized in the robot base frame, not world
XYZ. Never infer physical collision safety from this projection: the existing
ActionOnly preview and live substep protections remain authoritative.
"""
from __future__ import annotations

import numpy as np

from acoserl.boundary_guard import world_policy_displacement

SOURCE = 'PostGraspZClip'


def early_carry_eligible(cfg, *, holding, stable_grasp, held, mode,
                         carry_phase, assembled, assembly_contact, coupled,
                         reference_before_contact, policy_controls, clearance_m):
    """Select only early, independently verified free-space carry."""
    if not cfg.get('post_grasp_z_clip_enabled', False):
        return False
    if (not holding or not stable_grasp or not held or
            mode not in ('Handoff', 'Policy') or
            carry_phase not in ('ready', 'done') or assembled or
            assembly_contact or coupled or not reference_before_contact):
        return False
    budget = int(cfg['post_grasp_z_clip_max_controls'])
    if policy_controls < 0 or policy_controls >= budget:
        return False
    gap = float(clearance_m)
    if not np.isfinite(gap):
        raise ValueError('nonfinite carried-object clearance')
    if gap < float(cfg['carry_table_clearance_m']):
        raise ValueError('carried-object clearance below hard minimum')
    return gap < float(cfg['post_grasp_z_clip_release_clearance_m'])


def project_world_z(robot, physical_action):
    """Zero only negative requested WORLD Z, preserving world XY and wrist/grip.

    Returns (physical_action, audit or None). A returned audit means an actual
    projection occurred and the result must NOT count as unmodified Policy.
    Controller/action domain errors fail closed instead of saturating silently.
    """
    action = np.asarray(physical_action, dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('invalid seven-dimensional physical Policy action')
    world = world_policy_displacement(robot, action)
    if world[2] >= 0.:
        return action.copy(), None
    ctl = robot.part_controllers['right']
    rot = np.asarray(ctl.origin_ori, dtype=np.float64)
    scale = np.asarray(ctl.output_max, dtype=np.float64)
    safe_world = world.copy()
    safe_world[2] = 0.
    projected = action.copy()
    projected[:3] = (rot.T @ safe_world) / scale[:3]
    if (not np.isfinite(projected).all() or
            np.any(np.abs(projected[:3]) > 1. + 1e-6)):
        raise ValueError('projected Policy translation outside OSC action range')
    check = world_policy_displacement(robot, projected)
    if not np.allclose(check, safe_world, atol=1e-9, rtol=1e-9):
        raise ValueError('projected Policy world displacement inconsistent')
    return projected, dict(
        requested_world_translation_m=world.tolist(),
        projected_world_translation_m=check.tolist(),
        removed_downward_m=float(-world[2]))
