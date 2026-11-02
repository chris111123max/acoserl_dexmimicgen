"""Conservative, controller-frame-aware boundary protection during Policy cooldown.

This is a per-arm nonexpert safety operation, NOT Guide/Recovery and not an
Actor action. It requests inward movement only while near an existing world
workspace limit and measured/requested motion points outward. The ordinary
ActionOnly physics and substep preview remain authoritative for all commands.
"""
from __future__ import annotations

import numpy as np

SOURCE = "BoundaryGuard"
ENTER_DISTANCE_M = .025
RETREAT_DISTANCE_M = .035
REQUEST_OUTWARD_MIN_M = .0001
VELOCITY_OUTWARD_MIN_M_S = .005
MAX_CONSECUTIVE_CONTROLS = 16


def evaluate(position, lower, upper, commanded_world_delta, world_velocity,
             enter_distance_m=ENTER_DISTANCE_M,
             retreat_distance_m=RETREAT_DISTANCE_M):
    """Return an inward world target + complete risk evidence, or None.

    Arguments are in world coordinates. This routine does not claim a
    physically safe future trajectory; the real ActionOnly substep guard
    is still mandatory.
    """
    p = np.asarray(position, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    requested = np.asarray(commanded_world_delta, dtype=np.float64)
    velocity = np.asarray(world_velocity, dtype=np.float64)
    if any(a.shape != (3,) for a in (p, lo, hi, requested, velocity)):
        raise ValueError("boundary guard expects five world 3-vectors")
    if (not all(np.isfinite(a).all() for a in (p, lo, hi, requested, velocity))
            or np.any(hi <= lo) or not 0 < enter_distance_m < retreat_distance_m):
        raise ValueError("invalid boundary guard geometry or nonfinite motion")
    if np.any(p < lo) or np.any(p > hi):
        raise ValueError("already outside existing hard workspace")
    aim = p.copy()
    risks = []
    for axis in range(3):
        space = hi[axis]-lo[axis]
        if space <= 2*retreat_distance_m:
            raise ValueError("workspace too narrow for the configured retreat")
        low_d = float(p[axis]-lo[axis])
        high_d = float(hi[axis]-p[axis])
        toward_lower = (requested[axis] < -REQUEST_OUTWARD_MIN_M
                        or velocity[axis] < -VELOCITY_OUTWARD_MIN_M_S)
        toward_upper = (requested[axis] > REQUEST_OUTWARD_MIN_M
                        or velocity[axis] > VELOCITY_OUTWARD_MIN_M_S)
        if low_d <= enter_distance_m and toward_lower:
            aim[axis] = lo[axis]+retreat_distance_m
            risks.append(dict(axis='xyz'[axis],side='lower',distance_m=low_d,
                              policy_requested_delta_m=float(requested[axis]),
                              measured_velocity_m_s=float(velocity[axis])))
        elif high_d <= enter_distance_m and toward_upper:
            aim[axis] = hi[axis]-retreat_distance_m
            risks.append(dict(axis='xyz'[axis],side='upper',distance_m=high_d,
                              policy_requested_delta_m=float(requested[axis]),
                              measured_velocity_m_s=float(velocity[axis])))
    if not risks:
        return None
    return dict(target_world_position=aim, axes=risks,
                requested_world_delta=requested.copy(),
                measured_world_velocity=velocity.copy(),
                entering_distance_m=float(enter_distance_m),
                retreat_distance_m=float(retreat_distance_m))


def world_policy_displacement(robot, physical_action):
    """Invert the same OSC base-frame conversion used by servo().

    This is a proposed controller-frame translation, NOT a prediction of
    physical displacement. It is used only as one risk signal together
    with measured velocity and position. Invalid controller contracts fail.
    """
    ctl = robot.part_controllers['right']
    if (ctl.input_type != 'delta' or ctl.input_ref_frame != 'base'
            or ctl._goal_update_mode != 'achieved'):
        raise ValueError("unverified OSC controller input contract")
    rot = np.asarray(ctl.origin_ori, dtype=np.float64)
    scale = np.asarray(ctl.output_max, dtype=np.float64)
    proposal = np.asarray(physical_action, dtype=np.float64)
    if (rot.shape != (3,3) or scale.shape != (6,)
            or proposal.shape != (7,) or
            not all(np.isfinite(v).all() for v in (rot,scale,proposal))
            or np.any(scale[:3] <= 0)):
        raise ValueError("unverified OSC input scale/rotation")
    if not np.allclose(rot.T@rot, np.eye(3), atol=1e-4, rtol=1e-4):
        raise ValueError("OSC origin rotation not orthonormal")
    return rot@(proposal[:3]*scale[:3])
