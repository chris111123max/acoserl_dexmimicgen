"""Pure, per-arm post-grasp Guide thresholds. No simulator state or side effects.

The first N *completed real Policy cycles* after a confirmed stable grasp
use a tighter (5 mm -> 2 mm) pose-guidance hysteresis. Expert, safety and
partial/rejected controls do not consume the window.
"""
from __future__ import annotations


def in_window(completed_policy_steps: int, holding: bool, cfg: dict) -> bool:
    n = int(cfg.get('post_grasp_guide_policy_steps', 10))
    return bool(holding and n > 0 and 0 <= completed_policy_steps < n)


def trigger_distance_m(cfg: dict, post_grasp: bool) -> float:
    return float(cfg.get('post_grasp_trigger_position_m', .005) if post_grasp
                 else cfg['trigger_position_m'])


def guide_exit(cfg: dict, post_grasp: bool, position_error_m: float,
               rotation_error_rad: float, converged_after_progress: bool):
    """Return (exit_qualified, exact_target_reached).

    In the special post-grasp window, convergence alone cannot bypass the
    2 mm position limit, unlike the ordinary AutoSERL-style Guide path.
    """
    limit = float(cfg.get('post_grasp_exit_position_m', .002) if post_grasp
                  else cfg['exit_position_m'])
    reached = (position_error_m < limit and
               rotation_error_rad < float(cfg['exit_rotation_rad']))
    return bool(reached or (not post_grasp and converged_after_progress)), bool(reached)
