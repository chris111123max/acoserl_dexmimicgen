"""Per-arm, bounded motion slew at a verified grasp handback.

The manager calls arm() only *after a completed physical step* has entered
Handoff with an established real grasp. propose() is pure (does not consume
budget); complete_step() advances only after the next full physical step.

This is not a collision certificate. The existing ActionOnly physics preview,
force, workspace and grasp-loss safety gates remain authoritative.
"""
from __future__ import annotations

import numpy as np

SOURCE = 'HandoffGuard'


class HandoffBudgetExhausted(RuntimeError):
    """A Policy command never became continuous inside the bounded window."""


class HandoffMotionGuard:
    """Slew the active arm's six physical action coordinates; never the jaw."""

    def __init__(self, cfg):
        self.enabled = bool(cfg.get('handoff_motion_guard_enabled', True))
        self.delta = float(cfg.get('handoff_motion_max_action_delta', .05))
        self.minimum = int(cfg.get('handoff_motion_min_steps', 8))
        self.maximum = int(cfg.get('handoff_motion_max_steps', 32))
        if (not np.isfinite(self.delta) or not 0 < self.delta <= 1 or
                self.minimum < 1 or self.maximum < self.minimum):
            raise ValueError('invalid handoff motion guard configuration')
        self.reset()

    def reset(self):
        self._active = [False, False]
        self._last_motion = [None, None]
        self._controls = [0, 0]

    @staticmethod
    def _motion(motion):
        result = np.asarray(motion, dtype=float)
        if result.shape != (6,) or not np.isfinite(result).all() or np.any(abs(result) > 1.000001):
            raise ValueError('handoff action must be six finite, bounded coordinates')
        return result.copy()

    def active(self, ai):
        return bool(self._active[ai])

    def arm(self, ai, last_executed_motion):
        """Begin only from the last *executed* expert motion, not Actor intent."""
        if not self.enabled:
            return False
        self._last_motion[ai] = self._motion(last_executed_motion)
        self._controls[ai] = 0
        self._active[ai] = True
        return True

    def propose(self, ai, policy_motion):
        if not self._active[ai]:
            raise ValueError('handoff guard not armed')
        if self._controls[ai] >= self.maximum:
            raise HandoffBudgetExhausted('handoff_motion_guard_budget_exhausted')
        requested = self._motion(policy_motion)
        previous = self._last_motion[ai]
        executed = previous + np.clip(requested - previous, -self.delta, self.delta)
        if not np.isfinite(executed).all() or np.any(abs(executed) > 1.000001):
            raise ValueError('handoff guard proposed invalid motion')
        return dict(requested=requested.tolist(), executed=executed.tolist(),
                    previous=previous.tolist(), max_delta=self.delta,
                    control=int(self._controls[ai] + 1), budget=self.maximum,
                    limited=bool(np.max(abs(executed - requested)) > 1e-7))

    def complete_step(self, ai, proposal, real_executed_motion):
        """Return True only when the guarded Policy action was reached smoothly."""
        if not self._active[ai]:
            raise ValueError('completed handoff step was not armed')
        actual = self._motion(real_executed_motion)
        expected = self._motion(proposal['executed'])
        requested = self._motion(proposal['requested'])
        if not np.allclose(actual, expected, atol=2e-6, rtol=0):
            raise ValueError('handoff guard execution differs from its proposal')
        self._last_motion[ai] = actual
        self._controls[ai] += 1
        completed = (self._controls[ai] >= self.minimum and
                     np.allclose(actual, requested, atol=1e-7, rtol=0))
        if completed:
            self._active[ai] = False
        return completed

    def note_external(self, ai, real_executed_motion):
        """Keep continuity if an existing higher-priority safety guard ran."""
        if self._active[ai]:
            self._last_motion[ai] = self._motion(real_executed_motion)

    def cancel(self, ai):
        self._active[ai] = False
        self._last_motion[ai] = None
        self._controls[ai] = 0
