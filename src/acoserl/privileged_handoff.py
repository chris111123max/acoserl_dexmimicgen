"""Training-only privileged 1D gripper expert for the dual-arm AutoSERL baseline.

The visual Actor sees NO object poses, contact truth, or reference events.
This expert uses the immutable successful demo only for relative SE(3) pick
alignment and object-to-assembly-target release geometry. It never changes 6D
motion, replays demo timestamps, rewards, or simulator state.

Proposals are pure: commit event latches ONLY after a complete physical step.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


OBJECTS = ("base", "piece_1", "piece_2")


def _object_rotation(object_state):
    return Rotation.from_quat(
        np.roll(np.asarray(object_state["quaternion_wxyz"], dtype=float), -1)
    ).as_matrix()


def _relative_ee_to_object(ee, obj):
    rot = _object_rotation(obj)
    return (
        rot.T @ (np.asarray(ee["position"]) - np.asarray(obj["position"])),
        rot.T @ np.asarray(ee["rotation"]),
    )


def _rotation_error(actual, target):
    return float(Rotation.from_matrix(actual @ target.T).magnitude())


def effective_match(actor_grip, desired, actuator=None):
    """Panda gripper is sign-driven; zero is a valid HOLD at an end stop."""
    value = float(actor_grip)
    if not np.isfinite(value):
        raise ValueError("nonfinite Actor gripper command")
    if value * desired > 1e-7:
        return True
    if abs(value) > 1e-7 or actuator is None:
        return False
    actuator = np.asarray(actuator, dtype=float)
    endpoint = np.array([-1.0, 1.0]) if desired > 0 else np.array([1.0, -1.0])
    return bool(actuator.shape == (2,) and np.allclose(actuator, endpoint, atol=1e-5))


class PrivilegedHandoff:
    """One event latch per arm. Four controller modes remain unchanged.

    Internal phases are not additional control modes. The supervisor owns a
    gripper dimension only on a physically executed disagreement step.
    """

    PHASES = ("pregrasp", "closing", "holding", "releasing", "done")

    def __init__(self, points, trace, config):
        self.points = points
        self.trace = trace
        self.pick_position_m = float(config.get("privileged_pick_position_m", 0.005))
        self.pick_rotation_rad = float(config.get("privileged_pick_rotation_rad", 0.035))
        self.release_xy_m = float(config.get("privileged_release_xy_m", 0.005))
        self.release_z_m = float(config.get("privileged_release_z_m", 0.005))
        self.stable_grasp_steps = int(config["stable_grasp_window_steps"])
        if min(self.pick_position_m, self.pick_rotation_rad,
               self.release_xy_m, self.release_z_m) <= 0:
            raise ValueError("all privileged Handoff tolerances must be positive")
        if self.stable_grasp_steps < 1:
            raise ValueError("stable_grasp_window_steps must be positive")
        if any(k not in trace for k in ("positions", "rotations",
                                        "object_positions", "object_quaternions")):
            raise ValueError("reference trace lacks physical object poses")
        self.pick_references = []
        self.release_references = []
        for ai, arm in enumerate(("A", "B")):
            ref = points["arms"][arm]
            obj = ref["object"]
            if obj != ("piece_1", "piece_2")[ai]:
                raise ValueError("unexpected per-arm target object")
            close = int(ref["first_close_action"])
            release = int(ref["first_release_action"])
            if not 0 <= close < release < len(trace["positions"]):
                raise ValueError("invalid demonstration gripper event indices")
            pick_ee = dict(position=trace["positions"][close, ai],
                           rotation=trace["rotations"][close, ai])
            pick_obj = self._demo_object(obj, close)
            self.pick_references.append(_relative_ee_to_object(pick_ee, pick_obj))
            target = "base" if ai == 0 else "piece_1"
            release_obj = self._demo_object(obj, release)
            release_target = self._demo_object(target, release)
            target_rot = _object_rotation(release_target)
            self.release_references.append(
                target_rot.T @ (np.asarray(release_obj["position"]) -
                                np.asarray(release_target["position"]))
            )
        self.reset()

    def _demo_object(self, obj, frame):
        j = OBJECTS.index(obj)
        return dict(position=self.trace["object_positions"][frame, j],
                    quaternion_wxyz=self.trace["object_quaternions"][frame, j])

    def reset(self):
        self.phases = ["pregrasp", "pregrasp"]

    def reset_arm(self, ai):
        self.phases[ai] = "pregrasp"

    def pick_aligned(self, state, ai):
        obj = ("piece_1", "piece_2")[ai]
        p, r = _relative_ee_to_object(state["poses"][ai], state["objects"][obj])
        expected_p, expected_r = self.pick_references[ai]
        dp = float(np.linalg.norm(p - expected_p))
        dr = _rotation_error(r, expected_r)
        return bool(np.isfinite([dp, dr]).all() and
                    dp <= self.pick_position_m and dr <= self.pick_rotation_rad)

    def release_aligned(self, state, ai):
        if ai == 1 and not state["official_metrics"].get("first_piece_assembled", False):
            return False
        obj = ("piece_1", "piece_2")[ai]
        target = "base" if ai == 0 else "piece_1"
        goal = state["objects"][target]
        rel = _object_rotation(goal).T @ (
            np.asarray(state["objects"][obj]["position"]) -
            np.asarray(goal["position"])
        )
        deviation = rel - self.release_references[ai]
        return bool(np.isfinite(deviation).all() and
                    np.linalg.norm(deviation[:2]) <= self.release_xy_m and
                    abs(float(deviation[2])) <= self.release_z_m)

    def propose(self, state, policy_action, skipped=(False, False),
                actuators=(None, None)):
        """Return 14D copy, ownership bits and tentative phases.

        No phase is changed here: a refused or partial physics step cannot
        latch a close/release event or increment disagreement statistics.
        """
        proposed = np.asarray(policy_action, dtype=float).copy()
        if proposed.shape != (14,) or not np.isfinite(proposed).all():
            raise ValueError("expected finite 14D physical Actor action")
        overridden = [False, False]
        target = [None, None]
        next_phases = list(self.phases)
        events = []
        for ai in range(2):
            if skipped[ai] or self.phases[ai] == "done":
                continue
            phase = self.phases[ai]
            if phase == "pregrasp":
                if self.pick_aligned(state, ai):
                    next_phases[ai] = "closing"
                    events.append(("A", "B")[ai] + ":close")
                    desired = 1.0
                else:
                    desired = -1.0
            elif phase in ("closing", "holding"):
                if phase == "holding" and self.release_aligned(state, ai):
                    next_phases[ai] = "releasing"
                    events.append(("A", "B")[ai] + ":release")
                    desired = -1.0
                else:
                    desired = 1.0
            elif phase == "releasing":
                desired = -1.0
            else:
                raise ValueError("unknown privileged gripper phase")
            target[ai] = desired
            ix = 7 * ai + 6
            if not effective_match(proposed[ix], desired, actuators[ai]):
                proposed[ix] = desired
                overridden[ai] = True
        return proposed, overridden, target, next_phases, events

    def commit(self, next_phases, state, grasp_streak):
        """Commit only after a complete real control step."""
        if len(next_phases) != 2:
            raise ValueError("expected two proposed gripper phases")
        transitions = []
        for ai in range(2):
            old = self.phases[ai]
            phase = next_phases[ai]
            if old == "pregrasp" and phase == "closing":
                pass
            elif old == "holding" and phase == "releasing":
                pass
            elif old != phase:
                raise ValueError("invalid privileged Handoff phase transition")
            if phase == "closing" and grasp_streak[ai] >= self.stable_grasp_steps:
                phase = "holding"
            elif phase == "releasing":
                obj = ("piece_1", "piece_2")[ai]
                if not state["actual_grasps"][("A", "B")[ai]][obj]:
                    phase = "done"
            if phase != old:
                transitions.append(dict(arm=("A", "B")[ai], before=old, after=phase))
            self.phases[ai] = phase
        return transitions
