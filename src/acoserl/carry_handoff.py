"""Physics-gated post-grasp lift/clear/handback for the dual-arm supervisor.

A valid two-pad grasp does not prove that the object is free of the table.
These privileged simulator measurements are supervisor-only, never Actor
observations. Motion is real and ActionOnly keeps its existing authority.

Safety contract: no fabricated simulator state, no unbounded waiting, no
credit for a protected action as an unmodified Policy handback.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

LIFT = "CarryLift"
VERIFY = "CarryVerify"
SOURCES = frozenset((LIFT, VERIFY))


class CarryGateError(ValueError):
    pass


class CarryGate:
    """Independent per-arm state: clearance -> verify -> ready -> Policy."""

    def __init__(self, cfg):
        self.clearance = float(cfg.get("carry_table_clearance_m", .005))
        self.lift_step = float(cfg.get("carry_lift_step_m", .002))
        self.max_rise = float(cfg.get("carry_max_lift_m", .03))
        self.stable_steps = int(cfg.get("carry_clear_stable_steps", 3))
        self.max_controls = int(cfg.get("carry_handoff_max_controls", 32))
        self.slip_limit = float(cfg["passive_object_position_drift_m"])
        self.rotation_limit = float(cfg["passive_object_rotation_drift_rad"])
        self.step_cap = float(cfg["position_command_cap_m"])
        self.rotation_cap = float(cfg["rotation_command_cap_rad"])
        if (not all(np.isfinite(x) and x > 0 for x in
                    (self.clearance, self.lift_step, self.max_rise, self.slip_limit,
                     self.rotation_limit, self.step_cap, self.rotation_cap)) or
                self.lift_step > self.step_cap or self.stable_steps < 1 or
                self.max_controls < self.stable_steps + 1):
            raise ValueError("invalid carry handoff configuration")
        self.reset()

    def reset(self):
        self.baselines = [None, None]
        self.records = [None, None]

    @staticmethod
    def relative(state, ai, obj):
        pose, item = state["poses"][ai], state["objects"][obj]
        r = np.asarray(pose["rotation"], dtype=float)
        o = Rotation.from_quat(np.roll(item["quaternion_wxyz"], -1)).as_matrix()
        return (r.T @ (np.asarray(item["position"])-np.asarray(pose["position"])),
                r.T @ o)

    def on_grasp(self, ai, state, obj):
        pos, rot = self.relative(state, ai, obj)
        self.baselines[ai] = (pos.copy(), rot.copy())
        self.records[ai] = None

    def arm(self, ai, state, obj):
        if self.baselines[ai] is None:
            raise CarryGateError("grasp reference not confirmed")
        if self.records[ai] is not None and self.records[ai]["stage"] != "done":
            raise CarryGateError("carry handoff already active")
        self.records[ai] = dict(stage="clearance", object=obj, controls=0,
                                checks=0, z0=float(state["poses"][ai]["position"][2]))
        return True

    def active(self, ai):
        return self.records[ai] is not None and self.records[ai]["stage"] != "done"

    def phase(self, ai):
        r = self.records[ai]
        return r["stage"] if r is not None else None

    def drift(self, ai, state):
        record, baseline = self.records[ai], self.baselines[ai]
        if record is None or baseline is None:
            raise CarryGateError("missing carry baseline")
        pos, rot = self.relative(state, ai, record["object"])
        dp = float(np.linalg.norm(pos - baseline[0]))
        dr = float(Rotation.from_matrix(rot @ baseline[1].T).magnitude())
        if not np.isfinite([dp, dr]).all():
            raise CarryGateError("nonfinite carried-object pose")
        return dp, dr

    def validate(self, ai, state, clearance, table_contact, held,
                 allow_table_contact=False, allow_assembly=False):
        r = self.records[ai]
        if r is None:
            raise CarryGateError("carry state not armed")
        dp, dr = self.drift(ai, state)
        if not held:
            raise CarryGateError("confirmed grasp lost")
        if not allow_assembly and (dp > self.slip_limit or dr > self.rotation_limit):
            raise CarryGateError("held object relative transform changed")
        if (not np.isfinite(clearance) or
                (not allow_table_contact and
                 (table_contact or clearance < self.clearance))):
            raise CarryGateError("carried object table clearance unsafe")
        return dict(relative_translation_m=dp, relative_rotation_rad=dr,
                    table_clearance_m=float(clearance),
                    table_contact=bool(table_contact))

    def propose(self, ai, state, clearance, table_contact, held):
        r = self.records[ai]
        if r is None or r["stage"] == "done":
            raise CarryGateError("carry handoff is not active")
        if r["controls"] >= self.max_controls:
            raise CarryGateError("carry handoff budget exhausted")
        info = self.validate(ai, state, clearance, table_contact, held,
                             allow_table_contact=(r["stage"] == "clearance"))
        if r["stage"] in ("clearance", "verify") and (
                table_contact or clearance < self.clearance):
            rise = float(state["poses"][ai]["position"][2]) - r["z0"]
            if rise + self.lift_step > self.max_rise + 1e-9:
                raise CarryGateError("unverified lift would exceed maximum rise")
            kind = "lift"
        elif r["stage"] == "ready":
            # A confirmed stable carry may offer Policy actions to the
            # original physical safety stack; no extra command preference.
            kind = "policy"
        else:
            kind = "hold"
        return dict(kind=kind, stage=r["stage"], control=r["controls"] + 1,
                    **info)

    def complete(self, ai, proposal, state, clearance, table_contact, held,
                 unmodified_policy=True):
        """Commit only after a full real physical control cycle."""
        r = self.records[ai]
        if (r is None or r["stage"] != proposal["stage"] or
                r["controls"] + 1 != proposal["control"]):
            raise CarryGateError("stale or partial carry proposal")
        self.validate(ai, state, clearance, table_contact, held,
                      allow_table_contact=(proposal["kind"] == "lift"))
        r["controls"] += 1
        kind = proposal["kind"]
        clear = not table_contact and clearance >= self.clearance
        if kind == "lift":
            r["stage"] = "verify" if clear else "clearance"
            r["checks"] = 0
        elif kind == "hold":
            if not clear:
                # This can only occur after an unsafe movement; never pretend
                # such a step provided stable free-space evidence.
                raise CarryGateError("carry hold lost table clearance")
            r["checks"] += 1
            if r["checks"] >= self.stable_steps:
                r["stage"] = "ready"
        elif kind == "policy":
            if r["stage"] != "ready" or not clear:
                raise CarryGateError("unqualified Policy handback")
            # A world-Z-clipped action is a real constrained control step,
            # but it cannot certify an unmodified Policy handback.
            if not unmodified_policy:
                return False
            r["stage"] = "done"
            return True
        else:
            raise CarryGateError("unknown carry step")
        return False


def object_table_gap(env, obj, distance_cap=.05, data_override=None):
    """Read-only smallest collision-geom distance to a named table geom.

    MuJoCo's geom distance can be approximate on some mesh pairs; therefore
    no-contact + several measured hold steps are additionally required.
    No matching, queryable collision geometry -> fail closed.
    """
    import mujoco
    model = env.sim.model._model
    data = env.sim.data._data if data_override is None else data_override
    names = env.sim.model.geom_names
    obj_geoms = set(getattr(env, obj).contact_geoms)
    object_ids = [i for i, name in enumerate(names)
                  if name in obj_geoms and
                  (int(model.geom_contype[i]) or int(model.geom_conaffinity[i]))]
    table_ids = [i for i, name in enumerate(names)
                 if name and "table" in name.lower()
                 and (int(model.geom_contype[i]) or int(model.geom_conaffinity[i]))]
    if not object_ids or not table_ids or not hasattr(mujoco, "mj_geomDistance"):
        raise CarryGateError("table/object collision geometry unavailable")
    values = []
    for oi in object_ids:
        for ti in table_ids:
            try:
                distance = float(mujoco.mj_geomDistance(
                    model, data, oi, ti, float(distance_cap), None))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise CarryGateError("table collision distance unavailable") from exc
            if not np.isfinite(distance):
                raise CarryGateError("nonfinite table collision distance")
            values.append(distance)
    return min(values)


def object_static_contact(contact, obj):
    return any({e["a"], e["b"]} == {obj, "static"}
               for e in contact["edges"])


def upward_osc_action(robot, step_m):
    """World +Z physical controller request using the existing OSC contract."""
    ctl = robot.part_controllers["right"]
    ctl.update()
    if (ctl.input_type != "delta" or ctl.input_ref_frame != "base" or
            ctl._goal_update_mode != "achieved"):
        raise CarryGateError("unsupported carry OSC controller")
    rotation = np.asarray(ctl.origin_ori)
    scales = np.asarray(ctl.output_max)
    if (rotation.shape != (3, 3) or scales.shape != (6,) or
            not np.isfinite(rotation).all() or not np.isfinite(scales).all() or
            np.any(scales <= 0) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)):
        raise CarryGateError("invalid carry OSC scaling")
    translation = rotation.T @ np.array([0., 0., step_m]) / scales[:3]
    command = np.r_[translation, np.zeros(3)]
    if np.any(abs(command) > 1.000001):
        raise CarryGateError("lift request outside legal physical action range")
    return command

