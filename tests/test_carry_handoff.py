"""Post-grasp physical carry gate tests (pure controller logic, no simulator)."""
import numpy as np
import pytest
from acoserl.carry_handoff import CarryGate, CarryGateError


def state(obj_height=0.8, wrist_height=0.9):
    return {"poses": [
        {"position": np.array([0., 0., wrist_height]), "rotation": np.eye(3)},
        {"position": np.array([.2, 0., wrist_height]), "rotation": np.eye(3)}],
        "objects": {
            "piece_1": {"position": np.array([0., 0., obj_height]),
                        "quaternion_wxyz": np.array([1., 0., 0., 0.])},
            "piece_2": {"position": np.array([.2, 0., obj_height]),
                        "quaternion_wxyz": np.array([1., 0., 0., 0.])}}}


def gate():
    cfg = dict(carry_table_clearance_m=.005, carry_lift_step_m=.002,
               carry_max_lift_m=.03, carry_clear_stable_steps=3,
               carry_handoff_max_controls=8,
               passive_object_position_drift_m=.003,
               passive_object_rotation_drift_rad=.035,
               position_command_cap_m=.01,
               rotation_command_cap_rad=.02)
    return CarryGate(cfg)


def test_grasp_requires_reference_before_handoff():
    g = gate()
    with pytest.raises(CarryGateError):
        g.arm(0, state(), "piece_1")
    s = state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    assert g.phase(0) == "clearance"


def test_clearance_requires_real_lift_then_stable_hold_then_policy():
    g, s = gate(), state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    p = g.propose(0, s, 0., True, True)
    assert p["kind"] == "lift"
    s1 = state(obj_height=.802, wrist_height=.902)
    assert g.complete(0, p, s1, .006, False, True) is False
    assert g.phase(0) == "verify"
    for i in range(3):
        p = g.propose(0, s1, .006, False, True)
        assert p["kind"] == "hold"
        assert g.complete(0, p, s1, .006, False, True) is False
    assert g.phase(0) == "ready"
    p = g.propose(0, s1, .006, False, True)
    assert p["kind"] == "policy"
    assert g.complete(0, p, s1, .006, False, True) is True
    assert g.phase(0) == "done"


def test_carry_ready_allows_unmodified_policy_motion():
    g, s = gate(), state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    for _ in range(3):
        p = g.propose(0, s, .01, False, True)
        assert p["kind"] == "hold"
        g.complete(0, p, s, .01, False, True)
    assert g.phase(0) == "ready"
    p = g.propose(0, s, .01, False, True)
    assert p["kind"] == "policy"
    assert g.complete(0, p, s, .01, False, True)
    assert g.phase(0) == "done"


def test_carry_ready_still_rejects_failed_physical_clearance():
    g, s = gate(), state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    for _ in range(3):
        p = g.propose(0, s, .01, False, True)
        g.complete(0, p, s, .01, False, True)
    p = g.propose(0, s, .01, False, True)
    with pytest.raises(CarryGateError, match="clearance"):
        g.complete(0, p, s, 0., True, True)
    assert g.phase(0) == "ready"


def test_no_false_handoff_on_failed_contact_or_slip():
    g, s = gate(), state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    p = g.propose(0, s, 0., True, True)
    with pytest.raises(CarryGateError, match="grasp"):
        g.complete(0, p, s, .006, False, False)
    moved = state(obj_height=.81, wrist_height=.9)
    with pytest.raises(CarryGateError, match="relative transform"):
        g.complete(0, p, moved, .006, False, True)


def test_partial_or_stale_control_does_not_count():
    g, s = gate(), state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    p = g.propose(0, s, .01, False, True)
    assert g.records[0]["controls"] == 0
    bad = dict(p, control=9)
    with pytest.raises(CarryGateError, match="stale"):
        g.complete(0, bad, s, .01, False, True)
    assert g.records[0]["controls"] == 0


def test_both_arms_independent():
    g, s = gate(), state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    assert g.active(0) and not g.active(1)


def test_budget_stops_but_never_credits_handoff():
    g, s = gate(), state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    for _ in range(g.max_controls):
        # No measured lift progress: keep the finite safety budget.
        p = g.propose(0, s, 0., True, True)
        assert p["kind"] == "lift"
        g.complete(0, p, s, 0., True, True)
    with pytest.raises(CarryGateError, match="budget"):
        g.propose(0, s, 0., True, True)
    assert g.phase(0) != "done"


def test_invalid_parameters_and_reset():
    with pytest.raises(ValueError):
        CarryGate(dict(carry_table_clearance_m=-1,
                       passive_object_position_drift_m=.003,
                       passive_object_rotation_drift_rad=.035,
                       position_command_cap_m=.01,
                       rotation_command_cap_rad=.02))
    g, s = gate(), state()
    g.on_grasp(0, s, "piece_1")
    g.arm(0, s, "piece_1")
    g.reset()
    assert not g.active(0)
