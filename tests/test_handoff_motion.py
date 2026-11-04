"""Deterministic handoff-guard tests, without MuJoCo or training."""
import numpy as np
import pytest
from acoserl.handoff_motion import HandoffBudgetExhausted, HandoffMotionGuard


def guard(**kwargs):
    return HandoffMotionGuard({'handoff_motion_guard_enabled': True,
                               'handoff_motion_min_steps': 3,
                               'handoff_motion_max_steps': 8,
                               'handoff_motion_max_action_delta': .05,
                               **kwargs})


def test_first_policy_step_is_slewed_from_real_settle_command():
    x = guard()
    last = [0., .02, 0., 0., 0., 0.]
    x.arm(0, last)
    p = x.propose(0, [.69, -.5, .25, 0., .4, -.1])
    assert np.allclose(p['executed'], [.05, -.03, .05, 0., .05, -.05])
    assert p['limited'] and p['control'] == 1
    assert x.active(0) and x._controls[0] == 0  # propose is not a physical step
    x.complete_step(0, p, p['executed'])
    assert x._controls[0] == 1


def test_slew_never_exits_with_an_abrupt_jump():
    x = guard()
    x.arm(0, np.zeros(6))
    target = [.2, 0., 0., 0., 0., 0.]
    outputs = []
    for _ in range(4):
        p = x.propose(0, target)
        outputs.append(p['executed'][0])
        done = x.complete_step(0, p, p['executed'])
    assert np.allclose(outputs, [.05, .10, .15, .20])
    assert done and not x.active(0)


def test_minimum_window_is_observed_even_when_policy_is_identical():
    x = guard()
    x.arm(0, [0.]*6)
    for k in range(3):
        p = x.propose(0, [0.]*6)
        done = x.complete_step(0, p, p['executed'])
        assert done == (k == 2)


def test_budget_exhaustion_fails_closed_not_abruptly_policy():
    x = guard(handoff_motion_max_steps=3)
    x.arm(0, [0.]*6)
    for _ in range(3):
        p = x.propose(0, [1.]*6)
        x.complete_step(0, p, p['executed'])
    with pytest.raises(HandoffBudgetExhausted):
        x.propose(0, [1.]*6)
    assert x.active(0)


def test_one_arm_never_changes_the_other():
    x = guard()
    x.arm(0, [0.]*6)
    assert not x.active(1)
    p = x.propose(0, [1.]*6)
    assert p['executed'] == [.05]*6
    assert not x.active(1)


def test_external_safety_action_rebases_previous_action_but_not_budget():
    x = guard()
    x.arm(0, [0.]*6)
    x.note_external(0, [.4, 0., 0., 0., 0., 0.])
    p = x.propose(0, [1., 0., 0., 0., 0., 0.])
    assert np.isclose(p['executed'][0], .45)
    assert x._controls[0] == 0


def test_reset_cancels_pending_handbacks_and_rejects_bad_shape():
    x = guard()
    x.arm(0, [0.]*6)
    x.reset()
    assert not x.active(0)
    with pytest.raises(ValueError):
        x.arm(0, [0.]*7)


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        guard(handoff_motion_max_action_delta=0)
    with pytest.raises(ValueError):
        guard(handoff_motion_min_steps=9)
    with pytest.raises(ValueError):
        guard(handoff_motion_max_action_delta=float('nan'))


def test_disabled_guard_does_not_arm():
    x = HandoffMotionGuard({'handoff_motion_guard_enabled': False})
    assert not x.arm(0, [0.]*6)
    assert not x.active(0)


def test_executed_motion_mismatch_rejected():
    x = guard()
    x.arm(0, [0.]*6)
    p = x.propose(0, [1.]*6)
    with pytest.raises(ValueError):
        x.complete_step(0, p, [0.]*6)
