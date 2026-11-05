"""Contract tests for the training-only privileged gripper arbitration.

These tests do not run MuJoCo, NPU, rollout collection, or training.
"""
import numpy as np
import pytest

from acoserl.privileged_handoff import PrivilegedHandoff, effective_match


def demo_trace():
    poses = np.zeros((230, 2, 3), dtype=float)
    orientations = np.broadcast_to(np.eye(3), (230, 2, 3, 3)).copy()
    objects = np.zeros((230, 3, 3), dtype=float)
    object_quats = np.zeros((230, 3, 4), dtype=float)
    object_quats[..., 0] = 1.0
    return dict(positions=poses, rotations=orientations,
                object_positions=objects, object_quaternions=object_quats)


def points():
    return {'arms': {
        'A': {'object': 'piece_1', 'first_close_action': 45, 'first_release_action': 149},
        'B': {'object': 'piece_2', 'first_close_action': 58, 'first_release_action': 215},
    }}


def physical_state():
    pose = lambda: dict(position=np.zeros(3), rotation=np.eye(3))
    obj = lambda: dict(position=np.zeros(3), quaternion_wxyz=np.array([1., 0., 0., 0.]))
    return dict(poses=[pose(), pose()],
                objects={'base': obj(), 'piece_1': obj(), 'piece_2': obj()},
                actual_grasps={'A': {'piece_1': False},
                               'B': {'piece_2': False}},
                official_metrics={'first_piece_assembled': False})


def make_expert():
    cfg = dict(stable_grasp_window_steps=10, privileged_pick_position_m=.005,
               privileged_pick_rotation_rad=.035, privileged_release_xy_m=.005,
               privileged_release_z_m=.005)
    return PrivilegedHandoff(points(), demo_trace(), cfg)


def test_gripper_uses_effective_sign_not_float_equality():
    assert effective_match(.5, 1.)
    assert effective_match(-.2, -1.)
    assert not effective_match(-1., 1.)
    assert effective_match(0., 1., np.array([-1., 1.]))
    assert not effective_match(0., 1., np.array([0., 0.]))
    with pytest.raises(ValueError, match='nonfinite'):
        effective_match(np.nan, 1.)


def test_only_one_gripper_dimension_is_changed_and_latch_commits_after_physics():
    expert, state = make_expert(), physical_state()
    actor = np.linspace(-.9, .9, 14)
    actor[6], actor[13] = -1., 1.
    action, mismatch, goal, phases, events = expert.propose(
        state, actor, skipped=(False, True))
    assert mismatch == [True, False]
    assert goal == [1., None]
    assert events == ['A:close']
    np.testing.assert_array_equal(action[:6], actor[:6])
    np.testing.assert_array_equal(action[7:], actor[7:])
    assert action[6] == 1.
    assert expert.phases == ['pregrasp', 'pregrasp']  # no speculative state change
    expert.commit(phases, state, (0, 0))
    assert expert.phases == ['closing', 'pregrasp']
    state['objects']['piece_1']['position'][0] = .1
    correct = actor.copy()
    correct[6] = 1.
    action, mismatch, _, phases, _ = expert.propose(
        state, correct, skipped=(False, True))
    assert not mismatch[0] and action[6] == correct[6]
    expert.commit(phases, state, (10, 0))
    assert expert.phases[0] == 'holding'
    # Object not yet above base; Actor premature opening is corrected every step.
    for _ in range(12):
        action, mismatch, _, phases, _ = expert.propose(
            state, actor, skipped=(False, True))
        assert mismatch == [True, False] and action[6] == 1.
        expert.commit(phases, state, (10, 0))
    state['objects']['piece_1']['position'][0] = 0.
    action, mismatch, _, phases, _ = expert.propose(
        state, correct, skipped=(False, True))
    assert mismatch[0] and action[6] == -1.
    expert.commit(phases, state, (10, 0))
    assert expert.phases[0] == 'done'


def test_second_arm_release_requires_first_piece_assembled():
    expert, state = make_expert(), physical_state()
    expert.phases = ['done', 'holding']
    action = np.zeros(14)
    action[13] = 1.
    _, _, target, phases, _ = expert.propose(state, action)
    assert target[1] == 1. and phases[1] == 'holding'
    state['official_metrics']['first_piece_assembled'] = True
    _, _, target, phases, _ = expert.propose(state, action)
    assert target[1] == -1. and phases[1] == 'releasing'


def test_object_relative_pick_not_world_absolute():
    expert, state = make_expert(), physical_state()
    for ai, name in enumerate(('piece_1', 'piece_2')):
        displacement = np.array([.07, -.03, .02])
        state['objects'][name]['position'] += displacement
        state['poses'][ai]['position'] += displacement
        assert expert.pick_aligned(state, ai)
    state['poses'][0]['position'][0] += .006
    assert not expert.pick_aligned(state, 0)


def test_recovery_ownership_skips_both_grippers():
    expert, state = make_expert(), physical_state()
    actor = np.zeros(14)
    action, mismatch, goals, phases, _ = expert.propose(
        state, actor, skipped=(True, True))
    np.testing.assert_array_equal(action, actor)
    assert mismatch == [False, False] and goals == [None, None]
    assert phases == ['pregrasp', 'pregrasp']


def test_termination_is_successful_full_episode_with_strict_ten_limit():
    manager_mod = pytest.importorskip('acoserl.simple_autoserl')
    mgr = manager_mod.SimpleBimanualAutoSERL.__new__(
        manager_mod.SimpleBimanualAutoSERL)
    mgr.cfg = dict(termination_intervention_steps=10)
    mgr.auto_disabled = False
    mgr.guide_events = 2
    mgr.recovery_events = 1
    mgr.handoff_disagreements = 6
    mgr.intervention_control_steps = 55  # old per-control criterion is irrelevant
    mgr.episode_interventions = 3
    mgr.handbacks = [1, 1]
    mgr.boundary_guard_steps = 0
    mgr._stable_grasp = {0, 1}
    result = mgr.episode_result(True)
    assert result['intervention_count'] == 9
    assert result['disable_requested'] and mgr.auto_disabled
    mgr.auto_disabled = False
    mgr.handoff_disagreements = 7
    assert not mgr.episode_result(True)['disable_requested']
    mgr.handoff_disagreements = 6
    assert not mgr.episode_result(False)['disable_requested']
    mgr.boundary_guard_steps = 1
    assert not mgr.episode_result(True)['disable_requested']
