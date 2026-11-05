"""Deterministic tests for event proposal, confirmation and relative pose."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from acoserl.gripper_events import DemoGripperEvents, compose_relative

def pose(x=0, y=0, z=0, yaw=0):
    return {'position': np.array([x, y, z], dtype=float),
            'rotation': Rotation.from_euler('z', yaw).as_matrix()}

def make(randomized=False, max_steps=80):
    points = {'arms': {'A': {'first_close_action': 1, 'first_grasp': 2, 'p1': 2, 'first_release_action': 3},
                       'B': {'first_close_action': 2, 'first_grasp': 3, 'p1': 3, 'first_release_action': 4}}}
    pos = np.zeros((5, 2, 3))
    pos[1, 0] = [.2, 0, 0]
    pos[2, 0] = [.25, 0, 0]
    pos[2, 1] = [.3, 0, 0]
    pos[3, 1] = [.35, 0, 0]
    pos[3, 0] = [1, 0, 0]
    pos[4, 1] = [2, 0, 0]
    rot = np.broadcast_to(np.eye(3), (5, 2, 3, 3)).copy()
    g = DemoGripperEvents(points, pos, rot,
          {'fixed_initial_state': not randomized,
           'gripper_grasp_confirm_max_steps': max_steps,
           'gripper_release_stable_steps': 3})
    g.reset([pose(), pose()])
    return g

def commit(g, phases=None, a=None, b=None, stable=(False, False)):
    a = pose() if a is None else a
    b = pose() if b is None else b
    return g.complete_step(g.phases[:] if phases is None else phases,
                           [a, b], [a, b], grasp_confirmed=stable)

def begin(g, x):
    q = g.propose(0, x, -1, qualified_close=True)
    assert q.next_phase == 'closing'
    assert commit(g, ['closing', 'open'], a=x)['events'] == [(0, 'close_started')]

def confirm(g, x):
    for _ in range(9):
        commit(g, a=x)
        assert g.phases[0] == 'closing'
        assert g.grasp_anchor[0] is None
    assert commit(g, a=x, stable=(True, False))['events'] == [(0, 'grasp_confirmed')]
    assert g.phases[0] == 'holding'

def test_close_does_not_imply_stable_grasp():
    g = make()
    x = pose(.2)
    assert g.propose(0, x, 1).next_phase == 'closing'
    assert g.phases == ['open', 'open']
    begin(g, x)
    assert g.propose(0, pose(.4), -1).command == 1
    confirm(g, x)
    assert np.allclose(g.grasp_anchor[0]['position'], x['position'])
    assert g.propose(0, pose(.7), -1).command == 1

def test_unconfirmed_close_times_out():
    g = make(max_steps=10)
    x = pose(.2)
    begin(g, x)
    for _ in range(8):
        assert not commit(g, a=x)['grasp_timeouts']
    assert commit(g, a=x)['grasp_timeouts'] == [0]
    assert g.phases[0] == 'closing' and g.grasp_anchor[0] is None

def test_random_actor_can_select_new_pick_but_fixed_rejects():
    x = pose(.6, .4)
    assert make().propose(0, x, 1).next_phase == 'open'
    g = make(randomized=True)
    assert g.propose(0, x, 1).next_phase == 'closing'
    begin(g, x)
    confirm(g, x)
    assert np.allclose(g.relative_reference(0, 3)['position'], [1.35, .4, 0])

def test_rotation_and_reset():
    g = make(randomized=True)
    x = pose(.5, -.2, yaw=np.pi/2)
    begin(g, x)
    confirm(g, x)
    t = g.release_reference(0)
    assert np.allclose(t['position'], [.5, .55, 0])
    g.reset([pose(), pose()])
    assert g.release_reference(0) is None

def test_quiet_release_and_one_way_gripper():
    g = make()
    x = pose(.2)
    begin(g, x)
    confirm(g, x)
    target = g.release_reference(0)
    assert g.propose(0, target, -1).next_phase == 'holding'
    for _ in range(4):
        commit(g, a=target)
    assert g.release_qualified(0, target)
    assert g.propose(0, target, 1).event == 'release'
    assert commit(g, ['released', 'open'], a=target)['events'] == [(0, 'release')]
    assert g.propose(0, target, 1).command == -1

def test_second_release_waits_for_first_actual_anchor():
    g = make(randomized=True)
    x = pose(.2)
    begin(g, x)
    confirm(g, x)
    b = pose(.3)
    assert g.propose(1, b, 1).next_phase == 'closing'
    commit(g, ['holding', 'closing'], a=x, b=b)
    for _ in range(9):
        commit(g, a=x, b=b)
    commit(g, a=x, b=b, stable=(False, True))
    assert g.phases[1] == 'holding' and g.release_reference(1) is None
    release = g.release_reference(0)
    for _ in range(4):
        commit(g, a=release, b=b)
    assert g.propose(0, release, -1).next_phase == 'released'
    commit(g, ['released', 'holding'], a=release, b=b)
    assert np.allclose(g.release_reference(1)['position'], [1.95, 0, 0])

def test_se3_offset_rotates():
    q = compose_relative(pose(5, 6, yaw=np.pi/2), pose(1), pose(2))
    assert np.allclose(q['position'], [5, 7, 0])

def test_forbidden_event_transition():
    with pytest.raises(ValueError):
        commit(make(), ['released', 'open'])

def test_confirmed_anchor_observed_pose_not_preclose_with_rotation():
    g = make(randomized=True)
    attempt = pose(.2, 0, yaw=0)
    confirmed = pose(.220, .007, yaw=.23)
    begin(g, attempt)
    confirm(g, confirmed)
    assert np.allclose(g.grasp_attempt_anchor[0]['position'], attempt['position'])
    assert np.allclose(g.grasp_anchor[0]['position'], confirmed['position'])
    assert np.allclose(g.grasp_anchor[0]['rotation'], confirmed['rotation'])
    p1 = g.points['arms']['A']['p1']
    target = g.relative_reference(0, p1)
    assert np.allclose(target['position'], confirmed['position'])
    assert np.allclose(target['rotation'], confirmed['rotation'])
    release = g.release_reference(0)
    expected = confirmed['position'] + confirmed['rotation'] @ np.array([.75, 0., 0.])
    assert np.allclose(release['position'], expected)


def test_confirmed_anchor_is_independent_per_arm():
    g = make(randomized=True)
    attempt = pose(.3, 0)
    confirmed = pose(.325, .009, yaw=-.18)
    command = g.propose(1, attempt, 1.)
    assert command.next_phase == 'closing'
    commit(g, ['open', 'closing'], b=attempt)
    for _ in range(9):
        commit(g, b=confirmed)
    commit(g, b=confirmed, stable=(False, True))
    assert g.phases[1] == 'holding'
    assert g.grasp_anchor[0] is None
    assert np.allclose(g.grasp_anchor[1]['position'], confirmed['position'])
    p1 = g.points['arms']['B']['p1']
    assert np.allclose(g.relative_reference(1, p1)['position'], confirmed['position'])
    assert np.allclose(g.relative_reference(1, p1)['rotation'], confirmed['rotation'])
