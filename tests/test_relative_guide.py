"""Reference-rebase tests; no MuJoCo, training, or network needed."""
import numpy as np
from scipy.spatial.transform import Rotation
from acoserl.gripper_events import DemoGripperEvents
from acoserl.relative_guide import RelativeGuideReference

def pose(x=0, y=0, yaw=0):
    return {'position': np.array([x, y, 0.], dtype=float),
            'rotation': Rotation.from_euler('z', yaw).as_matrix()}

def make(randomized=True, margin=.12):
    pos = np.zeros((6, 2, 3))
    pos[:, 0, 0] = np.arange(6) * .1
    pos[:, 1, 0] = np.arange(6) * .12
    rot = np.broadcast_to(np.eye(3), (6, 2, 3, 3)).copy()
    points = {'arms': {
        'A': {'first_close_action': 1, 'first_grasp': 2, 'first_release_action': 4},
        'B': {'first_close_action': 2, 'first_grasp': 3, 'first_release_action': 5}}}
    cfg = {'fixed_initial_state': not randomized, 'free_workspace_margin_m': margin}
    ev = DemoGripperEvents(points, pos, rot, cfg)
    return ev, RelativeGuideReference(pos, rot, points, ev, cfg)

def anchor(events, ai, x, y=0, yaw=0):
    events.phases[ai] = 'holding'
    events.grasp_anchor[ai] = pose(x, y, yaw)

def test_random_pregrasp_never_uses_fixed_reference():
    ev, r = make()
    assert not r.can_guide(0) and not r.can_guide(1)
    assert not r.activate(0)
    assert np.array_equal(r.positions, r.demo_positions)
    _, fixed = make(randomized=False)
    assert fixed.can_guide(0)

def test_rebase_updates_full_carry_reference_not_original():
    ev, r = make()
    anchor(ev, 0, .15, .03)
    assert r.activate(0)
    assert r.can_guide(0)
    assert np.allclose(r.positions[2, 0], [.25, .03, 0])
    assert np.allclose(r.positions[4, 0], ev.release_reference(0)['position'])
    assert np.allclose(r.demo_positions[:, 0, 0], np.arange(6) * .1)

def test_rotated_rebase_tracks_pose():
    ev, r = make(margin=.7)
    anchor(ev, 0, .1, .01, yaw=.2)
    assert r.activate(0)
    expected = np.array([.1, .01, 0.]) + Rotation.from_euler('z', .2).as_matrix() @ np.array([.3, 0., 0.])
    assert np.allclose(r.positions[4, 0], expected)
    assert np.allclose(r.rotations[4, 0], Rotation.from_euler('z', .2).as_matrix())

def test_large_shift_rejects_and_leaves_workspace_unchanged():
    ev, r = make()
    anchor(ev, 0, .1, .4)
    original = r.positions.copy()
    lo, hi = r.workspace_bounds(0)
    assert not r.activate(0)
    assert r.rejected[0] == 'relative_reference_outside_original_workspace'
    assert not r.can_guide(0)
    assert np.array_equal(r.positions, original)
    lo2, hi2 = r.workspace_bounds(0)
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)

def test_b_guide_disabled_if_its_target_disagrees_with_a_release():
    ev, r = make()
    anchor(ev, 1, .28, .02)
    assert r.activate(1) and r.can_guide(1)
    ev.phases[0] = 'released'
    ev.first_release_anchor = pose(.4, .06)
    assert not r.release_consistent(1)
    assert not r.can_guide(1)

def test_reset_discards_relative_changes():
    ev, r = make()
    anchor(ev, 0, .13, .05)
    assert r.activate(0)
    r.reset()
    ev.reset([pose(), pose()])
    assert not r.can_guide(0)
    assert not any(r.ready)
    assert np.array_equal(r.positions, r.demo_positions)
