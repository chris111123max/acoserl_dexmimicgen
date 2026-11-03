"""Focused deterministic tests for relative two-event gripper control.

No training, simulation, network or checkpoint dependencies.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from acoserl.gripper_events import DemoGripperEvents, compose_relative


def pose(x=0., y=0., z=0., yaw=0.):
    return {"position": np.array([x, y, z], dtype=float),
            "rotation": Rotation.from_euler("z", yaw).as_matrix()}


def make():
    points = {"arms": {
        "A": {"first_close_action": 1, "first_release_action": 3},
        "B": {"first_close_action": 2, "first_release_action": 4}}}
    positions = np.zeros((5, 2, 3))
    positions[1, 0] = [.2, 0, 0]
    positions[2, 1] = [.3, 0, 0]
    positions[3, 0] = [1., 0, 0]
    positions[4, 1] = [2., 0, 0]
    rotations = np.broadcast_to(np.eye(3), (5, 2, 3, 3)).copy()
    g = DemoGripperEvents(points, positions, rotations,
                          {"gripper_release_stable_steps": 3})
    g.reset([pose(), pose()])
    return g


def commit(g, phases, a, b=None, previous=None):
    observed = [a, pose() if b is None else b]
    g.complete_step(phases, observed, previous or observed)


def test_early_close_blocked_and_qualified_close_latches():
    g = make()
    assert g.propose(0, pose(), 1.).command == -1.
    assert g.phases == ["open", "open"]
    p = pose(.2)
    proposed = g.propose(0, p, -1., qualified_close=True)
    assert proposed.event == "grasp" and proposed.command == 1.
    commit(g, ["holding", "open"], p)
    assert g.propose(0, pose(.5), -1.).command == 1.


def test_relative_release_rotates_and_requires_stable_pose():
    g = make()
    first = pose(.5, -.2, yaw=np.pi / 2)
    commit(g, ["holding", "open"], first)
    target = g.release_reference(0)
    assert np.allclose(target["position"], [.5, .6, 0.])
    assert np.allclose(target["rotation"], first["rotation"])
    assert not g.release_qualified(0, target)
    for _ in range(3):
        commit(g, ["holding", "open"], target)
    assert g.release_qualified(0, target)
    request = g.propose(0, target, 1.)
    assert request.event == "release" and request.command == -1.
    commit(g, ["released", "open"], target)
    assert g.propose(0, target, 1.).command == -1.


def test_second_release_waits_for_first():
    g = make()
    b = pose(.3)
    commit(g, ["open", "holding"], pose(), b)
    for _ in range(4):
        commit(g, ["open", "holding"], pose(), pose(2.))
    assert not g.release_qualified(1, pose(2.))
    first = pose(.2)
    commit(g, ["holding", "holding"], first, pose(2.))
    released_a = pose(1., .1)
    commit(g, ["released", "holding"], released_a, pose(2.))
    assert np.allclose(g.release_reference(1)["position"], [2., .1, 0.])
    assert not g.release_qualified(1, pose(2., .1))
    for _ in range(3):
        commit(g, ["released", "holding"], released_a, pose(2., .1))
    assert g.release_qualified(1, pose(2., .1))


def test_reset_discards_relative_anchors():
    g = make()
    commit(g, ["holding", "open"], pose(.2))
    assert g.release_reference(0) is not None
    g.reset([pose(), pose()])
    assert g.release_reference(0) is None
    assert g.phases == ["open", "open"]


def test_rigid_relative_transform_not_world_offset():
    target = compose_relative(pose(5., 6., yaw=np.pi/2),
                              pose(1.), pose(2.))
    assert np.allclose(target["position"], [5., 7., 0.])
