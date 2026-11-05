"""World-Z post-grasp projection: pure OSC/controller and gate regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from acoserl.post_grasp_z_clip import early_carry_eligible, project_world_z


def robot(rot=None):
    ctl = SimpleNamespace(
        input_type='delta', input_ref_frame='base',
        _goal_update_mode='achieved',
        origin_ori=np.eye(3) if rot is None else rot,
        output_max=np.ones(6))
    return SimpleNamespace(part_controllers={'right': ctl})


def config():
    return dict(post_grasp_z_clip_enabled=True,
                post_grasp_z_clip_max_controls=10,
                post_grasp_z_clip_release_clearance_m=.010,
                carry_table_clearance_m=.005)


def eligible(**updates):
    args = dict(holding=True, stable_grasp=True, held=True,
                mode='Handoff', carry_phase='ready', assembled=False,
                assembly_contact=False, coupled=False,
                reference_before_contact=True, policy_controls=0,
                clearance_m=.0052)
    args.update(updates)
    return early_carry_eligible(config(), **args)


def test_world_z_negative_is_zeroed_without_changing_xy_rotation_or_grip():
    proposed = np.array([.1, -.2, -.3, .4, -.1, .25, 1.])
    executed, audit = project_world_z(robot(), proposed)
    assert np.array_equal(executed[[0, 1, 3, 4, 5, 6]],
                          proposed[[0, 1, 3, 4, 5, 6]])
    assert executed[2] == 0.
    assert audit['removed_downward_m'] == pytest.approx(.3)
    assert proposed[2] == -.3


def test_rotated_base_clips_world_z_not_physical_action_third_coordinate():
    rot = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    proposed = np.array([.25, .4, .2, .5, .6, .7, -1.])
    executed, audit = project_world_z(robot(rot), proposed)
    assert audit is not None
    assert np.allclose(rot @ executed[:3], [.2, .4, 0.])
    assert np.array_equal(executed[3:], proposed[3:])


def test_zero_or_upward_world_z_is_not_modified():
    for z in (0., .7):
        proposed = np.array([.1, .2, z, .3, .4, .5, .6])
        executed, audit = project_world_z(robot(), proposed)
        assert np.array_equal(executed, proposed)
        assert audit is None


def test_projection_outside_osc_domain_refuses_instead_of_saturating():
    rot = np.array([[.8660254037844386, .5, 0.],
                    [0., 0., 1.],
                    [.5, -.8660254037844386, 0.]])
    proposed = np.array([1., 1., 0., 0., 0., 0., 1.])
    with pytest.raises(ValueError, match='outside OSC action range'):
        project_world_z(robot(rot), proposed)


def test_invalid_controller_contract_refuses():
    r = robot()
    r.part_controllers['right'].input_ref_frame = 'world'
    with pytest.raises(ValueError, match='unverified OSC'):
        project_world_z(r, np.zeros(7))


def test_clip_only_while_verified_early_carry():
    assert eligible()
    assert eligible(mode='Policy', carry_phase='done', policy_controls=9)
    for incompatible in (
        dict(holding=False), dict(stable_grasp=False), dict(held=False),
        dict(mode='Guide'), dict(carry_phase='clearance'),
        dict(assembled=True), dict(assembly_contact=True),
        dict(coupled=True), dict(reference_before_contact=False),
        dict(policy_controls=-1), dict(policy_controls=10),
        dict(clearance_m=.010)):
        assert not eligible(**incompatible), incompatible


def test_clearance_below_hard_minimum_refuses():
    with pytest.raises(ValueError, match='below hard minimum'):
        eligible(clearance_m=.004)


def test_gate_disabled_by_configuration():
    x = config()
    x['post_grasp_z_clip_enabled'] = False
    assert not early_carry_eligible(
        x, holding=True, stable_grasp=True, held=True,
        mode='Policy', carry_phase='done', assembled=False,
        assembly_contact=False, coupled=False,
        reference_before_contact=True, policy_controls=0,
        clearance_m=.0052)


def test_runtime_wiring_records_provenance_without_claiming_policy_handback():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    supervisor = (root/'src/acoserl/simple_autoserl.py').read_text()
    replay = (root/'src/acoserl/simple_replay.py').read_text()
    bridge = (root/'src/acoserl/simple_collection.py').read_text()
    assert 'POST_GRASP_Z_CLIP' in supervisor
    assert 'unmodified_policy=(issued[ai] == \'Policy\')' in supervisor
    assert 'post_grasp_z_clip=copy.deepcopy(z_clips)' in supervisor
    assert 'POST_GRASP_Z_CLIP' in replay
    assert 'POST_GRASP_Z_CLIP' in bridge
