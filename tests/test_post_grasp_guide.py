"""Regression checks for post-grasp Guide hysteresis and window accounting."""
from acoserl.post_grasp_guide import in_window, trigger_distance_m, guide_exit

CFG = dict(trigger_position_m=.02, exit_position_m=.005,
           exit_rotation_rad=.02,
           post_grasp_guide_policy_steps=10,
           post_grasp_trigger_position_m=.005,
           post_grasp_exit_position_m=.002)


def test_window_starts_after_grasp_and_ends_after_ten_completed_policy_cycles():
    assert not in_window(-1, True, CFG)
    assert not in_window(0, False, CFG)
    assert all(in_window(i, True, CFG) for i in range(10))
    assert not in_window(10, True, CFG)


def test_special_trigger_then_restored_normal_trigger():
    assert trigger_distance_m(CFG, True) == .005
    assert trigger_distance_m(CFG, False) == .02


def test_early_guide_needs_both_two_mm_and_rotation_threshold():
    assert guide_exit(CFG, True, .0019, .019, False) == (True, True)
    assert guide_exit(CFG, True, .0021, .019, True) == (False, False)
    assert guide_exit(CFG, True, .0019, .021, True) == (False, False)


def test_normal_guide_retains_existing_convergence_exit():
    assert guide_exit(CFG, False, .004, .019, False) == (True, True)
    assert guide_exit(CFG, False, .015, .019, True) == (True, False)
    assert guide_exit(CFG, False, .015, .019, False) == (False, False)


def test_runtime_wiring_when_patch_is_installed():
    """Catch accidental restoration of the 32-step gate in installed source."""
    from pathlib import Path
    import pytest
    main = Path(__file__).resolve().parents[1] / 'src/acoserl/simple_autoserl.py'
    if not main.exists():
        pytest.skip('patch bundle contains only changed/added files, not entire repository')
    text = main.read_text(encoding='utf-8')
    assert 'POLICY_AFTER_GUIDE_STEPS' not in text
    assert "arm.cooldown == 0" not in text
    assert "arm.cooldown <= 0" not in text
    assert "if arm.mode == 'Policy' and arm.cooldown" not in text
    assert 'per_arm_post_grasp_policy_controls=' in text
    assert "and not released_guide_this_cycle" in text
    assert 'guide_exit(' in text
