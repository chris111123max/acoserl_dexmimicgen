"""The executed handoff guard is Online Replay, never Dynamic Demo."""
import torch
import numpy as np

from acoserl.simple_replay import make_row
import acoserl.simple_replay as replay_module


def _row(monkeypatch, controls, grip_owners=None, grip_sources=None, failed=False):
    def raw_row(*args, **kwargs):
        return {
            'action_source': torch.tensor(0, dtype=torch.long),
            'expert_action_mask': torch.zeros(14, dtype=torch.bool),
            'pose_sources': torch.tensor([0, 0]),
            'gripper_sources': torch.tensor([0, 0]),
            'policy_dimension_mask': torch.ones(14, dtype=torch.bool),
            'quality_validated': torch.tensor(False),
        }
    monkeypatch.setattr(replay_module, 'raw_row', raw_row)
    decision = {
        'per_arm_control': controls,
        'per_arm_gripper_control': grip_owners or [False, False],
        'per_arm_gripper_source': grip_sources or ['Policy', 'Policy'],
        'mode': 'HandoffGuard',
    }
    payload = {'executed': True, 'supervisor_stop_reason': 'unsafe' if failed else None,
               'auto_intervention': decision, 'env_id': 0, 'episode_id': 0, 'frame': 2}
    row = make_row(None, np.zeros(14), None, 0, False, False, payload)
    return row


def test_guard_A_excludes_only_A_motion_from_policy_and_demo(monkeypatch):
    row = _row(monkeypatch, ['HandoffGuard', 'Policy'])
    assert int(row['action_source']) == 9
    assert row['pose_sources'].tolist() == [4, 0]
    assert row['policy_dimension_mask'].tolist() == [False]*6 + [True]*8
    assert not row['expert_action_mask'].any() and not row['quality_validated']


def test_guard_B_independent_of_A_policy(monkeypatch):
    row = _row(monkeypatch, ['Policy', 'HandoffGuard'])
    assert int(row['action_source']) == 10
    assert row['pose_sources'].tolist() == [0, 4]
    assert row['policy_dimension_mask'].tolist() == [True]*7 + [False]*6 + [True]
    assert not row['expert_action_mask'].any()


def test_guard_does_not_qualify_other_arm_guide_as_expert_demo(monkeypatch):
    row = _row(monkeypatch, ['HandoffGuard', 'Guide'])
    assert int(row['action_source']) == 9
    assert row['pose_sources'].tolist() == [4, 1]
    assert not row['quality_validated'] and not row['expert_action_mask'].any()


def test_guard_and_event_gripper_are_both_non_policy(monkeypatch):
    row = _row(monkeypatch, ['HandoffGuard', 'Policy'],
               [True, False], ['EventGuide', 'Policy'])
    assert int(row['action_source']) == 9
    assert row['gripper_sources'].tolist() == [4, 0]
    assert row['policy_dimension_mask'].tolist() == [False]*7 + [True]*7
    assert not row['quality_validated']


def test_gripper_only_safety_override_not_mislabeled_as_policy(monkeypatch):
    row = _row(monkeypatch, ['Policy', 'Policy'],
               [True, False], ['EventGuide', 'Policy'])
    assert int(row['action_source']) == 9
    assert row['pose_sources'].tolist() == [0, 0]
    assert row['gripper_sources'].tolist() == [4, 0]
    assert row['policy_dimension_mask'].tolist() == [True]*6 + [False] + [True]*7


def test_guard_does_not_turn_failure_into_qualified_demo(monkeypatch):
    row = _row(monkeypatch, ['HandoffGuard', 'Policy'], failed=True)
    assert not row['quality_validated'] and not row['expert_action_mask'].any()
