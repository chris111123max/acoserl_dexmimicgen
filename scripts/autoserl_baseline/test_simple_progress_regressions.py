"""Targeted Simple AutoSERL regressions; no MuJoCo or synthetic training."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict

from acoserl.simple_autoserl import (
    ARMS, ArmState, GUIDE_PROGRESS_WINDOW, POLICY_AFTER_GUIDE_STEPS,
    SimpleBimanualAutoSERL,
)
from acoserl.simple_replay import SimpleDemo


def row(frame, mask=(), episode=1, segment=1, env=0, done=False):
    expert = torch.zeros(14, dtype=torch.bool)
    for index in mask:
        expert[index] = True
    return TensorDict({
        'env_id': torch.tensor(env, dtype=torch.long),
        'episode': torch.tensor(episode, dtype=torch.long),
        'frame': torch.tensor(frame, dtype=torch.long),
        'takeover_id': torch.tensor(segment, dtype=torch.long),
        'source': torch.tensor(1, dtype=torch.long),
        'expert_action_mask': expert,
        'quality_validated': torch.tensor(bool(expert.any())),
        'quality_handoff_confirmed': torch.tensor(False),
        'complete_physics': torch.tensor(True),
        'next': TensorDict({
            'done': torch.tensor(done),
            'terminated': torch.tensor(done),
            'truncated': torch.tensor(False),
            'reward': torch.tensor(0., dtype=torch.float32),
            'bootstrap_allowed': torch.tensor(not done),
            'boundary_kind': torch.tensor(1 if done else 0, dtype=torch.long),
        }, batch_size=[]),
    }, batch_size=[])


def decision(a='Policy', b='Policy', a_id=0, b_id=0):
    return {'per_arm_control': [a, b],
            'arm_takeover_ids': [a_id, b_id]}


class GuideProgressTests(unittest.TestCase):
    def test_stalled_nearest_but_position_improves(self):
        arm = ArmState(30, 57)
        for i in range(GUIDE_PROGRESS_WINDOW):
            arm.guide_errors.append((.035 - i*.0015, .03))
        self.assertFalse(SimpleBimanualAutoSERL._guide_no_progress(arm))

    def test_stalled_nearest_but_rotation_improves(self):
        arm = ArmState(30, 57)
        for i in range(GUIDE_PROGRESS_WINDOW):
            arm.guide_errors.append((.030, .07-i*.002))
        self.assertFalse(SimpleBimanualAutoSERL._guide_no_progress(arm))

    def test_plateau_and_unqualified_release_not_handback(self):
        arm = ArmState(30, 57, mode='Guide', target=40, takeover=7, age=65)
        for _ in range(GUIDE_PROGRESS_WINDOW):
            arm.guide_errors.append((.03, .05))
        self.assertTrue(SimpleBimanualAutoSERL._guide_no_progress(arm))
        manager = object.__new__(SimpleBimanualAutoSERL)
        manager.arms = [arm, ArmState(38, 70)]
        manager.events = []
        manager._release_guide_to_policy(0, 'stalled_no_qualified_p0')
        self.assertEqual(arm.mode, 'Policy')
        self.assertEqual(arm.cooldown, POLICY_AFTER_GUIDE_STEPS)
        self.assertEqual(arm.age, 0)
        self.assertEqual(manager.events[-1]['event'], 'guide_released_to_policy')
        self.assertFalse(any(e.get('event') == 'policy_handback_confirmed'
                             for e in manager.events))

    def _manager(self, can_return, improving=False, cooldown=0):
        manager = object.__new__(SimpleBimanualAutoSERL)
        arm = ArmState(30, 57, mode='Guide', target=40, takeover=8,
                       age=45, cooldown=cooldown)
        for i in range(GUIDE_PROGRESS_WINDOW):
            arm.nearest.append(35)
            arm.guide_errors.append((.035-i*.001 if improving else .030, .04))
        manager.arms = [arm, ArmState(38, 70)]
        manager.cfg = dict(intervention_enabled=True, trigger_position_m=.02,
            position_command_cap_m=.01, rotation_command_cap_rad=.02,
            position_gain=2., rotation_gain=1., max_recovery_steps=350)
        manager.n_states = 230
        manager.positions = np.zeros((230, 2, 3))
        manager.sequence = 8
        manager.episode_interventions = 1
        manager.events = []
        manager._event_cursor = 0
        manager._issued = None
        manager._segment = 0
        manager._segment_signature = None
        manager._error_before = 0.
        manager.steps = 0
        manager.stop_reason = None
        manager.auto_disabled = False
        manager.completed = [False, False]
        manager.grasp_streak = [0, 0]
        manager.points = {'arms': {
            'A': {'object': 'piece_1'},
            'B': {'object': 'piece_2'}}}
        state = {
            'poses': [dict(position=np.zeros(3), rotation=np.eye(3))
                      for _ in ARMS],
            'actual_grasps': {'A': {'piece_1': False},
                              'B': {'piece_2': False}},
        }
        contact = {'dynamic_A_B_coupled': False, 'edges': []}
        manager.contacts = types.SimpleNamespace(read=lambda: contact)
        manager.physical_danger = lambda s, c: None
        manager._domain = lambda s, c, ai: (0, 51) if ai == 0 else None
        manager._measure = lambda s, ai, domain: (35, 40, .03, True)
        manager._can_return = lambda s, c, ai: can_return
        manager.pose = lambda ai, index: dict(
            position=np.zeros(3), rotation=np.eye(3))
        return manager, state

    def _choose(self, manager, state):
        fake_core = types.ModuleType('acoserl.bimanual_core')
        fake_core.snapshot = lambda env: state
        fake_core.servo = lambda env, targets, grips, **kwargs: np.zeros(14)
        with patch.dict(sys.modules, {'acoserl.bimanual_core': fake_core}):
            return manager.choose(np.zeros(14, dtype=float))

    def test_safe_guide_stall_enters_real_p0_state(self):
        manager, state = self._manager(can_return=True)
        action, dec = self._choose(manager, state)
        self.assertIsNotNone(action)
        self.assertEqual(dec['per_arm_control'], ['ReturnP0', 'Policy'])
        self.assertEqual(manager.arms[0].mode, 'ReturnP0')
        self.assertEqual(manager.arms[0].takeover, 9)
        self.assertEqual(manager.stop_reason, None)

    def test_unsafe_p0_falls_back_to_policy_with_cooldown(self):
        manager, state = self._manager(can_return=False)
        action, dec = self._choose(manager, state)
        self.assertTrue(np.array_equal(action, np.zeros(14)))
        self.assertEqual(dec['per_arm_control'], ['Policy', 'Policy'])
        self.assertEqual(manager.arms[0].cooldown, POLICY_AFTER_GUIDE_STEPS)
        self.assertIsNone(manager.stop_reason)

    def test_actively_improving_guide_not_interrupted(self):
        manager, state = self._manager(can_return=True, improving=True)
        action, dec = self._choose(manager, state)
        self.assertEqual(dec['per_arm_control'], ['Guide', 'Policy'])
        self.assertEqual(manager.arms[0].mode, 'Guide')

    def test_policy_cooldown_prevents_new_returnp0(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].mode = 'Policy'
        manager.arms[0].cooldown = 16
        action, dec = self._choose(manager, state)
        self.assertEqual(dec['per_arm_control'], ['Policy', 'Policy'])
        self.assertEqual(manager.arms[0].mode, 'Policy')
        self.assertIsNone(manager.stop_reason)


class DemoQualityTests(unittest.TestCase):
    def setUp(self):
        self.bank = SimpleDemo(capacity=64, gamma=.99, seed=7)
        self.mask_a = list(range(6))
        self.mask_b = list(range(7, 13))

    def test_unfinished_guide_is_not_expert_demo(self):
        for i in range(8):
            self.bank.observe(row(i, self.mask_a), decision=decision('Guide','Policy',5,0))
        self.assertEqual(self.bank.stats['accepted_rows'], 0)
        self.bank.close_episode(0, bootstrap=True, kind=7)
        self.assertEqual(len(self.bank.buffer), 0)
        self.assertEqual(self.bank.stats['unqualified_discarded_rows'], 8)

    def test_successful_handoff_qualifies_bounded_guide_tail(self):
        for i in range(5):
            self.bank.observe(row(i, self.mask_a), decision=decision('Guide','Policy',7,0))
        self.assertEqual(len(self.bank.buffer), 0)
        self.bank.observe(row(5), decision=decision(),
            handbacks=[dict(arm='A', takeover_id=7)])
        self.assertEqual(self.bank.stats['accepted_rows'], 5)
        self.bank.close_episode(0, bootstrap=True, kind=7)
        self.assertEqual(len(self.bank.buffer), 5)
        self.assertTrue(bool(self.bank.buffer[:]['quality_handoff_confirmed'].all()))

    def test_two_arm_guides_need_both_real_handbacks(self):
        for i in range(4):
            self.bank.observe(row(i, self.mask_a+self.mask_b),
                decision=decision('Guide','Guide',10,11))
        self.bank.observe(row(4), decision=decision(),
            handbacks=[dict(arm='A', takeover_id=10)])
        self.assertEqual(self.bank.stats['accepted_rows'], 0)
        self.bank.observe(row(5), decision=decision(),
            handbacks=[dict(arm='B', takeover_id=11)])
        self.assertEqual(self.bank.stats['accepted_rows'], 4)
        self.bank.close_episode(0, bootstrap=True, kind=7)
        self.assertEqual(len(self.bank.buffer), 4)
        self.assertEqual(self.bank.stats['dual_arm_rows'], 4)

    def test_aborted_guide_discards_candidates(self):
        for i in range(3):
            self.bank.observe(row(i, self.mask_a), decision=decision('Guide','Policy',12,0))
        self.bank.observe(row(3, self.mask_a), decision=decision('Guide','Policy',12,0),
            control_events=[dict(event='guide_released_to_policy',
                                 arm='A', takeover=12)])
        self.bank.observe(row(4), decision=decision(),
            handbacks=[dict(arm='A', takeover_id=12)])
        self.bank.close_episode(0, bootstrap=True, kind=7)
        self.assertEqual(len(self.bank.buffer), 0)
        self.assertEqual(self.bank.stats['failed_guide_discarded_rows'], 4)

    def test_pending_tail_is_bounded_per_worker(self):
        for i in range(40):
            self.bank.observe(row(i, self.mask_a), decision=decision('Guide','Policy',14,0))
        self.assertEqual(len(self.bank.pending[0]), 32)
        self.assertEqual(self.bank.stats['expired_candidate_rows'], 8)
        self.bank.close_episode(0)
        self.assertEqual(len(self.bank.buffer), 0)


if __name__ == '__main__':
    unittest.main()
