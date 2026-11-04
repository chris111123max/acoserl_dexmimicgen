"""Targeted Simple AutoSERL regressions; no MuJoCo or synthetic training."""
import sys
import types
import unittest
from collections import deque
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict

from acoserl.simple_autoserl import (
    ARMS, ArmState, GUIDE_PROGRESS_WINDOW,
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
        self.assertEqual(arm.cooldown, 0)
        self.assertEqual(arm.age, 0)
        self.assertEqual(manager.events[-1]['event'], 'guide_released_to_policy')
        self.assertFalse(any(e.get('event') == 'policy_handback_confirmed'
                             for e in manager.events))

    def _manager(self, can_return, improving=False, cooldown=0):
        manager = object.__new__(SimpleBimanualAutoSERL)
        # __new__ bypasses the real constructor; snapshot/servo below are
        # replaced by controlled stubs but choose() still reads manager.env.
        manager.env = object()
        arm = ArmState(30, 57, mode='Guide', target=40, takeover=8,
                       age=45, cooldown=cooldown)
        for i in range(GUIDE_PROGRESS_WINDOW):
            arm.nearest.append(35)
            arm.guide_errors.append((.035-i*.001 if improving else .030, .04))
        manager.arms = [arm, ArmState(38, 70)]
        manager.cfg = dict(intervention_enabled=True, trigger_position_m=.02,
            position_command_cap_m=.01, rotation_command_cap_rad=.02,
            position_gain=2., rotation_gain=1., max_recovery_steps=350,
            exit_position_m=.005, exit_rotation_rad=.02, stable_exit_steps=3,
            p0_alignment_m=.001, relative_grasp_motion_m=.0005)
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
        manager._issued_targets = [None, None]
        manager._stable_grasp = set()
        manager._touched = set()
        manager._assembly_anchor = None
        manager._assembly_rest = deque(maxlen=3)
        manager.handbacks = [0, 0]
        manager.intervention_control_steps = 0
        manager.steps = 0
        manager.stop_reason = None
        manager.auto_disabled = False
        manager.completed = [False, False]
        manager.grasp_streak = [0, 0]
        manager.points = {'arms': {
            'A': {'object': 'piece_1', 's0': 0, 'p0': 30, 'p1': 57},
            'B': {'object': 'piece_2', 's0': 0, 'p0': 38, 'p1': 70}}}
        state = {
            'poses': [dict(position=np.zeros(3), rotation=np.eye(3))
                      for _ in ARMS],
            'actual_grasps': {'A': {'piece_1': False},
                              'B': {'piece_2': False}},
            'official_metrics': {'first_piece_assembled': False,
                                 'task': False},
        }
        contact = {'dynamic_A_B_coupled': False, 'edges': []}
        manager.contacts = types.SimpleNamespace(read=lambda: contact)
        manager.gripper_events = types.SimpleNamespace(phases=['open', 'open'])
        manager._guide_gripper_event = lambda state, contact, ai, target: None
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

    def _after(self, manager, state, position_error=.04, rotation_error=.04):
        fake_core = types.ModuleType('acoserl.bimanual_core')
        fake_core.snapshot = lambda env: state
        fake_core.errors = lambda actual, target: {
            'position_m': position_error, 'rotation_rad': rotation_error}
        with patch.dict(sys.modules, {'acoserl.bimanual_core': fake_core}):
            return manager.after_step()

    def _complete_policy_step(self, manager, state):
        fake_core = types.ModuleType('acoserl.bimanual_core')
        fake_core.snapshot = lambda env: state
        fake_core.servo = lambda env, targets, grips, **kw: np.zeros(14)
        fake_core.errors = lambda actual, target: {
            'position_m': .04, 'rotation_rad': .04}
        with patch.dict(sys.modules, {'acoserl.bimanual_core': fake_core}):
            action, decision = manager.choose(np.zeros(14, dtype=float))
            after = manager.after_step()
        return action, decision, after

    def test_safe_guide_stall_enters_real_p0_state(self):
        manager, state = self._manager(can_return=True)
        action, dec = self._choose(manager, state)
        self.assertIsNotNone(action)
        self.assertEqual(dec['per_arm_control'], ['ReturnP0', 'Policy'])
        self.assertEqual(manager.arms[0].mode, 'ReturnP0')
        self.assertEqual(manager.arms[0].takeover, 9)
        self.assertEqual(manager.stop_reason, None)

    def test_unsafe_p0_falls_back_to_policy_without_cooldown(self):
        manager, state = self._manager(can_return=False)
        action, dec = self._choose(manager, state)
        self.assertTrue(np.array_equal(action, np.zeros(14)))
        self.assertEqual(dec['per_arm_control'], ['Policy', 'Policy'])
        self.assertEqual(manager.arms[0].cooldown, 0)
        self.assertIsNone(manager.stop_reason)

    def test_actively_improving_guide_not_interrupted(self):
        manager, state = self._manager(can_return=True, improving=True)
        action, dec = self._choose(manager, state)
        self.assertEqual(dec['per_arm_control'], ['Guide', 'Policy'])
        self.assertEqual(manager.arms[0].mode, 'Guide')

    def test_no_cooldown_allows_qualified_returnp0(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].mode = 'Policy'
        manager.arms[0].cooldown = 16  # stale telemetry must not gate recovery
        action, dec = self._choose(manager, state)
        self.assertEqual(dec['per_arm_control'], ['ReturnP0', 'Policy'])
        self.assertEqual(manager.arms[0].mode, 'ReturnP0')
        self.assertIsNone(manager.stop_reason)


    def test_guide_80th_control_releases_without_episode_stop(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].age = 79
        manager._issued = ('Guide', 'Policy')
        manager._issued_targets = [40, None]
        after = self._after(manager, state)
        self.assertIsNone(manager.stop_reason)
        self.assertEqual(manager.arms[0].mode, 'Policy')
        self.assertEqual(manager.arms[0].cooldown, 0)
        self.assertEqual(manager.arms[0].age, 0)
        self.assertFalse(after['handback'])
        self.assertEqual(after['handbacks'], [])
        self.assertEqual(after['control_events'][-1]['event'],
                         'guide_released_to_policy')
        self.assertEqual(after['control_events'][-1]['reason'],
                         'guide_max_controls')

    def test_qualified_recovery_immediate_after_handoff(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].mode = 'Policy'
        action, decision = self._choose(manager, state)
        self.assertEqual(decision['per_arm_control'], ['ReturnP0', 'Policy'])
        self.assertEqual(manager.arms[0].mode, 'ReturnP0')
        self.assertIsNone(manager.stop_reason)

    def test_s0_approach_stage_stall_chooses_initial_anchor(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].mode = 'Policy'
        manager._measure = lambda s, ai, domain: (15, 15, .03, True)
        manager._eligible_anchor = lambda s, c, ai, mode, anchor: True
        _, d = self._choose(manager, state)
        self.assertEqual(d['per_arm_control'], ['ReturnS0', 'Policy'])
        self.assertEqual(manager.arms[0].recovery_anchor, 0)
        self.assertEqual(manager.arms[0].recovery_goal, 57)

    def test_stalled_guide_before_p0_chooses_s0(self):
        manager, state = self._manager(can_return=True)
        manager._measure = lambda s, ai, domain: (15, 15, .03, True)
        manager._eligible_anchor = lambda s, c, ai, mode, anchor: True
        _, d = self._choose(manager, state)
        self.assertEqual(d['per_arm_control'], ['ReturnS0', 'Policy'])
        self.assertEqual(manager.arms[0].recovery_anchor, 0)
        self.assertTrue(any(e.get('event') == 'guide_stall_to_stage_recovery'
                            and e['kind'] == 'ReturnS0' for e in manager.events))

    def test_grasp_stage_stall_chooses_p0_anchor(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].mode = 'Policy'
        _, d = self._choose(manager, state)
        self.assertEqual(d['per_arm_control'], ['ReturnP0', 'Policy'])
        self.assertEqual(manager.arms[0].recovery_anchor, 30)
        self.assertEqual(manager.arms[0].recovery_goal, 57)

    def test_carry_stage_stall_chooses_p1_without_clearing_grasp(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].mode = 'Policy'
        state['actual_grasps']['A']['piece_1'] = True
        manager.grasp_streak[0] = 10
        for _ in range(10):
            manager.arms[0].relative.append((np.zeros(3), np.eye(3)))
        manager._eligible_anchor = lambda s, c, ai, mode, anchor: True
        _, d = self._choose(manager, state)
        self.assertEqual(d['per_arm_control'], ['ReturnP1', 'Policy'])
        self.assertEqual(manager.arms[0].recovery_anchor, 57)
        self.assertEqual(len(manager.arms[0].relative), 10)

    def test_non_target_contact_forbids_p1_return(self):
        manager, state = self._manager(can_return=True)
        state['actual_grasps']['A']['piece_1'] = True
        manager.grasp_streak[0] = 10
        for _ in range(10):
            manager.arms[0].relative.append((np.zeros(3), np.eye(3)))
        contact = {'edges': [{'a':'A','b':'base'}],
                   'dynamic_A_B_coupled': False}
        self.assertFalse(manager._eligible_anchor(
            state, contact, 0, 'ReturnP1', 57))

    def test_unstable_carry_not_rewound_as_empty_gripper(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].mode = 'Policy'
        state['actual_grasps']['A']['piece_1'] = True
        manager.grasp_streak[0] = 1
        self.assertIsNone(manager._phase_anchor(state, {'edges': [], 'dynamic_A_B_coupled': False}, 0, 35))
        _, d = self._choose(manager, state)
        self.assertEqual(d['per_arm_control'], ['Policy', 'Policy'])

    def test_assembly_contact_no_unqualified_rewind(self):
        manager, state = self._manager(can_return=True)
        manager.arms[0].mode = 'Policy'
        contact = {'edges': [{'a':'piece_1', 'b':'base'}], 'dynamic_A_B_coupled': False}
        self.assertIsNone(manager._phase_anchor(state, contact, 0, 35))

    def test_autoserl_convergence_exit_requires_real_displacement(self):
        manager, state = self._manager(can_return=False)
        manager.arms[0].guide_origin = np.array([-.006, 0, 0])
        for _ in range(5):
            manager._issued = ('Guide', 'Policy')
            manager._issued_targets = [40, None]
            observed = self._after(manager, state, .01, .01)
        self.assertEqual(manager.arms[0].mode, 'Handoff')
        self.assertEqual(observed['control_events'][-1]['event'], 'guide_exit_to_handoff')
        self.assertEqual(observed['control_events'][-1]['reason'],
                         'converged_after_progress')
        self.assertFalse(observed['handback'])
        self.assertIsNone(manager.stop_reason)

    def test_autoserl_static_pose_does_not_claim_convergence_exit(self):
        manager, state = self._manager(can_return=False)
        manager.arms[0].guide_origin = np.zeros(3)
        for _ in range(8):
            manager._issued = ('Guide', 'Policy')
            manager._issued_targets = [40, None]
            self._after(manager, state, .01, .01)
        self.assertEqual(manager.arms[0].mode, 'Guide')
        self.assertEqual(manager.handbacks, [0, 0])

    def test_stage_replay_end_is_true_p1(self):
        manager, state = self._manager(can_return=True)
        arm = manager.arms[0]
        arm.mode = 'Replay'
        arm.replay = 0
        arm.recovery_anchor = 0
        arm.recovery_goal = 57
        initial_window_start = arm.start
        for frame in range(57):
            manager._issued = ('Replay', 'Policy')
            manager._issued_targets = [arm.replay+1, None]
            self._after(manager, state)
        self.assertEqual(arm.replay, 57)
        self.assertEqual(arm.mode, 'Settle')
        self.assertEqual(arm.start, initial_window_start,
                         'expert replay must never advance Actor sliding window')
        self.assertIsNone(manager.stop_reason)



    def test_s0_anchor_alignment_starts_demo_from_zero(self):
        manager, state = self._manager(can_return=True)
        arm = manager.arms[0]
        arm.mode = 'ReturnS0'
        arm.recovery_anchor = 0
        arm.recovery_goal = 57
        arm.start = 11
        for _ in range(3):
            manager._issued = ('ReturnS0', 'Policy')
            manager._issued_targets = [0, None]
            self._after(manager, state, position_error=.0005,
                        rotation_error=.01)
        self.assertEqual(arm.mode, 'Replay')
        self.assertEqual(arm.replay, 0)
        self.assertEqual(arm.start, 11)
        self.assertIsNone(manager.stop_reason)

    def test_p1_alignment_skips_replay_and_requires_settle(self):
        manager, state = self._manager(can_return=True)
        arm = manager.arms[0]
        arm.mode = 'ReturnP1'
        arm.recovery_anchor = 57
        arm.recovery_goal = 57
        state['actual_grasps']['A']['piece_1'] = True
        state['objects'] = {'piece_1': {
            'position': np.zeros(3),
            'quaternion_wxyz': np.array([1., 0., 0., 0.])}}
        for _ in range(3):
            manager._issued = ('ReturnP1', 'Policy')
            manager._issued_targets = [57, None]
            self._after(manager, state, position_error=.003,
                        rotation_error=.01)
        self.assertEqual(arm.mode, 'Settle')
        self.assertEqual(arm.recovery_anchor, 57)
        self.assertIsNone(arm.replay)
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

    def test_stage_recovery_does_not_certify_superseded_guide(self):
        for i in range(3):
            self.bank.observe(row(i, self.mask_a, segment=11),
                decision=decision('Guide', 'Policy', 11, 0))
        self.bank.observe(row(3, self.mask_a, segment=12),
            decision=decision('ReturnP0', 'Policy', 12, 0),
            control_events=[dict(event='guide_stall_to_stage_recovery',
                arm='A', kind='ReturnP0', from_takeover=11,
                to_takeover=12)])
        self.bank.observe(row(4), decision=decision(),
            handbacks=[dict(arm='A', takeover_id=12)])
        self.bank.close_episode(0)
        self.assertEqual(self.bank.stats['failed_guide_discarded_rows'], 3)
        self.assertEqual(self.bank.stats['accepted_rows'], 1)
        self.assertEqual(len(self.bank.buffer), 1)
        self.assertEqual(int(self.bank.buffer[0]['frame']), 3)

    def test_pending_tail_is_bounded_per_worker(self):
        for i in range(40):
            self.bank.observe(row(i, self.mask_a), decision=decision('Guide','Policy',14,0))
        self.assertEqual(len(self.bank.pending[0]), 32)
        self.assertEqual(self.bank.stats['expired_candidate_rows'], 8)
        self.bank.close_episode(0)
        self.assertEqual(len(self.bank.buffer), 0)


if __name__ == '__main__':
    unittest.main()
