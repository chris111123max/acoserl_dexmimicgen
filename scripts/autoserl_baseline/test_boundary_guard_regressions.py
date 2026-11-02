"""Focused boundary-risk, mixed-arm provenance and Demo exclusion regressions.

The numerical motion tests here are deterministic contracts, NOT evidence
that an inward OSC correction will overcome real MuJoCo inertia. Codex must
run the bounded real-physics targeted test before formal training.
"""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from acoserl.boundary_guard import evaluate, world_policy_displacement
from acoserl.simple_autoserl import SimpleBimanualAutoSERL
from acoserl.simple_replay import SimpleDemo, make_row
from test_simple_progress_regressions import GuideProgressTests, row, decision


class RiskGeometryTests(unittest.TestCase):
    def test_upper_z_outward_policy_triggers_inward_target(self):
        risk = evaluate([0, 0, 1.13], [-1,-1,0], [1,1,1.14],
                        [0,0,.004], [0,0,.002])
        self.assertEqual([(r['axis'],r['side']) for r in risk['axes']], [('z','upper')])
        self.assertAlmostEqual(risk['target_world_position'][2], 1.105)

    def test_upper_z_outward_velocity_triggers_even_if_policy_inward(self):
        risk = evaluate([0,0,1.13], [-1,-1,0], [1,1,1.14],
                        [0,0,-.004], [0,0,.013])
        self.assertEqual(risk['axes'][0]['axis'], 'z')

    def test_inward_policy_and_velocity_do_not_intervene(self):
        self.assertIsNone(evaluate([0,0,1.13], [-1,-1,0], [1,1,1.14],
                                   [0,0,-.005], [0,0,-.001]))

    def test_far_from_boundary_does_not_intervene(self):
        self.assertIsNone(evaluate([0,0,1.05], [-1,-1,0], [1,1,1.14],
                                   [0,0,.015], [0,0,.2]))

    def test_invalid_world_frame_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([0,0,1.15], [-1,-1,0], [1,1,1.14], [0,0,.01], [0,0,.0])

    def test_controller_conversion_uses_same_base_to_world_rule_as_servo(self):
        ctl=types.SimpleNamespace(input_type='delta', input_ref_frame='base',
                _goal_update_mode='achieved', origin_ori=np.eye(3),
                output_max=np.array([.03,.03,.03,.05,.05,.05]))
        robot=types.SimpleNamespace(part_controllers={'right':ctl})
        self.assertTrue(np.allclose(world_policy_displacement(
            robot, np.array([0,0,.5,0,0,0,-1.])), [0,0,.015]))


class BoundaryOwnershipTests(unittest.TestCase):
    def fixture(self, b_guide=True):
        base=GuideProgressTests()
        m, state=base._manager(can_return=True)
        m.cfg['boundary_guard_enabled']=True
        m.cfg['free_workspace_margin_m']=.12
        m.cfg['boundary_guard_entry_m']=.025
        m.cfg['boundary_guard_retreat_m']=.035
        m.cfg['boundary_guard_max_consecutive']=16
        m.arms[0].mode='Policy'
        m.arms[0].cooldown=17
        m.arms[0].start=4
        m.boundary_guard_steps=0
        m._domain=lambda s,c,ai:(0,51)
        m._measure=lambda s,ai,d:(35,40,.03,False)
        if b_guide:
            m.arms[1].mode='Guide'
            m.arms[1].target=40
            m.arms[1].takeover=9
        else:
            m.arms[1].mode='Policy'
            m.arms[1].cooldown=17
        def candidate(ai,s,physical):
            if ai==0:
                return dict(
                    target_world_position=np.array([0,0,-.035]),
                    axes=[dict(axis='z',side='upper',distance_m=.005,
                               policy_requested_delta_m=.004,
                               measured_velocity_m_s=.01)],
                    requested_world_delta=np.array([0,0,.004]),
                    measured_world_velocity=np.array([0,0,.01]),
                    entering_distance_m=.025,retreat_distance_m=.035)
            return None
        m._boundary_safety_candidate=candidate
        fake=types.ModuleType('acoserl.bimanual_core')
        fake.snapshot=lambda env:state
        fake.servo=lambda env,targets,grips,**kwargs:np.array(
            [0,0,-.2,0,0,0,0, 0,0,-.1,0,0,0,0], dtype=float)
        fake.errors=lambda a,b:dict(position_m=.03,rotation_rad=.03)
        return m,state,fake

    def test_a_boundary_guard_does_not_steal_b_guide_or_grippers(self):
        m,state,fake=self.fixture(b_guide=True)
        request=np.zeros(14);request[6]=.4;request[13]=-.3
        with patch.dict(sys.modules,{'acoserl.bimanual_core':fake}):
            actual,decision=m.choose(request)
        self.assertEqual(decision['per_arm_control'],['BoundaryGuard','Guide'])
        self.assertEqual(decision['mode'],'Guide_AB')
        self.assertAlmostEqual(actual[2],-.2)
        self.assertAlmostEqual(actual[6],.4)
        self.assertAlmostEqual(actual[13],-.3)
        self.assertEqual(m.arms[0].mode,'Policy')
        self.assertEqual(m.arms[0].cooldown,17)
        self.assertFalse(any(e.get('event')=='policy_handback_confirmed'
                             for e in m.events))

    def test_successful_safety_step_does_not_spend_policy_cooldown_or_window(self):
        m,state,fake=self.fixture(b_guide=True)
        with patch.dict(sys.modules,{'acoserl.bimanual_core':fake}):
            _, d=m.choose(np.zeros(14))
            result=m.after_step()
        self.assertEqual(m.arms[0].start,4)
        self.assertEqual(m.arms[0].cooldown,17)
        self.assertEqual(m.arms[0].boundary_guard_total,1)
        self.assertEqual(m.boundary_guard_steps,1)
        self.assertEqual(m.intervention_control_steps,1)
        m.cfg['termination_intervention_steps']=0
        outcome=m.episode_result(True)
        self.assertTrue(outcome['assisted_success'])
        self.assertFalse(outcome['autonomous_success'])
        self.assertFalse(result['task_quality']['eligible'])
        self.assertFalse(result['handback'])
        self.assertEqual(result['boundary_guard_streak'][0],1)

    def test_safety_only_has_non_normal_mode_for_bridge(self):
        m,state,fake=self.fixture(b_guide=False)
        with patch.dict(sys.modules,{'acoserl.bimanual_core':fake}):
            actual,d=m.choose(np.zeros(14))
        self.assertEqual(d['mode'],'BoundarySafety')
        self.assertEqual(d['per_arm_control'],['BoundaryGuard','Policy'])
        self.assertNotEqual(float(actual[2]),0.)

    def test_out_of_action_domain_refuses_instead_of_clipping(self):
        m,state,fake=self.fixture(b_guide=False)
        lo=-np.ones(14);hi=np.ones(14);lo[2]=-.1
        m.action_bounds=(lo,hi)
        with patch.dict(sys.modules,{'acoserl.bimanual_core':fake}):
            actual,d=m.choose(np.zeros(14))
        self.assertIsNone(actual)
        self.assertEqual(d['mode'],'Safe_Stop')
        self.assertIn('boundary_guard_outside_action_domain_A',d['reason'])

    def test_active_contact_refuses_boundary_override_before_physics(self):
        m,state,fake=self.fixture(b_guide=False)
        m.contacts=types.SimpleNamespace(read=lambda:dict(
            dynamic_A_B_coupled=False,
            edges=[dict(a='A',b='piece_1')]))
        with patch.dict(sys.modules,{'acoserl.bimanual_core':fake}):
            actual,d=m.choose(np.zeros(14))
        self.assertIsNone(actual)
        self.assertIn('boundary_guard_unqualified_contact_A',d['reason'])
        self.assertEqual(d['mode'],'Safe_Stop')
        self.assertIsNone(m._issued)

    def test_real_row_preserves_nonpolicy_safety_pose_without_expert_credit(self):
        payload=dict(env_id=0,episode_id=1,frame=3,executed=True,
                     supervisor_stop_reason=None,
                     auto_intervention=decision('BoundaryGuard','Guide',0,7))
        fake_row=row(3,mask=list(range(7,13)))
        with patch('acoserl.simple_replay.raw_row',return_value=fake_row):
            out=make_row({},torch.zeros(14),{},0.,False,False,payload)
        self.assertEqual(int(out['action_source']),9)
        self.assertEqual(int(out['pose_sources'][0]),4)
        self.assertEqual(int(out['pose_sources'][1]),1)
        self.assertFalse(bool(out['quality_validated']))
        self.assertFalse(bool(out['expert_action_mask'].any()))
        self.assertFalse(bool(out['policy_dimension_mask'][2]))
        self.assertTrue(bool(out['policy_dimension_mask'][6]))
        self.assertFalse(bool(out['policy_dimension_mask'][9]))
        self.assertTrue(bool(out['policy_dimension_mask'][13]))

    def test_boundary_safety_drops_unverified_expert_tail(self):
        bank=SimpleDemo(capacity=64,gamma=.99,seed=7)
        bank.observe(row(0,mask=list(range(7,13)),segment=7),
                     decision=decision('Policy','Guide',0,7))
        self.assertEqual(len(bank.pending[0]),1)
        bank.observe(row(1),decision=decision('BoundaryGuard','Guide',0,7))
        self.assertEqual(bank.stats['boundary_guard_rows'],1)
        self.assertEqual(bank.stats['boundary_guard_discarded_candidate_rows'],1)
        self.assertNotIn(0,bank.pending)
        bank.observe(row(2),decision=decision(),
                     handbacks=[dict(arm='B',takeover_id=7)])
        bank.close_episode(0)
        self.assertEqual(len(bank.buffer),0)


if __name__=='__main__':
    unittest.main()
