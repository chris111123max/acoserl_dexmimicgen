"""Reviewed execution contract around the retained AutoSERL baseline.

Task guidance and hard execution safety have separate ownership. Ordinary Guide
owns six pose DOFs on one arm; Recovery may own both arms including static Hold.
A handoff requires one real Actor control, not a target-reached bookkeeping event.
"""
from __future__ import annotations
import copy
import hashlib
import time
from pathlib import Path
import numpy as np
from acoserl.osc_prediction import predict, project_independent_workspace

ROOT = Path(__file__).resolve().parents[2]


class ReviewedRuntimeMixin:
    def reset(self, initial_progress=(0, 0)):
        super().reset(initial_progress)
        self._risk_cache = None
        self._runtime_before = None
        self._handoff_owner = None
        self._forward_grasp = None
        self.runtime_stats = dict(control_proposals=0, safety_projections=0,
                                  final_refusals=0, choose_seconds=0.0,
                                  risk_cache_hits=0, handbacks_A=0, handbacks_B=0,
                                  forward_grasp_starts=0, forward_grasp_completions=0)
        self._workspace_bounds = None

    def v3_fingerprint(self):
        # Production sources are immutable during a run. The retained V3 code
        # already caches its hashes; do not reread extension files per action.
        if not hasattr(self, '_reviewed_fingerprint'):
            result = super().v3_fingerprint()
            for name in ('src/acoserl/reviewed_runtime.py', 'src/acoserl/bimanual_core.py',
                         'src/acoserl/substep_guard.py', 'src/acoserl/qualification.py',
                         'scripts/assembly_feasibility/common.py', 'src/acoserl/grasp_guidance.py'):
                result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            self._reviewed_fingerprint = result
        return copy.deepcopy(self._reviewed_fingerprint)

    def advanced_qualification(self, kind, state=None, contact=None):
        if not self.manager_cfg.get(kind, {}).get('enabled', False):
            return dict(allowed=False, kind=kind, reason=kind+'_disabled',
                        controller_available=True, proof={})
        return super().advanced_qualification(kind, state, contact)

    def predicted_risk(self, ai, physical_action, state):
        if self._workspace_bounds is None:
            self._workspace_bounds = (self.positions.min(axis=0)-self.cfg['free_workspace_margin_m'],
                                      self.positions.max(axis=0)+self.cfg['free_workspace_margin_m'])
        lower, upper = self._workspace_bounds[0][ai], self._workspace_bounds[1][ai]
        current = np.asarray(state['poses'][ai]['position'])
        estimate = predict(self.env, ai, physical_action)
        raw = estimate['command_displacement']
        path_low = current+estimate['trajectory_min']
        path_high = current+estimate['trajectory_max']
        margin = float(np.minimum(current-lower, upper-current).min())
        # Physical bounds are unchanged. The same 3mm prospective reserve is
        # used by the projector; no artificial instantaneous stop is assumed.
        lo = np.minimum(lower+.003, current)
        hi = np.maximum(upper-.003, current)
        crossing = bool(np.any(current+estimate['brake_low'] < lo-1e-8) or
                        np.any(current+estimate['brake_high'] > hi+1e-8) or
                        np.any(path_low < lo-1e-8) or np.any(path_high > hi+1e-8)
                        or np.any(current+raw < lo-1e-8) or np.any(current+raw > hi+1e-8))
        outward = bool(np.any((current-lower < self.manager_cfg['risk_margin_m']) & (raw < -1e-8))
                       or np.any((upper-current < self.manager_cfg['risk_margin_m']) & (raw > 1e-8)))
        return dict(arm='AB'[ai], risk=crossing or outward, margin_m=margin,
                    displacement=estimate['displacement'].tolist(),
                    command_displacement=raw.tolist(),
                    predicted_position=(current+estimate['displacement']).tolist(),
                    trajectory_min=path_low.tolist(), trajectory_max=path_high.tolist(),
                    actual_velocity=estimate['linear_velocity'].tolist(),
                    nominal_terminal_velocity=estimate['terminal_velocity'].tolist(),
                    nominal_brake_low=(current+estimate['brake_low']).tolist(),
                    nominal_brake_high=(current+estimate['brake_high']).tolist(),
                    lower=lower.tolist(), upper=upper.tolist(),
                    estimate=estimate['estimate'])

    def action_risks(self, action, state):
        # Exact repeated proposals within ONE choose call are common through
        # V3/V2/Guide. Cache geometry queries without caching across physics.
        cache = getattr(self, '_risk_cache', None)
        key = np.asarray(action, dtype=np.float64).tobytes()
        if cache is not None and key in cache:
            self.runtime_stats['risk_cache_hits'] += 1
            return copy.deepcopy(cache[key])
        result = super().action_risks(action, state)
        if cache is not None and len(cache) < 16:
            cache[key] = copy.deepcopy(result)
        return result

    def _project_independent_risks(self, policy, state, risks, indices):
        corrected = np.asarray(policy, dtype=float).copy()
        selected = tuple(sorted(set(indices)))
        assessment = self.contact_assessment(state)
        for ai in selected:
            risk = risks[ai]
            categories = set(risk['categories'])
            if categories == {'unplanned_grasp_release'}:
                corrected[7*ai+6] = 1.0
                continue
            reasons = []
            if assessment['category'] != 'Independent':
                reasons.append('contact_mode='+str(assessment['category']))
            held = [arm for arm in ('A', 'B') if any(state['actual_grasps'][arm].values())]
            if held: reasons.append('existing_grasps='+','.join(held))
            if not categories or not categories <= {'workspace', 'open_gripper_unplanned_approach'}:
                reasons.append('unsupported_categories='+','.join(sorted(categories)))
            if risk.get('gripper_contact', {}).get('command_reversal'):
                reasons.append('near_contact_gripper_reversal')
            if reasons:
                return None, 'independent_risk_projection_not_qualified:'+'AB'[ai]+':'+'|'.join(reasons)
            normals = []
            # A second nearest obstacle may become limiting after projection.
            # Re-evaluate the SAME arm and add its constraints, rather than
            # considering an unused guided-arm Actor action to be executable.
            for _ in range(4):
                normal = risk.get('gripper_contact', {}).get('normal')
                if normal is not None: normals.append(normal)
                corrected, reason = project_independent_workspace(
                    self.env, ai, corrected, state['poses'][ai]['position'],
                    self.positions[:, ai].min(0)-self.cfg['free_workspace_margin_m'],
                    self.positions[:, ai].max(0)+self.cfg['free_workspace_margin_m'],
                    self.manager_cfg['risk_margin_m'], getattr(self, 'action_bounds', None),
                    normals or None)
                if corrected is None: return None, 'AB'[ai]+':'+reason
                risk = self.action_risks(corrected, state)[ai]
                if not risk['risk']: break
                if set(risk['categories'])-{'workspace', 'open_gripper_unplanned_approach'}:
                    break
            if risk['risk']:
                return None, 'independent_projection_residual_risk:'+'AB'[ai]+':'+','.join(risk['categories'])
        return corrected, 'independent_full_interval_projection'

    def _final_screen(self, action, decision, policy, state):
        if action is None: return action, decision
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            self.stop('nonfinite_final_action'); return self.refuse()
        original = action.copy()
        if hasattr(self, 'action_bounds'):
            action = np.clip(action, *self.action_bounds)
        risks = self.action_risks(action, state)
        selected = [ai for ai, r in enumerate(risks) if r['risk']]
        if selected:
            # Strict evaluation never gets silent safety-projected assistance.
            if not self.cfg['intervention_enabled']:
                self.last_decision = decision
                decision['final_action_risks'] = risks
                self.stop('evaluation_action_risk_rejected'); return self.refuse()
            corrected, reason = self._project_independent_risks(action, state, risks, selected)
            if corrected is None:
                self.last_decision = decision
                decision['final_action_risks'] = risks
                self.stop('no_legal_projected_action:'+reason); return self.refuse()
            action = corrected
        remaining = self.action_risks(action, state)
        if any(r['risk'] for r in remaining):
            self.last_decision = decision; decision['remaining_risks'] = remaining
            self.stop('final_composed_action_residual_risk'); return self.refuse()
        if self.contact_assessment(state)['category'] == 'Independent':
            allowed, reason = self._independent_tcp_swept_clearance(action, state)
            if not allowed:
                self.last_decision = decision; self.stop(reason); return self.refuse()
        changed = [ai for ai in range(2) if np.max(abs(action[7*ai:7*ai+7]-original[7*ai:7*ai+7])) > 1e-7]
        if changed:
            source = decision['mode']
            prior=set(getattr(self, 'projected_arms', [])) if self.projected else set()
            self.projected = True
            self.projected_arms = sorted(prior | set(changed))
            guided = source.startswith('Guide_')
            decision['source_before_final_projection'] = source
            if not guided or self.active in changed:
                decision['mode'] = 'Constrained_'+('AB'[changed[0]] if len(changed)==1 else 'Joint')
            decision.update(action_projection_applied=True, final_projection_arms=changed,
                            final_projection_reason='full_interval_final_action_screen')
            self.runtime_stats['safety_projections'] += 1
        decision['executed_action'] = action.tolist()
        decision['actor_action_executed'] = action.tolist()
        decision['policy_action_accepted_unmodified'] = bool(np.array_equal(action, policy))
        decision['final_action_checked'] = True
        decision['final_action_risks'] = remaining
        self.last_decision = decision
        return action, decision

    def choose(self, policy_action):
        from acoserl.bimanual_core import snapshot
        start = time.perf_counter()
        trial=getattr(self,'_static_hold_trial',None)
        if trial is not None:
            trial.proposals+=1
            if trial.proposals>self.cfg['max_recovery_steps']+40:
                self.last_decision={'policy_action':np.asarray(policy_action).tolist()}
                self.stop('bounded_static_hold_trial_budget');return self.refuse()
        policy = np.asarray(policy_action, dtype=float)
        if policy.shape != (14,) or not np.isfinite(policy).all():
            raise ValueError('expected finite 14D Policy action')
        self._maybe_start_forward_grasp(policy)
        self._runtime_before = dict(intervention_steps=self.intervention_control_steps,
                                    touched=set(self.episode_touched),
                                    policy_steps=self.episode_policy_steps.copy())
        self._risk_cache = {}
        self.projected=False;self.projected_arms=[]
        self.runtime_stats['control_proposals'] += 1
        try:
            if not self.cfg['intervention_enabled']:
                # Evaluation/preparation must not inherit V3's automatic grip
                # retention or any hidden projected-Actor assistance.
                state=snapshot(self.env);contact=self.contacts.read()
                self.steps+=1;self._quality_context=None
                decision=dict(mode='Normal',execute=True,policy_action=policy.tolist(),
                              executed_action=policy.tolist(),indices=self.indices(),
                              collaboration=self.collaboration(state,contact),takeover_id=0)
                self.last_decision=decision
                danger=self.physical_danger(state,contact)
                if danger:
                    self.stop(danger);action,decision=self.refuse()
                else:
                    action=policy.copy()
            else:
                action, decision = super().choose(policy)
            action, decision = self._final_screen(action, decision, policy, snapshot(self.env))
            if action is None: self.runtime_stats['final_refusals'] += 1
            return action, decision
        finally:
            self._risk_cache = None
            self.runtime_stats['choose_seconds'] += time.perf_counter()-start

    def handback(self, result, index):
        if self._forward_grasp is not None:
            if not self._forward_grasp.ready:
                result.update(handback=False, grasp_guide_wait=self._forward_grasp.phase)
                return
            index = self._forward_grasp.end
        owner = dict(arm=self.active, takeover_id=self.takeover_id, source=self.mode, kind=self.recovery_kind)
        super().handback(result, index)
        if self.mode != 'Safe_Stop' and self.handoff_phase in ('Handoff_Ready', 'Recovery_Stabilizing'):
            self._handoff_owner = owner
            if self._forward_grasp is not None:
                self.runtime_stats['forward_grasp_completions'] += 1
                self._forward_grasp = None

    def after_step(self):
        from acoserl.action_provenance import describe
        phase = self.handoff_phase
        decision = self.last_decision
        local_before=self.local
        grasp_result = None
        if self._forward_grasp is not None:
            from acoserl.bimanual_core import snapshot
            grasp_result = self._forward_grasp.observe(self, snapshot(self.env))
            if self._forward_grasp.controls >= self._forward_grasp.max_controls and not self._forward_grasp.ready:
                self.stop('forward_grasp_guide_timeout')
        result = super().after_step()
        if grasp_result is not None:
            result['forward_grasp'] = grasp_result
        if local_before is not None and result.get('target_reached') and self.mode!='Safe_Stop':
            self._handoff_owner=dict(arm=self.active,takeover_id=self.takeover_id,source=decision.get('mode'))
        source = decision.get('mode', 'Normal')
        # Count physical intervention controls ONCE even if a Guide contains
        # a safety correction on the other arm (legacy mixins count both).
        if self._runtime_before is not None:
            controlled = source.startswith(('Guide_', 'Recover_', 'Constrained_'))
            self.intervention_control_steps = self._runtime_before['intervention_steps']+int(controlled)
            # Safety changes on an otherwise Policy-controlled arm are not
            # autonomous phase steps, even when the other arm is in Guide.
            stages=decision.get('collaboration',{}).get('per_arm',['unknown','unknown'])
            for ai in getattr(self,'projected_arms',[]) if self.projected else []:
                key=('AB'[ai],stages[ai])
                self.episode_touched.add(key)
                self.episode_policy_steps[key]=self._runtime_before['policy_steps'].get(key,0)
        if result.get('safe_handback') and phase in ('Handoff_Ready', 'Policy_Guarded'):
            owner = self._handoff_owner
            if owner is None:
                self.stop('handoff_missing_takeover_owner')
                result.update(handback=False, safe_handback=False)
            else:
                arm = 'AB'[owner['arm']]
                self.runtime_stats['handbacks_'+arm] += 1
                result.update(handback_arm=arm, handback_takeover_id=owner['takeover_id'], handback_kind=owner.get('kind'))
                decision['takeover_id'] = owner['takeover_id']
                result.setdefault('task_quality', {})['handoff_confirmed'] = True
                # One completed, safety-checked Actor action proves handback.
                # Do not suppress ALL guidance for another 20-40 random steps.
                self.handoff_phase = None; self.handoff_local = None; self.mode = 'Normal'
                self._handoff_owner = None
        if self.stop_reason is not None:
            result.update(handback=False, safe_handback=False)
        result['runtime_stats'] = self.runtime_stats.copy()
        return result

    def substep_safety(self, contact):
        from acoserl.bimanual_core import snapshot
        state = snapshot(self.env)
        # Uniform coverage: Policy, Guide, Recovery, projected Actor and
        # handoff all receive the SAME real workspace/force/contact checks.
        danger = self.physical_danger(state, contact)
        if danger: return danger
        return super().substep_safety(contact)

    def _recovery_permitted(self):
        # A bounded, external real-physics qualification trial is explicitly
        # separate from a production learner. It never authorizes the learner.
        return (bool(self.manager_cfg.get('qualified_recovery_enabled', False)) or
                getattr(self, '_static_hold_trial', None) is not None)

    def start_recovery(self, ai, state, kind, target):
        if kind != 'sliding' and not self._recovery_permitted():
            self.stop('recovery_disabled_no_current_qualification')
            return None
        return super().start_recovery(ai, state, kind, target)

    def start_local(self, *args, **kwargs):
        if not self._recovery_permitted():
            return False, 'recovery_disabled_no_current_qualification'
        return super().start_local(*args, **kwargs)

    def passive_hold_entry(self, active, state):
        from acoserl.qualification import external_hold_entry
        if not self._recovery_permitted():
            return {'allowed': False, 'reason': 'recovery_disabled_no_current_qualification'}
        entry = external_hold_entry(self, active, state)
        # Production Recovery requires the SOURCE-BOUND external certificate.
        # It must not fall through to the old unverified envelope JSON.
        if entry is None:
            return {'allowed': False, 'reason': 'external_hold_certificate_required'}
        return entry

    def _maybe_start_forward_grasp(self, policy):
        """Local forward acquisition, never a return-to-P0 or passive Hold."""
        if (not self.cfg['intervention_enabled'] or self.auto_disabled or
                not self.manager_cfg.get('grasp_guidance_enabled', False) or
                not self.manager_cfg.get('sliding_enabled', False) or
                self.handoff_phase is not None and self.handoff_phase != 'Recovering' or
                self.local is not None or self._forward_grasp is not None):
            return
        adopting = self.mode.startswith('Guide_') and self.recovery_kind == 'sliding'
        if self.mode != 'Normal' and not adopting:
            return
        if self.points.get('task') != 'TwoArmThreePieceAssembly':
            return
        from acoserl.bimanual_core import snapshot
        from acoserl.grasp_guidance import grasp_candidate, ForwardGraspGuide
        state = snapshot(self.env)
        candidates = [self.active] if adopting else list(range(2))
        for ai in candidates:
            if not adopting and self.cooldown[ai] > 0:
                continue
            if self.stage_mastered('AB'[ai], 'grasp_stabilization'):
                continue
            index, reason = grasp_candidate(self, state, ai)
            if index is None:
                continue
            # Reaching pregrasp is not itself a reason to override a healthy
            # Actor. Continue an already-needed Guide, or require a measured
            # stagnation / missing closure at the demonstrated closing event.
            points = self.points['arms']['AB'[ai]]
            missing_closure = index >= points['first_close_action'] and policy[7*ai+6] <= 0
            stalled = False if adopting else self.window_measure(ai, state, record_history=False)['stagnant']
            if not adopting and not (missing_closure or stalled):
                continue
            trigger = ('existing_guide_at_pregrasp' if adopting else
                       'missing_grasp_closure' if missing_closure else 'pregrasp_stagnation')
            # Do not silently extend a nearly exhausted existing Guide.
            needed = self.points['arms']['AB'[ai]]['p1'] - index + self.cfg['stable_grasp_window_steps'] + 6
            if adopting and self.age + needed >= self.cfg.get('max_guide_steps', 80):
                continue
            if not adopting:
                self.active = ai
                self.baseline = copy.deepcopy(state)
                self.age = 0
                self.takeover_sequence += 1
                self.takeover_id = self.takeover_sequence
                self.episode_interventions += 1
                self.decision_stats['starts'] += 1
            self.mode = 'Guide_' + 'AB'[ai]
            self.recovery_stage = 'Sliding_Guide'
            self.recovery_kind = 'forward_grasp'
            self.recovery_index = index
            self.replay_index = None
            self.stable = 0
            self.targets = copy.deepcopy(state['poses'])
            self.grips = [-1., -1.]
            self.hold_age = 0
            self.hold_limit = None
            self._forward_grasp = ForwardGraspGuide(self, ai, index)
            stages = self.collaboration(state, self.contacts.read())['per_arm']
            self.episode_touched.add(('AB'[ai], stages[ai]))
            self.runtime_stats['forward_grasp_starts'] += 1
            self.events.append(dict(event='forward_grasp_guide', step=self.steps,
                                    arm='AB'[ai], start_index=index,
                                    end_index=self._forward_grasp.end,
                                    adopts_current_guide=adopting, reason=reason, trigger=trigger))
            break

    def recovery_action(self, state):
        if self._forward_grasp is None:
            return super().recovery_action(state)
        if self.contact_assessment(state, self.contacts.read())['category'] != 'Independent':
            self.stop('forward_grasp_contact_mode_changed')
            return self.refuse()
        policy = np.asarray(self.last_decision['policy_action'], dtype=float)
        action = self._forward_grasp.action(self, state, policy)
        ai = self.active
        self.last_decision.update(mode=self.mode, active_arm='AB'[ai], execute=True,
            recovery_step=self.age+1, recovery_kind=self.recovery_kind,
            recovery_stage=self.recovery_stage, guide_phase=self._forward_grasp.phase,
            guide_gripper_owned=True, reference_index=self._forward_grasp.index,
            indices=self.indices(), executed_action=action.tolist(),
            per_arm_control=['Guide' if i == ai else 'Policy' for i in range(2)])
        # V3 progress protection and ReviewedRuntime._final_screen still run
        # on this COMPOSED action. No force, workspace or contact check removed.
        return action, self.last_decision

    def _task_quality(self):
        report = super()._task_quality()
        context = getattr(self, '_quality_context', None)
        source = (self.last_decision or {}).get('mode', '')
        legal = bool(context and context['legal'] and self.stop_reason is None and
                     source.startswith(('Guide_', 'Recover_')))
        before = float(report.get('error_before_m', float('nan')))
        after = float(report.get('error_after_m', float('nan')))
        report['eligible'] = bool(legal and np.isfinite([before, after]).all() and min(before, after) >= 0)
        report['reason'] = ('complete_legal_task_intervention' if report['eligible']
                            else 'not_a_qualified_task_intervention')
        report['measured_error_reduction_m'] = before-after
        return report
