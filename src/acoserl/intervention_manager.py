"""Bimanual control ownership, qualified pregrasp guidance and explicit refusals.

Carry-local and joint contact recovery are proposals only; this revision has no
controller or physical qualification for either and cannot execute them.
"""
import copy
from collections import deque
import json
from pathlib import Path
import numpy as np
from acoserl.transition_semantics import STAGES
from acoserl.two_point_supervisor import ARMS, snapshot, errors

ROOT = Path(__file__).resolve().parents[2]
MANAGER_DEFAULT = json.loads((ROOT / "configs/bimanual_autoserl/manager.json").read_text())

METHODS = {
    "plain_rlpd": dict(sliding_enabled=False, safety_recovery_enabled=False,
                       predict_workspace_risk=False, dynamic_demo_enabled=False),
    "independent_window": dict(sliding_enabled=True, safety_recovery_enabled=False,
                              predict_workspace_risk=False, dynamic_demo_enabled=False),
    "window_recovery": dict(sliding_enabled=True, safety_recovery_enabled=True,
                           predict_workspace_risk=False, dynamic_demo_enabled=False),
    "full_acoserl": dict(sliding_enabled=True, safety_recovery_enabled=True,
                        predict_workspace_risk=True, dynamic_demo_enabled=True),
}

def method_config(name):
    if name not in METHODS:
        raise ValueError(name)
    return {**copy.deepcopy(MANAGER_DEFAULT), **METHODS[name], "method": name}

class InterventionManagerMixin:
    def __init__(self, env, config=None, intervention_enabled=None):
        self.manager_cfg = copy.deepcopy(MANAGER_DEFAULT)
        if config and "manager" in config:
            self.manager_cfg.update(copy.deepcopy(config["manager"]))
        self.autonomy = {}
        self.clean_full_episodes = 0
        self.takeover_sequence = 0
        super().__init__(env, config, intervention_enabled)

    def reset(self, initial_progress=(0, 0)):
        super().reset(initial_progress)
        self.takeover_id = 0
        self.episode_touched = set()
        self.episode_completed = set()
        self.episode_phase_steps = {}
        self.episode_policy_steps = {}
        self.decision_stats = dict(candidates=0, starts=0, handbacks=0,
                                   refused={}, triggers={})
        self._previous_stages = ["approach_pregrasp"] * 2
        self.relative_grasp_history = [deque(maxlen=self.cfg["stable_grasp_window_steps"]) for _ in ARMS]

    def collaboration(self, state, contact):
        # Include real grasp relations in the graph even for low-force contacts.
        edges = {tuple(sorted((e["a"], e["b"]))) for e in contact["edges"]}
        for ai, arm in enumerate(ARMS):
            for obj, held in state["actual_grasps"][arm].items():
                if held:
                    edges.add(tuple(sorted((arm, obj))))
        raw = self.env.sim.data
        for i in range(raw.ncon):
            c = raw.contact[i]
            owners = tuple(sorted((self.contacts.owners[int(c.geom1)],
                                   self.contacts.owners[int(c.geom2)])))
            if c.dist <= 0 and owners[0] != owners[1] and "static" not in owners:
                edges.add(owners)
        graph = {}
        for a, b in edges:
            if "static" in (a, b):
                continue
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        reachable, todo = {"A"}, ["A"]
        while todo:
            for nxt in graph.get(todo.pop(), set()) - reachable:
                reachable.add(nxt)
                todo.append(nxt)
        coupled = "B" in reachable
        second = ("piece_1", "piece_2") in edges or ("base", "piece_2") in edges
        first = ("base", "piece_1") in edges
        stages = []
        for ai, arm in enumerate(ARMS):
            held = state["actual_grasps"][arm][self.points["arms"][arm]["object"]]
            if coupled:
                stage = "coupled"
            elif self.completed[ai]:
                stage = "release_completion"
            elif ai == 1 and second:
                stage = "second_assembly_contact"
            elif ai == 0 and first:
                stage = "first_assembly_contact"
            elif held:
                stage = "independent_carry" if self.grasp_streak[ai] >= 3 else "grasp_stabilization"
            else:
                stage = "approach_pregrasp"
            stages.append(stage)
        unfinished = [s for s in stages if s != "release_completion"]
        global_stage = "coupled" if coupled else max(unfinished, key=STAGES.index) if unfinished else "release_completion"
        return dict(stage=global_stage, per_arm=stages, coupled=coupled,
                    contact_edges=[list(x) for x in sorted(edges)])

    def predicted_risk(self, ai, physical_action, state):
        ctl = self.env.robots[ai].part_controllers["right"]
        assert ctl.input_type == "delta" and ctl.input_ref_frame == "base"
        assert np.allclose(ctl.output_max, -ctl.output_min)
        current = np.asarray(state["poses"][ai]["position"])
        from acoserl.osc_prediction import predict
        prediction=predict(self.env,ai,physical_action)
        displacement=prediction['displacement']
        lo = self.positions[:, ai].min(0) - self.cfg["free_workspace_margin_m"]
        hi = self.positions[:, ai].max(0) + self.cfg["free_workspace_margin_m"]
        margin = float(np.minimum(current - lo, hi - current).min())
        proposed = current + displacement
        # Nominal prediction is not a certificate: retain the raw setpoint as
        # an additional conservative screen, rather than interpreting it as motion.
        command=np.asarray(prediction['command_displacement'])
        crossing = bool(np.any(proposed < lo) or np.any(proposed > hi) or
                        np.any(current+command < lo) or np.any(current+command > hi))
        outward = bool(any(np.any((current-lo < self.manager_cfg["risk_margin_m"]) & (d < 0))
                       or np.any((hi-current < self.manager_cfg["risk_margin_m"]) & (d > 0)) for d in (displacement,command)))
        return dict(arm=ARMS[ai], risk=bool(crossing or outward), margin_m=margin,
                    predicted_position=proposed.tolist(), displacement=displacement.tolist(),
                    command_displacement=command.tolist(),estimate=prediction["estimate"])

    def advanced_qualification(self, kind, state=None, contact=None):
        # No implementation in this revision. Config alone never grants authority.
        if kind not in ("carry_local", "joint"):
            raise ValueError(kind)
        return dict(allowed=False, kind=kind, reason=kind + "_recovery_not_physically_validated",
                    enabled=False, controller_available=False)

    def request_joint(self, reason):
        qualification = self.advanced_qualification("joint")
        self.stop(qualification["reason"] + ":" + reason)
        return None, qualification

    def record_refusal(self, arm, kind, reason):
        key = arm + ":" + reason
        table = self.decision_stats["refused"]
        table[key] = table.get(key, 0) + 1
        self.last_decision.setdefault("deferred", []).append(
            dict(arm=arm, kind=kind, reason=reason))

    def start_recovery(self, ai, state, kind, target):
        super().start_recovery(ai, state, kind, target)
        if kind == "sliding":
            self.mode = "Guide_" + ARMS[ai]
        self.takeover_sequence += 1
        self.takeover_id = self.takeover_sequence
        self.decision_stats["starts"] += 1
        # Guide only owns the drifting arm. Qualified P0/P1 Recovery
        # owns its active arm and the static Hold arm.
        stages = self.collaboration(state, self.contacts.read())["per_arm"]
        if kind == "sliding":
            self.episode_touched.add((ARMS[ai], stages[ai]))
        else:
            self.episode_touched.update((arm, stage) for arm, stage in zip(ARMS, stages))

    def _annotate(self, action, decision, state, contact):
        coop = self.collaboration(state, contact)
        decision.update(collaboration=coop, stage=coop["per_arm"][self.active] if self.control_active() else coop["stage"],
                        takeover_id=self.takeover_id if self.control_active() else 0,
                        priority="hard_safety>workspace_risk>stagnation>window",
                        advanced_qualifications={k:self.advanced_qualification(k)
                                                 for k in ("carry_local", "joint")})
        return action, decision

    def choose(self, policy_action):
        policy_action = np.asarray(policy_action, dtype=float)
        assert policy_action.shape == (14,) and np.isfinite(policy_action).all()
        self.steps += 1
        state, contact = snapshot(self.env), self.contacts.read()
        self.last_decision = dict(mode_before=self.mode, indices=self.indices(),
                                  policy_action=policy_action.tolist(), handback=False,
                                  deferred=[])
        danger = self.physical_danger(state, contact)
        if danger:
            self.stop(danger)
        if self.mode == "Safe_Stop":
            return self._annotate(*self.refuse(), state, contact)
        coop = self.collaboration(state, contact)
        if coop["coupled"] and not self.contact_assessment(state,contact)["allowed"]:
            self.request_joint("measured_contact_chain")
            return self._annotate(*self.refuse(), state, contact)
        risks = [self.predicted_risk(ai, policy_action, state) for ai in range(2)]
        self.last_decision["policy_risks"] = risks
        enabled = self.cfg["intervention_enabled"] and not self.auto_disabled and self.manager_cfg["method"] != "plain_rlpd"
        if self.mode == "Normal" and enabled:
            measures = [self.window_measure(ai, state) for ai in range(2)]
            for ai, measure in enumerate(measures):
                if measure["pipoint"] is not None:
                    measure["pose_error"] = errors(state["poses"][ai], self.pose(ai, measure["pipoint"]))
            self.last_decision["per_arm"] = measures
            candidates = []
            for ai, m in enumerate(measures):
                arm = ARMS[ai]
                self.cooldown[ai] = max(0, self.cooldown[ai] - 1)
                if self.manager_cfg["predict_workspace_risk"] and risks[ai]["risk"]:
                    candidates.append((0, ai, "workspace_risk", self.points["arms"][arm]["p0"]))
                elif not self.cooldown[ai]:
                    if self.manager_cfg["safety_recovery_enabled"] and m["stagnant"]:
                        candidates.append((1, ai, "safety_recovery", self.points["arms"][arm]["p0"]))
                    elif (self.manager_cfg["sliding_enabled"] and
                          not self.stage_mastered(arm, coop["per_arm"][ai]) and
                          m["distance_m"] is not None and
                          m["distance_m"] > self.cfg["trigger_position_m"] and m["direction_ok"]):
                        candidates.append((2, ai, "sliding", m["pipoint"]))
            self.decision_stats["candidates"] += len(candidates)
            for priority, ai, kind, target in sorted(candidates):
                arm = ARMS[ai]
                triggers = self.decision_stats["triggers"]
                triggers[arm + ":" + kind] = triggers.get(arm + ":" + kind, 0) + 1
                if coop["coupled"] and not self.contact_assessment(state,contact)["allowed"]:
                    self.record_refusal(arm, kind, "joint_recovery_not_physically_validated")
                    if priority == 0:
                        self.request_joint("predicted_unsafe_action")
                    continue
                if any(state["actual_grasps"][arm].values()):
                    self.record_refusal(arm, kind, "carry_local_recovery_not_physically_validated")
                    if priority == 0:
                        self.stop("predicted_unsafe_action_no_qualified_carry_recovery")
                    continue
                allowed, reason = self.eligible(ai, state, contact, kind, target)
                if allowed:
                    # Workspace prevention is safety recovery, not a backwards sliding fallback.
                    actual_kind = "safety_recovery" if kind == "workspace_risk" else kind
                    self.start_recovery(ai, state, actual_kind, target)
                    self.last_decision["trigger_reason"] = kind
                    break
                self.record_refusal(arm, kind, reason)
                if priority == 0:
                    self.stop("predicted_unsafe_action_no_qualified_recovery:" + reason)
                    break
        if self.mode == "Safe_Stop":
            return self._annotate(*self.refuse(), state, contact)
        if self.control_active():
            action, decision = self.recovery_action(state)
            return self._annotate(action, decision, state, contact)
        self.last_decision.update(mode="Normal", enabled=bool(enabled), execute=True,
                                  indices=self.indices(), executed_action=policy_action.tolist())
        return self._annotate(policy_action.copy(), self.last_decision, state, contact)

    def after_step(self):
        controlling = self.control_active()
        before_stages = self.last_decision.get("collaboration", {}).get(
            "per_arm", ["unknown"] * 2)
        guide=controlling and self.mode.startswith("Guide_")
        for ai, (arm, stage) in enumerate(zip(ARMS, before_stages)):
            key = (arm, stage)
            self.episode_phase_steps[key] = self.episode_phase_steps.get(key, 0) + 1
            if controlling and (not guide or ai==self.active):
                self.episode_touched.add(key)
            else:
                # The other arm remains a Policy arm during ordinary Guide.
                self.episode_policy_steps[key] = self.episode_policy_steps.get(key, 0) + 1
        result = super().after_step()
        state, contact = snapshot(self.env), self.contacts.read()
        # Parent previously checked danger only while recovering. Policy actions
        # must also be labelled correctly when a FULL physical step ends unsafe.
        danger = self.physical_danger(state, contact)
        if danger and self.mode != "Safe_Stop":
            self.stop(danger)
        stages = self.collaboration(state, contact)["per_arm"]
        for ai, arm in enumerate(ARMS):
            if self.grasp_streak[ai] >= 3:
                self.episode_completed.add((arm, "approach_pregrasp"))
            obj = self.points["arms"][arm]["object"]
            history = self.relative_grasp_history[ai]
            if state["actual_grasps"][arm][obj]:
                history.append(state["poses"][ai]["rotation"].T @ (np.asarray(state["objects"][obj]["position"]) - state["poses"][ai]["position"]))
            else:
                history.clear()
            stable = len(history) == history.maxlen and max(np.linalg.norm(x-history[0]) for x in history) < self.cfg["relative_grasp_motion_m"]
            if stable and self.grasp_streak[ai] >= self.cfg["stable_grasp_window_steps"]:
                self.episode_completed.add((arm, "grasp_stabilization"))
            if stages[ai] in ("first_assembly_contact", "second_assembly_contact"):
                self.episode_completed.add((arm, "independent_carry"))
            if self.completed[ai]:
                self.episode_completed.add((arm, "first_assembly_contact" if ai == 0 else "second_assembly_contact"))
        if result.get("handback"):
            self.decision_stats["handbacks"] += 1
        result.update(stage=self.collaboration(state, contact)["stage"],
                      safe_handback=bool(result.get("handback") and self.mode == "Normal"
                                         and not danger),
                      takeover_id=self.takeover_id)
        return result

    def handback(self, result, index):
        if self.recovery_kind == "sliding":
            from acoserl.object_geometry import object_error
            state = snapshot(self.env)
            obj = self.points["arms"][ARMS[self.active]]["object"]
            change = object_error(state["objects"][obj], self.baseline["objects"][obj])
            gripper = np.asarray(self.env.robots[self.active].gripper["right"].current_action)
            contact=self.contacts.read()
            active_contact=any(ARMS[self.active] in (edge['a'],edge['b']) for edge in contact['edges'])
            if (any(state["actual_grasps"][ARMS[self.active]].values()) or
                    gripper.shape != (2,) or not np.isfinite(gripper).all() or active_contact or
                    self.contact_assessment(state,contact)['category']!='Independent' or
                    change["position_m"] > .003 or change["rotation_rad"] > .035):
                self.stop("guide_handback_grasp_or_object_stage_mismatch")
                result["handback"] = False
                return
        super().handback(result, index)

    def stage_mastered(self, arm, stage):
        return bool(stage in ("approach_pregrasp", "grasp_stabilization") and self.manager_cfg["autonomy_enabled"] and
                    self.autonomy.get(arm + ":" + stage, 0) >=
                    self.manager_cfg["stage_clean_completions"])

    def episode_result(self, stable_full_success, had_intervention=None,
                       full_policy_episode=False):
        if self.manager_cfg["autonomy_enabled"] and full_policy_episode:
            for arm, stage in self.episode_phase_steps:
                key = arm + ":" + stage
                clean = (arm, stage) in self.episode_completed and (arm, stage) not in self.episode_touched and self.episode_policy_steps.get((arm, stage), 0) > 0
                self.autonomy[key] = self.autonomy.get(key, 0) + 1 if clean else 0
        clean_full = bool(stable_full_success and self.intervention_control_steps == 0
                          and full_policy_episode and self.cfg["fixed_initial_state"])
        self.clean_full_episodes = self.clean_full_episodes + 1 if clean_full else 0
        if (self.manager_cfg["autonomy_enabled"] and
                self.clean_full_episodes >= self.manager_cfg["full_clean_successes"]):
            self.auto_disabled = True
        row = dict(stable_full_success=bool(stable_full_success),
                   assisted_success=bool(stable_full_success and self.intervention_control_steps),
                   autonomous_success=clean_full, intervention_events=self.episode_interventions,
                   intervention_control_steps=self.intervention_control_steps,
                   handbacks=self.decision_stats["handbacks"],
                   clean_full_episodes=self.clean_full_episodes,
                   original_l_term=self.cfg["termination_intervention_steps"],
                   original_low_intervention_criterion=bool(stable_full_success and
                       self.intervention_control_steps < self.cfg["termination_intervention_steps"]),
                   auto_disabled=self.auto_disabled, stage_autonomy=self.autonomy.copy())
        self.termination_log.append(row)
        return row
