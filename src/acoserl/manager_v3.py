"""V3 guarded handoff, always-on qualified assembly protection.

Opt-in during development. V3 qualification is deliberately independent of V2.
Every returned command is executed by the ordinary bridge and replay collector.
"""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from acoserl.two_point_supervisor import snapshot,ARMS,errors
from acoserl.progress_guard import AssemblyProgressGuard,velocities,progress_state
from acoserl.object_geometry import object_error

ROOT=Path(__file__).resolve().parents[2]

class HandoffHold:
    """Only the existing OSC hold, retaining the successful takeover's baseline."""
    def __init__(self,manager,source_mode):
        self.m=manager;self.ai=manager.active;self.pi=1-self.ai
        self.contact_mode=False;self.stage="Correct";self.source_mode=source_mode
        self.baseline=copy.deepcopy(manager.baseline)
        self.targets=copy.deepcopy(manager.targets);self.grips=copy.deepcopy(manager.grips)
        self.obj=manager.points["arms"][ARMS[self.ai]]["object"]
        self.expected_grasps=copy.deepcopy(snapshot(manager.env)["actual_grasps"])
        self.relative=self.rel(snapshot(manager.env))
        self.held=bool(self.expected_grasps[ARMS[self.ai]][self.obj])

    def rel(self,state):
        from acoserl.local_recovery import LocalRecovery
        return LocalRecovery.rel(self,state)

    def action(self):
        action=self.m.command(self.targets,self.grips)
        if hasattr(self.m,"action_bounds"):action=np.clip(action,*self.m.action_bounds)
        return action

class V3Mixin:
    def reset(self,initial_progress=(0,0)):
        super().reset(initial_progress)
        self.progress_guard=AssemblyProgressGuard(self)
        self.handoff_phase=None;self.handoff_local=None;self.handoff_stable=0
        self.handoff_age=0;self.guarded_steps=0;self.guarded_unmodified=0
        self.hold_age=0;self.hold_limit=None
        self.release_latch=False;self.release_age=0
        self.v3_stats=dict(target_reached=0,handoff_ready=0,handbacks=0,
            policy_proposals=0,unmodified_policy=0,projected_policy=0,
            safe_actor_steps=0,first_piece_violations=0)

    def v3_enabled(self):return bool(self.manager_cfg.get("v3_enabled",False))

    def v3_fingerprint(self):
        if not hasattr(self,"_v3_fingerprint"):
            files=["src/acoserl/"+name for name in ("manager_v3.py","progress_guard.py","hold_envelope.py",
                "local_recovery.py","manager_v2.py","contact_modes.py","two_point_supervisor.py",
                "intervention_manager.py","bimanual_autoserl.py")]
            files += ["configs/bimanual_autoserl/"+name for name in
                      ("v1.json","recovery_points.json","hold_envelope_v3r3.json","reference_trace.npz")]
            if self.manager_cfg.get("baseline_mode",False):files += ["src/acoserl/pregrasp_hold.py"]
            self._v3_fingerprint={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
        return copy.deepcopy(self._v3_fingerprint)

    def advanced_qualification(self,kind,state=None,contact=None):
        if not self.v3_enabled():return super().advanced_qualification(kind,state,contact)
        p=ROOT/"configs/bimanual_autoserl/qualification_v3.json"
        proof=json.loads(p.read_text()).get(kind,{}) if p.exists() else {}
        source=self.v3_fingerprint()
        okay=bool(self.manager_cfg.get(kind,{}).get("enabled",False)
                  and proof.get("passed") and proof.get("source_sha256")==source)
        return dict(allowed=okay,kind=kind,reason="qualified_v3_domain" if okay else kind+"_not_qualified_v3",
                    controller_available=True,proof=proof)

    def passive_hold_entry(self,active,state):
        entry=super().passive_hold_entry(active,state)
        if not self.v3_enabled():return entry
        if self.manager_cfg.get("baseline_mode",False):
            from acoserl.pregrasp_hold import qualified_entry
            qualified=qualified_entry(self,active,state,entry)
            if qualified is not None:return qualified
        from acoserl.hold_envelope import evaluate
        p=ROOT/"configs/bimanual_autoserl/hold_envelope_v3r3.json"
        if not p.exists():return {**entry,"allowed":False,"reason":"passive_hold_calibration_missing"}
        envelope=json.loads(p.read_text());ctl=self.env.robots[1-active].part_controllers["right"]
        if (not np.allclose(ctl.kp,np.full(6,150.)) or float(self.env.control_freq)!=envelope["control_hz"]
                or float(self.env.sim.model._model.opt.timestep)!=envelope["physics_dt_s"]
                or self.cfg["position_gain"]!=2. or self.cfg["rotation_gain"]!=1.
                or self.cfg["position_command_cap_m"]!=.01 or self.cfg["rotation_command_cap_rad"]!=.02):
            return {**entry,"allowed":False,"reason":"hold_controller_not_calibrated"}
        result=evaluate(state,entry,self.cfg,envelope)
        if result["allowed"] and result.get("requires_independent_active"):
            if self.contact_assessment(state)["category"]!="Independent":
                result.update(allowed=False,reason="calibrated_hold_requires_independent_active_dynamics")
        return result

    def start_recovery(self,ai,state,kind,target):
        if self.v3_enabled():
            if kind=='sliding':
                # Ordinary Guide never acquires static ownership of the
                # independent Policy arm.
                self.hold_age=0;self.hold_limit=None
            else:
                entry=self.passive_hold_entry(ai,state)
                if not entry["allowed"]:raise RuntimeError("recovery start without calibrated passive hold")
                self.hold_age=0;self.hold_limit=entry["calibration_horizon_controls"]
        return super().start_recovery(ai,state,kind,target)

    def start_local(self,ai,index=None,contact_mode=False,validation=False):
        if not self.v3_enabled():return super().start_local(ai,index,contact_mode,validation)
        state=snapshot(self.env);entry=self.passive_hold_entry(ai,state)
        if not entry["allowed"]:return False,entry["reason"]
        corrected=None;alignment=None
        if not contact_mode:
            from scipy.spatial.transform import Rotation
            obj=self.points["arms"][ARMS[ai]]["object"]
            relative=state["poses"][ai]["rotation"].T@Rotation.from_quat(np.roll(state["objects"][obj]["quaternion_wxyz"],-1)).as_matrix()
            target=Rotation.from_quat(np.roll(self.ref_object(obj,index)["quaternion_wxyz"],-1)).as_matrix()@relative.T
            nominal=self.pose(ai,index)["rotation"]
            delta=Rotation.from_matrix(target@nominal.T).as_rotvec();length=np.linalg.norm(delta)
            if length>self.cfg["rotation_command_cap_rad"]:delta*=self.cfg["rotation_command_cap_rad"]/length
            corrected=Rotation.from_rotvec(delta).as_matrix()@nominal
            if Rotation.from_matrix(corrected@state["poses"][ai]["rotation"].T).magnitude()>.035:
                return False,"object_alignment_exceeds_existing_rotation_capture"
            alignment=dict(applied_rotation_rad=float(np.linalg.norm(delta)),
                remaining_reference_rotation_rad=float(Rotation.from_matrix(target@corrected.T).magnitude()))
        allowed,reason=super().start_local(ai,index,contact_mode,validation)
        if not allowed:return allowed,reason
        self.hold_age=0;self.hold_limit=entry["calibration_horizon_controls"]
        if corrected is not None:
            self.local.targets[ai]["rotation"]=corrected;self.local.object_alignment=alignment
        return True,reason

    def physical_danger(self,state,contact):
        if self.v3_enabled() and hasattr(self,"progress_guard"):
            reason=self.progress_guard.check(state)["violation"]
            if reason:return reason
        return super().physical_danger(state,contact)

    def substep_safety(self,contact):
        if not self.v3_enabled():return super().substep_safety(contact)
        state=snapshot(self.env);assessment=self.contact_assessment(state,contact)
        if not assessment["allowed"]:return assessment["reason"]
        if self.local is not None and not self.local.contact_mode and assessment["category"]!="Independent":
            return "carry_contact_mode_changed"
        check=self.progress_guard.check(state)
        if check["violation"]:
            self.progress_guard.violations+=1;return check["violation"]
        holding=self.control_active() or self.handoff_phase=="Recovery_Stabilizing"
        if holding and self.active is not None and self.baseline is not None:
            failure,_=self.retention(state)
            if failure:return failure
        return None

    def release_budget(self):
        speed=float(self.env.robots[1].gripper["right"].speed)
        return int(np.ceil(2./speed))+self.cfg["release_settle_max_steps"]

    def pending_release_valid(self,state):
        if not self.manager_cfg.get("baseline_mode",False) or not self.release_latch:return False
        if self.release_age>=self.release_budget() or not self.completed[0]:return False
        # Once opening has been legally initiated, the object follows the
        # demonstrated contact/release interval, not a monotonic final-pose ball.
        # Keeping a6mm final-pose test here reversed the Panda gripper midway
        # through its10-command opening stroke on the successful demo itself.
        ids=range(self.points["assembly"][1]["first_contact"],self.n_states)
        nearest=min(ids,key=lambda j:object_error(state["objects"]["piece_2"],self.ref_object("piece_2",j))["position_m"])
        oe=object_error(state["objects"]["piece_2"],self.ref_object("piece_2",nearest))
        return bool(state["official_metrics"]["first_piece_assembled"] and oe["position_m"]<=.006 and oe["rotation_rad"]<=.1)

    def valid_release(self,ai,state):
        if ai==1 and self.pending_release_valid(state):return True
        return super().valid_release(ai,state)

    def release_supported(self,ai,state,action):
        if ai==0:
            return bool(state["official_metrics"]["first_piece_assembled"]
                        and self.support_evidence("piece_1")["supported"])
        if not self.completed[0] or action[13]>=0:return False
        if self.pending_release_valid(state):return True
        error=object_error(state["objects"]["piece_2"],self.ref_object("piece_2",self.n_states-1))
        v=velocities(self.env)["objects"]["piece_2"]
        if self.manager_cfg.get("baseline_mode",False):
            # The demonstrated release starts while the gripper still carries
            # load. Static-rest speed/half-weight tests are hold qualifications,
            # not a valid prerequisite for this supported assembly release.
            # Preserve the existing6mm/.1rad/recent-contact release protocol.
            contact=self.contacts.read()
            assembly_contact=any("piece_2" in {edge["a"],edge["b"]}
                and bool(({edge["a"],edge["b"]}-{"piece_2"})&{"piece_1","base"}) for edge in contact["edges"])
            legal=self.contact_assessment(state,contact)["allowed"]
            return bool(legal and (assembly_contact or self.support_age[1]<=4) and self.support_age[1]<=4
                and self.windows[1].end>=self.points["assembly"][1]["first_contact"]
                and error["position_m"]<=.006 and error["rotation_rad"]<=.1)
        # Existing final-pose limits; actual support must exist NOW, not a clock.
        # Piece2 may be supported by piece1, which support_evidence does not cover.
        support=0.;model=self.env.sim.model._model;data=self.env.sim.data._data
        gravity=np.asarray(model.opt.gravity);g=float(np.linalg.norm(gravity));up=-gravity/max(g,1e-12)
        weight=float(model.body_subtreemass[self.env.obj_body_id["piece_2"]]*g)
        for i in range(self.env.sim.data.ncon):
            c=self.env.sim.data.contact[i]
            owners={self.contacts.owners[int(c.geom1)],self.contacts.owners[int(c.geom2)]}
            if c.dist<=0 and "piece_2" in owners and owners.intersection({"piece_1","base"}):
                force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
                support+=float(max(force[0],0)*abs(np.dot(np.asarray(c.frame[:3]),up)))
        return bool(support>=weight*self.cfg["placed_support_weight_fraction"] and weight>0 and error["position_m"]<=.006 and error["rotation_rad"]<=.1
                    and np.linalg.norm(v["linear"])<=self.cfg["placed_object_linear_speed_m_s"]
                    and np.linalg.norm(v["angular"])<=self.cfg["placed_object_angular_speed_rad_s"])

    def action_risks(self,action,state):
        risks=super().action_risks(action,state)
        if not self.v3_enabled():return risks
        for ai,arm in enumerate(ARMS):
            if any(state["actual_grasps"][arm].values()) and action[7*ai+6]<0:
                legal=self.release_supported(ai,state,action)
                if legal and self.manager_cfg.get("baseline_mode",False):
                    risks[ai]["categories"]=[c for c in risks[ai]["categories"] if c!="unplanned_grasp_release"]
                    risks[ai]["risk"]=bool(risks[ai]["categories"])
                elif not legal:
                    if "unplanned_grasp_release" not in risks[ai]["categories"]:
                        risks[ai]["categories"].append("unplanned_grasp_release")
                    risks[ai]["risk"]=True
        return risks

    def _stability(self):
        state=snapshot(self.env);v=velocities(self.env);local=self.handoff_local
        failure,passive=self.retention(state)
        if self.local is not None:local=self.local
        relative=local.rel(state)
        slip=float(np.linalg.norm(relative[0]-local.relative[0]))
        from scipy.spatial.transform import Rotation
        turn=float(Rotation.from_matrix(relative[1]@local.relative[1].T).magnitude())
        held=getattr(local,"held",True)
        expected=getattr(local,"expected_grasps",None)
        grasp_ok=(state["actual_grasps"]==expected if expected is not None
                  else state["actual_grasps"][ARMS[local.ai]][local.obj])
        checks=dict(physical=self.physical_danger(state,self.contacts.read()) is None,
            passive_retention=failure is None,
            target=errors(state["poses"][local.ai],local.targets[local.ai])["position_m"]<=.001,
            target_rotation=errors(state["poses"][local.ai],local.targets[local.ai])["rotation_rad"]<=.02,
            grasp=grasp_ok,
            grasp_relative=not held or (slip<=.0005 and turn<=.035),
            closed=bool(np.allclose(self.env.robots[local.ai].gripper["right"].current_action,[-1.,1.] if held else [1.,-1.])),
            eef_linear=all(np.linalg.norm(x["linear"])<=.005 for x in v["eef"]),
            eef_angular=all(np.linalg.norm(x["angular"])<=.05 for x in v["eef"]),
            object_linear=np.linalg.norm(v["objects"][local.obj]["linear"])<=.005,
            object_angular=np.linalg.norm(v["objects"][local.obj]["angular"])<=.05)
        if self.completed[0]:
            checks["assembly_qualified"]=self.progress_guard.anchor is not None
            checks["assembly_intact"]=self.progress_guard.check(state)["violation"] is None
        return all(checks.values()),dict(checks=checks,velocities=v,passive_error=passive,
                                        held_translation_slip_m=slip,held_rotation_slip_rad=turn)

    def _protected(self,action,decision,policy):
        if action is None:return action,decision
        intervention=np.asarray(action).copy()
        state=snapshot(self.env);self._quality_context=None
        source=decision["mode"]
        if (source.startswith(("Recover_","Guide_")) and self.handoff_phase!="Recovery_Stabilizing"
                and (self.local is None or self.local.stage!="Brake")):
            if self.local is not None:target=copy.deepcopy(self.local.targets[self.active])
            elif self.recovery_stage=="Replay_P0_P1":target=self.pose(self.active,min(self.replay_index+1,self.n_states-1))
            elif self.targets is not None:target=copy.deepcopy(self.targets[self.active])
            else:target=None
            if target is not None:
                self._quality_context=dict(target=target,active=self.active,
                    error_before_m=errors(state["poses"][self.active],target)["position_m"],
                    legal=bool(self.contact_assessment(state)["allowed"] and not self.completed[self.active]))
        final,projection=self.progress_guard.project(intervention,state)
        decision.update(actor_action_original=np.asarray(policy).tolist(),
            intervention_action_proposed=intervention.tolist(),
            progress_protection=projection,first_piece_protection_activated=projection["protection"]["active"],
            handoff_phase=self.handoff_phase)
        if self.handoff_phase in ("Handoff_Ready","Policy_Guarded"):
            decision["takeover_id"]=self.takeover_id
        if final is None:
            self.last_decision=decision;self.stop(projection["refusal"]);return self.refuse()
        if decision.get("pre_super_projection_arms") and source.startswith(("Guide_","Recover_")):
            # An independent safety correction on the Policy arm is not
            # expert guidance on that arm; exclude the mixed step from Demo.
            decision.update(mode="Constrained_Joint",
                source_before_projection=source,action_projection_applied=True,
                projection_reason="independent_policy_risk_correction")
        if projection["applied"]:
            affected={i for i in range(2) if np.max(abs(final[7*i:7*i+7]-intervention[7*i:7*i+7]))>1e-7}
            affected.update(getattr(self,"projected_arms",[]) if self.projected else [])
            self.projected=True;self.projected_arms=sorted(affected)
            decision["source_before_projection"]=decision["mode"]
            # SafetyProjection never becomes expert guidance by relabeling.
            decision.update(mode="Constrained_Joint",projection_reason=projection["reason"],
                action_projection_applied=True,projection_magnitude=projection["magnitude"])
        else:decision.setdefault("action_projection_applied",False)
        if self.manager_cfg.get("baseline_mode",False):
            if final[13]>0:
                self.release_latch=False;self.release_age=0
            elif (final[13]<0 and state["actual_grasps"]["B"]["piece_2"]
                  and not self.release_latch and self.release_supported(1,state,final)):
                self.release_latch=True;self.release_age=0
            decision.update(legal_release_pending=self.release_latch,legal_release_age=self.release_age,
                            gripper_actuation_budget=self.release_budget())
        decision["actor_action_executed"]=final.tolist()
        decision["executed_action"]=final.tolist()
        decision["policy_action_accepted_unmodified"]=bool(np.array_equal(final,np.asarray(policy)))
        self.last_decision=decision
        return final,decision

    def choose(self,policy_action):
        if not self.v3_enabled():return super().choose(policy_action)
        policy=np.asarray(policy_action,dtype=float)
        if self.manager_cfg.get("baseline_mode",False) and self.release_latch and self.release_age>=self.release_budget() and not self.completed[1]:
            self.last_decision=dict(policy_action=policy.tolist())
            self.stop("recovery_timeout_legal_release_not_completed");return self.refuse()
        holding=self.control_active() or self.handoff_phase=="Recovery_Stabilizing"
        if holding and self.hold_limit is not None and self.hold_age>=self.hold_limit:
            self.last_decision=dict(policy_action=policy.tolist())
            self.stop("calibrated_passive_hold_horizon_exhausted");return self.refuse()
        if self.mode=="Safe_Stop":return super().choose(policy)
        if self.handoff_phase=="Recovery_Stabilizing":
            self.steps+=1;self.handoff_age+=1
            state=snapshot(self.env);danger=self.physical_danger(state,self.contacts.read())
            if danger:
                self.last_decision=dict(policy_action=policy.tolist());self.stop(danger);return self.refuse()
            action=self.handoff_local.action()
            decision=dict(mode=getattr(self.handoff_local,"source_mode",
                "Recover_Joint" if self.handoff_local.contact_mode else "Recover_Carry_"+ARMS[self.active]),
                execute=True,policy_action=policy.tolist(),takeover_id=self.takeover_id,
                indices=self.indices(),stage="second_assembly_contact" if self.handoff_local.contact_mode else "independent_carry",
                recovery_stage="Recovery_Stabilizing")
            return self._protected(action,decision,policy)
        guarded=self.handoff_phase in ("Handoff_Ready","Policy_Guarded")
        if guarded:
            self.mode="Normal";self.local=None
            self.cfg["intervention_enabled"]=True
        # A safely retainable gripper suggestion must not suppress all contact
        # stagnation/recovery checks. Preserve the original proposal separately.
        proposed=policy.copy();retained=[];original_risks=None
        if self.local is None and self.mode=="Normal":
            state=snapshot(self.env);original_risks=self.action_risks(policy,state)
            candidates=[i for i,r in enumerate(original_risks) if r["risk"]]
            if candidates and all(original_risks[i]["categories"]==["unplanned_grasp_release"] for i in candidates):
                if len(candidates)==1 or (self.contact_assessment(state)["category"]=="Independent" and all(self.grasp_streak[i]>=10 for i in candidates)):
                    for i in candidates:proposed[7*i+6]=1.;retained.append(i)
        action,decision=super().choose(proposed)
        decision["policy_action"]=policy.tolist()
        if original_risks is not None:decision["policy_risks_original"]=original_risks
        if action is not None and retained and decision["mode"]=="Normal":
            self.projected=True;self.projected_arms=retained
            self.takeover_sequence+=1
            decision.update(mode="Constrained_"+(ARMS[retained[0]] if len(retained)==1 else "Joint"),
                            takeover_id=self.takeover_sequence,projection_reason="unplanned_grasp_release_retained_before_trigger_evaluation")
        if self.control_active() and not guarded:self.handoff_phase="Recovering"
        if guarded:
            self.v3_stats["policy_proposals"]+=1
            decision["guarded_policy_proposal"]=True
            if self.control_active():
                self.handoff_phase="Recovering"
            elif action is not None:
                self.mode="Policy_Guarded"
                decision["handoff_first_proposal"]=self.handoff_phase=="Handoff_Ready"
        return self._protected(action,decision,policy)

    def v3_checkpoint_state(self):
        context=self.handoff_local
        return dict(enabled=self.v3_enabled(),handoff_phase=self.handoff_phase,
            stabilization_steps=self.handoff_stable,hold_age=self.hold_age,hold_limit=self.hold_limit,
            guarded_steps=self.guarded_steps,guarded_unmodified=self.guarded_unmodified,stats=copy.deepcopy(self.v3_stats),
            progress_anchor=copy.deepcopy(self.progress_guard.anchor),
            handoff_targets=copy.deepcopy(context.targets) if context is not None else None,
            handoff_grips=copy.deepcopy(context.grips) if context is not None else None,
            resume_policy="audit context only; close pending real rows and reset into a NEW physical episode")

    def handback(self,result,index):
        if not self.v3_enabled():return super().handback(result,index)
        source_mode=self.mode
        super().handback(result,index)
        if not result.get("handback") or self.mode=="Safe_Stop":return
        if source_mode.startswith("Guide_"):
            # No Recovery_Stabilizing static hold for Guide: the non-guided
            # arm was always Policy-controlled. Next command is a guarded
            # full Actor proposal without moving either arm to a hold pose.
            self.handoff_local=None;self.handoff_phase="Handoff_Ready"
            self.mode="Handoff_Ready"
            self.guarded_steps=0;self.guarded_unmodified=0
            self.v3_stats["target_reached"]+=1
            result.update(target_reached=True,handback=False,
                safe_handback=False,guide_handoff_no_passive_hold=True)
            return
        self.handoff_local=HandoffHold(self,source_mode)
        self.handoff_phase="Recovery_Stabilizing";self.mode="Recovery_Stabilizing"
        self.handoff_stable=0;self.handoff_age=0
        self.guarded_steps=0;self.guarded_unmodified=0
        self.v3_stats["target_reached"]+=1
        result.update(target_reached=True,handback=False,safe_handback=False)

    def _task_quality(self):
        context=getattr(self,"_quality_context",None)
        waiting=self.handoff_phase in ("Recovery_Stabilizing","Handoff_Ready")
        report=dict(eligible=False,error_before_m=0.,error_after_m=0.,
                    handoff_wait=waiting,
                    handoff_confirmed=bool(self.last_decision.get("handoff_first_proposal") and self.stop_reason is None),
                    reason="no_task_correction_evidence")
        if context is None:return report
        state=snapshot(self.env);after=errors(state["poses"][context["active"]],context["target"])["position_m"]
        report.update(error_before_m=context["error_before_m"],error_after_m=after)
        projected=self.last_decision["mode"].startswith("Constrained_")
        eligible=(context["legal"] and not projected and self.stop_reason is None
                  and context["error_before_m"]-after>1e-6)
        report.update(eligible=bool(eligible),reason="legal_measured_target_error_reduction" if eligible else "no_qualified_error_reduction")
        return report

    def after_step(self):
        if not self.v3_enabled():
            result=super().after_step();self.progress_guard.observe(snapshot(self.env));return result
        phase=self.handoff_phase
        if self.release_latch and self.last_decision.get("executed_action",[0]*14)[13]<0:self.release_age+=1
        if self.control_active() or phase=="Recovery_Stabilizing":self.hold_age+=1
        if phase=="Recovery_Stabilizing":
            self.intervention_control_steps+=1
            okay,metrics=self._stability()
            self.handoff_stable=self.handoff_stable+1 if okay else 0
            protection=self.progress_guard.observe(snapshot(self.env))
            if protection["violation"]:self.stop(protection["violation"])
            if self.handoff_stable>=3:
                self.handoff_phase="Handoff_Ready";self.mode="Handoff_Ready"
                self.v3_stats["handoff_ready"]+=1
            if self.handoff_age>=40:self.stop("handoff_stabilization_timeout")
            return dict(handback=False,safe_handback=False,handoff_phase=self.handoff_phase,
                stabilization=metrics,stable_steps=self.handoff_stable,indices=self.indices(),
                first_piece_protection=protection,failure=self.stop_reason,
                task_quality=self._task_quality())
        local=self.local
        result=super().after_step()
        protection=self.progress_guard.observe(snapshot(self.env))
        result["first_piece_protection"]=protection
        if protection["violation"]:
            self.v3_stats["first_piece_violations"]+=1;self.stop(protection["violation"])
        if local is not None and result.get("handback") and self.mode!="Safe_Stop":
            self.decision_stats["handbacks"]-=1
            local.source_mode="Recover_Joint" if local.contact_mode else "Recover_Carry_"+ARMS[self.active]
            self.local=None;self.handoff_local=local;self.handoff_phase="Recovery_Stabilizing"
            self.handoff_stable=0;self.handoff_age=0;self.mode="Recovery_Stabilizing"
            self.guarded_steps=0;self.guarded_unmodified=0
            self.v3_stats["target_reached"]+=1
            result.update(target_reached=True,handback=False,safe_handback=False)
        elif phase in ("Handoff_Ready","Policy_Guarded") and self.mode!="Safe_Stop" and self.local is None:
            first=phase=="Handoff_Ready"
            self.handoff_phase="Policy_Guarded";self.mode="Policy_Guarded"
            self.guarded_steps+=1;self.v3_stats["safe_actor_steps"]+=1
            unmodified=bool(self.last_decision.get("policy_action_accepted_unmodified"))
            self.guarded_unmodified+=int(unmodified)
            self.v3_stats["unmodified_policy"]+=int(unmodified)
            self.v3_stats["projected_policy"]+=int(not unmodified)
            if first:
                self.v3_stats["handbacks"]+=1;self.decision_stats["handbacks"]+=1
            result.update(handback=first,safe_handback=first,post_handoff_safe_steps=self.guarded_steps,
                policy_action_accepted_unmodified=unmodified)
            # Bounded guard; qualification tests still collect >=20 proposals.
            if self.guarded_steps>=20 and self.guarded_unmodified>=3:
                self.handoff_phase=None;self.mode="Normal"
            elif self.guarded_steps>=40:self.stop("guarded_policy_no_independent_feasible_continuation")
        if phase not in ("Handoff_Ready","Policy_Guarded") and self.last_decision.get("mode","").startswith("Constrained_"):
            result.update(handback=False,safe_handback=False)
        result["handoff_phase"]=self.handoff_phase
        result["task_quality"]=self._task_quality()
        return result
