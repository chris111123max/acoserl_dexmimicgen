"""V2 action shield, shared contact assessment and qualified local recovery."""
import copy,json,hashlib
from collections import deque
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from acoserl.two_point_supervisor import snapshot,ARMS,errors
from acoserl.contact_modes import assess
from acoserl.local_recovery import LocalRecovery

ROOT=Path(__file__).resolve().parents[2]

class V2Mixin:
    def reset(self,initial_progress=(0,0)):
        super().reset(initial_progress)
        self.local=None;self.projected=False;self.stage_trace={};self.contact_history=deque(maxlen=21);self.relative_grasp_rotation_history=[deque(maxlen=10) for _ in ARMS]
        self.global_capability=copy.deepcopy(getattr(self,"global_capability",{"stage_counts":{},"clean_full":0,"disable":False}))

    def contact_assessment(self,state=None,contact=None):
        return assess(self,state or snapshot(self.env),contact or self.contacts.read())

    def physical_danger(self,state,contact):
        a=self.contact_assessment(state,contact)
        if not a["allowed"]:return a["reason"]
        return super().physical_danger(state,{**contact,"dynamic_A_B_coupled":False})

    def substep_safety(self,contact):
        a=self.contact_assessment(contact=contact)
        if self.local is not None and not self.local.contact_mode and a["category"]!="Independent":return "carry_contact_mode_changed"
        return None if a["allowed"] else a["reason"]

    def advanced_qualification(self,kind,state=None,contact=None):
        p=ROOT/"configs/bimanual_autoserl/qualification_v2.json"
        data=json.loads(p.read_text()) if p.exists() else {}
        proof=data.get(kind,{})
        expected=hashlib.sha256((ROOT/"src/acoserl/local_recovery.py").read_bytes()).hexdigest()
        okay=bool(self.manager_cfg.get(kind,{}).get("enabled",False) and proof.get("passed") and proof.get("controller_sha256")==expected)
        return dict(allowed=okay,kind=kind,reason="qualified_bounded_domain" if okay else kind+"_not_qualified_v2",
                    controller_available=True,proof=proof)

    def start_local(self,ai,index=None,contact_mode=False,validation=False):
        state=snapshot(self.env);contact=self.contacts.read()
        allowed,reason=LocalRecovery.eligibility(self,ai,state,contact,contact_mode)
        if not allowed:return False,reason
        if not validation and not self.advanced_qualification("joint" if contact_mode else "carry_local")["allowed"]:
            return False,"physical_qualification_missing"
        if not validation:
            proof=self.advanced_qualification("joint" if contact_mode else "carry_local")["proof"]
            if not contact_mode:
                goal=self.pose(ai,index);certified=proof["target_poses"][ARMS[ai]]
                if np.linalg.norm(goal["position"]-np.asarray(certified["position"]))>1e-6 or Rotation.from_matrix(goal["rotation"]@np.asarray(certified["rotation"]).T).magnitude()>1e-6:return False,"unqualified_local_target_pose"
            if proof.get("requires_both_holding") and not all(state["actual_grasps"][a][self.points["arms"][a]["object"]] for a in ARMS):return False,"passive_empty_handoff_domain_not_qualified"
            entry=self.passive_hold_entry(ai,state);speed=entry.get("linear_speed_m_s",0)
            if not entry["allowed"]:
                brake=self.advanced_qualification("passive_brake")
                interval=brake["proof"].get("speed_intervals",{}).get(ARMS[ai])
                if not brake["allowed"] or interval is None or not interval[0]<=speed<=interval[1]:return False,"passive_brake_not_physically_qualified"
            elif speed>proof.get("direct_hold_max_speed",{}).get(ARMS[ai],0):return False,"outside_tested_direct_hold_speed"
        if not contact_mode:
            error=errors(state["poses"][ai],self.pose(ai,index))
            bound=.008 if validation else proof["capture_position_max"][ARMS[ai]]
            rotation_bound=.035 if validation else proof["capture_rotation_max"][ARMS[ai]]
            if error["position_m"]>bound or error["rotation_rad"]>rotation_bound:return False,"local_capture_envelope"
        self.active=ai;self.baseline=copy.deepcopy(state)
        self.local=LocalRecovery(self,ai,index,contact_mode)
        self.mode="Recover_Joint" if contact_mode else "Recover_Carry_"+ARMS[ai]
        self.takeover_sequence+=1;self.takeover_id=self.takeover_sequence
        self.age=0;self.episode_interventions+=1;self.decision_stats["starts"]+=1
        for arm,stage in zip(ARMS,self.collaboration(state,contact)["per_arm"]):self.episode_touched.add((arm,stage))
        return True,reason

    def action_risks(self,action,state):
        risks=[self.predicted_risk(i,action,state) for i in range(2)]
        for ai,risk in enumerate(risks):
            risk["categories"]=["workspace"] if risk["risk"] else []
            r=self.env.robots[ai];arm=ARMS[ai]
            site=self.env.sim.model.site_id2name(r.eef_site_id["right"])
            risk["linear_speed_m_s"]=float(np.linalg.norm(self.env.sim.data.get_site_xvelp(site)))
            # Do not forbid planned closing contact or held-object assembly.
            opening=action[7*ai+6]<0
            held=any(state["actual_grasps"][arm].values())
            planned_release=(ai==0 and state["official_metrics"]["first_piece_assembled"]) or (ai==1 and self.completed[0] and self.windows[1].end>=self.points["arms"]["B"]["first_release_action"] and self.contact_assessment(state)["category"]=="Legal_Object_Contact")
            if held and opening and not planned_release:
                risk["categories"].append("unplanned_grasp_release");risk["risk"]=True
            if not opening or held:continue
            displacement=np.asarray(risk["displacement"])
            candidates=[]
            model=self.env.sim.model;data=self.env.sim.data
            for obj in ("piece_1","piece_2","base"):
                center=np.asarray(state["objects"][obj]["position"])
                if np.linalg.norm(center-state["poses"][ai]["position"])>.14:continue
                for g in r.gripper["right"].contact_geoms:
                    gid=model.geom_name2id(g)
                    collision_geoms=[name for name in getattr(self.env,obj).contact_geoms
                        if (int(model.geom_contype[gid])&int(model.geom_conaffinity[model.geom_name2id(name)])) or (int(model.geom_contype[model.geom_name2id(name)])&int(model.geom_conaffinity[gid]))]
                    nearest=sorted(collision_geoms,
                        key=lambda name:np.linalg.norm(data.geom_xpos[model.geom_name2id(name)]-data.geom_xpos[gid]))[:12]
                    for name in nearest:
                        oid=model.geom_name2id(name)
                        gap=float(mujoco.mj_geomDistance(model._model,data._data,gid,oid,.03,None))
                        toward=center-data.geom_xpos[gid];toward/=max(np.linalg.norm(toward),1e-9)
                        command=np.asarray(risk['command_displacement'])
                        approach=max(float(np.dot(displacement,toward)),float(np.dot(command,toward)))
                        predicted=gap-approach
                        reversing=bool(np.asarray(r.gripper["right"].current_action)[0]<-.5)
                        if (gap<.025 and predicted<.006 and approach>0) or (reversing and gap<.003):
                            candidates.append(dict(object=obj,gap_m=gap,predicted_gap_m=predicted,normal=toward.tolist(),command_reversal=reversing))
            if candidates:
                risk["categories"].append("open_gripper_unplanned_approach")
                risk["gripper_contact"]=min(candidates,key=lambda x:x["predicted_gap_m"])
                risk["risk"]=True
        return risks

    def _independent_tcp_swept_clearance(self,action,state):
        """Conservative EEF sweep of the ACTUALLY composed 14D action."""
        p=[np.asarray(x['position']) for x in state['poses']]
        delta=[np.asarray(self.predicted_risk(i,action,state)['displacement'])
               for i in range(2)]
        rel=p[0]-p[1];motion=delta[0]-delta[1]
        t=np.clip(-float(rel@motion)/max(float(motion@motion),1e-12),0.,1.)
        if np.linalg.norm(rel+t*motion)<.12:
            return False,'independent_projection_tcp_clearance'
        return True,'independent_tcp_swept_clearance'

    def _project_independent_risks(self,policy,state,risks,indices):
        """Correct selected Policy arms, never a Guide arm's unused proposal.

        A separate screen in TwoPoint.recovery_action validates the final
        Guide_A+Policy_B or Policy_A+Guide_B command before physical stepping.
        """
        from acoserl.osc_prediction import project_independent_workspace
        corrected=np.asarray(policy,dtype=float).copy()
        chosen=tuple(sorted(set(indices)))
        assessment=self.contact_assessment(state)
        motion_projected=False
        for ai in chosen:
            risk=risks[ai];categories=set(risk['categories'])
            if categories=={'unplanned_grasp_release'}:
                corrected[7*ai+6]=1.
                continue
            refusals=[]
            if assessment['category']!='Independent':
                refusals.append('contact_mode='+str(assessment['category']))
            holding=[arm for arm in ARMS
                     if any(state['actual_grasps'][arm].values())]
            if holding:refusals.append('existing_grasps='+','.join(holding))
            if not categories or not categories.issubset(
                    {'workspace','open_gripper_unplanned_approach'}):
                refusals.append('unsupported_categories='+','.join(sorted(categories)))
            if risk.get('gripper_contact',{}).get('command_reversal'):
                refusals.append('near_contact_gripper_reversal')
            if refusals:
                # Keep the same fail-closed gate but expose WHICH physical
                # qualification was violated. Never silently weaken this test.
                return None,('independent_risk_projection_not_qualified:'
                             +ARMS[ai]+':'+'|'.join(refusals))
            position=np.asarray(state['poses'][ai]['position'])
            lower=self.positions[:,ai].min(0)-self.cfg['free_workspace_margin_m']
            upper=self.positions[:,ai].max(0)+self.cfg['free_workspace_margin_m']
            corrected,reason=project_independent_workspace(
                self.env,ai,corrected,position,lower,upper,
                self.manager_cfg['risk_margin_m'],
                getattr(self,'action_bounds',None),
                risk.get('gripper_contact',{}).get('normal'))
            if corrected is None:return None,ARMS[ai]+':'+reason
            motion_projected=True
        # The active Guide arm's original Actor proposal is never executed.
        # Check only projected arms HERE; the final joint command is checked
        # AFTER the real Guide action has replaced that arm's six pose DOFs.
        fresh=self.action_risks(corrected,state)
        remaining=[ARMS[i]+':'+','.join(fresh[i]['categories'])
                   for i in chosen if fresh[i]['risk']]
        if remaining:return None,'independent_projection_residual_risk:'+';'.join(remaining)
        guide_substitution=(self.mode.startswith('Guide_') and
                            self.active is not None and self.active not in chosen)
        if motion_projected and not guide_substitution:
            clear,reason=self._independent_tcp_swept_clearance(corrected,state)
            if not clear:return None,reason
        return corrected,'independent_risks_corrected'

    def _local_decision(self,policy):
        self.steps+=1;self.age+=1
        state=snapshot(self.env);danger=self.physical_danger(state,self.contacts.read())
        if danger:
            self.stop(danger);self.last_decision=dict(policy_action=policy.tolist())
            return self.refuse()
        action=self.local.action()
        self.last_decision=dict(mode=self.mode,execute=True,policy_action=policy.tolist(),
            executed_action=action.tolist(),indices=self.indices(),takeover_id=self.takeover_id,
            stage="second_assembly_contact" if self.local.contact_mode else "independent_carry",
            recovery_step=self.age,recovery_stage=self.local.stage,
            contact_assessment=self.contact_assessment())
        return action,self.last_decision

    def choose(self,policy_action):
        policy=np.asarray(policy_action,dtype=float)
        if self.local is not None and self.mode!="Safe_Stop":return self._local_decision(policy)
        state=snapshot(self.env)
        danger=self.physical_danger(state,self.contacts.read())
        if danger:
            self.steps+=1;self.last_decision=dict(policy_action=policy.tolist(),contact_assessment=self.contact_assessment(state))
            self.stop(danger);return self.refuse()
        early_risks=self.action_risks(policy,state)
        # An in-flight Guide does not freeze the other arm. Its fresh Policy
        # proposal must be checked each step and corrected only if unsafe.
        if self.mode.startswith('Guide_') and self.active is not None:
            self.projected=False;self.projected_arms=[]
            other=1-self.active
            proposal=policy
            if early_risks[other]['risk']:
                proposal,reason=self._project_independent_risks(
                    policy,state,early_risks,[other])
                if proposal is None:
                    self.steps+=1;self.last_decision=dict(
                        policy_action=policy.tolist(),policy_risks=early_risks,
                        guide_risk_refusal_context=dict(
                            guided_arm=ARMS[self.active],policy_arm=ARMS[other],
                            contact_category=self.contact_assessment(state)['category'],
                            actual_grasps=state['actual_grasps'],
                            projection_reason=reason))
                    self.stop('guide_passive_policy_risk:'+reason)
                    return self.refuse()
            final,decision=super().choose(proposal)
            decision['policy_action']=policy.tolist()
            if final is not None and proposal is not policy:
                self.projected=True;self.projected_arms=[other]
                decision['pre_super_projection_arms']=[other]
                decision['policy_risks_original']=early_risks
            return final,decision
        enabled=self.cfg["intervention_enabled"] and self.manager_cfg["method"]!="plain_rlpd"
        contact_candidate=None
        if enabled and not any(r["risk"] for r in early_risks) and self.mode=="Normal" and self.manager_cfg["safety_recovery_enabled"]:
            coop=self.collaboration(state,self.contacts.read())
            if coop["per_arm"][1]=="second_assembly_contact" and self.completed[0] and self.cooldown[1]==0:
                j=int(np.argmin(np.linalg.norm(self.positions[:,1]-state["poses"][1]["position"],axis=1)))
                self.contact_history.append(j)
                stuck=len(self.contact_history)==21 and np.linalg.norm(self.positions[self.contact_history[-1],1]-self.positions[self.contact_history[0],1])<.005
                near_force=self.contacts.read()["max_robot_contact_force_N"]>60
                if stuck or near_force:
                    allowed,reason=self.start_local(1,contact_mode=True)
                    contact_candidate=dict(trigger="contact_stagnation" if stuck else "contact_force_risk",allowed=allowed,reason=reason)
                    if allowed:return self._local_decision(policy)
                    if near_force:
                        self.last_decision=dict(policy_action=policy.tolist(),contact_recovery=contact_candidate)
                        self.stop("contact_force_risk_no_qualified_recovery");return self.refuse()
            else:self.contact_history.clear()
        # Only qualified bounded held-object domains can enable automatic local correction.
        if enabled and not any(r["risk"] for r in early_risks) and not self.global_capability.get("disable") and self.mode=="Normal" and self.manager_cfg["safety_recovery_enabled"]:
            for ai in range(2):
                if self.cooldown[ai]>0:continue
                if not state["actual_grasps"][ARMS[ai]][self.points["arms"][ARMS[ai]]["object"]]:continue
                measure=self.window_measure(ai,state,record_history=False)
                j=measure["pipoint"]
                if j is None:continue
                distance=measure["distance_m"]
                if .003<distance<=.008:
                    allowed,_=self.start_local(ai,j)
                    if allowed:return self._local_decision(policy)
        risks=early_risks
        active=[i for i,r in enumerate(risks) if r["risk"]]
        self.projected=False
        if enabled and self.mode=="Normal" and self.manager_cfg["predict_workspace_risk"] and active:
            shield_proof=ROOT/"configs/bimanual_autoserl/qualification_v2.json"
            proof=json.loads(shield_proof.read_text()).get("joint_grip_shield",{}) if shield_proof.exists() else {}
            shield_authorized=self.manager_cfg.get("development_validation",False) or (self.manager_cfg.get("joint_grip_shield",{}).get("enabled",False) and proof.get("passed") and max(r["linear_speed_m_s"] for r in risks)<=proof.get("max_speed_m_s",0) and proof.get("controller_sha256")==hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            joint_grips=(len(active)==2 and all(r["categories"]==["unplanned_grasp_release"] for r in risks)
                         and self.contact_assessment(state)["category"]=="Independent"
                         and all(self.grasp_streak[i]>=10 for i in range(2)))
            if joint_grips and shield_authorized:
                corrected=policy.copy();corrected[[6,13]]=1.
                final,decision=super().choose(corrected)
                decision["policy_action"]=policy.tolist();decision["policy_risks"]=risks
                if final is not None and decision["mode"]=="Normal":
                    self.projected=True;self.projected_arms=[0,1]
                    self.takeover_sequence+=1
                    decision.update(mode="Constrained_Joint",takeover_id=self.takeover_sequence,
                                    projection_reason="independent_held_objects_joint_grip_retention")
                return final,decision
            if len(active)>1:
                # Two independent free-space action risks are NOT evidence
                # of a mechanically coupled joint recovery requirement.
                corrected,reason=self._project_independent_risks(
                    policy,state,risks,active)
                if corrected is None:
                    self.steps+=1;self.last_decision=dict(
                        policy_action=policy.tolist(),policy_risks=risks,
                        independent_risk_refusal=reason)
                    self.stop("simultaneous_high_confidence_risks_no_joint_qualification")
                    return self.refuse()
                final,decision=super().choose(corrected)
                decision["policy_action"]=policy.tolist()
                decision["policy_risks_original"]=risks
                if final is not None:
                    self.projected=True;self.projected_arms=active
                    self.takeover_sequence+=1
                    decision["pre_super_projection_arms"]=active
                    if decision["mode"]=="Normal":
                        decision.update(mode="Constrained_Joint",
                            takeover_id=self.takeover_sequence,
                            projection_reason="independent_free_risk_correction")
                return final,decision
            ai=active[0]
            if risks[ai].get("gripper_contact",{}).get("command_reversal"):
                self.steps+=1;self.last_decision=dict(policy_action=policy.tolist(),policy_risks=risks)
                self.stop("near_contact_gripper_reversal_no_qualified_correction")
                return self.refuse()
            safe_free=not any(any(g.values()) for g in state['actual_grasps'].values()) and self.contact_assessment(state)['category']=='Independent'
            release_only=risks[ai]['categories']==['unplanned_grasp_release']
            if safe_free or release_only:
                corrected,reason=self._project_independent_risks(policy,state,risks,[ai])
                if corrected is None:
                    self.steps+=1;self.last_decision=dict(
                        policy_action=policy.tolist(),policy_risks=risks,
                        independent_risk_refusal=reason)
                    self.stop('no_legal_projected_action:'+reason);return self.refuse()
                final,decision=super().choose(corrected)
                decision["policy_action"]=policy.tolist();decision["policy_risks"]=risks
                if final is not None and decision["mode"]=="Normal":
                    self.projected=True;self.projected_arms=[ai]
                    self.takeover_sequence+=1
                    decision.update(mode="Constrained_"+ARMS[ai],takeover_id=self.takeover_sequence,
                                    projection_reason=risks[ai]["categories"])
                return final,decision
        final,decision=super().choose(policy)
        if contact_candidate:decision["contact_recovery"]=contact_candidate
        return final,decision

    def after_step(self):
        state=snapshot(self.env)
        for ai,arm in enumerate(ARMS):
            obj=self.points["arms"][arm]["object"];history=self.relative_grasp_rotation_history[ai]
            if state["actual_grasps"][arm][obj]:history.append(state["poses"][ai]["rotation"].T@Rotation.from_quat(np.roll(state["objects"][obj]["quaternion_wxyz"],-1)).as_matrix())
            else:history.clear()
        for ai,stage in enumerate(self.collaboration(state,self.contacts.read())["per_arm"]):
            lo,hi,_=self.legal_bounds(ai,state);ids=self.windows[ai].indices(hi)
            ids=[j for j in ids if j>=lo]
            marker=self.stage_trace.setdefault(ARMS[ai]+":"+stage,dict(entered=True,progressed=False,first=None))
            if ids:
                j=min(ids,key=lambda j:np.linalg.norm(self.positions[j,ai]-state["poses"][ai]["position"]))
                if marker["first"] is None:marker["first"]=j
                if j>=marker["first"] and np.linalg.norm(self.positions[j,ai]-self.positions[marker["first"],ai])>=.005:marker["progressed"]=True
        if self.local is not None:
            self.intervention_control_steps+=1
            result=self.local.after()
            if result["failure"]:self.stop(result["failure"])
            elif result["handback"]:
                self.mode="Normal";self.cooldown[self.active]=self.cfg["cooldown_steps"];self.contact_history.clear()
                self.decision_stats["handbacks"]+=1;self.local=None
            result["indices"]=self.indices()
            return result
        result=super().after_step()
        if self.projected:
            self.intervention_control_steps+=1;self.episode_interventions+=1
            for ai in self.projected_arms:
                arm=ARMS[ai]
                stage=self.collaboration(snapshot(self.env),self.contacts.read())["per_arm"][ai]
                self.episode_touched.add((arm,stage))
            result.update(handback=self.mode=="Normal",safe_handback=self.mode=="Normal")
        return result

    def stage_mastered(self,arm,stage):
        return self.global_capability["stage_counts"].get(arm+":"+stage,0)>=self.manager_cfg["stage_clean_completions"]

    def set_global_capability(self,value):
        self.global_capability=copy.deepcopy(value)
        # Only coordinator evidence can disable shared-policy guidance.
        self.auto_disabled=bool(value.get("disable",False))

    def episode_result(self,*args,**kwargs):
        row=super().episode_result(*args,**kwargs)
        self.auto_disabled=bool(self.global_capability.get("disable",False))
        row["auto_disabled"]=self.auto_disabled
        row["stage_events"]={}
        for arm,stage in self.episode_phase_steps:
            key=(arm,stage);completed=key in self.episode_completed
            row["stage_events"][arm+":"+stage]=dict(stage_entered=True,
                stage_progressed=self.stage_trace.get(arm+":"+stage,{}).get("progressed",False) or completed,stage_completed=completed,
                stage_stably_completed=completed and stage in ("approach_pregrasp","grasp_stabilization"),
                stage_policy_completed=completed and key not in self.episode_touched,
                stage_intervention_assisted=key in self.episode_touched)
        for key,event in row["stage_events"].items():
            if key.endswith(("first_assembly_contact","second_assembly_contact","independent_carry")):
                ai=0 if key.startswith("A:") else 1
                event["stage_stably_completed"]=bool(self.completed[ai] and (self.protected_first is not None if ai==0 else row["stable_full_success"]))
        if self.protected_first is not None:
            from acoserl.object_geometry import object_error
            current=snapshot(self.env)
            keep=object_error(current["objects"]["piece_1"],self.protected_first)
            valid_first=current["official_metrics"]["first_piece_assembled"] and keep["position_m"]<=self.cfg["assembled_object_drift_m"] and keep["rotation_rad"]<=self.cfg["assembled_object_rotation_drift_rad"]
            for name in ("A:first_assembly_contact","A:independent_carry"):
                if name in row["stage_events"]:row["stage_events"][name]["stage_stably_completed"] &= bool(valid_first)
        self.last_episode_outcome=row
        return row
