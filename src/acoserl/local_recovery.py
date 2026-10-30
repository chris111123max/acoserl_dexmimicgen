"""Local held-object and constrained insertion-unload recovery, action-only."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from acoserl.two_point_supervisor import snapshot,ARMS,errors
from acoserl.object_geometry import object_error

class LocalRecovery:
    def __init__(self,manager,ai,target_index=None,contact_mode=False):
        self.m=manager;self.ai=ai;self.pi=1-ai;self.contact_mode=contact_mode
        self.baseline=snapshot(manager.env);self.targets=copy.deepcopy(self.baseline["poses"])
        self.grips=[1. if any(self.baseline["actual_grasps"][a].values()) else -1. for a in ARMS]
        self.obj=manager.points["arms"][ARMS[ai]]["object"]
        self.relative=self.rel(self.baseline)
        self.target_index=target_index
        if contact_mode:self.targets[ai]["position"][2]+=.003
        else:self.targets[ai]=manager.pose(ai,target_index)
        self.stable=0;self.age=0;self.brake_steps=0
        self.stage="Correct" if manager.passive_hold_entry(ai,self.baseline)["allowed"] else "Brake"
        self.start_relation=self.baseline["poses"][self.pi]["rotation"].T@(self.targets[ai]["position"]-self.targets[self.pi]["position"])

    def rel(self,state):
        p=state["poses"][self.ai];o=state["objects"][self.obj]
        return (p["rotation"].T@(np.asarray(o["position"])-p["position"]),
                p["rotation"].T@Rotation.from_quat(np.roll(o["quaternion_wxyz"],-1)).as_matrix())

    @staticmethod
    def eligibility(m,ai,state,contact,contact_mode=False):
        from acoserl.contact_modes import assess
        a=ARMS[ai];obj=m.points["arms"][a]["object"]
        assessment=assess(m,state,contact)
        if not assessment["allowed"]:return False,assessment["reason"]
        if m.completed[ai]:return False,"completed_arm_must_not_rewind"
        if not state["actual_grasps"][a][obj]:return False,"correct_object_not_grasped"
        if m.grasp_streak[ai]<10:return False,"grasp_not_stable_10_steps"
        history=m.relative_grasp_history[ai];rotations=m.relative_grasp_rotation_history[ai]
        if len(history)<10 or max(np.linalg.norm(p-history[0]) for p in history)>.0005:return False,"initial_relative_grasp_translation_unstable"
        if len(rotations)<10 or max(Rotation.from_matrix(r@rotations[0].T).magnitude() for r in rotations)>.035:return False,"initial_relative_grasp_rotation_unstable"
        if not np.allclose(m.env.robots[ai].gripper["right"].current_action,[-1.,1.]):return False,"active_gripper_not_closed"
        if contact_mode:
            if ai!=1 or not m.completed[0]:return False,"contact_unload_requires_completed_A"
            if assessment["category"]!="Legal_Object_Contact":return False,"no_qualified_insertion_contact"
        elif assessment["category"]!="Independent":return False,"carry_requires_no_task_contact"
        if not contact_mode and any(obj in (edge["a"],edge["b"]) and "static" in (edge["a"],edge["b"]) for edge in contact["edges"]):return False,"held_object_environment_contact"
        if not contact_mode:
            if state["official_metrics"]["first_piece_assembled"]:return False,"carry_preassembly_domain_only"
            passive=1-ai
            if np.linalg.norm(state["poses"][ai]["position"]-state["poses"][passive]["position"])<.08:return False,"local_path_near_passive_arm"
            for other,held in state["actual_grasps"][ARMS[passive]].items():
                if held and np.linalg.norm(np.asarray(state["objects"][obj]["position"])-state["objects"][other]["position"])<.08:return False,"local_path_near_passive_object"
        entry=m.passive_hold_entry(ai,state)
        if not entry["allowed"]:
            speed=entry.get("linear_speed_m_s",float("inf"))
            if speed>.08:return False,"passive_brake_speed_outside_experimental_domain"
        return True,"eligible_for_directed_validation"

    def action(self):
        m=self.m;state=snapshot(m.env);targets=copy.deepcopy(self.targets)
        if self.stage=="Brake":
            targets=copy.deepcopy(self.baseline["poses"])
            for i,r in enumerate(m.env.robots):
                ctl=r.part_controllers["right"];site=m.env.sim.model.site_id2name(r.eef_site_id["right"])
                v=np.asarray(m.env.sim.data.get_site_xvelp(site));w=np.asarray(m.env.sim.data.get_site_xvelr(site))
                targets[i]["position"]-=np.clip(v*ctl.kd[:3]/ctl.kp[:3],-.002,.002)
                targets[i]["rotation"]=Rotation.from_rotvec(-w*ctl.kd[3:]/ctl.kp[3:]).as_matrix()@targets[i]["rotation"]
        action=m.command(targets,self.grips)
        if hasattr(m,"action_bounds"):action=np.clip(action,*m.action_bounds)
        return action

    def after(self):
        m=self.m;state=snapshot(m.env);self.age+=1
        failure,passive=m.retention(state)
        # retention operates on the same immutable baseline and active arm.
        actual=self.rel(state)
        slip=float(np.linalg.norm(actual[0]-self.relative[0]))
        rotation=float(Rotation.from_matrix(actual[1]@self.relative[1].T).magnitude())
        if not state["actual_grasps"][ARMS[self.ai]][self.obj]:failure="active_grasp_lost"
        if slip>.0005 or rotation>.035:failure="held_relative_transform_changed"
        contact=m.contacts.read();danger=m.physical_danger(state,contact)
        if danger:failure=danger
        if self.contact_mode:
            before=self.baseline["objects"]["piece_1"]
            if not state["official_metrics"]["first_piece_assembled"]:failure="first_assembly_progress_lost"
            if object_error(state["objects"]["piece_1"],before)["position_m"]>.002:failure="first_piece_displaced"
        if self.stage=="Brake":
            self.brake_steps+=1
            # Neither arm may wander during deceleration.
            for i in range(2):
                if errors(state["poses"][i],self.baseline["poses"][i])["position_m"]>.002:failure="brake_pose_drift"
            if m.passive_hold_entry(self.ai,state)["allowed"] and self.brake_steps>=3:self.stage="Correct"
            if self.brake_steps>=20 and self.stage=="Brake":failure="brake_timeout"
        error=errors(state["poses"][self.ai],self.targets[self.ai])
        expected=self.targets[self.ai]["position"]+self.targets[self.ai]["rotation"]@self.relative[0]
        object_target_error=float(np.linalg.norm(np.asarray(state["objects"][self.obj]["position"])-expected))
        reference_ok=True
        if not self.contact_mode:
            if m.contact_assessment(state,contact)["category"]!="Independent":failure="carry_contact_mode_changed"
            ref_error=object_error(state["objects"][self.obj],m.ref_object(self.obj,self.target_index))
            reference_ok=ref_error["position_m"]<=m.cfg["active_object_exit_position_m"] and ref_error["rotation_rad"]<=m.cfg["active_object_exit_rotation_rad"]
        okay=reference_ok and not failure and self.stage=="Correct" and error["position_m"]<=.001 and error["rotation_rad"]<=.02 and object_target_error<=.0015
        self.stable=self.stable+1 if okay else 0
        if self.age>=80:failure="local_recovery_timeout"
        return dict(handback=self.stable>=3 and not failure,safe_handback=self.stable>=3 and not failure,
                    failure=failure,passive_error=passive,held_translation_slip_m=slip,
                    held_rotation_slip_rad=rotation,object_target_error_m=object_target_error,
                    error=error,recovery_stage=self.stage,brake_steps=self.brake_steps)

    def checkpoint_state(self):
        return dict(active=self.ai,passive=self.pi,target_index=self.target_index,
                    contact_mode=self.contact_mode,baseline=self.baseline,targets=self.targets,
                    grips=self.grips,relative_transform=self.relative,age=self.age,
                    stable=self.stable,stage=self.stage,brake_steps=self.brake_steps,
                    resume_policy="audit_only; cancel takeover and reset at a new episode")
