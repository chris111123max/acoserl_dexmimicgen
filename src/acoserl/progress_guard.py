"""Measured assembly protection and local contact half-space projection.

This supervisor is not a learned dynamics certificate. New domains remain behind
V3 qualification; geometry prediction and actual substep safety are separate.
"""
import copy
import numpy as np
import mujoco
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.transform import Rotation
from acoserl.object_geometry import object_error


def velocities(env):
    eef=[];objects={}
    for robot in env.robots:
        site=env.sim.model.site_id2name(robot.eef_site_id["right"])
        eef.append(dict(linear=np.asarray(env.sim.data.get_site_xvelp(site)).copy(),
                        angular=np.asarray(env.sim.data.get_site_xvelr(site)).copy()))
    for name,bid in env.obj_body_id.items():
        v=np.zeros(6)
        mujoco.mj_objectVelocity(env.sim.model._model,env.sim.data._data,
                                mujoco.mjtObj.mjOBJ_BODY,bid,v,0)
        objects[name]=dict(linear=v[3:].copy(),angular=v[:3].copy())
    return dict(eef=eef,objects=objects)


def progress_state(env):
    return dict(objects={name:dict(position=env.sim.data.body_xpos[bid].copy(),
                   quaternion_wxyz=env.sim.data.body_xquat[bid].copy())
                   for name,bid in env.obj_body_id.items()},
                official_metrics={k:bool(v) for k,v in env._get_partial_task_metrics().items()})


def relative_pose(obj,base):
    r=Rotation.from_quat(np.roll(base["quaternion_wxyz"],-1))
    q=Rotation.from_quat(np.roll(obj["quaternion_wxyz"],-1))
    return dict(position=r.inv().apply(np.asarray(obj["position"])-base["position"]),
                quaternion_wxyz=np.roll((r.inv()*q).as_quat(),1))


class AssemblyProgressGuard:
    def __init__(self,manager):
        self.m=manager;self.anchor=None;self.streak=0;self.last={}
        self.activations=0;self.violations=0;self.last_load_time=None
        model=manager.env.sim.model
        self.object_geoms={name:[model.geom_name2id(g) for g in getattr(manager.env,name).contact_geoms]
                           for name in ("piece_1","piece_2","base")}

    def observe(self,state):
        m=self.m;v=velocities(m.env);support=m.support_evidence("piece_1")
        stable=(m.completed[0] and state["official_metrics"]["first_piece_assembled"]
                and not any(state["actual_grasps"]["A"].values()) and support["supported"]
                and np.linalg.norm(v["objects"]["piece_1"]["linear"])<=m.cfg["placed_object_linear_speed_m_s"]
                and np.linalg.norm(v["objects"]["piece_1"]["angular"])<=m.cfg["placed_object_angular_speed_rad_s"])
        self.streak=self.streak+1 if stable else 0
        if self.anchor is None and self.streak>=3:
            self.anchor=dict(piece=copy.deepcopy(state["objects"]["piece_1"]),
                base=copy.deepcopy(state["objects"]["base"]),
                relative=relative_pose(state["objects"]["piece_1"],state["objects"]["base"]),
                support=support,velocities=v,grasps=copy.deepcopy(state["actual_grasps"]),
                sim_time=state["sim_time"])
            self.activations+=1
        self.last=self.check(state)
        return self.last

    def check(self,state):
        m=self.m
        # Legacy anchor remains enforced immediately, before stable qualification.
        anchor=self.anchor
        target=anchor["piece"] if anchor else m.protected_first
        if target is None:return dict(active=False,qualified=False,violation=None)
        error=object_error(state["objects"]["piece_1"],target)
        relative_error=(object_error(relative_pose(state["objects"]["piece_1"],state["objects"]["base"]),anchor["relative"])
                        if anchor else error)
        support=m.support_evidence("piece_1")
        reason=None
        if not state["official_metrics"]["first_piece_assembled"]:reason="completed_piece1_progress_lost"
        elif (error["position_m"]>m.cfg["assembled_object_drift_m"]
              or relative_error["position_m"]>m.cfg["assembled_object_drift_m"]
              or error["rotation_rad"]>m.cfg["assembled_object_rotation_drift_rad"]
              or relative_error["rotation_rad"]>m.cfg["assembled_object_rotation_drift_rad"]):
            reason="completed_piece1_displaced"
        elif anchor and object_error(state["objects"]["base"],anchor["base"])["position_m"]>m.cfg["base_drift_m"]:
            reason="protected_base_displaced"
        elif anchor and not support["supported"]:
            transfer=False
            if (m.manager_cfg.get("baseline_mode",False) and m.pending_release_valid(state)
                    and not any(state["actual_grasps"]["A"].values())):
                pairs={frozenset((e["a"],e["b"])) for e in m.contacts.read()["edges"]}
                first_second=frozenset(("piece_1","piece_2")) in pairs
                grounded=any(frozenset(("piece_2",other)) in pairs for other in ("base","static"))
                held=state["actual_grasps"]["B"]["piece_2"]
                transfer=bool(first_second and (grounded or held))
                if transfer:
                    # A measured, task-specific load path during the already
                    # authorized finite opening stroke. Keep the direct-force
                    # evidence unchanged and never move the saved pose anchor.
                    support={**support,"task_transfer_route":["piece_1","piece_2","ground" if grounded else "B"],
                             "task_transfer_qualified":True}
            if not transfer:
                # MuJoCo changes contact pairs at an integration boundary. The
                # successful raw demo has one isolated 2ms zero-contact sample.
                # Permit ONLY that single solver interval during qualified release;
                # a second unsupported sample stops. Pose/force limits never change.
                dt=float(m.env.sim.model._model.opt.timestep)
                v=velocities(m.env)['objects']['piece_1']
                gravity=float(np.linalg.norm(m.env.sim.model._model.opt.gravity))
                predicted=error['position_m']+np.linalg.norm(v['linear'])*dt+.5*gravity*dt*dt
                rotation=error['rotation_rad']+np.linalg.norm(v['angular'])*dt
                age=float(state['sim_time']-self.last_load_time) if self.last_load_time is not None else float('inf')
                bridge=bool(m.manager_cfg.get('baseline_mode',False) and m.pending_release_valid(state)
                    and not any(state['actual_grasps']['A'].values()) and age<=dt+1e-9
                    and predicted<=m.cfg['assembled_object_drift_m']
                    and rotation<=m.cfg['assembled_object_rotation_drift_rad'])
                if bridge:support={**support,'single_solver_interval_bridge':True,'unsupported_age_s':age,'next_substep_position_bound_m':float(predicted)}
                else:reason="completed_piece1_support_lost"
            else:self.last_load_time=float(state['sim_time'])
        if support['supported']:self.last_load_time=float(state['sim_time'])
        return dict(active=True,qualified=anchor is not None,violation=reason,
                    pose_error=error,relative_error=relative_error,support=support)

    def legal_task_motion(self,ai,action,state):
        """Allow demonstrated task contact; contact is not a free-space obstacle.

        OSC inputs are SETPOINT deltas, not instantaneous 50ms displacements.
        Use the existing fixed critically-damped OSC response for a nominal
        one-period check, then retain the actual 2ms safety monitor. This is not
        a calibrated contact-dynamics certificate or a recovery controller.
        """
        m=self.m;env=m.env
        if not m.manager_cfg.get("baseline_mode",False):return dict(allowed=False,reason="legacy_projection")
        if not m.completed[0] or not state["official_metrics"]["first_piece_assembled"]:
            return dict(allowed=False,reason="no_completed_first_assembly")
        assessment=m.contact_assessment(state)
        if not assessment["allowed"]:return dict(allowed=False,reason=assessment["reason"])
        from acoserl.osc_prediction import predict
        nominal=predict(env,ai,action);dp=nominal['displacement'];dw=nominal['angular_displacement']
        increment=Rotation.from_rotvec(dw).as_matrix()
        released_B=(ai==1 and not any(state["actual_grasps"]["B"].values())
                    and (m.completed[1] or m.valid_release(1,state)))
        if ai==0 or released_B:
            if any(state["actual_grasps"]["AB"[ai]].values()):return dict(allowed=False,reason="released_arm_still_holding")
            goal=state["poses"][ai]["position"]+dp
            ids=np.arange(m.points["arms"]["AB"[ai]]["first_release_action"],m.n_states)
            distance=np.linalg.norm(m.positions[ids,ai]-goal,axis=1);j=int(ids[np.argmin(distance)])
            rotation=float(Rotation.from_matrix(increment@state["poses"][ai]["rotation"]@m.rotations[j,ai].T).magnitude())
            okay=float(distance.min())<=m.cfg["exit_position_m"] and rotation<=m.cfg["exit_rotation_rad"]
        else:
            if not state["actual_grasps"]["B"]["piece_2"]:return dict(allowed=False,reason="B_not_holding_piece2")
            actual=state["objects"]["piece_2"];eef=state["poses"][1]["position"]
            goal=eef+dp+increment@(np.asarray(actual["position"])-eef)
            # Compare in base coordinates; reference search never changes either
            # arm's cursor or declares a grasp/assembly event.
            base=state["objects"]["base"];base_r=Rotation.from_quat(np.roll(base["quaternion_wxyz"],-1)).as_matrix()
            goal_rel=base_r.T@(goal-base["position"])
            ids=np.arange(m.points["arms"]["B"]["first_grasp"],m.n_states)
            centers=m.trace["object_positions"][ids,2]-m.trace["object_positions"][ids,0]
            br=Rotation.from_quat(np.roll(m.trace["object_quaternions"][ids,0],-1,axis=1)).as_matrix()
            rel=np.einsum("nij,nj->ni",br.transpose(0,2,1),centers)
            distance=np.linalg.norm(rel-goal_rel,axis=1);j=int(ids[np.argmin(distance)])
            actual_r=increment@Rotation.from_quat(np.roll(actual["quaternion_wxyz"],-1)).as_matrix()
            ref_r=Rotation.from_quat(np.roll(m.trace["object_quaternions"][j,2],-1)).as_matrix()
            ref_b=Rotation.from_quat(np.roll(m.trace["object_quaternions"][j,0],-1)).as_matrix()
            rotation=float(Rotation.from_matrix((base_r.T@actual_r)@(ref_b.T@ref_r).T).magnitude())
            okay=float(distance.min())<=m.cfg["active_object_exit_position_m"] and rotation<=m.cfg["active_object_exit_rotation_rad"]
        return dict(allowed=bool(okay),reason="reference_compatible_normal_task_motion" if okay else "outside_legal_task_corridor",
            diagnostic_reference=j,nominal_position_error_m=float(distance.min()),nominal_rotation_error_rad=rotation,
            model="fixed_OSC_one_period_nominal_not_contact_certificate")

    def project(self,action,state):
        """Closest bounded OSC command under measured near-contact inequalities.

        Contact normal inequalities forbid additional commanded penetration.
        A narrow tangential/rotation budget limits shear at established contact.
        It is an eligibility-limited geometry approximation, NOT a calibrated
        error bound; real tests must qualify it before production activation.
        """
        m=self.m;env=m.env;model=env.sim.model;data=env.sim.data
        final=np.asarray(action,dtype=float).copy();checks=self.check(state)
        report=dict(applied=False,reason=[],constraints=[],protection=checks,
                    prediction="linearized_contact_geometry_not_dynamics_certificate")
        if checks["violation"]:return None,{**report,"refusal":checks["violation"]}
        if not checks["active"]:return final,report
        if self.anchor is None:
            v=velocities(env)["objects"]["piece_1"]
            if (not checks["support"]["supported"]
                    or np.linalg.norm(v["linear"])>m.cfg["placed_object_linear_speed_m_s"]
                    or np.linalg.norm(v["angular"])>m.cfg["placed_object_angular_speed_rad_s"]):
                return None,{**report,"refusal":"first_piece_protection_not_stably_qualified"}
            # Provisional geometry constraints remain active while three genuine
            # consecutive control observations establish stable qualification.
            # This is never sufficient to authorize recovery or handoff.
            report["provisional_qualification"]=True
        if any(state["actual_grasps"]["A"].values()):
            return None,{**report,"refusal":"protected_piece_A_grasp_domain_unqualified"}
        fixed=self.object_geoms["piece_1"]+self.object_geoms["base"]
        for ai,arm in enumerate(("A","B")):
            legal=self.legal_task_motion(ai,action,state)
            report.setdefault("normal_task_motion",[]).append({"arm":arm,**legal})
            if legal["allowed"]:continue
            moving=[gid for gid,owner in m.contacts.owners.items() if owner==arm]
            moving += [gid for obj,held in state["actual_grasps"][arm].items()
                       if held for gid in self.object_geoms[obj]]
            ctl=env.robots[ai].part_controllers["right"]
            origin=np.asarray(ctl.origin_ori);scales=np.asarray(ctl.output_max)
            pairs=[]
            for g in moving:
                ids=np.asarray(fixed,dtype=int)
                distance=np.linalg.norm(data.geom_xpos[ids]-data.geom_xpos[g],axis=1)
                lower=distance-model.geom_rbound[ids]-model.geom_rbound[g]
                for j in np.flatnonzero(lower<.008):
                    h=int(ids[j])
                    if not ((int(model.geom_contype[g])&int(model.geom_conaffinity[h]))
                            or (int(model.geom_contype[h])&int(model.geom_conaffinity[g]))):continue
                    pairs.append((float(lower[j]),h,g))
            rows=[];lower_bounds=[];minimum=.008
            for _,fixed_gid,moving_gid in sorted(pairs)[:192]:
                points=np.zeros(6)
                gap=float(mujoco.mj_geomDistance(model._model,data._data,
                                                fixed_gid,moving_gid,.008,points))
                if gap>=.008:continue
                difference=points[3:]-points[:3];norm=np.linalg.norm(difference)
                if norm<1e-9:continue
                # Signed gap means penetration reverses the nearest-point vector.
                normal=difference/norm*(1 if gap>=0 else -1)
                radius=points[3:]-state["poses"][ai]["position"]
                jac=np.concatenate((normal@origin*scales[:3],np.cross(radius,normal)@origin*scales[3:]))
                rows.append(jac);lower_bounds.append(-max(0.,gap-.0005))
                minimum=min(minimum,gap)
                if gap<.001:
                    # Preserve normal separation while limiting tangential shear.
                    # The demonstrated insertion-unload coordinate is world +Z.
                    # Do not prohibit axial unloading as "tangential shear" at
                    # a vertical side wall. Only lateral shear is constrained.
                    vertical=np.array([0.,0.,1.]);t=np.cross(vertical,normal)
                    tangents=(t/np.linalg.norm(t),) if np.linalg.norm(t)>1e-6 else (np.array([1.,0.,0.]),np.array([0.,1.,0.]))
                    for tangent in tangents:
                        j=np.concatenate((tangent@origin*scales[:3],np.cross(radius,tangent)@origin*scales[3:]))
                        rows.extend((j,-j));lower_bounds.extend((-.0001,-.0001))
            if not rows:continue
            sl=slice(7*ai,7*ai+6);original=final[sl].copy()
            mat=np.asarray(rows);lb=np.asarray(lower_bounds)
            if minimum<.002:
                for k in range(3,6):
                    j=np.zeros(6);j[k]=1
                    rows.extend((j,-j));lower_bounds.extend((-.003/scales[k],-.003/scales[k]))
                mat=np.asarray(rows);lb=np.asarray(lower_bounds)
            if hasattr(m,"action_bounds"):
                bounds=list(zip(m.action_bounds[0][sl],m.action_bounds[1][sl]))
            else:bounds=[(-1.,1.)]*6
            result=minimize(lambda x:.5*np.sum((x-original)**2),np.clip(original,*np.asarray(bounds).T),
                jac=lambda x:x-original,method="SLSQP",bounds=bounds,
                constraints=[LinearConstraint(mat,lb,np.inf)],options=dict(ftol=1e-12,maxiter=80))
            if not result.success or np.min(mat@result.x-lb)<-1e-8:
                return None,{**report,"refusal":"no_feasible_contact_projection", "solver":result.message}
            final[sl]=result.x
            report["constraints"].append(dict(arm=arm,count=len(lb),minimum_gap_m=minimum,
                residual_min=float(np.min(mat@result.x-lb)),command_before=original.tolist(),command_after=result.x.tolist()))
        report["applied"]=bool(np.max(abs(final-action))>1e-7)
        if report["applied"]:report["reason"].append("preserve_completed_piece1_contact_constraints")
        report["magnitude"]=float(np.linalg.norm(final-action))
        return final,report
