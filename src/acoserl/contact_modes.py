"""Task-contact classification shared by control-step and substep safety."""
import numpy as np
from scipy.spatial.transform import Rotation

def assess(manager,state,contact):
    pairs={tuple(sorted((e["a"],e["b"]))) for e in contact["edges"]}
    grasp=state["actual_grasps"]
    wrong=[(a,o) for a in ("A","B") for o,h in grasp[a].items()
           if h and o!={"A":"piece_1","B":"piece_2"}[a]]
    reason=None
    if contact["max_robot_contact_force_N"]>manager.cfg["robot_force_stop_N"]:reason="robot_force_limit"
    elif wrong:reason="unexpected_grasp_relation"
    elif pairs.intersection({("A","B"),("A","static"),("B","static")}):reason="robot_environment_collision"
    for ai,arm in enumerate(("A","B")):
        command=np.asarray(manager.env.robots[ai].gripper["right"].current_action)
        opening=command.shape==(2,) and command[0]>.5
        legal_release=manager.completed[ai] or (ai==0 and state["official_metrics"]["first_piece_assembled"])
        if ai==1 and manager.last_decision is not None:
            legal_release=legal_release or manager.valid_release(ai,state)
        touching=any(arm in pair and bool(set(pair)&{"base","piece_1","piece_2"}) for pair in pairs)
        if touching and opening and not any(grasp[arm].values()) and not legal_release:
            reason="open_gripper_unplanned_object_contact_"+arm
    if reason:return dict(category="Unsafe_Collision",reason=reason,allowed=False,edges=sorted(pairs))
    # Raw geometry and real grasp edges, not just the >0.1N graph.
    coop=manager.collaboration(state,contact)
    if coop["coupled"]:
        return dict(category="Unknown_Coupling",reason="unqualified_dynamic_contact_chain",
                    allowed=False,edges=coop["contact_edges"],cooperative_controller_verified=False)
    task_pairs=pairs.intersection({("base","piece_1"),("base","piece_2"),("piece_1","piece_2")})
    if task_pairs.intersection({("base","piece_2"),("piece_1","piece_2")}) and not (manager.completed[0] and state["official_metrics"]["first_piece_assembled"]):
        return dict(category="Unknown_Coupling",reason="second_contact_before_first_assembly_complete",allowed=False,edges=coop["contact_edges"])
    relative={}
    for a,b in task_pairs:
        actual=np.asarray(state["objects"][a]["position"])-state["objects"][b]["position"]
        ia=manager.reference_objects.index(a);ib=manager.reference_objects.index(b)
        positions=manager.trace["object_positions"][:,ia]-manager.trace["object_positions"][:,ib]
        match=int(np.argmin(np.linalg.norm(positions-actual,axis=1)))
        qa=state["objects"][a]["quaternion_wxyz"];qb=state["objects"][b]["quaternion_wxyz"]
        actual_rot=Rotation.from_quat(np.roll(qb,-1)).inv()*Rotation.from_quat(np.roll(qa,-1))
        ra=Rotation.from_quat(np.roll(manager.trace["object_quaternions"][match,ia],-1))
        rb=Rotation.from_quat(np.roll(manager.trace["object_quaternions"][match,ib],-1))
        rel_error=(actual_rot.inv()*(rb.inv()*ra)).magnitude()
        error=float(np.linalg.norm(positions[match]-actual))
        relative[a+":"+b]=dict(position_error_m=error,rotation_error_rad=float(rel_error),diagnostic_reference=match)
        if error>.012 or rel_error>.15:
            return dict(category="Unknown_Coupling",reason="object_contact_relative_pose_unqualified",
                        allowed=False,relative=relative,edges=coop["contact_edges"])
    return dict(category="Legal_Object_Contact" if task_pairs else "Independent",
                reason="reference_compatible_object_contact" if task_pairs else "separate_grasps",
                allowed=True,relative=relative,edges=coop["contact_edges"],
                actual_grasps=grasp,first_piece_completed=bool(state["official_metrics"]["first_piece_assembled"]))
