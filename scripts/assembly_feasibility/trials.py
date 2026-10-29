"""Predeclared local-recovery feasibility experiment, not a learned policy."""
from common import *
import argparse,copy,h5py
CFG=json.loads((ROOT/'configs/assembly_feasibility/protocol.json').read_text())
SURVEY=json.loads((OUT/'contact_survey.json').read_text())['records']

def reference(index):
    with h5py.File(REFERENCE,'r') as f:
        return [{'position':f[arm+'/eef_position'][index],
                 'rotation':Rotation.from_quat(f[arm+'/eef_quaternion_xyzw'][index]).as_matrix()} for arm in ['trajectory_A','trajectory_B']]

def object_error(a,b):
    qa=np.asarray(a['quaternion_wxyz']);qb=np.asarray(b['quaternion_wxyz'])
    angle=2*np.arccos(np.clip(abs(np.dot(qa,qb)),0,1))
    return {'position_m':float(np.linalg.norm(np.asarray(a['position'])-b['position'])),'rotation_rad':float(angle)}

def constrained(contact,active,grasps):
    held=[n for n,v in grasps[active].items() if v]
    for edge in contact['edges']:
        a,b=edge['a'],edge['b']
        if any(n in (a,b) and other in (a,b) for n in held for other in ['base','piece_1','piece_2'] if other!=n):
            return 'active_held_object_in_assembly_contact'
        if active in (a,b) and (b if a==active else a) in ['base']:
            return 'active_gripper_contacts_base'
    if contact['dynamic_A_B_coupled']:return 'dynamic_contact_chain_connects_both_arms'
    return None

def run_one(anchor,active,offset,repetition,mode='recovery',tag='main',control_recovery_steps=None):
    ai=ARMS.index(active);pi=1-ai;passive=ARMS[pi]
    trial_id=f'{tag}_{mode}_{anchor}_{active}_{offset[0]:+.3f}_{offset[1]:+.3f}_{offset[2]:+.3f}_r{repetition}'
    output=OUT/tag/(trial_id+'.json')
    if output.exists():raise FileExistsError(output)
    env,demo,runner=setup();log=[];result={'trial_id':trial_id,'anchor':anchor,'active_arm':active,'passive_arm':passive,
       'offset_world_m':offset,'repetition':repetition,'mode':mode,'protocol':CFG,'recovery_success':False,'status':'running'}
    progress=[anchor,anchor]
    def step(action,phase):
        try:
            r=runner.step(action)
        except SafetyStop as error:
            log.append({'phase':phase,'action':action,'indices':progress.copy(),'stopped_mid_control_interval':str(error),**snapshot(env),'contacts':runner.contact.read()})
            raise
        log.append({'phase':phase,'action':action,'indices':progress.copy(),**r})
        return r
    try:
        # Prefix is executed, not restored from an intermediate recorded state.
        for i,action in enumerate(demo['actions'][:anchor]):
            progress[:]=[i+1,i+1];step(action,'original_action_prefix')
        initial=snapshot(env);gate=constrained(runner.contact.read(),active,initial['actual_grasps'])
        result['anchor_snapshot']=initial;result['initial_gate_reason']=gate
        if gate:
            result['status']='rejected_before_perturbation';result['failure_reason']=gate
            return result
        goals=reference(anchor)
        grips=[float(demo['actions'][anchor-1,6]),float(demo['actions'][anchor-1,13])]
        def safety(contact):
            if contact['dynamic_A_B_coupled']:return 'coupled_dynamic_contact_during_local_control'
            if contact['max_robot_contact_force_N']>CFG['robot_contact_force_stop_N']:return 'robot_contact_force_limit'
            if mode in ['recovery','hold_control']:
                reason=constrained(contact,active,initial['actual_grasps'])
                if reason:return reason
            return None
        runner.safety=safety
        def command(targets):
            return servo(env,targets,grips,max_position_step=CFG['max_command_position_step_m'],max_rotation_step=CFG['max_command_rotation_step_rad'],position_gain=CFG['position_feedback_gain'],rotation_gain=CFG['rotation_feedback_gain'])
        settle_stable=0
        for i in range(CFG['settle_steps']):
            r=step(command(goals),'settle_pose_hold')
            for arm in ARMS:
                for obj,held in initial['actual_grasps'][arm].items():
                    if held and not r['actual_grasps'][arm][obj]:raise SafetyStop('grasp_lost_during_settle_'+arm+'_'+obj)
            if any(value and not r['official_metrics'][key] for key,value in initial['official_metrics'].items()):raise SafetyStop('progress_lost_during_settle')
            okay=all(errors(r['poses'][j],goals[j])['position_m']<=CFG['recovery_position_tolerance_m'] and errors(r['poses'][j],goals[j])['rotation_rad']<=CFG['recovery_rotation_tolerance_rad'] and np.linalg.norm(env.sim.data.qvel[env.robots[j]._ref_joint_vel_indexes])<CFG['settle_joint_velocity_l2_rad_s'] for j in range(2))
            settle_stable=settle_stable+1 if okay else 0
            if settle_stable>=CFG['settle_pose_stable_steps']:break
        if settle_stable<CFG['settle_pose_stable_steps']:raise SafetyStop('initial_hold_not_settled')
        result['settle_steps_used']=i+1
        baseline=snapshot(env);result['hold_baseline']=baseline
        # Reject an initial hold that already lost a demonstrated grasp or assembly milestone.
        for arm in ARMS:
            for obj,held in initial['actual_grasps'][arm].items():
                if held and not baseline['actual_grasps'][arm][obj]:raise SafetyStop('grasp_lost_during_initial_hold_'+arm+'_'+obj)
        for key,value in initial['official_metrics'].items():
            if value and not baseline['official_metrics'][key]:raise SafetyStop('progress_lost_during_initial_hold_'+key)
        def retention(r):
            passive_error=errors(r['poses'][pi],baseline['poses'][pi])
            if passive_error['position_m']>CFG['passive_position_bound_m']:return 'passive_position_drift'
            if passive_error['rotation_rad']>CFG['passive_rotation_bound_rad']:return 'passive_orientation_drift'
            if np.linalg.norm(np.asarray(r['qpos'])[env.robots[pi]._ref_joint_pos_indexes]-np.asarray(baseline['qpos'])[env.robots[pi]._ref_joint_pos_indexes])>CFG['passive_joint_l2_bound_rad']:return 'passive_joint_drift'
            for arm in ARMS:
                for obj,held in baseline['actual_grasps'][arm].items():
                    if held and not r['actual_grasps'][arm][obj]:return 'grasp_lost_'+arm+'_'+obj
            for key,value in baseline['official_metrics'].items():
                if value and not r['official_metrics'][key]:return 'official_progress_lost_'+key
            for obj,held in baseline['actual_grasps'][passive].items():
                if held:
                    delta=object_error(r['objects'][obj],baseline['objects'][obj])
                    if delta['position_m']>CFG['passive_object_position_bound_m'] or delta['rotation_rad']>CFG['passive_object_rotation_bound_rad']:return 'passive_object_progress_changed_'+obj
            if baseline['official_metrics']['first_piece_assembled']:
                if object_error(r['objects']['piece_1'],baseline['objects']['piece_1'])['position_m']>CFG['assembled_object_position_bound_m']:return 'assembled_piece_1_displaced'
            if object_error(r['objects']['base'],baseline['objects']['base'])['position_m']>CFG['base_position_bound_m']:return 'base_displaced'
            return None
        perturbed=copy.deepcopy(goals)
        if mode=='recovery':perturbed[ai]['position']=perturbed[ai]['position']+np.asarray(offset)
        for i in range(CFG['perturb_steps']):
            r=step(command(perturbed),'perturb' if mode=='recovery' else 'hold_control')
            reason=retention(r)
            if reason:raise SafetyStop(reason)
        injected=errors(poses(env)[ai],goals[ai]);result['error_after_perturbation']=injected
        if mode=='recovery' and injected['position_m']<CFG['minimum_effective_perturbation_m']:
            raise SafetyStop('ineffective_perturbation')
        stable=0
        for i in range(CFG['max_recovery_steps']):
            r=step(command(goals),'recover' if mode=='recovery' else 'hold_control')
            reason=retention(r)
            if reason:raise SafetyStop(reason)
            error=errors(r['poses'][ai],goals[ai])
            okay=error['position_m']<=CFG['recovery_position_tolerance_m'] and error['rotation_rad']<=CFG['recovery_rotation_tolerance_rad']
            stable=stable+1 if okay else 0
            reached=(stable>=CFG['stable_steps_required'])
            matched_time=(control_recovery_steps is None or i+1>=control_recovery_steps)
            if reached and matched_time:
                result['recovery_steps']=i+1;result['recovery_success']=True;break
        result['final_reference_error']=errors(poses(env)[ai],goals[ai])
        if not result['recovery_success']:raise SafetyStop('recovery_timeout')
        result['local_end_snapshot']=snapshot(env)
        result['status']='locally_recovered';result['independent_indices_during_recovery']=progress.copy()
        # Resume original joint suffix as a distinct functional progress check.
        runner.safety=lambda contact: 'coupled_dynamic_contact_on_resume' if contact['dynamic_A_B_coupled'] else 'robot_force_on_resume' if contact['max_robot_contact_force_N']>CFG['robot_contact_force_stop_N'] else None
        for i in range(anchor,len(demo['actions'])):
            progress[:]=[i+1,i+1];step(demo['actions'][i],'resume_original_joint_suffix')
        result['task_success_after_resume']=bool(env._check_success());result['status']='completed'
    except SafetyStop as error:
        result['status']='stopped';result['failure_reason']=str(error)
    except Exception as error:
        result['status']='error';result['failure_reason']=repr(error)
        raise
    finally:
        result['direct_state_setter_attempts_after_initialization']=runner.physical_writes_attempted
        result['initial_state_restores']=1
        result['trace']=log
        if 'hold_baseline' in result:
            b=result['hold_baseline'];controlled=[r for r in log if r['phase'] in ['perturb','recover','hold_control']]
            result['max_passive_position_drift_m']=max((errors(r['poses'][pi],b['poses'][pi])['position_m'] for r in controlled),default=0)
            result['max_passive_orientation_drift_rad']=max((errors(r['poses'][pi],b['poses'][pi])['rotation_rad'] for r in controlled),default=0)
            result['max_active_reference_error_m']=max((errors(r['poses'][ai],goals[ai])['position_m'] for r in controlled),default=0)
            result['passive_objects_final_change']={obj:object_error(result.get('local_end_snapshot',snapshot(env))['objects'][obj],b['objects'][obj]) for obj,held in b['actual_grasps'][passive].items() if held}
        save(output,result);env.close()
        print(trial_id,result['status'],result.get('failure_reason'),result.get('recovery_steps'),result.get('task_success_after_resume'),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--tag',default='main');a=p.parse_args()
    if a.pilot:
        for arm in ARMS:run_one(80,arm,CFG['offsets_world_m'][0],0,tag=a.tag)
    else:
        for anchor in CFG['independent_stage_anchors']:
            for arm in ARMS:
                for offset in CFG['offsets_world_m']:
                    for repetition in range(CFG['repetitions']):run_one(anchor,arm,offset,repetition,tag=a.tag)
