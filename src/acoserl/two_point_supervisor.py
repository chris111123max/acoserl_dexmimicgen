"""Paper section 3.3 mechanisms, with conservative two-arm contact eligibility.

No simulator state setter is used here. Every replay action is returned to the
ordinary environment step and collector, including Return_P0 and Replay_P0_P1.
"""
import copy,json,sys
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from acoserl.trajectories import ReferenceCursor,SlidingReferenceWindow
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assembly_feasibility'))
from common import snapshot,errors,ARMS
from acoserl.object_geometry import object_error
ROOT=Path(__file__).resolve().parents[2]


def paper_direction(positions,cpoint,current,pipoint,epsilon=1e-6):
    """Tangent at GLOBAL cpoint, not pipoint. Stationary tangent cancels sliding."""
    j=min(cpoint+1,len(positions)-1);i=cpoint if j>cpoint else max(0,cpoint-1)
    tangent=positions[j]-positions[i];toward=positions[pipoint]-current
    n=float(np.linalg.norm(tangent));m=float(np.linalg.norm(toward))
    if n<epsilon:return False,None,'undefined_stationary_tangent'
    if m<epsilon:return False,None,'already_at_pipoint'
    cosine=float(np.clip(np.dot(tangent,toward)/(n*m),-1,1))
    return cosine>=0,cosine,'forward' if cosine>=0 else 'past_pipoint_cancelled'


class TwoPointMixin:
    def __init__(self,env,config=None,intervention_enabled=None):
        self.points=json.loads((ROOT/'configs/bimanual_autoserl/recovery_points.json').read_text())
        self.trace=np.load(ROOT/self.points['reference_path'])
        self.positions=self.trace['positions'];self.rotations=self.trace['rotations'];self.n_states=len(self.positions)
        assert self.n_states==len(self.trace['actions'])+1
        self.ref_grasps=self.trace['grasp'];self.reference_objects=['base','piece_1','piece_2']
        super().__init__(env,config,intervention_enabled)

    def control_active(self):
        return self.mode.startswith(('Recover_', 'Guide_')) and self.active is not None

    def reset(self,initial_progress=(0,0)):
        super().reset(initial_progress)
        self.cursors=[ReferenceCursor(self.n_states,int(i)) for i in initial_progress]
        self.windows=[SlidingReferenceWindow(self.points['arms'][a]['window_length'],self.n_states,int(i)) for a,i in zip(ARMS,initial_progress)]
        self.history=[deque(maxlen=self.cfg['stagnation_window']+1) for _ in ARMS]
        self.grasp_streak=[0,0];self.was_grasped=[False,False];self.completed=[False,False]
        self.intervention_control_steps=0;self.recovery_stage=None;self.plan=[];self.replay_index=None;self.grasp_stability=deque(maxlen=self.cfg['stable_grasp_window_steps'])
        self.initial_snapshot=snapshot(self.env);self.protected_first=None;self.release_wait=[0,0];self.support_age=[999,999]

    def pose(self,ai,idx):
        assert 0<=idx<self.n_states
        return {'position':self.positions[idx,ai].copy(),'rotation':self.rotations[idx,ai].copy()}

    def ref_object(self,name,idx):
        oi=self.reference_objects.index(name)
        return {'position':self.trace['object_positions'][idx,oi].copy(),'quaternion_wxyz':self.trace['object_quaternions'][idx,oi].copy()}

    def gripper_command(self,ai,state):
        if 'gripper_commands' in state:return np.asarray(state['gripper_commands'][ai])
        return np.asarray(self.env.robots[ai].gripper['right'].current_action)

    def legal_bounds(self,ai,state):
        a=ARMS[ai];p=self.points['arms'][a];obj=p['object']
        held=state['actual_grasps'][a][obj]
        if self.completed[ai]:return p['p1'],self.n_states-1,'released_completed'
        if held and self.grasp_streak[ai]>=3:
            last=int(np.flatnonzero(self.ref_grasps[:,ai,self.reference_objects.index(obj)])[ -1])
            command=self.gripper_command(ai,state)
            if command.shape==(2,) and command[0]<=-.999:last=min(last,p['first_release_action'])
            boundary=self.points['assembly'][ai]['first_contact']
            edges=self.contacts.read()['edges']
            interacting=any(obj in {e['a'],e['b']} and bool(({e['a'],e['b']}-{obj}) & {'base','piece_1'}) for e in edges)
            event_valid=(state['official_metrics']['first_piece_assembled'] or interacting) if ai==0 else (self.completed[0] and (interacting or state['official_metrics']['task']))
            if boundary is not None and not event_valid:last=min(last,boundary-1)
            return p['first_grasp'],last,'carry_or_assembly'
        cap=p['first_contact']-1
        command=self.gripper_command(ai,state)
        if command.shape==(2,) and command[0]>=.999:cap=min(cap,p['first_close_action'])
        return 0,cap,'pregrasp'

    def valid_release(self,ai,state):
        # Measured demo release precedes strict task success by one control step.
        # Only allow a short settle with recent real support, final-pose agreement,
        # explicit opening action and a completed first assembly; never mark success.
        if ai!=1 or not self.completed[0] or self.release_wait[ai]>self.cfg['release_settle_max_steps']:return False
        if self.support_age[ai]>4 or self.windows[ai].end<self.points['assembly'][1]['first_contact']:return False
        if self.last_decision['policy_action'][13]>=0:return False
        oe=object_error(state['objects']['piece_2'],self.ref_object('piece_2',self.n_states-1))
        return oe['position_m']<=.006 and oe['rotation_rad']<=.1

    def physical_danger(self,state,contact):
        if not all(np.isfinite(p['position']).all() for p in state['poses']):return 'nonfinite_physics'
        if contact['max_robot_contact_force_N']>self.cfg['robot_force_stop_N']:return 'robot_force_limit'
        if contact['dynamic_A_B_coupled']:return 'unverified_dynamic_A_B_coupling'
        for ai,arm in enumerate(ARMS):
            p=np.asarray(state['poses'][ai]['position']);lo=self.positions[:,ai].min(0)-self.cfg['free_workspace_margin_m'];hi=self.positions[:,ai].max(0)+self.cfg['free_workspace_margin_m']
            if np.any(p<lo) or np.any(p>hi):return 'workspace_exit_'+arm
            if any(held for o,held in state['actual_grasps'][arm].items() if o!=self.points['arms'][arm]['object']):return 'unexpected_grasp_'+arm
            if self.was_grasped[ai] and not any(state['actual_grasps'][arm].values()) and not self.completed[ai] and not self.valid_release(ai,state):return 'unplanned_grasp_loss_'+arm
        for edge in contact['edges']:
            pair={edge['a'],edge['b']}
            if pair in [{'A','static'},{'B','static'},{'A','B'}]:return 'robot_environment_collision'
            for ai,arm in enumerate(ARMS):
                if arm not in pair or not pair.intersection({'base','piece_1','piece_2'}):continue
                command=np.asarray(self.env.robots[ai].gripper['right'].current_action)
                opening=command.shape==(2,) and command[0]>.5
                if opening and not any(state['actual_grasps'][arm].values()) and not self.completed[ai] and not self.valid_release(ai,state):return 'open_gripper_unplanned_object_contact_'+arm
        if self.protected_first is not None:
            if not state['official_metrics']['first_piece_assembled']:return 'completed_piece1_progress_lost'
            oe=object_error(state['objects']['piece_1'],self.protected_first)
            if oe['position_m']>self.cfg['assembled_object_drift_m'] or oe['rotation_rad']>self.cfg['assembled_object_rotation_drift_rad']:return 'completed_piece1_displaced'
        return None

    def window_measure(self,ai,state,record_history=True):
        lo,hi,stage=self.legal_bounds(ai,state);w=self.windows[ai]
        if lo>w.floor:w.event(lo)
        ids=w.indices(hi)
        # Grasp-state compatibility blocks nearest searches across irreversible events.
        if stage!='released_completed':ids=[j for j in ids if np.array_equal(self.ref_grasps[j,ai],[state['actual_grasps'][ARMS[ai]][o] for o in self.reference_objects])]
        held_objects=[o for o,h in state['actual_grasps'][ARMS[ai]].items() if h]
        for obj in held_objects:
            oi=self.reference_objects.index(obj);actual=state['objects'][obj];eef=state['poses'][ai]
            relative=eef['rotation'].T@(np.asarray(actual['position'])-eef['position'])
            relative_rotation=eef['rotation'].T@Rotation.from_quat(np.roll(actual['quaternion_wxyz'],-1)).as_matrix()
            legal=[]
            for j in ids:
                rel=self.rotations[j,ai].T@(self.trace['object_positions'][j,oi]-self.positions[j,ai])
                rotation=self.rotations[j,ai].T@Rotation.from_quat(np.roll(self.trace['object_quaternions'][j,oi],-1)).as_matrix()
                if np.linalg.norm(relative-rel)<=self.cfg['object_window_position_m'] and Rotation.from_matrix(relative_rotation@rotation.T).magnitude()<=self.cfg['object_window_rotation_rad']:legal.append(j)
            ids=legal
        distances=np.linalg.norm(self.positions[:,ai]-state['poses'][ai]['position'],axis=1)
        cpoint=int(np.argmin(distances))
        if record_history:self.history[ai].append(cpoint)
        pipoint=min(ids,key=lambda j:float(distances[j])) if ids else None
        h=self.history[ai];p=self.points['arms'][ARMS[ai]]
        stagnant=len(h)>self.cfg['stagnation_window'] and np.linalg.norm(self.positions[h[-1],ai]-self.positions[h[0],ai])<self.cfg['exit_position_m'] and p['p0']<=cpoint<=p['p1']
        okay,cosine,reason=paper_direction(self.positions[:,ai],cpoint,state['poses'][ai]['position'],pipoint) if pipoint is not None else (False,None,'no_legal_window')
        return {'arm':ARMS[ai],'window_start':ids[0] if ids else None,'window_end':w.end,'length':w.length,'stage':stage,'legal_end':hi,'pipoint':pipoint,'distance_m':float(distances[pipoint]) if pipoint is not None else None,'cpoint':cpoint,'direction_ok':okay,'direction_cosine':cosine,'direction_reason':reason,'stagnant':bool(stagnant)}

    def eligible(self,ai,state,contact,kind,target):
        arm=ARMS[ai];obj=self.points['arms'][arm]['object'];p=self.points['arms'][arm]
        if not p['validated'] and not self.cfg.get('development_probe',False):return False,'recovery_interval_not_validated'
        if self.completed[ai]:return False,'completed_arm_must_not_rewind'
        if contact['dynamic_A_B_coupled']:return False,'coupled_single_arm_recovery_refused'
        if any(state['actual_grasps'][arm].values()):return False,'held_object_long_return_not_validated'
        phase,reason=self.phase(state,contact)
        if phase in ['assembly_contact','coupled']:return False,'assembly_barrier:'+reason
        if any(arm in {e['a'],e['b']} for e in contact['edges']):return False,'active_arm_has_contact_no_free_return'
        oe=object_error(state['objects'][obj],self.ref_object(obj,p['p0']))
        if oe['position_m']>.003 or oe['rotation_rad']>.035:return False,'target_object_not_at_validated_pick_pose'
        # Ordinary sliding Guide is single-arm feedback, NOT an arm-freezing
        # P0 recovery. The other arm retains its Policy action and must pass
        # whole-joint action screening before either arm moves.
        if kind=='sliding':
            if target>=p['first_contact']:return False,'guide_target_crosses_contact_barrier'
            if self.contact_assessment(state,contact)['category']!='Independent':
                return False,'guide_requires_independent_free_space'
            start=np.asarray(state['poses'][ai]['position'])
            goal=np.asarray(self.pose(ai,target)['position'])
            other=np.asarray(state['poses'][1-ai]['position'])
            delta=goal-start
            alpha=np.clip(np.dot(other-start,delta)/max(float(np.dot(delta,delta)),1e-12),0.,1.)
            if np.linalg.norm(start+alpha*delta-other)<.12:
                return False,'guide_tcp_corridor_near_other_arm'
            return True,'eligible_single_arm_free_guide'
        entry=self.passive_hold_entry(ai,state);self.last_decision['passive_hold_entry']=entry
        if not entry['allowed']:return False,entry['reason']
        # Free pregrasp captures were observed to drift 2.13mm when the local
        # damping screen estimated only 1.57mm. Add a full control interval
        # here; keep the separately qualified carry/braking controller intact.
        extra=max(0.,1./float(self.env.control_freq)-entry['physics_latency_s'])
        pos=entry['predicted_peak_position_m']+entry['linear_speed_m_s']*extra
        rot=entry['predicted_peak_rotation_rad']+entry['angular_speed_rad_s']*extra
        held=[o for o,h in state['actual_grasps'][ARMS[1-ai]].items() if h]
        bounds={o:pos+float(np.linalg.norm(np.asarray(state['objects'][o]['position'])-state['poses'][1-ai]['position']))*rot for o in held}
        okay=pos<=self.cfg['passive_position_drift_m'] and rot<=self.cfg['passive_rotation_drift_rad'] and all(v<=self.cfg['passive_object_position_drift_m'] for v in bounds.values()) and (not held or rot<=self.cfg['passive_object_rotation_drift_rad'])
        entry.update(pregrasp_control_interval_allowed=bool(okay),pregrasp_position_bound_m=float(pos),pregrasp_rotation_bound_rad=float(rot),pregrasp_object_bounds_m=bounds)
        if not okay:return False,'pregrasp_passive_hold_not_quasistatic'
        if target>=p['first_contact']:return False,'sliding_target_crosses_contact_barrier'
        current=state['poses'][ai]['position'];goal=self.pose(ai,target)['position']
        height=max(current[2],goal[2],float(self.trace['object_positions'][0,:,2].max())+.10)
        path=[current,np.array([current[0],current[1],height]),np.array([goal[0],goal[1],height]),goal]
        obstacles=[state['poses'][1-ai]['position']]+[np.asarray(state['objects'][o]['position']) for o,h in state['actual_grasps'][ARMS[1-ai]].items() if h]
        for start,end in zip(path,path[1:]):
            for center in obstacles:
                direction=end-start;alpha=np.clip(np.dot(center-start,direction)/max(np.dot(direction,direction),1e-12),0,1)
                if np.linalg.norm(start+alpha*direction-center)<.08:return False,'return_corridor_near_passive_arm_or_object'
        return True,'eligible_free_pregrasp'

    def start_recovery(self,ai,state,kind,target):
        self.active=ai;self.baseline=copy.deepcopy(state);self.mode='Recover_'+ARMS[ai];self.age=0;self.stable=0
        self.episode_interventions+=1;self.recovery_kind=kind;self.recovery_index=target;self.targets=copy.deepcopy(state['poses'])
        self.grips=[1. if any(state['actual_grasps'][a].values()) else -1. for a in ARMS]
        self.grips[ai]=-1.
        # Empty-hand, object-consistent, elevated L path; not an enlarged local capture ball.
        goal=self.pose(ai,target);current=state['poses'][ai];height=max(current['position'][2],goal['position'][2],float(self.trace['object_positions'][0,:,2].max())+.10)
        p_up=copy.deepcopy(current);p_up['position'][2]=height
        p_over=copy.deepcopy(goal);p_over['position'][2]=height
        if kind=='sliding':
            # Track the legal window target directly. The normal second-arm
            # Policy proposal will be spliced back in at every control step.
            self.plan=[];self.targets[ai]=goal
            self.recovery_stage='Sliding_Guide';self.replay_index=None
        else:
            self.plan=[p_up,p_over,goal];self.recovery_stage='Return_P0';self.replay_index=None
        self.events.append({'event':'start','step':self.steps,'arm':ARMS[ai],'kind':kind,'target':target,'plan':[p['position'].tolist() for p in self.plan]})

    def choose(self,policy_action):
        policy_action=np.asarray(policy_action,dtype=float);assert policy_action.shape==(14,) and np.isfinite(policy_action).all()
        self.steps+=1;state=snapshot(self.env);contact=self.contacts.read()
        self.last_decision={'mode_before':self.mode,'indices':self.indices(),'policy_action':policy_action.tolist(),'handback':False}
        danger=self.physical_danger(state,contact)
        if danger:self.stop(danger)
        if self.mode=='Safe_Stop':return self.refuse()
        enabled=self.cfg['intervention_enabled'] and not self.auto_disabled
        if self.mode=='Normal' and enabled:
            measures=[self.window_measure(ai,state) for ai in range(2)];self.last_decision['per_arm']=measures
            triggers=[]
            for ai,m in enumerate(measures):
                self.cooldown[ai]=max(0,self.cooldown[ai]-1)
                if self.cooldown[ai]:continue
                if m['stagnant'] and not any(state['actual_grasps'][ARMS[ai]].values()):triggers.append((ai,'safety_recovery',self.points['arms'][ARMS[ai]]['p0']))
                elif m['distance_m'] is not None and m['distance_m']>self.cfg['trigger_position_m'] and m['direction_ok']:triggers.append((ai,'sliding',m['pipoint']))
            self.last_decision['deferred']=[]
            for ai,kind,target in sorted(triggers,key=lambda x:0 if x[1]=='safety_recovery' else 1):
                allowed,reason=self.eligible(ai,state,contact,kind,target)
                if allowed:self.start_recovery(ai,state,kind,target);break
                self.last_decision['deferred'].append({'arm':ARMS[ai],'kind':kind,'reason':reason})
        if self.control_active():return self.recovery_action(state)
        self.last_decision.update(mode='Normal',enabled=bool(enabled),execute=True,indices=self.indices(),executed_action=policy_action.tolist())
        return policy_action.copy(),self.last_decision

    def refuse(self):
        self.last_decision.update(mode='Safe_Stop',execute=False,reason=self.stop_reason)
        return None,self.last_decision

    def recovery_action(self,state):
        ai=self.active
        if self.recovery_stage=='Sliding_Guide':
            # Guide owns ONLY the active arm. In particular, do not replace a
            # moving, independently controlled arm with a captured static pose.
            action=np.asarray(self.last_decision['policy_action'],dtype=float).copy()
            sl=slice(ai*7,ai*7+7)
            guided=self.command(self.targets,self.grips)
            action[sl]=guided[sl]
            self.last_decision['per_arm_control']=[
                'Guide' if i==ai else 'Policy' for i in range(2)]
            # V2/V3 action checks are applied to the actual composed 14D
            # command. Unsafe active-guide or passive-policy proposals fail
            # closed; never confuse this with qualification for static Hold.
            if self.contact_assessment(state,self.contacts.read())['category']!='Independent':
                self.stop('guide_contact_mode_changed')
                return self.refuse()
            risks=self.action_risks(action,state)
            if any(r['risk'] for r in risks):
                self.last_decision['guide_composed_action_risks']=risks
                self.stop('guide_composed_action_risk_unqualified')
                return self.refuse()
        elif self.recovery_stage=='Replay_P0_P1':
            # Literal demonstrated action at t: s_t -> s_(t+1). Other arm holds its captured pose.
            action=self.command(self.targets,self.grips);sl=slice(ai*7,ai*7+7)
            nominal=self.trace['actions'][self.replay_index,sl].copy()
            correction_targets=copy.deepcopy(self.targets);correction_targets[ai]=self.pose(ai,self.replay_index)
            correction=self.command(correction_targets,self.grips)[sl]
            nominal[:6]=np.clip(nominal[:6]+correction[:6],-1,1)
            action[sl]=nominal
        else:
            self.targets[ai]=self.plan[0];action=self.command(self.targets,self.grips)
        if hasattr(self,'action_bounds'):
            bounded=np.clip(action,*self.action_bounds);self.last_decision['controller_saturated_dimensions']=np.flatnonzero(np.abs(action-bounded)>1e-8).tolist();action=bounded
        self.last_decision.update(mode=self.mode,active_arm=ARMS[ai],execute=True,recovery_step=self.age+1,recovery_kind=self.recovery_kind,recovery_stage=self.recovery_stage,replay_index=self.replay_index,indices=self.indices(),executed_action=action.tolist())
        return action,self.last_decision

    def retention(self,state):
        ai=self.active;pi=1-ai;baseline=self.baseline
        pe=errors(state['poses'][pi],baseline['poses'][pi])
        if pe['position_m']>self.cfg['passive_position_drift_m'] or pe['rotation_rad']>self.cfg['passive_rotation_drift_rad']:return 'passive_pose_changed',pe
        if state['actual_grasps'][ARMS[pi]]!=baseline['actual_grasps'][ARMS[pi]]:return 'passive_grasp_changed',pe
        for obj,held in baseline['actual_grasps'][ARMS[pi]].items():
            if held:
                oe=object_error(state['objects'][obj],baseline['objects'][obj])
                if oe['position_m']>self.cfg['passive_object_position_drift_m'] or oe['rotation_rad']>self.cfg['passive_object_rotation_drift_rad']:return 'passive_object_changed',pe
        if object_error(state['objects']['base'],baseline['objects']['base'])['position_m']>self.cfg['base_drift_m']:return 'base_displaced',pe
        if baseline['official_metrics']['first_piece_assembled']:
            oe=object_error(state['objects']['piece_1'],baseline['objects']['piece_1'])
            if not state['official_metrics']['first_piece_assembled'] or oe['position_m']>self.cfg['assembled_object_drift_m'] or oe['rotation_rad']>self.cfg['assembled_object_rotation_drift_rad']:return 'completed_piece1_changed',pe
        return None,pe

    def after_step(self):
        state=snapshot(self.env)
        edges=self.contacts.read()['edges']
        for ai,a in enumerate(ARMS):
            held=state['actual_grasps'][a][self.points['arms'][a]['object']]
            self.grasp_streak[ai]=self.grasp_streak[ai]+1 if held else 0
            released=(ai==0 and state['official_metrics']['first_piece_assembled'] or ai==1 and state['official_metrics']['task']) and not held
            if released:self.completed[ai]=True
            obj=self.points['arms'][a]['object'];supported=any(obj in {e['a'],e['b']} and bool(({e['a'],e['b']}-{obj}) & {'base','piece_1','static'}) for e in edges)
            self.support_age[ai]=0 if supported else self.support_age[ai]+1
            self.release_wait[ai]=self.release_wait[ai]+1 if self.was_grasped[ai] and not held and not released else 0
            if self.grasp_streak[ai]>=3:self.was_grasped[ai]=True
        if self.completed[0] and self.protected_first is None:
            speed=np.asarray(self.env.sim.data.cvel[self.env.obj_body_id['piece_1']])
            if np.linalg.norm(speed[3:])<=self.cfg['placed_object_linear_speed_m_s'] and np.linalg.norm(speed[:3])<=self.cfg['placed_object_angular_speed_rad_s']:self.protected_first=copy.deepcopy(state['objects']['piece_1'])
        projected=set(getattr(self,"projected_arms",[])) if getattr(self,"projected",False) else set()
        guide=self.control_active() and self.recovery_stage=='Sliding_Guide'
        for ai,w in enumerate(self.windows):
            lo,hi,stage=self.legal_bounds(ai,state)
            # During Guide only the guided arm's window is frozen. The
            # independently controlled Policy arm must keep its own progress.
            frozen=self.control_active() and (not guide or ai==self.active)
            if not frozen and lo>w.floor:w.event(lo)
            w.tick(hi,frozen or ai in projected);self.cursors[ai].index=max(self.cursors[ai].index,w.end)
        result={'handback':False,'indices':self.indices(),'completed':self.completed.copy()}
        if not self.control_active():return result
        self.age+=1;self.intervention_control_steps+=1
        ai=self.active;p=self.points['arms'][ARMS[ai]]
        danger=self.physical_danger(state,self.contacts.read())
        if self.recovery_stage=='Sliding_Guide':
            # No passive-retention test: the other arm has never been put in
            # Hold. Its normal motion is checked by joint and substep safety.
            result.update(recovery_stage=self.recovery_stage,passive_control='Policy',
                          retention_failure=danger)
            if danger:self.stop(danger);return result
            e=errors(state['poses'][ai],self.targets[ai])
            okay=(e['position_m']<self.cfg['exit_position_m'] and
                  e['rotation_rad']<self.cfg['exit_rotation_rad'])
            self.stable=self.stable+1 if okay else 0
            result.update(error=e,stable_guide_steps=self.stable)
            if self.stable>=self.cfg['stable_exit_steps']:
                self.handback(result,None)
            elif self.age>=self.cfg.get('max_guide_steps',80):
                self.stop('guide_timeout_no_safe_handback')
            return result
        failure,pe=self.retention(state)
        if danger:failure=danger
        result.update(passive_error=pe,recovery_stage=self.recovery_stage,retention_failure=failure)
        if failure:self.stop(failure);return result
        if self.recovery_stage=='Replay_P0_P1':
            self.replay_index+=1
            result['replay_pose_error']=errors(state['poses'][ai],self.pose(ai,self.replay_index))
            if self.replay_index>=p['p1']:
                self.recovery_stage='Stable_Grasp';self.targets[ai]=self.pose(ai,p['p1']);self.grips[ai]=1.;self.plan=[self.targets[ai]];self.stable=0;self.grasp_stability.clear()
        elif self.recovery_stage=='Stable_Grasp':
            e=errors(state['poses'][ai],self.targets[ai]);held=state['actual_grasps'][ARMS[ai]][p['object']]
            oe=object_error(state['objects'][p['object']],self.ref_object(p['object'],p['p1']))
            okay=held and np.allclose(self.env.robots[ai].gripper['right'].current_action,[-1,1]) and e['position_m']<=self.cfg['exit_position_m'] and e['rotation_rad']<=self.cfg['exit_rotation_rad'] and oe['position_m']<=self.cfg['active_object_exit_position_m'] and oe['rotation_rad']<=self.cfg['active_object_exit_rotation_rad']
            rel=state['poses'][ai]['rotation'].T@(np.asarray(state['objects'][p['object']]['position'])-state['poses'][ai]['position'])
            self.grasp_stability.append(rel)
            stable_relative=len(self.grasp_stability)==self.grasp_stability.maxlen and max(np.linalg.norm(x-self.grasp_stability[0]) for x in self.grasp_stability)<self.cfg['relative_grasp_motion_m']
            self.stable=self.stable+1 if okay and stable_relative else 0;result.update(error=e,active_object_error=oe,stable_grasp_steps=self.stable,relative_grasp_stable=bool(stable_relative))
            if self.stable>=self.cfg['stable_exit_steps']:self.handback(result,p['p1'])
        else:
            e=errors(state['poses'][ai],self.plan[0]);last=len(self.plan)==1
            # P0 replay alignment is tighter than paper's ordinary sliding exit.
            tol=self.cfg['p0_alignment_m'] if self.recovery_kind=='safety_recovery' and last else self.cfg['exit_position_m']
            okay=e['position_m']<tol and e['rotation_rad']<self.cfg['exit_rotation_rad'];self.stable=self.stable+1 if okay else 0;result['error']=e
            if self.stable>=self.cfg['stable_exit_steps']:
                self.plan.pop(0);self.stable=0
                if not self.plan:
                    if self.recovery_kind=='safety_recovery':self.recovery_stage='Replay_P0_P1';self.replay_index=p['p0']
                    else:self.handback(result,None)
        if self.control_active() and self.age>=self.cfg['max_recovery_steps']:self.stop('recovery_timeout_no_safe_handback')
        return result

    def handback(self,result,index):
        ai=self.active
        if index is not None:self.windows[ai].event(index);self.cursors[ai].index=max(self.cursors[ai].index,index)
        result['indices']=self.indices()
        self.mode='Normal';self.cooldown[ai]=self.cfg['cooldown_steps'];self.history[ai].clear();result['handback']=True
        self.events.append({'event':'handback','arm':ARMS[ai],'kind':self.recovery_kind,'steps':self.age,'indices':self.indices()})

    def episode_result(self,stable_full_success,had_intervention=None,full_policy_episode=False):
        # Original l_term counts intervention CONTROL STEPS, not takeover episodes.
        qualifies=bool(stable_full_success and full_policy_episode and self.intervention_control_steps<self.cfg['termination_intervention_steps'] and self.cfg['fixed_initial_state'])
        if qualifies:self.auto_disabled=True
        row={'stable_full_success':bool(stable_full_success),'intervention_control_steps':self.intervention_control_steps,'intervention_events':self.episode_interventions,'l_term':self.cfg['termination_intervention_steps'],'fixed_initial_state':self.cfg['fixed_initial_state'],'auto_disabled':self.auto_disabled}
        self.termination_log.append(row);return row
