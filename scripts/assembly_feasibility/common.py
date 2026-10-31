"""Measurement and existing OSC command utilities; real actions only."""
from pathlib import Path
import json,hashlib,os,re
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from acoserl.data import load_demo
from acoserl.envs import create,restore_initial
ROOT=Path(__file__).resolve().parents[2]
RUN_LABEL=os.environ.get('ASSEMBLY_RUN_LABEL','')
assert not RUN_LABEL or re.fullmatch(r'[A-Za-z0-9_-]+',RUN_LABEL)
OUT=ROOT/'outputs/assembly_feasibility'
DOC=ROOT/'docs/assembly_feasibility'
if RUN_LABEL:OUT=OUT/'runs'/RUN_LABEL;DOC=DOC/'runs'/RUN_LABEL
DATA=ROOT/'datasets/generated/two_arm_three_piece_assembly.hdf5'
REFERENCE=ROOT/'datasets/single_demo/two_arm_three_piece_assembly_demo_0.hdf5'
ARMS=['A','B']

def save(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(str(path)+'.partial')
    temporary.write_text(json.dumps(data,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)))
    temporary.replace(path)

def poses(env):
    values=[]
    for r in env.robots:
        site=r.eef_site_id['right']
        values.append({'position':env.sim.data.site_xpos[site].copy(),'rotation':env.sim.data.site_xmat[site].reshape(3,3).copy()})
    return values

def objects(env):
    return {name:{'position':env.sim.data.body_xpos[bid].copy(),'quaternion_wxyz':env.sim.data.body_xquat[bid].copy()} for name,bid in env.obj_body_id.items()}

def snapshot(env):
    return {'sim_time':float(env.sim.data.time),'poses':poses(env),'objects':objects(env),
            'qpos':env.sim.data.qpos.copy(),'qvel':env.sim.data.qvel.copy(),
            'official_metrics':{k:bool(v) for k,v in env._get_partial_task_metrics().items()},
            'actual_grasps':{arm:{name:bool(env._check_grasp(env.robots[i].gripper['right'],getattr(env,name).contact_geoms)) for name in ['base','piece_1','piece_2']} for i,arm in enumerate(ARMS)}}

class Contacts:
    def __init__(self,env):
        self.env=env;self.owners={}
        model=env.sim.model
        for gid,name in enumerate(model.geom_names):
            owner=None;bid=int(model.geom_bodyid[gid])
            while bid>0:
                if bid in env.obj_body_id.values():
                    owner=next(k for k,v in env.obj_body_id.items() if v==bid);break
                bname=model.body_id2name(bid) or ''
                if bname.startswith(('robot0_','gripper0_')):owner='A';break
                if bname.startswith(('robot1_','gripper1_')):owner='B';break
                bid=int(model.body_parentid[bid])
            self.owners[gid]=owner or 'static'
    def read(self):
        env=self.env;edges={};details=[];max_force=0.0;max_robot_force=0.0
        for i in range(env.sim.data.ncon):
            c=env.sim.data.contact[i];a=self.owners[int(c.geom1)];b=self.owners[int(c.geom2)]
            if a==b:continue
            force=np.zeros(6);mujoco.mj_contactForce(env.sim.model._model,env.sim.data._data,i,force)
            normal=float(max(force[0],0));max_force=max(max_force,normal)
            if 'A' in (a,b) or 'B' in (a,b):max_robot_force=max(max_robot_force,normal)
            if normal<0.1:continue
            pair=tuple(sorted((a,b)));edges[pair]=max(edges.get(pair,0),normal)
            details.append({'owners':pair,'geom1':env.sim.model.geom_id2name(int(c.geom1)),
                            'geom2':env.sim.model.geom_id2name(int(c.geom2)),'normal_force_N':normal,'distance_m':float(c.dist)})
        graph={}
        for a,b in edges:
            if 'static' in (a,b):continue
            graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
        reachable={'A'};todo=['A']
        while todo:
            for node in graph.get(todo.pop(),set())-reachable:reachable.add(node);todo.append(node)
        return {'edges':[{'a':a,'b':b,'max_normal_force_N':v} for (a,b),v in sorted(edges.items())],
                'dynamic_A_B_coupled':'B' in reachable,'max_external_contact_force_N':max_force,'max_robot_contact_force_N':max_robot_force,'details':details}

class SafetyStop(RuntimeError):
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.evidence=evidence or {}

class ActionOnly:
    """Frozen live physics; private-data preview, continuous live safety checks."""
    def __init__(self,env):
        self.env=env;self.contact=Contacts(env);self.log=[];self.physical_writes_attempted=0;self.safety=None
        self.preview=None;self.last_attempt=None
        def denied(*args,**kwargs):
            self.physical_writes_attempted+=1
            raise RuntimeError('Direct physical state restoration is forbidden after initial setup')
        for name in ['set_state','set_state_from_flattened','reset']:
            setattr(env.sim,name,denied)
        env.reset=denied;env.reset_from_xml_string=denied
    def step(self,action):
        from acoserl.substep_guard import WorkspacePreview,refresh_positions
        action=np.asarray(action,dtype=float)
        assert np.isfinite(action).all() and np.max(np.abs(action))<=1.000001
        sim=self.env.sim;method='step2' if self.env.lite_physics else 'step'
        original=getattr(sim,method);start_time=float(sim.data.time)
        owner=getattr(self.safety,'__self__',None)
        if owner is not None and hasattr(owner,'positions') and self.preview is None:
            self.preview=WorkspacePreview(self.env,owner)
        attempt=dict(start_time=start_time,action=action.tolist(),executed_substeps=0,
                     partial_physics=False,complete=False,substeps=[],stop_reason=None,physical_state_writes=0)
        self.last_attempt=attempt
        def fail(reason):
            attempt.update(stop_reason=reason,physical_time=float(sim.data.time),
                           partial_physics=bool(float(sim.data.time)>start_time))
            if self.preview is not None:attempt['preview_totals']=self.preview.audit()
            raise SafetyStop(reason,attempt)
        def monitored():
            # At this point robosuite has updated ctrl for this physics step.
            if self.preview is not None:
                reason,prediction=self.preview.before(method)
                if reason:
                    attempt['rejected_preview']=prediction
                    fail(reason)
            original()
            refresh_positions(sim.model._model,sim.data._data)
            contact=self.contact.read()
            attempt['executed_substeps']+=1
            record=dict(index=attempt['executed_substeps'],time=float(sim.data.time),
                        positions=[p['position'].tolist() for p in poses(self.env)],
                        linear_velocities=[np.asarray(sim.data.get_site_xvelp(sim.model.site_id2name(r.eef_site_id['right']))).tolist() for r in self.env.robots],
                        force_sample_time=float(sim.data.time)-float(sim.model._model.opt.timestep),
                        robot_force_N=contact['max_robot_contact_force_N'],
                        external_force_N=contact['max_external_contact_force_N'],
                        edges=contact['edges'],coupled=contact['dynamic_A_B_coupled'])
            # At most one control interval of evidence is retained in RAM.
            attempt['substeps'].append(record)
            if self.preview is not None:
                actual=np.asarray(record['positions'])
                predicted=np.asarray(prediction['predicted_positions'])
                record['preview_error_m']=float(np.max(np.linalg.norm(actual-predicted,axis=1)))
                if record['preview_error_m']>1e-6:fail('substep_preview_does_not_match_live_physics')
            if self.safety:
                reason=self.safety(contact)
                if reason:fail(reason)
        setattr(sim,method,monitored)
        try:self.env.step(action)
        finally:setattr(sim,method,original)
        attempt.update(complete=True,physical_time=float(sim.data.time))
        result=snapshot(self.env);result['contacts']=self.contact.read()
        result['substep_coupled']=any(s['coupled'] for s in attempt['substeps'])
        result['substep_max_external_force_N']=max((s['external_force_N'] for s in attempt['substeps']),default=0.)
        result['substep_max_robot_force_N']=max((s['robot_force_N'] for s in attempt['substeps']),default=0.)
        result['substep_contact_edges']=sorted({tuple(sorted((e['a'],e['b']))) for s in attempt['substeps'] for e in s['edges']})
        return result

def setup():
    demo=load_demo(DATA,'demo_0');env=create(DATA);restore_initial(env,demo)
    return env,demo,ActionOnly(env)

def servo(env,targets,grippers,max_position_step=0.002,max_rotation_step=0.02,position_gain=1.0,rotation_gain=1.0):
    """Bounded proportional pose feedback in the existing base-frame OSC inputs."""
    current=poses(env);action=[]
    for i,r in enumerate(env.robots):
        ctl=r.part_controllers['right'];ctl.update()
        assert ctl.input_type=='delta' and ctl.input_ref_frame=='base' and ctl._goal_update_mode=='achieved'
        dp=position_gain*(targets[i]['position']-current[i]['position']);norm=np.linalg.norm(dp)
        if norm>max_position_step:dp*=max_position_step/norm
        dw=rotation_gain*Rotation.from_matrix(targets[i]['rotation']@current[i]['rotation'].T).as_rotvec();norm=np.linalg.norm(dw)
        if norm>max_rotation_step:dw*=max_rotation_step/norm
        local=np.concatenate((ctl.origin_ori.T@dp,ctl.origin_ori.T@dw))
        assert np.allclose(ctl.output_max,-ctl.output_min)
        action.extend((local/ctl.output_max).tolist()+[float(grippers[i])])
    return np.asarray(action)

def errors(actual,target):
    return {'position_m':float(np.linalg.norm(actual['position']-target['position'])),
            'rotation_rad':float(Rotation.from_matrix(actual['rotation']@target['rotation'].T).magnitude())}
