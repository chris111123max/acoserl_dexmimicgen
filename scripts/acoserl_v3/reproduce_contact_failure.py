"""Exact replay of the original Contact_B_6 failure with real substep kinematics.
Only canonical reset is permitted; no new Actor policy replaces the saved proposals.
"""
import argparse,copy,json,sys,hashlib
from pathlib import Path
import numpy as np
import mujoco,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,save
from acoserl.bimanual_autoserl import BimanualAutoSERL
from acoserl.object_geometry import object_error
from resfit.rl_finetuning.utils.normalization import ActionScaler
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
 out=ROOT/'outputs/acoserl_v3'/args.tag;out.mkdir(parents=True,exist_ok=False)
 source=ROOT/'outputs/acoserl_v2/local_physics_v3/Contact_B_6.jsonl'
 original=[json.loads(s) for s in source.read_text().splitlines()]
 E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False);M.manager_cfg['development_validation']=True
 norm=json.loads((ROOT/'docs/ascend16/normalization.json').read_text())
 scale=ActionScaler(torch.tensor(norm['effective_action_min']),torch.tensor(norm['effective_action_max']),action_scale=0,min_range_per_dim=.1)
 M.action_bounds=(scale.limits.min.numpy(),scale.limits.max.numpy())
 results=[];substeps=[];i=-1;first_bad=None;max_pose_error=0.;max_action_error=0.
 def measure():
  s=snapshot(E);s['eef_velocities']=[]
  for robot in E.robots:
   site=E.sim.model.site_id2name(robot.eef_site_id['right'])
   s['eef_velocities'].append(dict(linear=np.asarray(E.sim.data.get_site_xvelp(site)),angular=np.asarray(E.sim.data.get_site_xvelr(site))))
  s['object_velocities']={}
  for name,bid in E.obj_body_id.items():
   velocity=np.zeros(6);mujoco.mj_objectVelocity(E.sim.model._model,E.sim.data._data,mujoco.mjtObj.mjOBJ_BODY,bid,velocity,0)
   s['object_velocities'][name]=dict(angular=velocity[:3].copy(),linear=velocity[3:].copy())
  s['protected_first']=copy.deepcopy(M.protected_first)
  s['protected_error']=object_error(s['objects']['piece_1'],M.protected_first) if M.protected_first is not None else None
  return s
 def monitor(contact):
  nonlocal first_bad
  reason=M.substep_safety(contact)
  if i>=264:
   s=measure();err=s['protected_error'];bad=err is not None and (err['position_m']>M.cfg['assembled_object_drift_m'] or err['rotation_rad']>M.cfg['assembled_object_rotation_drift_rad'])
   item=dict(control_index_zero_based=i,sim_time=s['sim_time'],protected_error=err,objects=s['objects'],eef_velocities=s['eef_velocities'],object_velocities=s['object_velocities'],contact=contact,actual_grasps=s['actual_grasps'])
   substeps.append(item)
   if bad and first_bad is None:first_bad=copy.deepcopy(item)
  return reason
 R.safety=monitor
 try:
  for i,row in enumerate(original):
   if row['mode']=='Recover_Joint' and (i==0 or original[i-1]['mode']!='Recover_Joint'):
    ok,why=M.start_local(1,80,contact_mode=True,validation=True);assert ok,why
   if i>0 and original[i-1]['after'].get('safe_handback'):M.cfg['intervention_enabled']=True
   if 214<=i<279:
    # Reproduce the original outer servo's controller-cache update order.
    for robot in E.robots:robot.part_controllers['right'].update()
   before=measure();action,decision=M.choose(np.asarray(row['policy']))
   if action is None:raise RuntimeError('unexpected pre-step refusal:'+str(M.stop_reason))
   final=scale.unscale(scale.scale(torch.tensor(action,dtype=torch.float32))).numpy()
   ae=float(np.max(np.abs(final-np.asarray(row['action']))));max_action_error=max(max_action_error,ae)
   assert ae<2e-6,('action replay mismatch',i,ae)
   R.step(final);after=M.after_step();state=measure()
   pe=max(float(np.linalg.norm(state['poses'][a]['position']-row['poses'][a]['position'])) for a in range(2));max_pose_error=max(max_pose_error,pe)
   assert pe<1e-6,('physical replay mismatch',i,pe)
   if i>=264:results.append(dict(control_index_zero_based=i,before=before,after_state=state,policy_original=row['policy'],controller_proposal=action,executed=final,mode=decision['mode'],decision=decision,recovery_result=after,stop_reason=M.stop_reason))
   if M.mode=='Safe_Stop':break
  result=dict(reproduced=M.stop_reason=='completed_piece1_displaced' and i==len(original)-1,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),original_control_steps=len(original),executed_control_steps=i+1,last_control_index_zero_based=i,stop_reason=M.stop_reason,first_substep_violation=first_bad,max_pose_replay_error_m=max_pose_error,max_action_replay_error=max_action_error,physical_writes=R.physical_writes_attempted)
  save(out/'result.json',result);save(out/'control_trace.json',results);save(out/'substeps.json',substeps)
  print(json.dumps({k:v for k,v in result.items() if k!='first_substep_violation'}),flush=True)
 finally:E.close()
if __name__=='__main__':main()
