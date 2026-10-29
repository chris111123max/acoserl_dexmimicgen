"""One untouched demo, one canonical reset, then only actual original actions."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,save,Contacts,ARMS
out=ROOT/'outputs/autoserl_two_point';out.mkdir(exist_ok=True)
env,demo,runner=setup();contacts=Contacts(env);rows=[]
def capture(index):
 s=snapshot(env);s['state_index']=index;s['contact']=contacts.read();s['eef_velocity']=[];s['gripper_command']=[]
 for robot in env.robots:
  name=env.sim.model.site_id2name(robot.eef_site_id['right']);s['eef_velocity'].append({'linear':env.sim.data.get_site_xvelp(name).copy(),'angular':env.sim.data.get_site_xvelr(name).copy()});s['gripper_command'].append(np.asarray(robot.gripper['right'].current_action).copy())
 rows.append(s)
try:
 capture(0)
 for i,a in enumerate(demo['actions']):runner.step(a);capture(i+1)
 original=np.asarray(demo['states']);replayed=np.asarray([r['qpos'] for r in rows]);state_errors=[]
 # Recorded HDF5 has N states, not N+1. Verify each recorded pre-action state.
 for i,recorded in enumerate(original):state_errors.append(float(np.max(np.abs(recorded[1:1+env.sim.model.nq]-replayed[i]))))
 result={'recorded_actions':len(demo['actions']),'recorded_states':len(original),'actual_replayed_states':len(rows),'recorded_pre_action_qpos_max_error':max(state_errors),'direct_physical_writes_after_reset':runner.physical_writes_attempted,'rows':rows}
 save(out/'demo_audit.json',result)
 np.savez_compressed(out/'reference.npz',positions=np.asarray([[p['position'] for p in r['poses']] for r in rows]),rotations=np.asarray([[p['rotation'] for p in r['poses']] for r in rows]),actions=demo['actions'],grasp=np.asarray([[[r['actual_grasps'][a][o] for o in ['base','piece_1','piece_2']] for a in ARMS] for r in rows]),object_positions=np.asarray([[r['objects'][o]['position'] for o in ['base','piece_1','piece_2']] for r in rows]),object_quaternions=np.asarray([[r['objects'][o]['quaternion_wxyz'] for o in ['base','piece_1','piece_2']] for r in rows]))
 print('AUDIT_COMPLETE',len(rows),max(state_errors),rows[-1]['official_metrics'],flush=True)
finally:env.close()
