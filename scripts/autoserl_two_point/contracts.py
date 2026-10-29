"""Pure numerical/metadata contracts, NOT simulated recovery successes."""
import copy,json
from pathlib import Path
from collections import deque
import numpy as np
from acoserl.trajectories import SlidingReferenceWindow
from acoserl.two_point_supervisor import paper_direction
from acoserl.bimanual_autoserl import BimanualAutoSERL,DEFAULT
ROOT=Path(__file__).resolve().parents[2];checks={}
w=SlidingReferenceWindow(27,230);assert w.indices(51)==[0]
for _ in range(40):w.tick(51)
assert w.indices(51)==list(range(14,41));w.tick(51,True);assert w.end==40
v=SlidingReferenceWindow(32,230);v.tick(63);assert v.end==1 and w.end==40
for _ in range(300):w.tick(51)
assert w.end==51;w.event(57);assert w.end==57 and w.floor==57
for _ in range(300):w.tick(229)
assert w.indices(229)[-1]==229
checks['trailing_window_independence_event_caps_no_rewind_final_state']=True
p=np.array([[0.,0,0],[1.,0,0],[2.,0,0]])
assert paper_direction(p,0,np.array([-1.,0,0]),0)[0]
assert paper_direction(p,0,np.array([0.,-1,0]),0)[0]
assert not paper_direction(p,0,np.array([1.,0,0]),0)[0]
assert not paper_direction(np.zeros((3,3)),0,np.ones(3),0)[0]
checks['direction_at_global_cpoint_90_inclusive_reverse_and_zero_tangent_cancel']=True
m=BimanualAutoSERL.__new__(BimanualAutoSERL);m.points=json.loads((ROOT/'configs/bimanual_autoserl/recovery_points.json').read_text());m.trace=np.load(ROOT/m.points['reference_path']);m.positions=m.trace['positions'];m.n_states=len(m.positions);m.ref_grasps=m.trace['grasp'];m.cfg=DEFAULT;m.completed=[False,False];m.grasp_streak=[0,0]
state={'gripper_commands':[[1.,-1.],[1.,-1.]],'actual_grasps':{a:{o:False for o in ['base','piece_1','piece_2']} for a in ['A','B']},'poses':[{'position':m.positions[0,i].copy()} for i in range(2)]}
assert m.legal_bounds(0,state)[1]==m.points['arms']['A']['first_close_action']
state['gripper_commands'][0]=[.8,-.8]
assert m.legal_bounds(0,state)[1]==m.points['arms']['A']['first_contact']-1
state['gripper_commands'][0]=[1.,-1.]
checks['actual_gripper_event_caps_prevent_closure_jump']=True
for distance,expected in [(.019,False),(.021,True)]:
 m.windows=[SlidingReferenceWindow(27,230),SlidingReferenceWindow(32,230)];m.history=[deque(maxlen=21),deque(maxlen=21)];state['poses'][0]['position']=m.positions[0,0]+[distance,0,0];r=m.window_measure(0,state);assert (r['distance_m']>m.cfg['trigger_position_m'])==expected
assert m.cfg['trigger_position_m']==.020 and m.cfg['exit_position_m']==.005
checks['position_thresholds_20mm_trigger_5mm_exit_not_orientation']=True
m.windows=[SlidingReferenceWindow(27,230,30),SlidingReferenceWindow(32,230)];m.history=[deque(maxlen=21),deque(maxlen=21)];state['poses'][0]['position']=m.positions[30,0].copy()
for k in range(20):assert not m.window_measure(0,state)['stagnant']
assert m.window_measure(0,state)['stagnant']
checks['stagnation_20_step_lag_global_reference_point_5mm']=True
for count,expected in [(9,True),(10,False)]:
 q=BimanualAutoSERL.__new__(BimanualAutoSERL);q.cfg=DEFAULT;q.intervention_control_steps=count;q.episode_interventions=1;q.termination_log=[];q.auto_disabled=False;q.episode_result(True,full_policy_episode=True);assert q.auto_disabled==expected
checks['l_term_counts_executed_intervention_control_steps_strict_less_10']=True
m.was_grasped=[False,False];m.protected_first=None;state['poses']=[{'position':m.positions[0,i].copy()} for i in range(2)]
contact={'max_robot_contact_force_N':0.,'dynamic_A_B_coupled':False,'edges':[]}
assert m.physical_danger(state,contact) is None
assert m.physical_danger(state,{**contact,'max_robot_contact_force_N':80.01})=='robot_force_limit'
assert m.physical_danger(state,{**contact,'dynamic_A_B_coupled':True})=='unverified_dynamic_A_B_coupling'
checks['safe_free_space_distinct_from_force_and_coupling_danger']=True
assert DEFAULT['passive_position_drift_m']==.002 and DEFAULT['passive_object_position_drift_m']==.003
checks['passive_limits_unchanged']=True
path=ROOT/'outputs/autoserl_two_point/contracts.json';path.write_text(json.dumps({'passed':True,'kind':'pure numerical and metadata; no physical recovery claimed','checks':checks},indent=2)+'\n');print(path)
