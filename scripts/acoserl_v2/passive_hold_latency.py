"""Replay recorded real Actor prefixes to diagnose and reject underestimated passive holds."""
import argparse,json,sys,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,snapshot,save
from acoserl.bimanual_autoserl import BimanualAutoSERL
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
out=ROOT/'outputs/acoserl_v2'/a.tag;out.mkdir(exist_ok=False)
source=ROOT/'outputs/ascend16/v2_bound_20261029b_full_official/workers'
results=[]
for name in ('env07','env00'):
 rows=[json.loads(s) for s in (source/(name+'.jsonl')).read_text().splitlines() if json.loads(s).get('event')=='step']
 i=next(i for i,r in enumerate(rows) if (r.get('recovery_result') or {}).get('retention_failure')=='passive_pose_changed')
 old=rows[i];episode=old['episode_id'];prefix=[r for r in rows[:i] if r['episode_id']==episode]
 E,D,R=setup();M=BimanualAutoSERL(E,intervention_enabled=False);R.safety=M.substep_safety
 result=dict(source=name,episode=episode,frame=old['frame'],passed=False)
 try:
  for row in prefix:
   action=np.asarray(row['executed_action_physical']);final,_=M.choose(action);R.step(final);M.after_step()
  state=snapshot(E);previous=prefix[-1]
  pose_delta=max(float(np.linalg.norm(p['position']-np.asarray(q))) for p,q in zip(state['poses'],previous['eef_positions']))
  active=1 if old['action_source'].endswith('B') else 0
  gate=M.passive_hold_entry(active,state);M.cfg['intervention_enabled']=True
  proposal=np.asarray(old['auto_intervention']['policy_action'])
  final,decision=M.choose(proposal)
  result.update(reproduced_prefix_position_error_m=pose_delta,old_gate=old['auto_intervention']['passive_hold_entry'],old_measured=old['recovery_result']['passive_error'],new_gate=gate,decision=copy.deepcopy(decision))
  assert pose_delta<1e-6,'canonical prefix did not reproduce actual worker'
  assert decision.get('passive_hold_entry',{}).get('pregrasp_control_interval_allowed') is False,'unsafe pregrasp hold still admitted'
  assert final is not None and decision['mode']=='Normal','ordinary guide refusal should defer to safe RL exploration'
  R.step(final);after=M.after_step()
  assert M.mode!='Safe_Stop',M.stop_reason
  result.update(passed=True,executed_action=final,after=after)
 except BaseException:
  import traceback
  result['error']=traceback.format_exc()
 finally:
  result['physical_writes']=R.physical_writes_attempted;results.append(result);save(out/(name+'.json'),result);E.close()
save(out/'result.json',dict(passed=all(x['passed'] for x in results),cases=results))
print(json.dumps({x['source']:x['passed'] for x in results}),flush=True)
assert all(x['passed'] for x in results)
