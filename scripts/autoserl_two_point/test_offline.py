"""Check all 224 stored actions, future observations and three-step strict labels."""
import json
from pathlib import Path
import torch
from acoserl.ascend_data import offline,replay
from acoserl.single_demo_rlpd import demo_arrays,IMAGES
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
 rb,std,scale,stats=offline(128);state,images,actions=demo_arrays();rows=rb[:]
 for t in range(224):
  m=min(3,224-t);row=rows[t]
  assert int(row['steps_to_next_obs'])==m and int(row['env_id'])==-1
  torch.testing.assert_close(row['action'],scale.scale(torch.from_numpy(actions[t])))
  torch.testing.assert_close(row['obs','observation.state'],std.standardize(torch.from_numpy(state[t])))
  torch.testing.assert_close(row['next','obs','observation.state'],std.standardize(torch.from_numpy(state[t+m])))
  for k in IMAGES:
   assert torch.equal(row['obs',k],torch.from_numpy(images[k][t]))
   assert torch.equal(row['next','obs',k],torch.from_numpy(images[k][t+m]))
  expected=.99**(223-t) if t>=221 else 0.
  assert abs(float(row['next','reward'])-expected)<1e-7
  assert bool(row['next','terminated'])==(t>=221) and bool(row['nonterminal'])==(t<221)
 out=ROOT/'outputs/autoserl_two_point/offline_protocol';out.mkdir(exist_ok=False)
 rb.dumps(out/'buffer');loaded=replay(224,128);loaded.loads(out/'buffer')
 for key in rows.keys(True,True):assert torch.equal(rows[key],loaded[:][key]),key
 result={'passed':True,'original_valid_rows':224,'n_step_rows':len(rb),'n_step':3,'all_actions_current_observations_and_future_images_exact':True,'all_strict_labels_and_bootstrap_correct':True,'buffer_reload_exact':True}
 (ROOT/'outputs/autoserl_two_point/offline_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
