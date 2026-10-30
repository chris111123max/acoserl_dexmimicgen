"""Contract-only regression: an unsafe complete step cannot retain success reward."""
import json,types
from pathlib import Path
import numpy as np,torch
from acoserl.autoserl_env import InterventionBridge
from resfit.rl_finetuning.utils.normalization import ActionScaler
from acoserl.transition_semantics import Boundary
root=Path(__file__).resolve().parents[2]
n=json.loads((root/"docs/ascend16/normalization.json").read_text())
scale=ActionScaler(torch.tensor(n["effective_action_min"]),torch.tensor(n["effective_action_max"]),action_scale=0,min_range_per_dim=.1)
results=[]
for strict in (False,True):
 for stopped in (False,True):
  b=InterventionBridge.__new__(InterventionBridge);b.scaler=scale
  class Manager:
   stop_reason=None;mode="Normal"
   def choose(self,a):return a.copy(),dict(mode="Normal",policy_action=a.tolist())
   def after_step(self):
    self.stop_reason="completed_piece1_displaced" if stopped else None
    self.mode="Safe_Stop" if stopped else "Normal"
    return {}
  b.manager=Manager()
  b.env=types.SimpleNamespace(step=lambda a:({},float(strict),strict,False,dict(success=strict)))
  obs,r,t,tr,info=b.step(np.zeros(14,dtype=np.float32))
  assert r==float(strict and not stopped)==float(info["success"])
  if stopped:assert t and not tr and info["boundary_kind"]==int(Boundary.PHYSICAL_FAILURE) and not info["bootstrap_allowed"]
  results.append(dict(strict_environment_success=strict,safety_stop=stopped,reward=r,terminated=t,truncated=tr))
out=root/"outputs/acoserl_v3/bridge_reward_contract.json";assert not out.exists()
out.write_text(json.dumps(dict(passed=True,cases=results,contract_only=True),indent=2)+"\n")
print(json.dumps(results))
