"""Single-demo data bridge to unchanged upstream ResFiT; no RL losses here.

Only one-step returns are supported. next.done marks episode boundaries;
next.terminated alone disables bootstrap, including at real time limits.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import h5py
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage, RandomSampler
from resfit.dexmg.environments.dexmg import RobosuiteGymWrapper
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from acoserl.data import load_demo
from acoserl.envs import restore_initial, action_layout

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/generated/two_arm_three_piece_assembly.hdf5'
CAMERAS = ['agentview', 'robot0_eye_in_hand', 'robot1_eye_in_hand']
IMAGES = ['observation.images.' + c for c in CAMERAS]
STATE = [f'robot{i}_{part}' for i in range(2) for part in ['eef_pos','eef_quat','gripper_qpos']]


def demo_arrays():
    # Deliberately never iterate over data's episode groups.
    with h5py.File(DATA,'r') as f:
        g=f['data/demo_0']
        assert 'rewards' not in g and 'dones' not in g and 'next_obs' not in g
        state=np.concatenate([g['obs'][k][:] for k in STATE],axis=1).astype(np.float32)
        images={out: np.transpose(g['obs'][cam+'_image'][:],(0,3,1,2)).copy()
                for cam,out in zip(CAMERAS,IMAGES)}
        actions=g['actions'][:].astype(np.float32)
        for side,start in [('right',0),('left',7)]:
            block=np.concatenate([g['action_dict'][side+'_'+k][:] for k in ['rel_pos','rel_rot_axis_angle','gripper']],axis=1)
            np.testing.assert_allclose(actions[:,start:start+7],block,atol=1e-7,rtol=0)
        return state,images,actions


def scalers(n):
    state,images,actions=demo_arrays(); x=torch.from_numpy(state[:n]); a=torch.from_numpy(actions[:n])
    stats={'state_mean':x.mean(0).tolist(),'state_std_population':x.std(0,correction=0).tolist(),
           'action_min':a.min(0).values.tolist(),'action_max':a.max(0).values.tolist(),'frames_used':n}
    std=StateStandardizer(x.mean(0),x.std(0,correction=0),min_std=.1)
    scale=ActionScaler(a.min(0).values,a.max(0).values,action_scale=0,min_range_per_dim=.1)
    recovered=scale.unscale(scale.scale(torch.from_numpy(actions)))
    err=float((recovered-torch.from_numpy(actions)).abs().max())
    assert err<2e-6,err
    assert scale.limits.min.min()>=-1-1e-6 and scale.limits.max.max()<=1+1e-6
    stats.update({'effective_state_std':std._std.tolist(),'effective_action_min':scale.limits.min.tolist(),
                  'effective_action_max':scale.limits.max.tolist(),'all_229_action_roundtrip_max_error':err,
                  'policy_domain':'[-1,1] mapped to safeguarded single-demo per-dimension range; not full environment action range'})
    return std,scale,stats


def normalize(raw,std):
    obs={'observation.state':std.standardize(torch.as_tensor(raw['observation.state'],dtype=torch.float32))}
    for key in IMAGES:
        v=torch.as_tensor(raw[key])
        obs[key]=v.clone() if v.dtype==torch.uint8 else (v*255).round().clamp(0,255).to(torch.uint8)
        assert obs[key].shape==(3,84,84)
    assert obs['observation.state'].shape==(18,)
    return obs


def transition(obs,action,nxt,reward,terminated,truncated,source,frame,episode):
    assert not (terminated and truncated)
    assert np.isfinite(reward)
    return TensorDict({'obs':TensorDict(obs,batch_size=[]),'action':action.clone().float(),
      'source':torch.tensor(source),'frame':torch.tensor(frame),'episode':torch.tensor(episode),
      'next':TensorDict({'obs':TensorDict(nxt,batch_size=[]),'reward':torch.tensor(reward,dtype=torch.float32),
         'done':torch.tensor(terminated or truncated),'terminated':torch.tensor(terminated),'truncated':torch.tensor(truncated)},batch_size=[])},batch_size=[]).unsqueeze(0)
