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


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16*1024*1024), b''): h.update(block)
    return h.hexdigest()


def snapshot():
    """Protect every existing tracked file, all HDFs, and upstream algorithm."""
    names = subprocess.check_output(['git','ls-tree','-r','--name-only','e78e49587f44'], cwd=ROOT, text=True).splitlines()
    tracked = {n:sha(ROOT/n) for n in names}
    data = {}
    for p in sorted((ROOT/'datasets').rglob('*.hdf5')):
        st=p.stat(); record={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino}
        if p == DATA or p.parent.name == 'single_demo': record['sha256']=sha(p)
        data[str(p.relative_to(ROOT))]=record
    repo = ROOT/'third_party/residual-offpolicy-rl'
    paths = subprocess.check_output(['git','ls-files','resfit/rl_finetuning/off_policy',
        'resfit/rl_finetuning/config/rlpd.py','resfit/rl_finetuning/utils/rb_transforms.py'], cwd=repo,text=True).splitlines()
    algorithm={n:sha(repo/n) for n in paths}
    for n in paths:
        assert (repo/n).read_bytes()==subprocess.check_output(['git','show','HEAD:'+n],cwd=repo),n
    return {'tracked_at_baseline':tracked,'data':data,'algorithm':algorithm}


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


def environment(seed):
    np.random.seed(seed)
    return RobosuiteGymWrapper('TwoArmThreePieceAssembly', camera_size=84)


def audit(out):
    """Offline label reconstruction from recorded states; never training state reset."""
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    before=snapshot(); write_json(out/'preservation_before.json',before)
    state,images,actions=demo_arrays(); d=load_demo(DATA,'demo_0'); w=environment(20261027)
    low_errors=[]; image_checks=[]; rewards=[]; successes=[]
    try:
        restore_initial(w.env,d)
        for t,s in enumerate(d['states']):
            render_frame=t in [0,80,180,221,222,228]
            for camera in CAMERAS: w.env.modify_observable(camera+'_image','enabled',render_frame)
            if t%40==0: print('AUDIT FRAME',t,flush=True)
            w.env.sim.set_state_from_flattened(s); w.env.sim.forward()
            # Refresh all observable caches at each recorded state, not just qpos.
            raw=w.env._get_observations(force_update=True)
            now=np.concatenate([raw[k] for k in STATE]).astype(np.float32)
            low_errors.append(np.abs(now-state[t]).tolist())
            rewards.append(float(w.env.reward())); successes.append(bool(w.env._check_success()))
            if t in [0,80,180,221,222,228]:
                for cam,key in zip(CAMERAS,IMAGES):
                    stored=images[key][t].transpose(1,2,0).astype(float); live=raw[cam+'_image'].astype(float)
                    direct=float(abs(stored-live).mean()); flipped=float(abs(stored[::-1]-live).mean())
                    assert direct<5 and direct<flipped/5,(t,cam,direct,flipped)
                    image_checks.append({'frame':t,'camera':cam,'direct_mae_uint8':direct,'flipped_mae_uint8':flipped})
        first=successes.index(True)
        assert first==222 and not any(successes[:first])
        assert all(r==float(s) for r,s in zip(rewards,successes))
        assert np.max(low_errors)<.002
        controller_contract=[]
        expected=d['metadata']['env_kwargs']['controller_configs']['body_parts']['right']
        for robot in w.env.robots:
            cfg=robot.part_controller_config['right']
            for k,v in expected.items(): assert cfg[k]==v,(k,cfg[k],v)
            controller_contract.append({k:cfg[k] for k in expected})
        for camera in CAMERAS: w.env.modify_observable(camera+'_image','enabled',True)
        # Real timeout, not a synthetic reward/done example. Horizon changes only for this probe.
        w.reset(seed=20261027); w.env.horizon=2
        timeout=[]
        for _ in range(2):
            obs,r,term,trunc,info=w.step(np.zeros(14,np.float32))
            timeout.append({'reward':r,'terminated':term,'truncated':trunc,'success':info['success']})
        assert timeout[-1]=={'reward':0.,'terminated':False,'truncated':True,'success':False}
        # Real action-only successful prefix: canonical initial restore ONCE, no mid-prefix state writes.
        restore_initial(w.env,d); w.env.horizon=500
        for camera in CAMERAS: w.env.modify_observable(camera+'_image','enabled',False)
        prefix_rewards=[]
        for t in range(first-1):
            _,r,done,info=w.env.step(d['actions'][t]);prefix_rewards.append(float(r))
            assert not done and r==0,(t,r,done)
        for camera in CAMERAS: w.env.modify_observable(camera+'_image','enabled',True)
        _,r,term,trunc,info=w.step(d['actions'][first-1])
        print('SUCCESS PREFIX',first,'steps',r,term,trunc,flush=True)
        assert r==1 and term and not trunc and info['success']
        result={'source':str(DATA),'demo':'demo_0','source_frames':len(state),'effective_transitions':first,
          'current_frame_indices':[0,first-1],'next_frame_indices':[1,first],
          'excluded_actions':len(actions)-first,'first_success_state_index':first,
          'reward_at_recorded_state':rewards,'success_at_recorded_state':successes,
          'terminal_transition_index':first-1,'label_rule':'r_t=task.reward(s[t+1]); terminated_t=task._check_success(s[t+1]); stop at first success',
          'missing_fields':['rewards','dones','next_obs'],'final_action_missing_next_frame':True,
          'state_order':STATE,'state_shape':list(state.shape),'state_audit_max_abs_by_dimension':np.max(low_errors,axis=0).tolist(),
          'image_checks':image_checks,'image_transform':'HWC RGB uint8 -> CHW RGB uint8; no flip, no resize',
          'action_order':action_layout(w.env),'recorded_action_dict_verified':True,
          'controllers':controller_contract,'timeout_probe':timeout,
          'success_probe':{'reward':r,'terminated':term,'truncated':trunc},
          'success_probe_action_only_prefix_steps':first,'success_prefix_prior_rewards':prefix_rewards,
          'reward_predicate_caveat':'Uses unchanged official assembly success predicate, including previously documented gripper-string bug.',
          'offline_state_restoration':'Only used to audit recorded observations and derive labels, not online collection.'}
        write_json(out/'semantics.json',result)
    finally: w.close()
    after=snapshot(); assert before==after
    write_json(out/'preservation_after_audit.json',after)
    print('AUDIT PASSED: 222 transitions; timeout reward 0, success reward 1',flush=True)
    return result


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


def buffer(capacity,batch_size,gamma):
    # Unchanged upstream transform: using terminated as its boundary key is safe ONLY at n_steps=1.
    return ReplayBuffer(storage=LazyTensorStorage(capacity,device='cpu',ndim=1),sampler=RandomSampler(),
       transform=MultiStepTransform(n_steps=1,gamma=gamma,done_key='terminated'),
       pin_memory=False,batch_size=batch_size)


def offline_buffer(semantics,std,scale,batch_size,gamma):
    state,images,actions=demo_arrays(); n=semantics['effective_transitions']; rb=buffer(n,batch_size,gamma)
    observations=[normalize({'observation.state':state[t],**{k:v[t] for k,v in images.items()}},std) for t in range(n+1)]
    for t in range(n):
        rb.add(transition(observations[t],scale.scale(torch.from_numpy(actions[t])),observations[t+1],
            semantics['reward_at_recorded_state'][t+1],semantics['success_at_recorded_state'][t+1],False,0,t,0))
    assert len(rb)==n
    stored=rb[:]
    assert stored.batch_size==torch.Size([n])
    torch.testing.assert_close(stored['obs','observation.state'],std.standardize(torch.from_numpy(state[:n])))
    torch.testing.assert_close(stored['next','obs','observation.state'],std.standardize(torch.from_numpy(state[1:n+1])))
    for k in IMAGES:
        assert torch.equal(stored['obs',k],torch.from_numpy(images[k][:n]))
        assert torch.equal(stored['next','obs',k],torch.from_numpy(images[k][1:n+1]))
    torch.testing.assert_close(scale.unscale(stored['action']),torch.from_numpy(actions[:n]),atol=2e-6,rtol=0)
    assert stored['next','reward'].sum()==1 and stored['next','reward'][-1]==1
    assert stored['next','terminated'].sum()==1 and not stored['nonterminal'][-1]
    assert stored['nonterminal'][:-1].all() and not stored['next','truncated'].any()
    return rb
