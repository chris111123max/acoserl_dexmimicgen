"""Apply audited environment-only compatibility changes to the pinned upstream."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
repo=root/'third_party/residual-offpolicy-rl'
commit='66262b06a0caf159fb001504949b7b48ccda5f64'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
path=repo/'resfit/dexmg/environments/dexmg.py'
original=subprocess.check_output(['git','-C',str(repo),'show','HEAD:resfit/dexmg/environments/dexmg.py'],text=True)
s=original
s=s.replace('class RobosuiteGymWrapper:', 'class RobosuiteGymWrapper(gym.Env):',1)
start=s.rfind('\n',0,s.index('Idxs'))+1
end=s.index('\n\n    """',start)
s=s[:start]+'''    Action dimensions are discovered from the actual robosuite robots.
    For the fixed dual-Panda benchmark: robot0 / right action_dict block first,
    robot1 / left block second; each block is six normalized OSC_POSE delta
    inputs in the recorded base frame followed by one Panda gripper command.
    These are scaled by the original controller; inputs are not raw metres.
'''+s[end:]
old='            "TwoArmThreePieceAssembly",\n            "TwoArmThreading",'
assert old in s
s=s.replace(old,old+'\n            "TwoArmTransport",',1)
old='        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.render_gpu_device_id)'
assert old in s
s=s.replace(old,'        if os.environ.get("MUJOCO_GL", "egl") == "egl":\n            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.render_gpu_device_id)',1)
old='''        if env_name == "TwoArmTransport":
            env_kwargs["env_configuration"] = "opposed"
            env_kwargs["controller_configs"]["body_parts"]["right"]["input_ref_frame"] = "world"
'''
assert old in s
s=s.replace(old,'''        if env_name in ("TwoArmThreading", "TwoArmThreePieceAssembly", "TwoArmTransport"):
            # Match the existing official HDF5 benchmark metadata. Do not rotate
            # Transport actions into a different frame or instantiate the built-in task.
            env_kwargs["env_configuration"] = "parallel"
''',1)
old='''        # For robosuite environments, we'll ignore these for now
        obs = self.env.reset()'''
assert old in s
s=s.replace(old,'''        # Gymnasium RNG bookkeeping; robosuite's own randomization is unchanged.
        super().reset(seed=seed)
        obs = self.env.reset()''',1)
old='''        # Terminate episode on success (reward == 1) to shortcircuit successful rollouts
        success = reward == 1.0
        terminated_scalar = bool(success)
        truncated_scalar = bool(done)  # Robosuite returns done when timeout
'''
assert old in s
s=s.replace(old,'''        # Use the original task predicate, independent of reward scaling.
        success = bool(self.env._check_success())
        terminated_scalar = success
        truncated_scalar = bool(done) and not success  # timeout without task success
        info = {**info, "success": success}
''',1)
# Missing required observations must not silently shorten the policy state.
old='''        processed_obs = {}

        # Extract robot state'''
assert s.count(old)==2
s=s.replace(old,'''        self._validate_required_observations(obs)
        processed_obs = {}

        # Extract robot state''')
needle='    def _get_expected_image_keys(self, env_name: str):'
assert needle in s
s=s.replace(needle,'''    def _validate_required_observations(self, obs):
        required = self._get_expected_low_dim_keys(self.env_name) + self.expected_image_keys
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Missing required observations for {self.env_name}: {missing}")

'''+needle,1)
allowed=[original,s]
if path.read_text() not in allowed:
    raise RuntimeError('Existing local edits differ; refusing to overwrite')
path.write_text(s)
train=repo/'resfit/rl_finetuning/scripts/train_rlpd_dexmg.py'
original_train=subprocess.check_output(['git','-C',str(repo),'show','HEAD:resfit/rl_finetuning/scripts/train_rlpd_dexmg.py'],text=True)
new_train=original_train.replace('os.environ["MUJOCO_GL"] = "egl"','os.environ.setdefault("MUJOCO_GL", "egl")',1)
assert train.read_text() in [original_train,new_train]
train.write_text(new_train)
patch=subprocess.check_output(['git','-C',str(repo),'diff','--','resfit/dexmg/environments/dexmg.py','resfit/rl_finetuning/scripts/train_rlpd_dexmg.py'])
(root/'requirements/resfit/environment_compatibility.patch').write_bytes(patch)
print('Environment compatibility patch saved; actor, critic, loss, buffer, updates unchanged.')
