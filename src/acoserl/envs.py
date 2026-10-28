"""Create matching recorded environments, without modifying task controls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from .data import metadata

def official_playback():
    path = Path(__file__).resolve().parents[2] / 'third_party/dexmimicgen/scripts/playback_datasets.py'
    spec = importlib.util.spec_from_file_location('dexmg_playback', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create(path):
    import robosuite
    import dexmimicgen  # registers the task classes
    meta = metadata(path)
    kwargs = deepcopy(meta['env_kwargs'])
    kwargs.update(env_name=meta['env_name'], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    kwargs.pop('env_lang', None)  # follows official playback
    return robosuite.make(**kwargs)

def restore_initial(env, demo):
    official_playback().reset_to(env, {'states': demo['states'][0], 'model': demo['model'], 'ep_meta': demo['ep_meta']})

def action_layout(env):
    offsets = []
    start = 0
    for index, robot in enumerate(env.robots):
        stop = start + robot.action_dim
        offsets.append({'robot_index': index, 'robot_name': robot.name if hasattr(robot, 'name') else type(robot).__name__,
                        'start': start, 'stop': stop, 'arms': list(robot.arms),
                        'controller_type': type(robot.composite_controller).__name__,
                        'part_action_slices': {k: list(v) for k, v in robot._action_split_indexes.items()}})
        start = stop
    if start != env.action_dim or len(offsets) != 2:
        raise ValueError(f'Expected two independent robots, got {offsets}; dimension {env.action_dim}')
    return offsets
