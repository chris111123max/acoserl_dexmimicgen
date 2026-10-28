"""Integrity inspection, deterministic single-demo extraction and formal replay."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from acoserl.data import inspect, load_demo
from acoserl.envs import create, restore_initial, action_layout

ROOT = Path(__file__).resolve().parents[1]

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))

def check(path):
    manifest = json.loads((ROOT / 'docs/hf_manifest.json').read_text())
    expected = next(x for x in manifest if x['path'] == 'generated/' + path.name)
    report = inspect(path, expected)
    write_json(ROOT / 'docs' / (path.stem + '_data.json'), report)
    return report

def poses(env):
    from robosuite.utils.transform_utils import mat2quat
    result = []
    for robot in env.robots:
        if len(robot.arms) != 1:
            raise ValueError('This benchmark expects one arm per robot')
        site = robot.eef_site_id[robot.arms[0]]
        result.append((env.sim.data.site_xpos[site].copy(), mat2quat(env.sim.data.site_xmat[site].reshape(3, 3)).copy()))
    return result

def replay(path, demo_id, mode, report_tag=None):
    demo = load_demo(path, demo_id)
    env = create(path)
    try:
        restore_initial(env, demo)
        if demo['actions'].shape[1] != env.action_dim:
            raise ValueError('Recorded and environment action dimensions disagree')
        errors, successes = [], []
        qpos_errors, qvel_errors = [], []
        nq = env.sim.model.nq
        for i in range(len(demo['actions'])):
            if mode == 'actions':
                env.step(demo['actions'][i])
                if i + 1 < len(demo['states']):
                    actual = env.sim.get_state().flatten()
                    expected = demo['states'][i + 1]
                    errors.append(float(np.linalg.norm(actual - expected)))
                    qpos_errors.append(float(np.linalg.norm(actual[1:1+nq] - expected[1:1+nq])))
                    qvel_errors.append(float(np.linalg.norm(actual[1+nq:] - expected[1+nq:])))
            else:
                env.sim.set_state_from_flattened(demo['states'][i])
                env.sim.forward()
            successes.append(bool(env._check_success()))
        report = {'dataset': str(path), 'demo_id': demo_id, 'mode': mode, 'steps': len(successes),
                  'success_any': any(successes), 'success_final': successes[-1],
                  'first_success_step': next((i for i, s in enumerate(successes) if s), None),
                  'state_error_l2_mean': float(np.mean(errors)) if errors else None,
                  'state_error_l2_max': max(errors) if errors else None,
                  'state_error_l2_final_comparable': errors[-1] if errors else None,
                  'state_error_per_step': errors, 'success_per_step': successes,
                  'qpos_error_l2_max': max(qpos_errors) if qpos_errors else None,
                  'qvel_error_l2_max': max(qvel_errors) if qvel_errors else None,
                  'nq': int(nq), 'nv': int(env.sim.model.nv),
                  'control_freq': env.control_freq, 'model_timestep': float(env.sim.model.opt.timestep),
                  'action_layout': action_layout(env),
                  'initial_state_only_restored': mode == 'actions',
                  'success_criterion': type(env).__name__ + '._check_success (unmodified official implementation)',
                  'note': 'L2 mixes state units; exact reproducibility and task success are separate criteria.'}
        write_json(ROOT / 'docs' / (path.stem + '_' + demo_id + '_' + mode + '_replay' + ('_' + report_tag if report_tag else '') + '.json'), report)
        return report
    finally:
        env.close()

def extract(path, demo_id):
    demo = load_demo(path, demo_id)
    env = create(path)
    try:
        restore_initial(env, demo)
        layout = action_layout(env)
        available_signals = {}
        with h5py.File(path, 'r') as source:
            source_demo = source['data'][demo_id]
            for key, value in source_demo.get('datagen_info/subtask_term_signals', {}).items():
                signal = value[()]
                available_signals[key] = {'origin': 'original datagen_info/subtask_term_signals',
                                         'transition_indices': (np.flatnonzero(np.diff(signal) != 0) + 1).tolist()}
            for item in layout:
                for side in ('left', 'right'):
                    key = 'action_dict/' + side + '_rel_pos'
                    if key in source_demo:
                        v = source_demo[key][()]
                        start = item['start']
                        if v.ndim == 2 and np.allclose(v, demo['actions'][:, start:start+v.shape[1]]):
                            item['source_action_dict_side'] = side
        if demo['actions'].shape[1] != env.action_dim:
            raise ValueError('Invalid recorded action dimension')
        positions, quats = [[], []], [[], []]
        success = []
        objects = {}
        for state in demo['states']:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            for i, (position, quat) in enumerate(poses(env)):
                if quats[i] and np.dot(quats[i][-1], quat) < 0:
                    quat = -quat
                positions[i].append(position)
                quats[i].append(quat)
            success.append(bool(env._check_success()))
            obs = env._get_observations(force_update=True)
            for k, v in obs.items():
                a = np.asarray(v)
                if not k.startswith('robot') and a.ndim <= 1 and np.issubdtype(a.dtype, np.number):
                    objects.setdefault(k, []).append(a.copy())
        if not any(success):
            raise ValueError(f'{demo_id} does not satisfy recorded-state success; do not label it successful')
        output = ROOT / 'datasets/single_demo' / (path.stem + '_' + demo_id + '.hdf5')
        if output.exists():
            raise FileExistsError(output)
        n = len(demo['actions'])
        with h5py.File(output, 'x') as f:
            f.attrs.update(source_dataset=str(path), source_demo=demo_id, env_args=json.dumps(demo['metadata']),
                           selection_basis='First explicitly requested candidate with verified recorded-state task success',
                           quaternion_order='xyzw', pose_frame='MuJoCo world', pose_origin='derived from recorded simulation states and grip_site',
                           action_layout=json.dumps(layout), model_file=demo['model'])
            if demo['ep_meta'] is not None:
                f.attrs['ep_meta'] = demo['ep_meta']
            f.create_dataset('joint_actions', data=demo['actions'])
            f.create_dataset('joint_states', data=demo['states'])
            f.create_dataset('time_index', data=np.arange(n))
            f.create_dataset('time_seconds', data=np.arange(n) / env.control_freq)
            f.create_dataset('recorded_state_success', data=success)
            for i, arm in enumerate(('trajectory_A', 'trajectory_B')):
                g = f.create_group(arm)
                g.attrs['robot_index'] = i
                g.create_dataset('time_index', data=np.arange(n))
                g.create_dataset('eef_position', data=positions[i])
                g.create_dataset('eef_quaternion_xyzw', data=quats[i])
                g.create_dataset('actions', data=demo['actions'][:, layout[i]['start']:layout[i]['stop']])
            g = f.create_group('object_state_derived')
            for k, v in objects.items():
                g.create_dataset(k, data=v)
            g = f.create_group('original_low_dim_obs')
            for k, v in demo['obs'].items():
                g.create_dataset(k, data=v)
            events = f.create_group('task_events')
            events.attrs['annotation_status'] = 'Original subtask signals retained; synchronization barriers unannotated'
            with h5py.File(path, 'r') as source:
                source_demo = source['data'][demo_id]
                for key in ('datagen_info', 'action_dict'):
                    if key in source_demo:
                        source.copy(source_demo[key], f, name='original_' + key)
                if 'datagen_info/subtask_term_signals' in source_demo:
                    source.copy(source_demo['datagen_info/subtask_term_signals'], events, name='original_subtask_term_signals')
        event_path = ROOT / 'configs' / (path.stem + '_events.json')
        if not event_path.exists():
            write_json(event_path, {'source_demo': demo_id, 'annotation_status': 'synchronization barriers unannotated', 'events': [],
                                   'available_original_subtask_signals': available_signals,
                                   'schema': {'event_id': 'string', 'reference_index_A': 'int', 'reference_index_B': 'int',
                                              'barrier': 'bool', 'coupled_contact': 'bool', 'evidence': 'string'}})
        report = {'dataset': str(path), 'demo_id': demo_id, 'output': str(output), 'length': n,
                  'action_layout': layout, 'recorded_state_success_any': any(success), 'recorded_state_success_final': success[-1],
                  'pose_origin': 'derived', 'quaternion_order': 'xyzw', 'pose_frame': 'world',
                  'object_fields_derived': list(objects), 'event_annotations': str(event_path),
                  'original_subtask_signals': available_signals}
        write_json(ROOT / 'docs' / (path.stem + '_selection.json'), report)
        return report
    finally:
        env.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['inspect', 'extract', 'replay'])
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--demo', default='demo_0')
    p.add_argument('--mode', choices=['states', 'actions'], default='actions')
    p.add_argument('--report-tag', default=None)
    a = p.parse_args()
    path = a.dataset.resolve()
    if a.command == 'inspect':
        r = check(path)
        print(json.dumps({k: v for k, v in r.items() if k not in ('episodes', 'metadata')}, indent=2))
    elif a.command == 'extract':
        print(json.dumps(extract(path, a.demo), indent=2))
    else:
        r = replay(path, a.demo, a.mode, a.report_tag)
        print(json.dumps({k: v for k, v in r.items() if k not in ('success_per_step', 'state_error_per_step')}, indent=2))
        if not r['success_any']:
            raise SystemExit(2)
