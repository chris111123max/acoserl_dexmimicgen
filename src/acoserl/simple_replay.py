"""Original + real intervention demo pool for the simple dual-arm baseline.

All tensors retain the existing ResFiT replay schema. Per-arm ownership is
explicit, never inferred from whether Actor and expert happened to agree.
"""
from pathlib import Path
import copy
import json
import torch
from acoserl.ascend_data import raw_row, replay
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.simple_autoserl import EXPERT


def make_row(obs, action, nxt, reward, term, trunc, payload):
    metadata = copy.deepcopy(payload)
    decision = metadata.get('auto_intervention') or {}
    sources = decision.get('per_arm_control', ['Policy', 'Policy'])
    if len(sources) != 2 or any(s not in EXPERT | {'Policy'} for s in sources):
        raise ValueError('unknown simple-baseline action owner')
    # raw_row creates the common tensor schema. It is not used to infer the
    # passive arm as Hold, which the simple baseline never executes.
    metadata['auto_intervention'] = dict(decision, mode='Normal')
    row = raw_row(obs, action, nxt, reward, term, trunc,
                  payload['env_id'], payload['episode_id'], payload['frame'], metadata=metadata)
    mask = torch.zeros(14, dtype=torch.bool)
    pose, grip = [0, 0], [0, 0]
    for ai, source in enumerate(sources):
        if source in EXPERT:
            pose[ai] = 1 if source == 'Guide' else 2
            mask[7*ai:7*ai+6] = True
            if source != 'Guide':
                grip[ai] = 2; mask[7*ai+6] = True
    valid = bool(payload['executed'] and not payload.get('supervisor_stop_reason'))
    row['action_source'].fill_(13 if mask.any() else 0)
    row['expert_action_mask'] = mask if valid else torch.zeros_like(mask)
    row['pose_sources'] = torch.tensor(pose, dtype=torch.long)
    row['gripper_sources'] = torch.tensor(grip, dtype=torch.long)
    row['policy_dimension_mask'] = ~mask
    row['quality_validated'] = torch.tensor(valid and bool(mask.any()))
    return row


class SimpleDemo:
    """Official-style step admission, with isolated real n-step segments."""
    protocol = 'simple_bimanual_demo_v1'

    def __init__(self, capacity=5000, gamma=.99, original_quota=96, positive_quota=2, seed=0):
        self.capacity, self.gamma = capacity, gamma
        self.original_quota, self.positive_quota = original_quota, positive_quota
        self.buffer = replay(capacity, None)
        self.stream = IsolatedNStep(3, gamma)
        self.last = {}
        self.generator = torch.Generator().manual_seed(seed)
        self.stats = dict(accepted_rows=0, rejected_rows=0, sampled_original=0, sampled_dynamic=0,
                          expert_dimensions=0, dual_arm_rows=0)
        self.last_sample = {}
        self._positive = None

    def close_episode(self, env, bootstrap=False, kind=4):
        for row in self.stream.close(env, bootstrap=bootstrap, boundary_kind=kind):
            self.buffer.add(row)
        self.stream.episodes.pop(env, None); self.stream.frames.pop(env, None)
        self.stream.closed.discard(env); self.last.pop(env, None)

    def observe(self, row, **_):
        env, episode, frame, segment = (int(row[k]) for k in ('env_id', 'episode', 'frame', 'takeover_id'))
        identity = (episode, frame, segment)
        previous = self.last.get(env)
        if previous == identity:
            raise ValueError('duplicate real intervention transition')
        if previous is not None and previous != (episode, frame-1, segment):
            ended = bool(row['next', 'done'])
            self.close_episode(env, bool(row['next', 'bootstrap_allowed']) if ended else True,
                               int(row['next', 'boundary_kind']) if ended else 7)
        complete = bool(row['complete_physics'])
        dimensions = int(row['expert_action_mask'].sum())
        if not complete or not bool(row['quality_validated']) or not 0 < dimensions <= 14:
            self.stats['rejected_rows'] += 1
            terminal = bool(row['next', 'done'])
            self.close_episode(env, bool(row['next', 'bootstrap_allowed']) if terminal else complete,
                               int(row['next', 'boundary_kind']) if terminal else 7)
            return
        item = row.clone(); item['source'].fill_(2)
        for transition in self.stream.append(item):
            self.buffer.add(transition)
        self.last[env] = identity
        self.stats['accepted_rows'] += 1
        self.stats['expert_dimensions'] += dimensions
        self.stats['dual_arm_rows'] += int(dimensions > 7)
        if bool(row['next', 'done']):
            self.close_episode(env, bool(row['next', 'bootstrap_allowed']), int(row['next', 'boundary_kind']))

    def sample_demo(self, original, count):
        if count != 128 or len(original) != 224:
            raise ValueError('expected immutable 224-row original and 128 demo batch')
        n = self.original_quota if len(self.buffer) else count
        if self._positive is None:
            self._positive = torch.nonzero(original[:]['next', 'reward'] > 0).flatten()
        forced = self.positive_quota
        if forced and not len(self._positive):
            raise ValueError('original demonstration has no positive reward')
        parts = []
        if forced:
            idx = self._positive[torch.randint(len(self._positive), (forced,), generator=self.generator)]
            parts.append(original[idx].clone())
        if n > forced:
            parts.append(original[torch.randint(len(original), (n-forced,), generator=self.generator)].clone())
        if n < count:
            parts.append(self.buffer[torch.randint(len(self.buffer), (count-n,), generator=self.generator)].clone())
        batch = torch.cat(parts, 0)
        self.stats['sampled_original'] += n; self.stats['sampled_dynamic'] += count-n
        self.last_sample = dict(original=n, dynamic=count-n, forced_positive=forced)
        return batch[torch.randperm(count, generator=self.generator)]

    def save(self, directory):
        directory = Path(directory); directory.mkdir(exist_ok=False)
        if len(self.buffer):
            self.buffer.dumps(directory/'buffer')
        torch.save(dict(protocol=self.protocol, capacity=self.capacity, gamma=self.gamma,
                        original_quota=self.original_quota, positive_quota=self.positive_quota,
                        stream=self.stream, last=self.last, stats=self.stats, last_sample=self.last_sample,
                        generator=self.generator.get_state()), directory/'state.pt')
        (directory/'metadata.json').write_text(json.dumps(dict(rows=len(self.buffer), protocol=self.protocol)))

    def load(self, directory, discard_pending=True):
        directory = Path(directory)
        state = torch.load(directory/'state.pt', map_location='cpu', weights_only=False)
        for key in ('protocol', 'capacity', 'gamma', 'original_quota', 'positive_quota'):
            if state[key] != getattr(self, key):
                raise ValueError('incompatible simple Demo checkpoint: '+key)
        self.buffer = replay(self.capacity, None)
        if (directory/'buffer').exists():
            self.buffer.loads(directory/'buffer')
        self.stream, self.last = state['stream'], state['last']
        self.stats, self.last_sample = state['stats'], state['last_sample']
        self.generator.set_state(state['generator'])
        if discard_pending:
            for env in list(self.stream.queues):
                self.close_episode(env, True, 6)
