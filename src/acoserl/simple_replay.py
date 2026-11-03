"""Original + real intervention demo pool for the simple dual-arm baseline.

All tensors retain the existing ResFiT replay schema. Per-arm ownership is
explicit, never inferred from whether Actor and expert happened to agree.
"""
from collections import deque
from pathlib import Path
import copy
import json
import torch
from acoserl.ascend_data import raw_row, replay
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.simple_autoserl import ARMS, EXPERT, SAFETY_ONLY


def make_row(obs, action, nxt, reward, term, trunc, payload):
    metadata = copy.deepcopy(payload)
    decision = metadata.get('auto_intervention') or {}
    sources = decision.get('per_arm_control', ['Policy', 'Policy'])
    gripper_owners = decision.get('per_arm_gripper_control', [False, False])
    if len(sources) != 2 or any(s not in EXPERT | SAFETY_ONLY | {'Policy'} for s in sources):
        raise ValueError('unknown simple-baseline action owner')
    if len(gripper_owners) != 2 or any(not isinstance(v, bool) for v in gripper_owners):
        raise ValueError('invalid explicit Guide gripper ownership')
    if any(gripper_owners[i] and sources[i] != 'Guide' for i in range(2)):
        raise ValueError('non-Guide claims Guide gripper ownership')
    # raw_row creates the common tensor schema. It is not used to infer the
    # passive arm as Hold, which the simple baseline never executes.
    metadata['auto_intervention'] = dict(decision, mode='Normal')
    row = raw_row(obs, action, nxt, reward, term, trunc,
                  payload['env_id'], payload['episode_id'], payload['frame'], metadata=metadata)
    mask = torch.zeros(14, dtype=torch.bool)
    policy_owned = torch.ones(14, dtype=torch.bool)
    pose, grip = [0, 0], [0, 0]
    safety_arms = [ai for ai, source in enumerate(sources) if source in SAFETY_ONLY]
    for ai, source in enumerate(sources):
        if source in SAFETY_ONLY:
            pose[ai] = 4  # SafetyProjection, NEVER expert or Policy.
            policy_owned[7*ai:7*ai+6] = False
        elif source in EXPERT:
            pose[ai] = 1 if source == 'Guide' else 2
            mask[7*ai:7*ai+6] = True
            policy_owned[7*ai:7*ai+6] = False
            if source != 'Guide' or gripper_owners[ai]:
                grip[ai] = 1 if source == 'Guide' else 2
                mask[7*ai+6] = True
                policy_owned[7*ai+6] = False
    valid = bool(payload['executed'] and not payload.get('supervisor_stop_reason'))
    # Off-policy Online Replay retains the genuine complete action with
    # provenance; the safety intervention never acquires demo eligibility.
    if safety_arms:
        code = 11 if len(safety_arms) == 2 else (9 if safety_arms[0] == 0 else 10)
    else:
        code = 13 if mask.any() else 0
    row['action_source'].fill_(code)
    row['expert_action_mask'] = mask if valid and not safety_arms else torch.zeros_like(mask)
    row['pose_sources'] = torch.tensor(pose, dtype=torch.long)
    row['gripper_sources'] = torch.tensor(grip, dtype=torch.long)
    row['policy_dimension_mask'] = policy_owned
    row['quality_validated'] = torch.tensor(valid and bool(mask.any()) and not safety_arms)
    return row


class SimpleDemo:
    """Admit only actually executed interventions with a verified Policy handback.

    The original demonstration remains available at all times. A bounded tail
    of candidate expert actions is held per worker until all contributing
    arms complete their own handbacks; failed or unqualified Guides never
    become expert demos merely by running for many control steps.
    """
    protocol = 'simple_bimanual_demo_handback_v3_boundary_guard'
    pending_tail = 32

    def __init__(self, capacity=5000, gamma=.99, original_quota=96, positive_quota=2, seed=0):
        self.capacity, self.gamma = capacity, gamma
        self.original_quota, self.positive_quota = original_quota, positive_quota
        self.buffer = replay(capacity, None)
        self.stream = IsolatedNStep(3, gamma)
        self.last = {}
        self.pending = {}
        self.pending_episode = {}
        self.qualified = {}
        self.generator = torch.Generator().manual_seed(seed)
        self.stats = dict(accepted_rows=0, rejected_rows=0, sampled_original=0,
            sampled_dynamic=0, expert_dimensions=0, dual_arm_rows=0,
            staged_rows=0, policy_only_rows=0, expired_candidate_rows=0,
            failed_guide_discarded_rows=0, unqualified_discarded_rows=0,
            verified_handbacks=0, dropped_on_restore_rows=0,
            boundary_guard_rows=0, boundary_guard_discarded_candidate_rows=0)
        self.last_sample = {}
        self._positive = None

    def _close_stream(self, env, bootstrap=False, kind=4):
        for row in self.stream.close(env, bootstrap=bootstrap, boundary_kind=kind):
            self.buffer.add(row)
        self.stream.episodes.pop(env, None)
        self.stream.frames.pop(env, None)
        self.stream.closed.discard(env)
        self.last.pop(env, None)

    def close_episode(self, env, bootstrap=False, kind=4):
        """Close accepted n-step tails and discard unverified expert candidates."""
        self._close_stream(env, bootstrap, kind)
        self.stats['unqualified_discarded_rows'] += len(self.pending.pop(env, ()))
        self.qualified.pop(env, None)
        self.pending_episode.pop(env, None)

    def _admit(self, row):
        """Append one verified real row without crossing gaps or source changes."""
        env, episode, frame, segment = (
            int(row[k]) for k in ('env_id', 'episode', 'frame', 'takeover_id'))
        identity = (episode, frame, segment)
        previous = self.last.get(env)
        if previous == identity:
            raise ValueError('duplicate qualified intervention transition')
        if previous is not None and previous != (episode, frame - 1, segment):
            self._close_stream(env, bootstrap=True, kind=7)
        item = row.clone()
        item['source'].fill_(2)
        # Evidence is from the eventual complete, safe Policy handback, not
        # inferred from a mask or a nominal correction.
        item['quality_handoff_confirmed'].fill_(True)
        for transition in self.stream.append(item):
            self.buffer.add(transition)
        self.last[env] = identity
        self.stats['accepted_rows'] += 1
        dimensions = int(item['expert_action_mask'].sum())
        self.stats['expert_dimensions'] += dimensions
        self.stats['dual_arm_rows'] += int(dimensions > 7)
        if bool(row['next', 'done']):
            self._close_stream(env, bool(row['next', 'bootstrap_allowed']),
                               int(row['next', 'boundary_kind']))

    def observe(self, row, *, decision=None, handbacks=None, control_events=None,
                safe=True, **_):
        env, episode = int(row['env_id']), int(row['episode'])
        previous_episode = self.pending_episode.get(env)
        if previous_episode is not None and previous_episode != episode:
            # A worker reset without its expected episode-close event must
            # never qualify candidates from the previous physical episode.
            self.close_episode(env, bootstrap=True, kind=7)
        self.pending_episode[env] = episode
        safe = bool(safe and row['complete_physics'])
        source_controls = (decision or {}).get('per_arm_control', ())
        if any(source in SAFETY_ONLY for source in source_controls):
            # No safety override (even with B Guide) can be admitted as a
            # dynamic expert demonstration. Discard all pending segments that
            # might otherwise be certified by an unrelated later handback.
            self.stats['boundary_guard_rows'] += 1
            self.stats['boundary_guard_discarded_candidate_rows'] += len(self.pending.get(env, ()))
            self.pending.pop(env, None)
            self.qualified.pop(env, None)
            self._close_stream(env, bootstrap=True, kind=7)
            if bool(row['next','done']):
                self.close_episode(env, bool(row['next','bootstrap_allowed']),
                                   int(row['next','boundary_kind']))
            return

        # A Guide abandoned due to stalled progress or exhausted controls is
        # NOT a handback. Its rows, including dual-arm mixed rows, are invalid.
        failed = set()
        for event in control_events or ():
            if event.get('event') == 'guide_released_to_policy':
                failed.add((ARMS.index(event['arm']), int(event['takeover'])))
            elif event.get('event') in ('guide_stall_to_p0', 'guide_stall_to_stage_recovery'):
                # Any Guide replaced by a stage Recovery did not complete
                # its own safe Policy handback. Its tail must never be
                # implicitly certified by the subsequent Recovery handback.
                failed.add((ARMS.index(event['arm']), int(event['from_takeover'])))
        q = self.pending.get(env)
        if failed and q:
            retained = deque(maxlen=self.pending_tail)
            for item, owners in q:
                if owners & failed:
                    self.stats['failed_guide_discarded_rows'] += 1
                else:
                    retained.append((item, owners))
            self.pending[env] = retained

        dimensions = int(row['expert_action_mask'].sum())
        if dimensions and bool(row['quality_validated']) and safe:
            if decision is None:
                raise ValueError('cannot qualify intervention with missing per-arm provenance')
            sources = decision.get('per_arm_control', ())
            ids = decision.get('arm_takeover_ids', ())
            if len(sources) != 2 or len(ids) != 2:
                raise ValueError('incomplete per-arm takeover identity')
            owners = frozenset((ai, int(ids[ai])) for ai, source in enumerate(sources)
                               if source in EXPERT)
            if not owners or any(takeover <= 0 for _, takeover in owners):
                raise ValueError('unqualified expert takeover identity')
            if not owners & failed:
                q = self.pending.setdefault(env, deque(maxlen=self.pending_tail))
                if len(q) == q.maxlen:
                    q.popleft()
                    self.stats['expired_candidate_rows'] += 1
                q.append((row.clone(), owners))
                self.stats['staged_rows'] += 1
            else:
                self.stats['failed_guide_discarded_rows'] += 1
        elif dimensions:
            self.stats['rejected_rows'] += 1
        else:
            self.stats['policy_only_rows'] += 1

        if safe:
            qualified = self.qualified.setdefault(env, set())
            for handback in handbacks or ():
                arm = handback['arm']
                takeover = int(handback['takeover_id'])
                if arm not in ARMS or takeover <= 0:
                    raise ValueError('invalid physical handback record')
                qualified.add((ARMS.index(arm), takeover))
                self.stats['verified_handbacks'] += 1
            q = self.pending.get(env)
            # Process only a chronologically ordered qualified prefix. Mixed
            # A/B actions wait for BOTH arm IDs; no bad-arm demo is relabeled.
            while q and q[0][1].issubset(qualified):
                item, _ = q.popleft()
                self._admit(item)
        if bool(row['next', 'done']):
            self.close_episode(env, bool(row['next', 'bootstrap_allowed']),
                               int(row['next', 'boundary_kind']))

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
        # Live unverified candidate rows are intentionally NOT checkpointed:
        # worker physics restarts from a new canonical episode.
        pending_counts = {str(env): len(q) for env, q in self.pending.items() if q}
        torch.save(dict(protocol=self.protocol, capacity=self.capacity, gamma=self.gamma,
                        original_quota=self.original_quota, positive_quota=self.positive_quota,
                        stream=self.stream, last=self.last, stats=self.stats,
                        last_sample=self.last_sample, pending_counts=pending_counts,
                        generator=self.generator.get_state()), directory/'state.pt')
        (directory/'metadata.json').write_text(json.dumps(dict(
            rows=len(self.buffer), protocol=self.protocol,
            unqualified_candidate_rows=sum(pending_counts.values()),
            pending_candidates_are_not_resumable=True)))

    def load(self, directory, discard_pending=True):
        directory = Path(directory)
        state = torch.load(directory/'state.pt', map_location='cpu', weights_only=False)
        for key in ('protocol', 'capacity', 'gamma', 'original_quota', 'positive_quota'):
            if state[key] != getattr(self, key):
                raise ValueError('incompatible simple Demo checkpoint: '+key)
        dropped = sum(state.get('pending_counts', {}).values())
        if dropped and not discard_pending:
            raise ValueError('cannot resume unverified candidates across physical reset')
        self.buffer = replay(self.capacity, None)
        if (directory/'buffer').exists():
            self.buffer.loads(directory/'buffer')
        self.stream, self.last = state['stream'], state['last']
        self.stats, self.last_sample = state['stats'], state['last_sample']
        self.generator.set_state(state['generator'])
        self.pending = {}
        self.pending_episode = {}
        self.qualified = {}
        if discard_pending:
            # Saved dynamic rows must match exactly on load. Uncommitted
            # n-step tails cannot cross a worker physics reset, and are
            # discarded with an explicit count rather than fabricated.
            self.stats['dropped_on_restore_rows'] += dropped
            self.stats['dropped_nstep_on_restore_rows'] = (
                self.stats.get('dropped_nstep_on_restore_rows', 0)
                + self.stream.pending())
            self.stream = IsolatedNStep(3, self.gamma)
            self.last = {}
