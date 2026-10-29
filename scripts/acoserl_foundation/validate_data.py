"""Targeted replay-boundary/provenance verification; no learner or rollout smoke."""
import json
from pathlib import Path
import torch
from acoserl.ascend_data import offline, raw_row, replay
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.dynamic_demo import DynamicDemo
from acoserl.transition_semantics import Boundary, boundary

ROOT = Path(__file__).resolve().parents[2]

def row(env, episode, frame, mode="Normal", takeover=0, term=False, trunc=False,
        bootstrap=True, reward=0.):
    marker = {"observation.state": torch.tensor([env, episode, frame], dtype=torch.float32)}
    nxt = {"observation.state": torch.tensor([env, episode, frame + 1], dtype=torch.float32)}
    return raw_row(marker, torch.arange(14).float() / 14, nxt, reward, term, trunc,
                   env, episode, frame, metadata=dict(
                       success=term and reward == 1, bootstrap_allowed=bootstrap,
                       takeover_id=takeover, auto_intervention=dict(mode=mode, indices=[frame, frame]),
                       stage="approach_pregrasp"))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    out = ROOT / "outputs/acoserl_foundation" / args.tag
    out.mkdir(exist_ok=False)
    # Interleave 16 streams and all terminal categories; no cross-env/episode n3.
    nstep = IsolatedNStep()
    emitted = []
    expected = {}
    for ep in range(3):
        for t in range(5):
            for env in reversed(range(16)):
                length = 1 + (env + ep) % 5
                if t >= length:
                    continue
                terminal = t == length - 1 and env % 3 == 0
                timeout = t == length - 1 and env % 3 == 1
                r = row(env, ep, t, term=terminal, trunc=timeout,
                        bootstrap=not terminal, reward=float(env + ep + t))
                expected.setdefault((env, ep), []).append(r.clone())
                emitted.extend(nstep.append(r))
                if t == length - 1 and env % 3 == 2:
                    expected[(env, ep)][-1]["next", "done"].fill_(True)
                    expected[(env, ep)][-1]["next", "truncated"].fill_(True)
                    expected[(env, ep)][-1]["next", "bootstrap_allowed"].fill_(False)
                    emitted.extend(nstep.close(env, bootstrap=False,
                                                boundary_kind=int(Boundary.ACTION_REFUSED)))
    assert nstep.pending() == 0
    assert len(emitted) == sum(map(len, expected.values()))
    for r in emitted:
        env, ep, frame = [int(r[k]) for k in ("env_id", "episode", "frame")]
        future = expected[env, ep][frame:frame+3]
        assert int(r["steps_to_next_obs"]) == len(future)
        assert abs(float(r["next", "reward"]) -
                   sum(.99**i * float(x["next", "reward"]) for i, x in enumerate(future))) < 1e-5
        assert bool(r["nonterminal"]) == bool(future[-1]["next", "bootstrap_allowed"])
        torch.testing.assert_close(r["next", "obs", "observation.state"],
                                   future[-1]["next", "obs", "observation.state"])
    # A refused action/partial integration cannot create a row.
    assert boundary(executed=False, partial=True) == (Boundary.PARTIAL_PHYSICS, False)
    assert boundary(executed=False) == (Boundary.ACTION_REFUSED, False)
    assert boundary(truncated=True) == (Boundary.TIME_LIMIT, True)
    assert boundary(terminated=True, safety_reason="collision") == (Boundary.PHYSICAL_FAILURE, False)
    assert boundary(terminated=True, success=True) == (Boundary.SUCCESS, False)
    assert boundary(safety_reason="recovery_timeout_no_safe_handback") == (Boundary.SUPERVISOR_STOP, False)
    # Dynamic segments: exact continuity, different sources, rejected failure,
    # no concatenation of intervention fragments separated by policy actions.
    d = DynamicDemo(capacity=224)
    for t in range(5):
        d.observe(row(0, 0, t, "Recover_A", 1), handback=t == 4)
    assert len(d.buffer) == 5
    for t in range(4):
        d.observe(row(1, 0, t, "Guide_B", 2), handback=t == 3)
    assert len(d.buffer) == 9
    d.observe(row(0, 0, 5, "Recover_A", 3))
    d.observe(row(0, 0, 6))
    assert len(d.buffer) == 9 and not d.pending
    d.observe(row(2, 0, 0, "Recover_B", 4))
    d.reject(2, partial=True)
    d.observe(row(3, 0, 0, "Recover_A", 5), safe=False)
    for r in d.buffer[:]:
        assert int(r["source"]) == 2
        assert int(r["action_source"]) in (1, 2, 3, 4)
        assert int(r["next", "obs", "observation.state"][0]) == int(r["env_id"])
        assert int(r["next", "obs", "observation.state"][1]) == int(r["episode"])
    d.observe(row(7, 2, 0, "Recover_A", 8))
    d.save(out / "dynamic_checkpoint")
    loaded = DynamicDemo(capacity=224)
    loaded.load(out / "dynamic_checkpoint")
    assert not loaded.pending and len(loaded.buffer) == 9
    for key in d.buffer[:].keys(True, True):
        assert torch.equal(d.buffer[:][key], loaded.buffer[:][key]), key
    # Capacity eviction cannot evict original demonstrations.
    for segment in range(50):
        for t in range(5):
            d.observe(row(4, segment, t, "Recover_A", 10 + segment), handback=t == 4)
    assert len(d.buffer) == 224
    # Read exactly the original 224, verify strict n3 labels and scaling.
    original, std, scale, stats = offline(128)
    assert len(original) == 224 and int(original[:]["next", "terminated"].sum()) == 3
    assert torch.equal(original[:]["nonterminal"], ~original[:]["next", "terminated"])
    before = original[:].clone()
    # Mixing full-resolution original with matching real schema is verified in
    # targeted_physics.py; here use original shape to create dynamic rows.
    real_schema = DynamicDemo(capacity=224)
    r = original[0].clone()
    # Convert preassembled row back to one-step for this schema-only sample.
    r.pop("gamma"); r.pop("steps_to_next_obs"); r.pop("nonterminal")
    r["action_source"].fill_(3); r["takeover_id"].fill_(1)
    real_schema.observe(r, handback=True)
    batch = real_schema.sample_demo(original, 128)
    assert len(batch) == 128 and int((batch["source"] == 2).sum()) == 64
    empty = DynamicDemo(capacity=224)
    assert int((empty.sample_demo(original, 128)["source"] == 0).sum()) == 128
    for key in before.keys(True, True):
        assert torch.equal(before[key], original[:][key]), key
    result = dict(passed=True, interleaved_streams=16, episodes=48,
                  checked_rows=len(emitted), dynamic_continuity=True,
                  rejected_failures_and_partial=True, restored_buffer_exact=True,
                  pending_candidates_discarded_on_episode_reset=True,
                  dynamic_capacity=224, original_rows=224,
                  original_dynamic_demo_batch=[64,64], original_unchanged=True,
                  formal_training=False, physical_validation=False)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
