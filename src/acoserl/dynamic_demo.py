"""Admit only contiguous, physically complete interventions with safe handback.

Replay implementation and sampling remain upstream TorchRL components.
Online Replay is independent: rejected candidates must still be collected there.
"""
from pathlib import Path
import json
import torch
from acoserl.ascend_data import replay
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.transition_semantics import Boundary

class DynamicDemo:
    def __init__(self, capacity=5000, original_fraction=.5, n=3, gamma=.99):
        assert capacity > 0 and 0 <= original_fraction <= 1
        self.buffer = replay(capacity, 128)
        self.original_fraction = original_fraction
        self.n, self.gamma = n, gamma
        self.pending = {}
        self.stats = dict(accepted_segments=0, accepted_rows=0, rejected_segments=0,
                          rejected_rows=0, rejected_partial_steps=0)
        self.max_segment_steps = 350

    def reject(self, env, partial=False):
        rows = self.pending.pop(env, [])
        if rows:
            self.stats["rejected_segments"] += 1
            self.stats["rejected_rows"] += len(rows)
        self.stats["rejected_partial_steps"] += int(partial)

    def observe(self, row, handback=False, safe=True):
        env, ep, frame, source, takeover = [int(row[k]) for k in
            ("env_id", "episode", "frame", "action_source", "takeover_id")]
        if source not in (1, 2, 3, 4, 5):
            self.reject(env)
            return
        if not safe or not bool(row["complete_physics"]):
            self.reject(env)
            self.stats["rejected_rows"] += 1
            return
        rows = self.pending.setdefault(env, [])
        if rows and (int(rows[-1]["episode"]) != ep or
                     int(rows[-1]["frame"]) + 1 != frame or
                     int(rows[-1]["takeover_id"]) != takeover):
            self.reject(env)
            rows = self.pending.setdefault(env, [])
        rows.append(row.clone())
        if len(rows) > self.max_segment_steps:
            self.reject(env)
            return
        # A failure or time limit before handback does not become a demonstration.
        if bool(row["next", "done"]) and not handback:
            self.reject(env)
            return
        if not handback:
            return
        assembler = IsolatedNStep(self.n, self.gamma)
        for item in rows:
            item = item.clone()
            item["source"].fill_(2)
            for out in assembler.append(item):
                self.buffer.add(out)
        # A continuous intervention segment is not an entire task success.
        for out in assembler.close(env, bootstrap=True,
                                   boundary_kind=int(Boundary.DEMO_SEGMENT_END)):
            self.buffer.add(out)
        self.stats["accepted_segments"] += 1
        self.stats["accepted_rows"] += len(rows)
        self.pending.pop(env)

    def sample_demo(self, original, count):
        assert len(original) == 224
        dynamic_count = round(count * (1 - self.original_fraction)) if len(self.buffer) else 0
        parts = []
        if count - dynamic_count:
            parts.append(original.sample(count - dynamic_count).clone())
        if dynamic_count:
            parts.append(self.buffer.sample(dynamic_count).clone())
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(exist_ok=False)
        if len(self.buffer):
            self.buffer.dumps(directory / "buffer")
        torch.save(dict(pending=self.pending, stats=self.stats,
                        original_fraction=self.original_fraction,
                        capacity=self.buffer._storage.max_size,
                        n=self.n, gamma=self.gamma), directory / "state.pt")
        (directory / "metadata.json").write_text(json.dumps(
            dict(rows=len(self.buffer), **self.stats), indent=2))

    def load(self, directory, discard_pending=True):
        directory = Path(directory)
        state = torch.load(directory / "state.pt", weights_only=False, map_location="cpu")
        assert state["original_fraction"] == self.original_fraction
        assert state["n"] == self.n and state["gamma"] == self.gamma
        assert state["capacity"] == self.buffer._storage.max_size
        self.stats = state["stats"]
        self.pending = state["pending"]
        if (directory / "buffer").exists():
            self.buffer.loads(directory / "buffer")
        if discard_pending:
            for env in list(self.pending):
                self.reject(env)
