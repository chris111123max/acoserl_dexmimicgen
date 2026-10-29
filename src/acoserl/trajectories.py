"""Independent reference cursors retain the joint demonstration time index."""
from dataclasses import dataclass
import h5py

@dataclass
class ReferenceCursor:
    length: int
    index: int = 0

    def advance(self, steps=1):
        if steps < 0:
            raise ValueError('Use seek for recovery')
        self.index = min(self.length - 1, self.index + steps)
        return self.index

    def seek(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        self.index = index

def load_references(path):
    with h5py.File(path, 'r') as f:
        result = {arm: {k: v[()] for k, v in f[arm].items()} for arm in ('trajectory_A', 'trajectory_B')}
    return result, {arm: ReferenceCursor(len(data['time_index'])) for arm, data in result.items()}

@dataclass
class SingleDemoReference:
    source_demo: str
    source_dataset: str
    env_metadata: dict
    joint_actions: object
    joint_states: object
    time_index: object
    arms: dict
    object_poses: dict
    original_subtask_signals: dict
    cursors: dict


def load_single_reference(path):
    """Load exactly one extracted demo with its joint and task context."""
    import json
    arms, cursors = load_references(path)
    with h5py.File(path, 'r') as f:
        result = SingleDemoReference(
            source_demo=f.attrs['source_demo'], source_dataset=f.attrs['source_dataset'],
            env_metadata=json.loads(f.attrs['env_args']), joint_actions=f['joint_actions'][()],
            joint_states=f['joint_states'][()], time_index=f['time_index'][()],
            arms=arms, object_poses={k:v[()] for k,v in f.get('original_datagen_info/object_poses',{}).items()},
            original_subtask_signals={k:v[()] for k,v in f.get('task_events/original_subtask_term_signals',{}).items()},
            cursors=cursors)
    return result

@dataclass
class SlidingReferenceWindow:
    """Trailing L-point window. Endpoint initially zero; legal event cap is external."""
    length: int
    total_states: int
    end: int = 0
    floor: int = 0

    def indices(self, legal_end):
        stop = min(self.end, legal_end, self.total_states - 1)
        start = max(self.floor, stop - self.length + 1)
        return list(range(start, stop + 1)) if start <= stop else []

    def tick(self, legal_end, frozen=False):
        if not frozen:
            self.end = max(self.end, min(self.end + 1, legal_end, self.total_states - 1))

    def event(self, index):
        self.floor = max(self.floor, index)
        self.end = max(self.end, self.floor)
