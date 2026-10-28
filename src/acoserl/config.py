"""Configuration readers with explicit reference bounds and event evidence."""
import json
from pathlib import Path

def load_config(path):
    path = Path(path).resolve()
    with path.open() as stream:
        result = json.load(stream)
    return result

def load_events(path, reference_length_A, reference_length_B):
    config = load_config(path)
    seen = set()
    for event in config['events']:
        event_id = event['event_id']
        if event_id in seen:
            raise ValueError(f'Duplicate event {event_id}')
        seen.add(event_id)
        for arm, length in [('A',reference_length_A),('B',reference_length_B)]:
            index = event['reference_index_'+arm]
            if not isinstance(index,int) or not 0 <= index < length:
                raise ValueError(f'{event_id}: invalid {arm} index {index}')
        if not isinstance(event['barrier'],bool) or not isinstance(event['coupled_contact'],bool):
            raise ValueError(f'{event_id}: event flags must be booleans')
        if not event.get('evidence'):
            raise ValueError(f'{event_id}: event requires observed task-state evidence')
    return config
