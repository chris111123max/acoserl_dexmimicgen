"""Render the Chinese report using measured evidence, without touching a learner."""
from pathlib import Path
import json
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
template=(ROOT/'docs/pretrained_visual/report_template.md').read_text(encoding='utf-8')
tag=(ROOT/'logs/pretrained_visual/formal.tag').read_text().strip();out=ROOT/'outputs/ascend16'/tag
config=json.loads((out/'config.json').read_text())
p=out/'learning_curve.jsonl';latest=json.loads(p.read_text().splitlines()[-1]) if p.exists() else None
status=('16 workers initialized; actual_interactions='+str(latest['actual_interactions'])+', critic_updates='+str(latest['critic_updates'])+', actor_updates='+str(latest['actor_updates'])+', FPS='+str(round(latest['actual_steps_s'],3))) if latest else 'learner alive, initializing four batches; no collection/update claimed yet'
status='as of '+datetime.now().astimezone().isoformat()+'; '+status
p=out/'initialization.json';init=json.loads(p.read_text()) if p.exists() else {}
fresh=json.dumps({k:init.get(k) for k in ['fresh_rl','fresh_online_replay','online_rows','offline_rows']})
text=template.replace('__RUN_TAG__',tag).replace('__SOURCE_COMMIT__',config['source_commit']).replace('__STATUS__',status).replace('__FRESH__',fresh).replace('__STATUS_FILE__','Current bounded status snapshot: docs/pretrained_visual/formal_status.json.')
(ROOT/'docs/pretrained_visual/REPORT_ZH.md').write_text(text,encoding='utf-8')
print('REPORT_WRITTEN',len(text),tag,status)
