"""Read-only, bounded I/O status for the new formal run."""
from pathlib import Path
import json,time
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'logs/autoserl_two_point';tag=(base/'formal.tag').read_text().strip();out=ROOT/'outputs/ascend16'/tag;r={'time':time.time(),'tag':tag}
for name in ['learner','launcher']:
 p=base/(name+'.pid')
 if p.exists():pid=int(p.read_text());r[name+'_pid']=pid;r[name+'_alive']=Path(f'/proc/{pid}').exists()
for name in ['summary.json','config.json']:
 p=out/name
 if p.exists():r[name]=json.loads(p.read_text())
for name in ['learning_curve.jsonl','updates.jsonl','npu_telemetry.jsonl']:
 p=out/name
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-65536));rows=f.read().splitlines()
  for line in reversed(rows):
   try:r[name]=json.loads(line);break
   except json.JSONDecodeError:pass
p=out/'workers/startup.jsonl'
if p.exists():r['startup']=[{'event':x['event'],'group':x.get('group'),'seconds':x.get('seconds')} for x in map(json.loads,p.read_text().splitlines())]
print(json.dumps(r,indent=2))
