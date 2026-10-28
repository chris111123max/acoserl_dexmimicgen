"""Read-only formal learner status; works without importing torch or NPU."""
import json,os,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
tag=(root/'logs/ascend16/formal.tag').read_text().strip();out=root/'outputs/ascend16'/tag
result={'tag':tag,'time':time.time()}
for name in ['formal','learner']:
 path=root/f'logs/ascend16/{name}.pid'
 if path.exists():
  pid=int(path.read_text());result[name+'_pid']=pid;result[name+'_alive']=Path(f'/proc/{pid}').exists()
for name in ['summary.json','resume_verified.json']:
 path=out/name
 if path.exists():result[name]=json.loads(path.read_text())
path=out/'workers/startup.jsonl'
if path.exists():result['startup']=[json.loads(s) for s in path.read_text().splitlines()]
for name in ['learning_curve.jsonl','npu_telemetry.jsonl']:
 path=out/name
 if path.exists():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-32768));lines=f.read().splitlines()
  if lines:result[name]=json.loads(lines[-1])
print(json.dumps(result,indent=2))
