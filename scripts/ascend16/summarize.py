"""Bounded-memory summary of actual persisted training records (no torch import)."""
import argparse,json,time,collections
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--tag');a=p.parse_args()
tag=a.tag or (root/'logs/ascend16/formal.tag').read_text().strip();out=root/'outputs/ascend16'/tag
result={'tag':tag,'time':time.time(),'workers':[]}
for name in ['config.json','summary.json']:
 f=out/name
 if f.exists():result[name]=json.loads(f.read_text())
for name in ['learning_curve.jsonl','updates.jsonl']:
 f=out/name
 if f.exists():
  with f.open('rb') as h:h.seek(max(0,f.stat().st_size-32768));lines=h.read().splitlines()
  if lines:result[name]=json.loads(lines[-1])
for f in sorted((out/'workers').glob('env*.jsonl')):
 stats={'env_id':int(f.stem[3:]),'real_steps':0,'resets':0,'episodes':0,'interventions':0,'recoveries':0,'safe_stops':0,'reasons':collections.Counter(),'lengths':collections.Counter()}
 with f.open() as h:
  for line in h:
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if r['event']=='reset_ready':stats['resets']+=1
   elif r['event']=='episode_end':
    stats['episodes']+=1;stats['lengths'][r['length']]+=1
    if r.get('reason'):stats['safe_stops']+=1;stats['reasons'][r['reason']]+=1
   elif r['event']=='step' and r.get('executed'):
    stats['real_steps']+=1;d=r.get('auto_intervention') or {};after=r.get('recovery_result') or {}
    if d.get('mode','').startswith('Recover_') and d.get('recovery_step')==1:stats['interventions']+=1
    if after.get('handback'):stats['recoveries']+=1
 result['workers'].append(stats)
telemetry=out/'npu_telemetry.jsonl';util=[];hbm=[]
if telemetry.exists():
 with telemetry.open() as f:
  for line in f:
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   for key,target in [('aicore_percent',util),('hbm_used_mb',hbm)]:
    if isinstance(r.get(key),(int,float)):target.append(r[key])
result['npu_statistics']={'samples':len(util),'aicore_mean_percent':sum(util)/len(util) if util else None,'aicore_max_percent':max(util) if util else None,'hbm_peak_mb':max(hbm) if hbm else None}
result['evaluations']=[json.loads(f.read_text()) for f in sorted(out.glob('evaluation_*.json'))]
path=out/'current_statistics.json';temp=path.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
print(json.dumps({k:v for k,v in result.items() if k not in ['config.json','summary.json','evaluations']},indent=2))
