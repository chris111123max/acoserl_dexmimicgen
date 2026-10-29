"""Paired policies, same canonical reset, original bridge/scaler/strict protocol."""
from pathlib import Path
import json,subprocess,time,argparse,numpy as np
from acoserl.parallel_autoserl import AutoSERLWorker,WorkerFactory
from acoserl.bimanual_autoserl import LegacyBimanualAutoSERL
ROOT=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('--tag',default='paired_verified');a=p.parse_args();OUT=ROOT/'outputs/autoserl_two_point'/a.tag;OUT.mkdir(exist_ok=False)
norm=json.loads((ROOT/'docs/ascend16/normalization.json').read_text());old=json.loads(subprocess.check_output(['git','show','bef4907:configs/bimanual_autoserl/v1.json'],cwd=ROOT,text=True));results=[]
for mode in ['before','after']:
 w=AutoSERLWorker(WorkerFactory(0,90210,norm,str(OUT/mode),True))
 try:
  if mode=='before':w.bridge.manager=LegacyBimanualAutoSERL(w.w.env,old,True)
  else:w.bridge.manager.cfg['development_probe']=True
  for seed in [71,72,73,74]:
   w.reset(seed=seed);rng=np.random.default_rng(seed);record={'method':mode,'seed':seed,'complete_steps':0,'partial_attempts':0,'interventions':0,'recoveries':0,'stage_flags':{},'windows':[]};start=time.perf_counter();reason=None
   for t in range(100):
    action=rng.uniform(-1,1,14).astype(np.float32);obs,r,term,trunc,info=w.step(action);x=json.loads(info['payload']);record['complete_steps']+=int(x['executed']);record['partial_attempts']+=int(bool(x.get('partial_physics_step',False)));d=x.get('auto_intervention') or {};after=x.get('recovery_result') or {}
    record['interventions']+=int(d.get('recovery_step')==1);record['recoveries']+=int(after.get('handback',False));record['windows'].append(d.get('indices'))
    for k,v in x['task_metrics'].items():record['stage_flags'][k]=bool(record['stage_flags'].get(k,False) or v)
    if term or trunc:reason=x.get('supervisor_stop_reason');break
   else:w.collector_cut('bounded_episode_100_steps')
   record.update(reason=reason,seconds=time.perf_counter()-start,proposals=t+1);results.append(record);print(json.dumps(record),flush=True)
 finally:w.close()
(OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
