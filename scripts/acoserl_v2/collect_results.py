"""Summarize preserved real V2 runs; never infer success from process existence."""
import argparse,json,statistics
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
def read(path):
 return json.loads(path.read_text()) if path.exists() else {}
def records(path):
 if not path.exists():return []
 out=[]
 for line in path.read_text().splitlines():
  try:out.append(json.loads(line))
  except json.JSONDecodeError:pass
 return out
def collect(tag):
 p=ROOT/'outputs/ascend16'/tag;s=read(p/'summary.json');c=s.get('counters',{});cfg=read(p/'config.json')
 initial=read(p/'initialization.json');curves=records(p/'learning_curve.jsonl')
 if not c and curves:c=curves[-1]
 logs=[x for f in (p/'workers').glob('env[0-9][0-9].jsonl') for x in records(f) if x.get('event')=='step']
 sources=Counter(x.get('action_source') for x in logs if x.get('executed'));stops=Counter(x['supervisor_stop_reason'] for x in logs if x.get('supervisor_stop_reason'))
 starts=set();handbacks=[];actor_follow=0
 for f in (p/'workers').glob('env[0-9][0-9].jsonl'):
  rows=[x for x in records(f) if x.get('event')=='step']
  for i,x in enumerate(rows):
   d=x.get('auto_intervention') or {};tid=d.get('takeover_id')
   if tid and x.get('action_source')!='Policy':starts.add((x['env_id'],x['episode_id'],tid))
   if (x.get('recovery_result') or {}).get('safe_handback'):
    handbacks.append((x['env_id'],x['episode_id'],tid,x['frame']))
    actor_follow+=int(i+1<len(rows) and rows[i+1]['episode_id']==x['episode_id'] and rows[i+1].get('executed') and rows[i+1].get('action_source') in ('Policy','Constrained_A','Constrained_B','Constrained_Joint'))
 telemetry=records(p/'npu_telemetry.jsonl');util=[x['aicore_percent'] for x in telemetry if x.get('aicore_percent') is not None];hbm=[x['hbm_used_mb'] for x in telemetry if x.get('hbm_used_mb') is not None]
 updates=records(p/'updates.jsonl');grad={k:[x[k] for x in updates if k in x] for k in ['train/encoder_grad_norm','train/actor_grad_norm','train/critic_grad_norm']}
 startup=records(p/'workers/startup.jsonl');evaluations=[read(f) for f in p.glob('evaluation_*.json')]
 wall=s.get('training_seconds',0);timing=s.get('timing',curves[-1].get('timing',{}) if curves else {})
 return dict(tag=tag,status=s.get('status','incomplete'),error=s.get('error'),method=cfg.get('manager_config',{}).get('method'),admission=cfg.get('manager_config',{}).get('admission_mode'),source_commit=cfg.get('source_commit'),counters=c,online_rows=s.get('online_rows'),offline_rows=s.get('offline_rows'),dynamic_demo_rows=s.get('dynamic_demo_rows'),dynamic_demo_stats=s.get('dynamic_demo_stats'),valid_steps_s=s.get('valid_steps_s'),updates_s=s.get('updates_s'),utd_effective=s.get('utd_effective'),wall_seconds=s.get('wall_seconds'),training_seconds=wall,timing_seconds=timing,initial_actor_hash=initial.get('actor_hash'),initial_critic_hash=initial.get('critic_hash'),all_sixteen_ready=any(x.get('event')=='all_sixteen_ready' for x in startup),startup_events=[{k:x.get(k) for k in ['event','time','group','ids']} for x in startup],workers=s.get('workers'),actual_action_sources=dict(sources),distinct_takeovers=len(starts),safe_handbacks=len(handbacks),handback_then_actor_steps=actor_follow,stop_reasons=dict(stops),restoration_checks=s.get('restoration_checks'),visual_audit=read(p/'visual_audit.json'),npu=dict(samples=len(telemetry),utilization_mean=statistics.mean(util) if util else None,utilization_peak=max(util) if util else None,hbm_peak_mb=max(hbm) if hbm else None,allocated_peak_bytes=s.get('peak_npu_allocated_bytes')),gradient_norm_ranges={k:[min(v),max(v)] if v else None for k,v in grad.items()},evaluations=evaluations,complete_logged_transitions=sum(bool(x.get('executed')) for x in logs),partial_attempts=sum(bool(x.get('partial_physics_step')) for x in logs),mean_worker_step_seconds=statistics.mean([x['step_seconds'] for x in logs if x.get('executed')]) if logs else None,intervention_steps=sum(n for name,n in sources.items() if name!='Policy'),checkpoint=s.get('checkpoint'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('tags',nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=[collect(t) for t in a.tags];a.output.write_text(json.dumps(result,indent=2)+'\n')
 for x in result:print({k:x[k] for k in ['tag','status','complete_logged_transitions','safe_handbacks','handback_then_actor_steps','dynamic_demo_rows','valid_steps_s']})
