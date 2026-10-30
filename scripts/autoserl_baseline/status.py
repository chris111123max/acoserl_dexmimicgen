"""Compact read-only status for the CURRENT baseline run; no Torch/NPU import."""
import json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text()) if p.exists() else None
def tail(p):
 if not p.exists():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-65536));rows=f.read().splitlines()
 return json.loads(rows[-1]) if rows else None
logs=ROOT/'logs/autoserl_baseline';tag=(logs/'formal.tag').read_text().strip();out=ROOT/'outputs/ascend16'/tag;pid=int((logs/'learner.pid').read_text());proc=Path(f'/proc/{pid}/cmdline')
config=read(out/'config.json');curve=tail(out/'learning_curve.jsonl');tele=tail(out/'npu_telemetry.jsonl');initial=read(out/'initialization.json');summary=read(out/'summary.json')
startup=out/'workers/startup.jsonl';events=[json.loads(s) for s in startup.read_text().splitlines()] if startup.exists() else [];ready=next((e for e in events if e['event']=='all_sixteen_ready'),None)
workers=ready.get('workers',[]) if ready else [];alive=[w['pid'] for w in workers if Path('/proc/'+str(w['pid'])).exists()]
pid_correct=proc.exists() and tag in proc.read_bytes().decode().split('\0') and '--mode' in proc.read_bytes().decode().split('\0')
gate=read(ROOT/'docs/acoserl_foundation/ACCEPTANCE_AUTOSERL_BASELINE.json');exact=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in gate['tested_source_sha256'].items())
result=dict(tag=tag,output=str(out),time=time.time(),learner_pid=pid,learner_alive_and_command_matches=pid_correct,source_commit=config.get('source_commit') if config else None,tested_source_exact=exact,initialization=initial,ready_workers=len(workers),alive_workers=len(alive),worker_pids=alive,latest_curve=curve,latest_npu_telemetry=tele,status=summary.get('status') if summary else 'initializing')
health=(curve or {}).get('collection_health',{});npu_context=bool(tele and str(pid) in tele.get('raw',''))
result['startup_confirmed']=bool(pid_correct and len(alive)==16 and exact and curve and curve['valid_real_steps']>=1056 and curve['actor_updates']>=32 and curve['critic_updates']>=32 and curve['online_rows']>0 and npu_context and initial and initial['fresh_rl'] and initial['fresh_online_replay'] and initial['offline_rows']==224 and health.get('reasons',{}).get('no_validated_independent_phase',0)==0)
print(json.dumps(result,indent=2))
