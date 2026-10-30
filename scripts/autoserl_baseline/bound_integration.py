"""End the directed acceptance once real handback/continuation coverage is present.
This only requests the learner's existing graceful save path; no physics altered.
"""
import time,json,os,signal,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--tag',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=root/'outputs/ascend16'/a.tag
expected=['--mode','integration','--tag',a.tag]
while True:
 cmd=Path(f'/proc/{a.pid}/cmdline')
 if not cmd.exists():break
 words=cmd.read_bytes().decode().split('\0');assert all(w in words for w in expected),'unexpected learner PID'
 curve=out/'learning_curve.jsonl'
 if curve.exists():
  d=json.loads(curve.read_text().splitlines()[-1])
  if d['valid_real_steps']>=1024:
   rows=[json.loads(x) for x in (out/'workers/env00.jsonl').read_text().splitlines()]
   safe=[i for i,r in enumerate(rows) if r['event']=='step' and r['executed'] and (r.get('recovery_result') or {}).get('safe_handback')]
   if safe:
    last=rows[safe[-1]];post=[r for r in rows[safe[-1]+1:] if r['event']=='step' and r['executed'] and r['episode_id']==last['episode_id']]
    if len(post)>=5:
     evidence=dict(reason='directed_coverage_complete_after_real_Actor_handback_and_five_continuation_steps',learner_pid=a.pid,valid_steps_at_request=d['valid_real_steps'],handback_frame=last['frame'],post_handback_complete_steps=len(post),stop_signal='SIGTERM_graceful_checkpoint_not_SIGKILL',maximum_valid_physics_budget=2048)
     (out/'bounded_stop_request.json').write_text(json.dumps(evidence,indent=2)+'\n');os.kill(a.pid,signal.SIGTERM);break
 time.sleep(5)
