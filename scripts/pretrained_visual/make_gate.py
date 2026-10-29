"""Open formal deployment only from recorded, completed physical/NPU validations."""
import argparse,json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text())
def rows(p):return [json.loads(s) for s in (ROOT/p).read_text().splitlines() if s]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
visual='resnet18_spatial84';run=Path('outputs/ascend16')/a.run
bench=read('outputs/pretrained_visual/benchmark_'+visual+'.json');assert bench['passed']
s=read(run/'summary.json');assert s['status']=='completed' and s['passed'] and s['config']['visual_encoder']==visual
assert s['config']['num_envs']==16 and s['config']['startup_groups']==[4,4,4,4] and s['config']['n_step']==3 and s['config']['utd']==1
assert s['offline_rows']==224 and s['counters']['critic_updates']==s['counters']['valid_real_steps']>0
assert len(s['workers'])==16 and all(not w['npu_plugin_loaded'] and w['real_steps']>0 for w in s['workers'])
starts=rows(run/'workers/startup.jsonl');assert [r['group'] for r in starts if r['event']=='batch_ready']==[0,1,2,3] and starts[-1]['event']=='all_sixteen_ready'
audit=read(run/'visual_audit.json');assert audit['frozen_backbones_unchanged'] and audit['backbones_eval'] and len(audit['adapter_grad_norms'])==3 and all(x>0 for x in audit['adapter_grad_norms']) and all(audit['changed_tensors'].values())
ck=read('outputs/pretrained_visual/checkpoint_acceptance.json');assert ck['passed'] and ck['run']==a.run
for path in ['outputs/pretrained_visual/nstep_acceptance.json','outputs/pretrained_visual/offline_acceptance.json']:assert read(path)['passed']
bridges={}
for arm in ['A','B']:
 b=read(f'outputs/autoserl_two_point/visual_bridge_{arm}_{arm}/result.json');assert b['passed'] and any(e['event']=='handback' for e in b['events']);bridges[arm]={k:v for k,v in b.items() if k!='rows'}
protected=read('docs/pretrained_visual/preservation.json')
for path,sha in protected.items():
 if isinstance(sha,str):assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
for path,sha in protected['official_algorithm'].items():assert hashlib.sha256((ROOT/'third_party/residual-offpolicy-rl'/path).read_bytes()).hexdigest()==sha,path
telemetry=[r for r in rows(run/'npu_telemetry.jsonl') if 'aicore_percent' in r]
perf={k:s[k] for k in ['actual_steps_s','updates_s','timing','peak_npu_allocated_bytes','training_seconds','wall_seconds','utd_effective']}
perf.update(aicore_peak=max(r['aicore_percent'] for r in telemetry),aicore_mean_including_startup=statistics.mean(r['aicore_percent'] for r in telemetry),hbm_peak_mb=max(r['hbm_used_mb'] for r in telemetry))
result={'passed':True,'selected_visual':visual,'pretrained_validation_passed':True,'selected_utd':1,'pressure_run':str(run),'performance':perf,'counters':s['counters'],'workers':s['workers'],'visual_audit':audit,'bridge_recoveries':bridges,'checkpoint_reload':ck,'invariants_preserved':True,'fresh_formal_required':True,'resume_physics':'new episode boundary only'}
(ROOT/'docs/pretrained_visual/acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'docs/pretrained_visual/deployment_gate.json').write_text(json.dumps({k:result[k] for k in ['passed','selected_visual','pretrained_validation_passed','selected_utd','pressure_run']},indent=2)+'\n')
print('ACCEPTED',perf)
