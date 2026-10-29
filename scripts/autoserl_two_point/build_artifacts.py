"""Summarize real evidence and authorize only this new run after all gates pass."""
from pathlib import Path
import json,hashlib,time,sys,importlib.metadata,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/autoserl_two_point';DOC=ROOT/'docs/autoserl_two_point'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
required=[OUT/'final_verified_A_s4.json',OUT/'final_verified_B_s4.json',OUT/'final_gripper_barriers.json']
deadline=time.time()+600
while not all(p.exists() for p in required):
 if time.time()>deadline:raise RuntimeError('Acceptance evidence not finished')
 time.sleep(2)
probes=[]
for arm in ['A','B']:
 for label,seed in [('accepted',1),('accepted',2),('accepted',3),('final_verified',4)]:
  p=OUT/f'{label}_{arm}_s{seed}.json';d=read(p);assert d['passed'] and d['direct_physical_writes']==0
  recovery=[r for r in d['records'] if r['decision']['mode']=='Recover_'+arm]
  probes.append({'path':str(p.relative_to(ROOT)),'arm':arm,'seed':seed,'nominal_offset_mm':seed+2 if seed<4 else 5,'post_preparation_steps':d['actual_control_steps'],'total_actual_steps':d.get('physical_total_control_steps',(30 if arm=='A' else 58)+100+d['actual_control_steps']),'handbacks':d['handbacks'],'max_passive_position_m':d['passive_max_position_drift_m'],'policy_steps_after_handback':d['handback_policy_steps'],'active_window_frozen_during_recovery':len({r['decision']['indices'][0 if arm=='A' else 1] for r in recovery})==1})
assert all(p['active_window_frozen_during_recovery'] for p in probes)
for arm in ['A','B']:assert read(OUT/f'bounded_bridge_{arm}/result.json')['passed']
for p in ['contracts.json','nstep_acceptance.json','offline_acceptance.json','checkpoint_acceptance.json','safety_real_contact_fixed.json']:assert read(OUT/p)['passed']
full=read(OUT/'final_gripper_barriers.json');assert full['strict_success'] and full['steps']==229 and full['physical_writes']==0
for p in ['parked_A_passiveheldTrue.json','parked_B_passiveheldFalse.json']:assert read(OUT/p)['passed']
pressure_root=ROOT/'outputs/ascend16/autoserl2_validation_20261028_1';pressure=read(pressure_root/'summary.json');assert pressure['passed'] and pressure['counters']['critic_updates']==pressure['counters']['valid_real_steps'] and len(pressure['workers'])==16 and all(not w['npu_plugin_loaded'] for w in pressure['workers'])
startup=[json.loads(x) for x in (pressure_root/'workers/startup.jsonl').read_text().splitlines()];assert [x['group'] for x in startup if x['event']=='batch_ready']==[0,1,2,3]
paired=read(OUT/'paired_verified/summary.json')
resources={'cpu_cores':int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())/int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text()),'memory_limit_bytes':int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text()),'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['torch','torch-npu','torchrl','tensordict','mujoco','numpy']}}
telemetry=[json.loads(x) for x in (pressure_root/'npu_telemetry.jsonl').read_text().splitlines()];aicore=[r['aicore_percent'] for r in telemetry if 'aicore_percent'in r];hbm=[r['hbm_used_mb'] for r in telemetry if 'hbm_used_mb'in r]
resources.update(aicore_peak_percent=max(aicore),aicore_all_samples_mean_percent=sum(aicore)/len(aicore),hbm_peak_mb=max(hbm))
expected='e12cf336e23b362d8c5317b6e195df472d32bb424a797ab1858e8aa20fabe752';actual=sha(ROOT/'datasets/generated/two_arm_three_piece_assembly.hdf5');assert actual==expected
preservation=read(OUT/'preservation.json');preservation['raw_assembly_sha256']=actual
summary={'passed':True,'controlled_probes':probes,'paired_episodes':paired,'full_demo':{k:v for k,v in full.items() if k not in ['records','final','events']},'pressure':pressure,'runtime':resources,'checkpoint':read(OUT/'checkpoint_acceptance.json'),'preservation':preservation,'source_base':'bef4907','existing_uncommitted_deletions_preserved':183}
write(DOC/'acceptance.json',summary)
for p in ['contracts.json','nstep_acceptance.json','offline_acceptance.json','checkpoint_acceptance.json','preservation.json']:write(DOC/p,read(OUT/p))
write(DOC/'startup16.json',startup)
write(DOC/'deployment_gate.json',{'passed':True,'selected_utd':1,'start_authorized':False,'method':'Bimanual AutoSERL V1 section3.3 adaptation','acceptance_path':'docs/autoserl_two_point/acceptance.json','checkpoint_scope':'learner/optimizer/replay/RNG load and one real NPU update; full 16-process resume not tested; new episode boundary required','time':time.time()})
files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.log','.npz'] and 'offline_protocol' not in p.parts]
write(DOC/'artifact_manifest.json',{'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]})
print('ALL_ACCEPTANCE_PASSED',resources,flush=True)
