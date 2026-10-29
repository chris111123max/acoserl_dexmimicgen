"""Publish compact evidence; keep raw logs and replay outside Git."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'outputs/acoserl_foundation';docs=ROOT/'docs/acoserl_foundation'
names=['physical_Recover_A_v3','physical_Recover_B_v4','physical_Guide_A_v4','physical_Guide_B_v4','physical_free_v4','physical_risk_v5']
evidence={'formal_training_approved':False,'physical':{},'data':json.loads((base/'data_boundaries_final_v2/result.json').read_text()),'contracts':json.loads((base/'contract_acceptance.json').read_text()),'stages':json.loads((base/'stage_gates_v2/result.json').read_text()),'previous_run':json.loads((base/'previous_run_analysis.json').read_text())}
table=['|Case|Passed|Total control steps|Admitted intervention rows|Passive EEF max mm|Passive object max mm|','|---|---|---:|---:|---:|---:|']
for name in names:
 p=base/name/'result.json'
 if not p.exists():raise RuntimeError('still pending: '+name)
 d=json.loads(p.read_text());summary={k:v for k,v in d.items() if k not in ('baseline','intervention_events')}
 previous=None;baseline=None;mode_before=None;objmax=0.
 for line in (base/name/'worker/env00.jsonl').open():
  row=json.loads(line)
  if row['event']!='step':continue
  mode=row.get('action_source')
  if mode in ('Recover_A','Recover_B','Guide_A','Guide_B'):
   if mode!=mode_before:baseline=previous
   if baseline:
    arm='B' if mode.endswith('A') else 'A'
    for obj,held in baseline['supervisor_grasps'][arm].items():
     if held:
      change=np.linalg.norm(np.array(row['supervisor_objects'][obj]['position'])-baseline['supervisor_objects'][obj]['position']);objmax=max(objmax,float(change))
  previous=row;mode_before=mode
 summary['passive_object_max_position_m']=objmax
 evidence['physical'][name]=summary
 table.append('|'+name+'|'+str(d.get('passed'))+'|'+str(d.get('actual_steps','n/a'))+'|'+str(d.get('dynamic_rows','n/a'))+'|'+str(round(d.get('max_passive_error_m',0)*1000,6))+'|'+str(round(objmax*1000,6))+'|')
old=json.loads((base/'inspection.json').read_text());safety=ROOT/'configs/bimanual_autoserl/v1.json';evidence['safety_thresholds_unchanged']=hashlib.sha256(safety.read_bytes()).hexdigest()==old['hashes']['configs/bimanual_autoserl/v1.json']
evidence['physical_buffer_reload']=json.loads((base/'physical_buffer_reload.json').read_text())
(docs/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
s=(docs/'report_template.md').read_text().replace('{{RESULTS}}','\n'.join(table))
(docs/'RESULTS_ZH.md').write_text(s)
print(json.dumps({'physical_passes':sum(d['passed'] for d in evidence['physical'].values()),'cases':len(names),'safety_unchanged':evidence['safety_thresholds_unchanged']}))
