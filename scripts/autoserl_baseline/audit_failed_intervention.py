import json,collections,statistics,pathlib
root=pathlib.Path(__file__).resolve().parents[2];run=root/'outputs/ascend16/autoserl_baseline_formal_20261030_133346';out=root/'outputs/autoserl_intervention_fix';out.mkdir(exist_ok=True)
triggers=collections.Counter();rejects=collections.Counter();simultaneous=collections.Counter();coupling=collections.Counter();sources=collections.Counter();ends=collections.Counter();speeds=[];lengths=[];grasp=collections.Counter()
for f in sorted((run/'workers').glob('env[0-9][0-9].jsonl')):
 for line in f.open():
  d=json.loads(line)
  if d['event']=='intervention_episode':triggers.update((d.get('decisions') or {}).get('triggers',{}))
  if d['event']=='episode_end' and d.get('reason')!='operator_stop':ends[d.get('reason')]+=1;lengths.append(d['length'])
  if d['event']!='step':continue
  dec=d.get('auto_intervention') or {}
  for x in dec.get('deferred',[]):rejects[x.get('arm','?')+':'+str(x.get('kind'))+':'+str(x.get('reason'))]+=1
  if d.get('executed'):sources[d['action_source']]+=1
  gate=dec.get('passive_hold_entry')
  if gate:speeds.append({**gate,'actual_grasps':d.get('supervisor_grasps'),'actual_contact_coupled':d.get('supervisor_contacts',{}).get('dynamic_A_B_coupled')})
  if d.get('supervisor_stop_reason')=='simultaneous_high_confidence_risks_no_joint_qualification':
   risks=dec.get('policy_risks',[]);simultaneous[' | '.join(','.join(x.get('categories',[])) for x in risks)]+=1;coupling[str(d['supervisor_contacts']['dynamic_A_B_coupled'])]+=1
result={'final':json.loads((run/'summary.json').read_text())['counters'],'candidate_triggers_ended_episodes':dict(triggers),'per_decision_candidate_rejections':dict(rejects),'simultaneous_risk_classes':dict(simultaneous),'simultaneous_measured_graph_coupling':dict(coupling),'actual_sources':dict(sources),'episode_end_reasons':dict(ends),'natural_episode_count':len(lengths),'natural_episode_median':statistics.median(lengths),'hold_gate_observations':len(speeds),'hold_speed_m_s_quantiles':{str(q):sorted(x['linear_speed_m_s'] for x in speeds)[int(q*(len(speeds)-1))] for q in (0,.5,.9,1)},'prediction_not_actual_drift':True,'no_new_physics':True}
(out/'failed_training_root_cause.json').write_text(json.dumps(result,indent=2)+'\n');(out/'hold_gate_samples.json').write_text(json.dumps(speeds,indent=2)+'\n');print(json.dumps(result))
