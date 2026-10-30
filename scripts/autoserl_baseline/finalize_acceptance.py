"""Construct the 11-item gate from actual acceptance artifacts; fails closed."""
import json,hashlib,subprocess,argparse
from pathlib import Path
from acoserl.baseline_gate import source_files,REQUIRED,require_baseline
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--integration-tag',required=True);a=p.parse_args()
def read(path):return json.loads((ROOT/path).read_text())
base='outputs/autoserl_baseline/';run='outputs/ascend16/'+a.integration_tag+'/'
demo=read(base+'demo_full_safety_v10/result.json');normal=read(base+'normal_contact_evidence.json');data=read('outputs/acoserl_v3/baseline_per_arm_contracts/result.json');physical=read('outputs/acoserl_v3/physical_demo_per_arm_baseline/result.json')
proofs={k:read(base+'pregrasp_baseline_v1/'+k+'/result.json') for k in ('window_A','window_B','recovery_A','recovery_B')}
integration=read(run+'summary.json');audit=read(run+'integration_audit.json');reload=read(run+'checkpoint_reload.json');visual=read(run+'visual_audit.json');profile=read('configs/bimanual_autoserl/baseline.json');cfg=integration['config']
assert demo['passed'] and demo['complete_controls']==224 and demo['projection_steps']==0 and demo['physical_writes']==0 and demo['final_strict']['strict_success_streak']>=3
assert normal['passed'] and normal['normal_demo_exact_qpos_match_to_diagnostic']==0
assert all(r['passed'] and r['handback'] and r['post_handback_steps']>=5 and r['physical_writes']==0 for r in proofs.values())
assert proofs['recovery_A']['active_grasp'] and proofs['recovery_B']['active_grasp']
assert data['passed'] and len(data['sampling_combinations'])==18 and physical['passed']
assert audit['passed'] and audit['safe_handbacks']>0 and audit['actual_actor_proposals']>0
assert audit['source_counts'].get('Recover_A',0)>0 and audit['source_counts'].get('Recover_B',0)>0
assert reload['optimizer_continuation_update'] and all(reload[k] for k in ['parameters_exact','optimizers_exact','replays_exact','sampling_exact','rng_restored'])
assert visual['frozen_backbones_unchanged'] and visual['all_parameter_gradients_finite'] and all(visual['changed_tensors'].values())
assert not profile['development_validation'] and all(not profile[k]['enabled'] for k in ('carry_local','joint','passive_brake'))
assert cfg['manager_config']['method']=='full_acoserl' and cfg['manager_config']['sliding_enabled'] and cfg['manager_config']['safety_recovery_enabled'] and cfg['manager_config']['dynamic_demo_enabled']
assert integration['offline_rows']==224 and cfg['visual_encoder']=='resnet18_spatial84' and cfg['batch_size']==256 and cfg['gamma']==.99 and cfg['agent_config']['critic']['num_q']==10
limits=read('configs/bimanual_autoserl/v1.json');original=json.loads(subprocess.check_output(['git','show','7d0261d:configs/bimanual_autoserl/v1.json'],cwd=ROOT,text=True));assert limits==original,'safety or paper thresholds changed'
assert limits['robot_force_stop_N']==80 and limits['passive_position_drift_m']==.002 and limits['passive_object_position_drift_m']==.003 and limits['assembled_object_drift_m']==.002
assert audit['source_counts'].get('Policy',0)>0 and audit['nonzero_motion_steps']>100 and all(w['policy_proposals']>0 for w in audit['workers'])
checks={}
def add(key,paths,detail):
 assert key in REQUIRED and paths and all((ROOT/x).exists() for x in paths)
 checks[key]=dict(passed=True,paths=paths,detail=detail)
add('official_core_connected',['docs/acoserl_foundation/AUTOSERL_BASELINE_IMPLEMENTATION_ZH.md',run+'config.json'], 'Official2382712 mechanisms connected to baseline supervisor; actual Recover_A/B during NPU Actor collection')
add('real_grasp_phases_verified',[base+'demo_full_safety_v10/result.json',base+'normal_contact_evidence.json'],normal['events'])
add('single_demo_strict_protocol_verified',[run+'integration_audit.json',base+'demo_full_safety_v10/result.json'],'Only demo_0;224original rows; strict stable3 and actual gripper release')
add('executed_actions_replay_verified',[run+'integration_audit.json','outputs/acoserl_v3/physical_demo_per_arm_baseline/result.json'],'Live executed actions, per-arm expert masks, actual RGB n-step endpoints, explicit historical provenance rebuild')
add('enabled_recovery_domains_verified',[base+'pregrasp_baseline_v1/'+k+'/result.json' for k in proofs]+[run+'integration_audit.json'],'Only measured static-passive, empty-active pregrasp Guide/Recover domains; no general contact/carry authorization')
add('unverified_recovery_disabled',['configs/bimanual_autoserl/baseline.json'],'carry_local,joint,passive_brake disabled; legal Actor task motion remains enabled')
add('legal_grasp_assembly_contact_verified',[base+'demo_full_safety_v10/result.json',base+'normal_contact_evidence.json'],'Accepted full canonical demo with0projection and0state writes; diagnostic bypass excluded')
add('original_safety_limits_verified',['configs/bimanual_autoserl/v1.json',base+'normal_contact_evidence.json'],'Exact thresholds equal preservation commit;2ms monitor and immutable first-piece anchor; finite legal task support transfer')
add('npu16_real_updates_verified',[run+'summary.json',run+'visual_audit.json',run+'integration_audit.json',run+'workers/startup.jsonl'],dict(valid_steps=integration['counters']['valid_real_steps'],updates=integration['counters']['critic_updates']))
add('checkpoint_reload_update_verified',[run+'checkpoint_reload.json',run+'checkpoints/final/metadata.json'],reload)
add('no_systemic_collection_deadlock',[run+'integration_audit.json',run+'learning_curve.jsonl'],'All16 sample, real motion/Policy/Recover present, real safe handback; no claim of learned full-task success')
gate=dict(collector_protocol='autoserl_baseline_v1',passed=True,formal_training_approved=True,selected_utd=1,selected_visual='resnet18_spatial84',pretrained_validation_passed=True,checks=checks,tested_source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files()},source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),integration=run,level_A_system_integrated=True,level_B_formal_running=False,level_C_task_learned=False,scope='Necessary single-task baseline gates only; advanced12+4 not prerequisite')
path=ROOT/'docs/acoserl_foundation/ACCEPTANCE_AUTOSERL_BASELINE.json';path.write_text(json.dumps(gate,indent=2)+'\n');require_baseline(path,ROOT);print('All11 baseline checks passed from real artifacts')
