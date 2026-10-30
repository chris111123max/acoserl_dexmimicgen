"""Save the first scheduled V3 execution's real evidence and explicit unfinished gates."""
from pathlib import Path
import json,subprocess,hashlib
from datetime import datetime,timezone,timedelta
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/acoserl_foundation'
if (D/'ACCEPTANCE_V3.json').exists():raise FileExistsError('Round1 evidence already exists; do not overwrite later acceptance or reports')
def read(p):return json.loads((ROOT/p).read_text())
def write(n,x):(D/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
root=read('outputs/acoserl_v3/original_contact_b6_v2/result.json')
trace=read('outputs/acoserl_v3/original_contact_b6_v2/control_trace.json')
audit=read('outputs/acoserl_v3/provenance_audit_v1/result.json')
contract=read('outputs/acoserl_v3/data_contracts_v2/result.json')
assert root['reproduced'] and root['physical_writes']==0 and root['max_action_replay_error']==root['max_pose_replay_error_m']==0
assert audit['passed'] and contract['passed']
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
files=['src/acoserl/dynamic_demo.py','src/acoserl/transition_semantics.py','scripts/acoserl_v3/reproduce_contact_failure.py','scripts/acoserl_v3/audit_v2_demo_sources.py','scripts/acoserl_v3/validate_data.py']
hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
assert all(subprocess.check_output(['git','show',sha+':'+f],cwd=ROOT)==(ROOT/f).read_bytes() for f in files)
unchanged=not subprocess.check_output(['git','diff','867d474728e032c3782d5ac4104d5a6b95aec93d','--','configs/bimanual_autoserl/v1.json','configs/ascend16/formal.json'],cwd=ROOT)
fields=['source_commit_verified','contact_failure_reproduced','contact_failure_root_cause_fixed','first_piece_progress_protected','handoff_protocol_verified','contact_recovery_12_of_12_passed','contact_actor_continuation_verified','independent_contact_cases_verified','passive_hold_prediction_verified','safety_thresholds_unchanged','dynamic_demo_provenance_verified','safety_projection_excluded_from_demo','nstep_checkpoint_verified','real_actor_intervention_verified','npu_16env_integration_verified']
gate={f:False for f in fields};gate.update(source_commit_verified=True,contact_failure_reproduced=True,safety_thresholds_unchanged=unchanged,safety_projection_excluded_from_demo=True)
evidence={f:dict(passed=gate[f],status='verified_partial_round1' if gate[f] else 'pending',paths=[],parameters={},summary='尚未完成V3强制验收；不以V2结果代替。') for f in fields}
evidence['source_commit_verified'].update(paths=list(hashes),parameters=dict(commit=sha,sha256=hashes),summary='已提交源码与实际执行验证时源码一致。')
evidence['contact_failure_reproduced'].update(paths=['outputs/acoserl_v3/original_contact_b6_v2/result.json','outputs/acoserl_v3/original_contact_b6_v2/control_trace.json','outputs/acoserl_v3/original_contact_b6_v2/substeps.json'],parameters=dict(original_case='Contact_B_6',real_control_steps=284,substep_seconds=.002),summary='原动作、末端位姿逐步精确一致，第284控制步再次触发completed_piece1_displaced。')
evidence['safety_thresholds_unchanged'].update(paths=['configs/bimanual_autoserl/v1.json','configs/ascend16/formal.json'],parameters=dict(force_N=80,passive_eef_m=.002,passive_object_m=.003,protected_piece1_m=.002),summary='安全阈值与正式超参数文件相对V2没有修改。')
evidence['safety_projection_excluded_from_demo'].update(paths=['outputs/acoserl_v3/provenance_audit_v1/result.json','outputs/acoserl_v3/data_contracts_v2/result.json'],parameters=dict(historical_complete_rows=92,projection_rows=52,admission_modes=2,quota_combinations=18),summary='历史真实投影样本在两个V3接纳模式均排除；旧污染协议重载被拒绝。尚未证明全部任务纠偏质量规则。')
evidence['dynamic_demo_provenance_verified'].update(paths=['outputs/acoserl_v3/provenance_audit_v1/result.json'],summary='历史来源和动作逐字段核对通过；新物理Guide/Recover质量、三段动作链与交还后任务进展验收待完成。')
evidence['nstep_checkpoint_verified'].update(paths=['outputs/acoserl_v3/data_contracts_v2/result.json'],summary='18组合、投影断段和数据采样重载契约通过；V3真实NPU更新与整模型checkpoint验证待完成。')
gate.update(collector_protocol='acoserl_v3',passed=False,formal_training_approved=False,implementation_commit=sha,tested_source_sha256=hashes,checks=evidence,updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),automation_id='aco-serl-v3',stage='first scheduled execution; partial implementation',new_rl_training_effective_steps=0,new_rl_actor_updates=0,new_rl_critic_updates=0)
write('ACCEPTANCE_V3.json',gate)
state=dict(automation_id='aco-serl-v3',schedule_hours=6,status='ACTIVE',branch='codex/acoserl-v3',request_path=str(D/'REQUEST_V3_ZH.md'),implementation_commit=sha,completed=['V2 documents/config/source inspection','exact original failed-case physics replay','historical replay provenance audit','default projection exclusion and incompatible checkpoint refusal','18 quota/data-boundary contracts'],current_processes=[],formal_started=False,formal_start_verified=False,next_actions=['Read ROOT_CAUSE_V3_ZH.md and V3 acceptance evidence before editing.','Implement ongoing piece1/base support protection with pre-action feasible motion constraints; preserve useful Actor motion, do not zero all actions.','Replace immediate local handback with Recovering/Stabilizing/Handoff_Ready/Policy_Guarded/Normal protocol; use true Actor proposals for at least20 guarded checks in physical acceptance.','Calibrate passive-hold model error on separate calibration/validation domains; preserve original failed prefixes and all thresholds.','Finish three-action provenance and task-error/quality gating, then collect real Guide/Recover RGB transitions for both admission modes.','Freeze qualified code and run original12 contact cases plus4 independent cases; failed modes stay disabled.','Execute bounded16-env NPU integration <=2048 valid steps with real task interventions, mixed Replay and save/reload/update.','Only after every mandatory V3 check is evidenced and passed, launch fresh formal training automatically and verify the ten startup conditions, then delete automation aco-serl-v3.'],limitations=['Control/handback/protection changes not yet implemented or physically qualified.','Current controller code is still V2; joint recovery remains disabled.','Historical projection replay audit is not new physics collection or RL training.','Old validate_p0 expects the V2 unsafe-row admission semantics; its preserved failure is expected under the new rule. Use scripts/acoserl_v3/validate_data.py for V3.','The reproduction harness must preserve outer servo controller-cache update order during controls214..278; first attempt failed at control215 from harness mismatch and is retained.'])
(ROOT/'outputs/acoserl_v3/continuation.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
last=trace[-1];ready=trace[-5];bad=root['first_substep_violation']
root_text=f"""# V3首轮：原Contact_B_6失败的确定性定位

日期：2026-10-30（Asia/Shanghai）。原V2代码基线867d474728e032c3782d5ac4104d5a6b95aec93d。原始日志未改写，物理复现使用正常demo_0初始化和完全相同的动作前缀，没有随机挑选替代场景。

## 实际复现

原始284条完整动作全部执行，动作最大差异0、末端位置最大差异0，物理状态直接写入0。原始五个Actor提议来自已保存的真实NPU Actor轨迹，本次为定向动作重放，不伪称重新推理了20个Actor动作。

监测保存恢复前10步、5步恢复及5步Actor续行，另按2ms保存物体/末端真实速度、接触力、抓取、物体位姿和受保护首件误差。原失败在第5个Actor提议之后提前终止，没有补造后续15个提议。

第一次复现尝试在第215控制步出现11.66微米偏差，未计为通过；根因是原外层servo会先调用OSC.update更新缓存，复现最初漏了这一步。补齐相同调用顺序后，284步逐动作、逐末端位姿精确复现。未放宽复现容差。

## 第一个违反原约束的位置

- 控制步：零基{bad['control_index_zero_based']}，即第284步，交还后的第5个Actor提议。
- 2ms采样首次观测超限：仿真时间{bad['sim_time']:.3f}s，首件位置误差{bad['protected_error']['position_m']*1000:.6f}mm，超过原2mm。
- 该控制步开始：位置误差{last['before']['protected_error']['position_m']*1000:.6f}mm，首件线速度{np.linalg.norm(last['before']['object_velocities']['piece_1']['linear']):.9f}m/s。
- 完整50ms控制步结束：位置误差{last['after_state']['protected_error']['position_m']*1000:.6f}mm，旋转误差{last['after_state']['protected_error']['rotation_rad']:.9f}rad（未超过原.035rad）。
- first_piece_assembled在该步前后仍为True，因此只看任务事件会漏掉物理进度损伤；应同时保护位姿、支撑与速度。

原Actor14维物理提议：
```json
{json.dumps(last['policy_original'])}
```
实际执行14维动作：
```json
{json.dumps(last['executed'])}
```
两者仅有float32归一化往返产生的微小表示差异，没有施加装配接触运动约束。

首件执行前位姿：
```json
{json.dumps(last['before']['objects']['piece_1'])}
```
首件执行后位姿：
```json
{json.dumps(last['after_state']['objects']['piece_1'])}
```

## 确定的软件缺口

1. V2Mixin.after_step在LocalRecovery达到目标后直接清除local并进入Normal，没有独立的稳定、首个Actor动作资格和持续受保护交还协议。
2. V2Mixin.action_risks主要约束工作空间、夹爪和张爪接近；没有对已装配piece_1/base关系施加动作前的接触位姿/支撑约束。首个Actor动作只被修改B夹爪，后续位姿运动被放行。
3. B夹爪的planned_release使用窗口达到示范release索引加Legal_Object_Contact作为资格，这不能替代真实第二件稳定支撑/末态验证。它是另一个需修复的资格缺口；本失败期间B仍维持抓取，不能仅凭该分支把整次失败归因于已掉落物体。
4. TwoPointMixin.physical_danger中completed_piece1_displaced确实存在；InterventionManagerMixin.after_step在完整步结束后才调用它。
5. V2Mixin.substep_safety只调用contact_assessment及carry接触模式检查，没有首件位姿/支撑保护。因此14.184s采样已超限，仍执行到14.200s才由动作后检查停止。

恢复交还时（控制279开始），A/B末端线速度分别{np.linalg.norm(ready['before']['eef_velocities'][0]['linear']):.9f}/{np.linalg.norm(ready['before']['eef_velocities'][1]['linear']):.9f}m/s，首件线速度{np.linalg.norm(ready['before']['object_velocities']['piece_1']['linear']):.9f}m/s，首件误差{ready['before']['protected_error']['position_m']*1000:.6f}mm。该案例不支持“恢复器未到位或首件一开始已剧烈漂移”的解释；关键失效链为交还后接触运动未受保护、首件速度和误差累积、事后门禁才发现越界。

这些数据确定了原失败步骤和缺失的控制前/物理子步保护，不代表保护控制器已经实现或通过。不得以简单新增Safe_Stop代替V3要求的可行动作约束。

证据：outputs/acoserl_v3/original_contact_b6_v2/result.json、control_trace.json、substeps.json。测试源码：scripts/acoserl_v3/reproduce_contact_failure.py。
"""
(D/'ROOT_CAUSE_V3_ZH.md').write_text(root_text)
(D/'IMPLEMENTATION_V3_ZH.md').write_text(f"""# ACo-SERL V3 首轮实现进度

日期2026-10-30；分支codex/acoserl-v3；实现提交{sha}。任务全文保存在REQUEST_V3_ZH.md。每6小时在原对话继续执行，自动任务id=aco-serl-v3；确认正式训练真实启动并通过10项初始运行核验后才删除任务。

本轮只对已完成部分报告通过，没有启动正式训练。完整剩余步骤保存在outputs/acoserl_v3/continuation.json和ACCEPTANCE_V3.json。

## 已修改源码

- transition_semantics.py：TASK_INTERVENTION_SOURCES与SAFETY_PROJECTION_SOURCES明确区分，source_family把Constrained_A/B/Joint归为SafetyProjection，不伪装成Guide/Recover。
- dynamic_demo.py：默认排除SafetyProjection，保留Online链路；不完整动作、重复和乱序检查保留。安全已失败的完整行不进入Demo，Online仍保留真实转移。增加分来源观测/接纳与投影排除计数。
- 两种接纳模式均保留。投影或Policy切断任务干预连续段，禁止n-step跨不连续片段。
- 动态checkpoint新增接纳协议版本。V2含投影样本的旧动态Demo默认拒绝导入，不删除或改写旧数据。新正式训练仍必须从空Online/Dynamic启动。
- reproduce_contact_failure.py：用同一真实初始状态、原动作前缀和控制器调用次序复现失败，补录物体/末端速度与2ms接触监测。
- audit_v2_demo_sources.py：逐条连接真实worker日志与Online/Dynamic Replay，核对动作及来源；不根据52/92猜测来源。
- validate_data.py：V3数据契约，保留18组配额与采样重载检查，并增加投影断段/排除和不合格行排除。

## 尚未实现/验收

装配持续保护可行动作控制器、正式交还状态机、保持预测误差标定、任务纠偏质量指标及三段动作链、12+4真实接触场景、V3完整16环境NPU训练链路均待完成。不得把当前数据过滤修复等同于V3全验收。

现有RLPD、视觉编码器、稀疏奖励、原始HDF5与224条离线数据、3-step、10-Q、16环境及安全阈值均未改变。joint恢复继续禁用。未push公开仓库。
""")
(D/'VALIDATION_V3_ZH.md').write_text(f"""# V3 首轮真实验证

1. 原失败复现：284完整物理控制步，动作/末端差异均0，直接物理状态写入0。原先诊断尝试在第215步因测试器调用顺序差异中止，保留日志，不计通过。
2. 精确失败：第284控制步，14.184s首次采样超过2mm，完整步结束2.300353mm。细节见ROOT_CAUSE_V3_ZH.md。
3. 历史来源审计：96次提议、92条完整交互、4次未执行拒绝；实际来源40 Policy、52 Constrained_A。Online及两种Dynamic样本的执行动作逐条核对一致。历史两种Dynamic中的52条全部是SafetyProjection。
4. 新过滤历史回归：两个V3模式均排除这52条，Demo新增0；旧污染协议加载被拒绝，原文件未变。这是历史真实数据审计，不是新增物理数据或新RL训练。
5. V3契约：2模式×3原始配额×3正回报配额共18组通过，原始224/正回报3保持，投影断段不跨接、动态采样与保存重载一致。构造数据只用于契约。
6. 新物理Guide/Recover质量采集、接触12/12及4个独立场景、至少20个真实Actor交还提议、保持预测校准和V3 NPU更新均未验收；不能使用V2结果填充V3通过字段。

新RL有效步0，Actor/Critic更新0。当前只完成原失败定向物理重放及数据逻辑修复。全部强制门禁通过后会自动启动正式训练，不再询问用户；目前formal_training_approved=false。
""")
(D/'TRAINING_V3_ZH.md').write_text("""# V3 正式训练状态

尚未启动正式训练。没有训练PID、新训练模型或训练Replay。

原因：ACCEPTANCE_V3.json中除源码核验、原失败复现、安全阈值不变和投影排除外，其余强制门禁尚未通过。尤其装配进度保护控制器、完整交还协议、12+4接触场景、保持预测校准和新16环境NPU链路仍待完成。

用户已授权全部门禁通过后直接启动；不需要再次询问。启动后必须核验PID、NPU、16worker就绪、有效步与Actor/Critic更新持续增长、Online增长、Dynamic来源、配置日志和初始安全状态。完成这些核验才报告成功启动并删除自动任务aco-serl-v3。命令提交或进程短暂出现不算成功。

既有正式配置为500000实际交互/36000秒、UTD1、16环境、batch256、3-step/.99、10-Q、stddev.1，未改变。将以全新Actor/Critic及空Online/Dynamic开始，不使用专项模型或Replay作为预训练。
""")
print(json.dumps(dict(commit=sha,automation_id='aco-serl-v3',reproduced=root['reproduced'],projection_exclusion=True,formal_training_approved=False,pending=[f for f in fields if not gate[f]]),ensure_ascii=True))
