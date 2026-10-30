# 单任务 AutoSERL 基线真实验收报告

本轮范围以 REQUEST_BASELINE_ZH.md 为准。结论为系统集成状态A通过；本报告不把示范成功或恢复成功写成Actor已学会任务。正式训练启动状态在 AUTOSERL_BASELINE_TRAINING_ZH.md 独立记录。

## 完整示范与合法接触

outputs/autoserl_baseline/demo_full_safety_v10 从canonical demo_0初态出发，224个完整真实控制步，0物理状态写入、0动作投影，A/B均真实抓取并最终释放，通过原有assembly_real_grippers_stable3_v1严格成功协议。首次接触对应后状态A52/B64，首次真实抓取A53/B65，首件装配事件104，完整严格成功后状态224。P0/P1不是把首次接触索引直接当安全点。

最早本轮demo_current_safety_v1仍有26个投影、最终失败；修复合法任务接触及有限夹爪释放后，新的成功回放没有投影。对照诊断v9与验收v10全部224步qpos逐值完全相同，证明最终保护修复没有偷偷更改示范物理动作。

首件保护误报定位在11.012 s，物体位移0.218315 mm、速度0.016876 m/s，仅一个2 ms子步无接触。诊断v9只为识别原因暂时跳过该类支撑拒绝，明确不能作为安全验收；最终v10使用正常保护，没有诊断旁路。真实承载转移和孤立求解接触间隙的规则见实现报告；2 mm首件、3 mm被动持物、80 N及原姿态限制保持。

证据：demo_full_safety_v10/result.json、controls.json；normal_contact_evidence.json；全部失败版本和support_samples均保留在outputs/autoserl_baseline。

## 四项专项物理测试

| 测试 | 真实总控制步 | 结果 | 抓取与交还 |
|---|---:|---|---|
| A窗口引导 |178|通过|Guide_A后安全交还，继续5步；空手自由空间 |
| B窗口引导 |212|通过|Guide_B后安全交还，继续5步；空手自由空间 |
| A P0→P1 |230|通过|真实抓piece_1，Recover_A后安全交还及5步继续 |
| B P0→P1 |312|通过|真实抓piece_2，Recover_B后安全交还及5步继续；被动A原已持piece_1 |

总步数包含实际示范前缀和100步受控准备，不能当作全部恢复开销。A前缀30，B前缀58；B准备中A已实际抓取，通过保持留住该进度。全部0状态写入、0安全停机。提议动作来自明确标注的可复现测试控制器，不能冒充Actor。证据目录pregrasp_baseline_v1的四个子目录。

仅后续装配/释放保护发生改动，这些规则在抓取前域不活跃；当前生产配置另在以下NPU集成中验证，不以development_validation开关放行。高级持物、接触、联合恢复仍未生产授权，旧测试只保留为研究证据。

## 16环境NPU真实联合验收

运行：outputs/ascend16/autoserl_baseline_npu_v1。

- 16个worker按4+4+4+4完成初始化和reset，四批耗时56.76、54.25、50.33、54.79秒。四组在运行期间同时派发动作，各worker均采样并保持独立Episode/监督器；worker未加载torch_npu。
- 328步实际诊断准备写入Online，来源DiagnosticPreparation，不是Actor自主Episode；正式训练禁止此入口。
- 其后80轮16环境联合采样。训练有效物理步1578，部分控制步0，未执行提议30次不伪造transition。Actor/Critic均更新1578次，UTD严格1。
- Online1578，原始离线224，Dynamic147。其中Recover_A67、Recover_B80；434条SafetyProjection全部排除，额外专家维度总数1029=147×7，被动臂没有冒充专家。
- 所有真实来源：诊断准备328、Recover_A74、Recover_B80、完整Policy662、Constrained_A434。排除准备后的1250步中，完整Policy52.96%、恢复12.32%、安全投影34.72%；不能把投影称作任务引导。662个完整Policy步均有非零物理运动。
- A由真实NPU Actor触发，完成稳定性门禁，frame224以未修改的完整策略动作安全交还，并继续5步策略动作。B也由真实NPU Actor自动触发，但有界收尾时尚在恢复中；B完整交还的证据来自上一节物理专项，本次不能写成NPU中B也完成全链路。
- 达到定向覆盖后，独立监控器使用SIGTERM请求现有安全收尾，不使用SIGKILL。stop_reason=operator_stop是有证据的有界结束，见bounded_stop_request.json。
- 关闭主动干预的独立短评估1个Episode、15步，因workspace_exit_A终止，完整成功0、A/B抓取0；这不是正式20回合性能评估。训练和评估共1593有效物理步，小于2048。
- 42个自然结束Episode长20–26步，均值22.167、中位21，1–2步Episode为0；不含16个最终collector切尾及两个人工准备场景。安全原因workspace_exit_A12次、simultaneous_high_confidence_risks_no_joint_qualification30次；未验证联合恢复继续拒绝。没有no_validated_independent_phase初始误停。
- 原先用户报告3909步中多为1–2步是历史背景；本轮不是严格配对的学习效果对照。可靠的修复前后对照是同一canonical示范从26次投影失败到0投影严格成功，以及当前实际Episode分布。

## 数据、梯度和checkpoint

integration_audit.json逐行核对1578个实际动作、三动作来源、专家mask和3-step奖励/gamma/边界；1404个可与后续同Episode观测相连的端点RGB逐值相等，没有跨环境或Episode。原始224条保持，数据合约18组合通过。两个Dynamic接纳模式及原始/动态、正奖励quota的合约验证不等于额外RL对照实验。

历史29条真实RGB恢复数据另做显式逐臂元数据重建审计，原动作/观测/奖励不变，两个接纳模式各6条合格恢复、20条投影排除；不静默恢复旧协议checkpoint。证据physical_demo_per_arm_baseline/result.json。

当前checkpoint重载验证模型、优化器、Online/Offline/Dynamic、采样和RNG完全一致，随后实际完成一次优化器更新；总更新仍1578，预算债务0。完整结果保存在checkpoints/pre_restore、checkpoints/final，未调用官方清理目录逻辑。MuJoCo恢复边界是新Episode，不宣称中间物理状态位级续训。

冻结主干完全未变；Actor14个、Critic15个、适配层6个参数张量发生实际更新；三路适配层梯度范数约2.22e-4、2.18e-4、2.07e-4。参数和梯度有限。Actor末层14行Adam二阶矩均非零，包含两臂夹爪输出6/13，证明完整14维参与梯度更新。

## 硬件与性能

Python3.10.20、PyTorch2.3.1、torch_npu2.3.1、MuJoCo3.2.3；沿用既有CANN8.0.RC2和驱动26.1.1环境。CPU cgroup quota2200000/period100000=22核配额，虽affinity可见192个逻辑CPU，不能当192核资源；内存限制208240902144字节。OSMesa离屏渲染，worker线程1，主进程集中npu:0，对应物理设备3的910B3。

验收墙钟1779.72 s，采集/更新阶段含准备、评估与保存为1501.72 s，整体有效训练FPS和更新/s均1.0508。主要累计计时：16环境step223.14 s、reset23.35 s、Replay采样146.42 s、CPU→NPU21.27 s、更新505.27 s、批量推理3.13 s；其余包含诊断准备/保存/评估等，不能只用网络时间冒充总吞吐。

283次npu-smi采样（包含初始化和准备空闲），AICore平均26.31%、峰值99%，HBM峰值5167 MB；PyTorch峰值分配1599265792字节。确实在NPU训练，不是CPU长期更新。没有改UTD、网络或损失以提高数字。正式持续吞吐需继续从正式曲线测量。

## 适用范围与尚未证明事项

允许已验证的空手抓取前Guide/P0返回、真实P0→P1抓取与安全交还，且被动保持域、物体取物位、无危险接触/耦合及返回走廊均满足。正常Actor抓取、持物和装配可执行，但不代表任意状态可单臂恢复。第一件已完成时不能撤销锚点或回退其任务进度。

carry_local、joint、passive_brake保持禁用。规则化的正常接触走廊和有限支撑转移只在该任务示范域有证据，不能当通用接触动力学证书。短验收未证明策略会自主抓取/装配，也未证明样本效率优于普通RLPD。随机新Actor的无干预完整成功仍为0，任务学习状态C未达到。

## 复现命令

先source scripts/ascend16/activate.sh。完整回放使用scripts/autoserl_baseline/replay_demo.py --tag新的唯一标签；四项专项使用pregrasp_physics.py --tag新的唯一标签；验收命令、控制来源及预算见运行config和logs/autoserl_baseline/integration_v1.log。audit_integration.py --tag autoserl_baseline_npu_v1可对持久化实际数据重审，不新增物理步或梯度更新；summarize_evaluation.py --run 对应运行目录生成阶段指标；finalize_acceptance.py从真实结果构造11项门禁，不能跳过失败。

## Worker statistics (Episode includes final collector cuts)

| Env | Valid steps | Resets | Episodes | Task takeovers | Handbacks | Projection steps | Safe_Stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 | 230 | 1 | 1 | 1 | 1 | 0 | 0 |
| 01 | 258 | 1 | 1 | 1 | 0 | 0 | 0 |
| 02 | 79 | 4 | 4 | 0 | 0 | 33 | 3 |
| 03 | 79 | 4 | 4 | 0 | 0 | 31 | 3 |
| 04 | 78 | 4 | 4 | 0 | 0 | 31 | 3 |
| 05 | 78 | 4 | 4 | 0 | 0 | 29 | 3 |
| 06 | 77 | 4 | 4 | 0 | 0 | 30 | 3 |
| 07 | 78 | 4 | 4 | 0 | 0 | 30 | 3 |
| 08 | 77 | 4 | 4 | 0 | 0 | 30 | 3 |
| 09 | 78 | 4 | 4 | 0 | 0 | 31 | 3 |
| 10 | 77 | 4 | 4 | 0 | 0 | 32 | 3 |
| 11 | 77 | 4 | 4 | 0 | 0 | 31 | 3 |
| 12 | 77 | 4 | 4 | 0 | 0 | 31 | 3 |
| 13 | 78 | 4 | 4 | 0 | 0 | 31 | 3 |
| 14 | 79 | 4 | 4 | 0 | 0 | 33 | 3 |
| 15 | 78 | 4 | 4 | 0 | 0 | 31 | 3 |
