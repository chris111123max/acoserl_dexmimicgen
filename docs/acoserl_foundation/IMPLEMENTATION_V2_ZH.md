# ACo-SERL V2 实现说明

本轮为代码契约、定向物理实验及总有效训练步不超过1024的真实集成，不执行冒烟、BC或长期正式训练，不推送GitHub。A=robot0右臂，B=robot1左臂。

## 版本和固定设置

服务器分支codex/acoserl-v2，基础5c637f7664fdecffe0a23dc08fdbc9cae87206b7，实现提交5ec667f3c937a11234cef7fa7b6ebd97822af37c。公开b0c2c033e13310e963ce0769959d1a557019ca3d仅fetch比较，未checkout覆盖。限定src/acoserl、scripts/ascend16/train.py、configs/bimanual_autoserl比较，两份基础源码一致，仅reference_trace.npz不同；其他历史产物差异不能作为覆盖依据。

保留demo_0及224条有效离线transition（3条正回报）、gamma=.99、3-step、10-Q、在线128/示范128、UTD1、14动作、18低维、3路84RGB。继续原有冻结ResNet18空间编码器（单路81×128），未更换视觉模型、奖励、网络、损失或优化算法。Actor/Critic/适配层从随机初始化开始，Online Replay为空。

## 官方机制对照

官方缓存third_party/autoserl固定提交238271246e39ef0cf7e06a94392c4b2d184a9af2，已实际阅读wrappers.py:auto_intervention_wrapper及examples/train_rlpd.py Actor/Learner。

|机制|官方实际执行|本项目与双臂扩展|
|---|---|---|
|窗口|窗口内位置最近点引导；全局最近点用于方向；th2=.020/th1=.005，cosθ>=0|two_point_supervisor.window_measure：独立A/B窗口、实际抓取和阶段过滤，零方向防护；姿态不替代普通位置触发|
|窗口时钟|window_length=P1-P0+1，闭区间点数；无干预时随控制步前移|按现有任务约定L=P1-P0，最多L个点（A27/B32），比官方闭区间P1-P0+1少一点；事件限制合法末端；Guide/Recover保持双方窗口，Constrained冻结受修改臂，不强迫另一臂回退|
|停滞|20项全轨迹最近参考位姿历史，首末位置变化<5mm|保留参考进展判据，使用21个采样点跨20个控制间隔，比官方20项数组多一点；不用末端实际位移替代|
|P0/P1|记录恢复区间，返回P0再重放至P1|保留既有已验证真实OSC返回/重放；持物时禁止返回抓取前P0，另用LocalRecovery|
|动态示范|Actor以info.intervene_action替换动作，同一真实transition加入Online和intvn_data_store|final动作入Online；official接纳全部完整合法干预行；safe_handback仅安全交还连续段；未执行/部分子步无假行|
|RLPD|Learner在线/示范50:50|复用ResFiT更新，各128；示范内部原始最低配额和正回报最低采样配额，不改变奖励|
|终止干预|SpaceMouse左键给成功奖励，成功done且干预控制步数<10永久关闭窗口|不照搬人工奖励；共享Actor全局统计，多个严格无干预成功才退出，阶段稳定自主完成可减少普通引导；风险门禁始终保留。这是保守双臂扩展，非逐字复现|

## 接触分类与动作风险

contact_modes.assess输入实际接触图、抓取、任务阶段、相对位姿、接触力，输出category/allowed/reason。Independent和参考相容的Legal_Object_Contact可以按控制资格继续；Unknown_Coupling、Unsafe_Collision拒绝。Legal_Cooperative_Contact当前没有验证资格，不授权真正动态耦合恢复。

80N力限、机器人碰撞、错误抓取、非计划张爪接触保留。物体接触需参考相对位姿相容（12mm/.15rad），第二件接触需首件实际完成。这是规则适用域，非全空间安全证明。manager_v2.physical_danger/substep_safety及intervention_manager.choose共享分类，消除上下层矛盾；未知真实A-B接触链仍停。

manager_v2.action_risks检查OSC坐标下预测位移/工作空间余量、实际碰撞几何间距与开合方向、非计划释放。仲裁顺序为硬危险、高置信动作风险、持物稳定性、停滞、窗口；同级按确定A/B顺序。

Constrained_A/B实施合法投影或保留闭合命令，最终动作入Replay。Constrained_Joint仅限两个独立正确持物时同时保留闭合夹爪命令；不是联合装配恢复。其他双臂同时风险无资格即停。近接触夹爪反转无已验证修正则拒绝。

Policy、Guide、Recover、Carry、Constrained动作分源。worker风险JSONL保留18维状态、原Actor动作、最终动作、类别、余量、执行状态和拒绝原因。危险未执行提议不伪造成transition。

## 持物恢复和被动臂减速

local_recovery.LocalRecovery以不可移动的入口快照保持被动臂，用末端—物体相对变换约束滑移，不重播整条示范，不写物理状态。进入需正确抓取稳定10步、相对平移<=.5mm/旋转<=.035rad、闭合命令、非耦合、持物无静态接触、首件装配前、双臂/所持物体相距>=80mm。

生产资格仅支持双臂分别稳定持物、已测目标位姿邻域。索引80是复现实验位置；运行检查目标实际位姿与捕获范围。其他carry邻域未验证，不能外推。

直接保持不合格时，仅Recover_B的被动A实测速度约.06379–.07302m/s区间允许Brake。沿用OSC kp/kd做有界速度反馈，至少3步准静态再Correct，减速最多20步，总恢复80步。没有改增益或取消速度门禁。主动A减速仍禁用。

被动EEF<=2mm、持物<=3mm原约束保留。交还需主动<=1mm/.02rad、持物目标<=1.5mm、滑移<=.5mm/.035rad及原参考物体出口约束，连续3步合格。qualification_v2保存控制器hash，代码变化不沿用旧证据。

装配原型为B第二件接触后3mm竖直卸载，A已完成且保持，首件漂移<=2mm。12次中1次Actor续行破坏首件，joint.enabled=false；不是可用的双臂动态耦合恢复。

## 动态Replay与全局能力

dynamic_demo.DynamicDemo实现official、safe_handback两模式，env/episode/frame/takeover隔离连续性，Policy间隔切段，重复行去重。safe_handback失败段仅留Online。原始224独立不变；原始/动态128/0、96/32、64/64和正回报最低0/2/4均可配。默认96/32和2，动态不足用原始补齐。重复采样不是新交互；所有方法采用同一内部采样设置时比较，0可关闭正回报过采样。

isolated_nstep按环境、Episode和连续片段隔离。成功/物理失败不bootstrap，正常时间/采集截止bootstrap，监督器安全失败不bootstrap；执行前拒绝/部分子步不产生完整行，只关闭已有真实前缀。干预段结束不等于任务成功。Worker明确区分严格任务谓词、稀疏reward、安全合格Episode success，未将terminated直接当成功。

global_capability.GlobalCapability统一汇总共享Actor的进入/推进/完成/稳定完成/自主完成/干预辅助。短暂接触不算成功，worker局部数仅诊断，不单独关闭干预。阶段5次稳定自主完成减少普通引导，完整5次严格无干预成功才全局退出；风险安全不退出。

## 持久化与入口

ascend_checkpoint保存模型、优化器、Online/Original/Dynamic、n-step尾部、采样器/RNG、来源、全局能力、worker及接管上下文、配置/参考/控制器hash。逐内容比较模型/优化器/Replay/采样，加载后再做真实梯度更新。恢复必须新Episode，旧接管上下文仅审计，不宣称MuJoCo中间状态位级续训。

scripts/acoserl_v2/validate_p0.py、upstream_oracle.py执行契约；contact_trace.py、risk_physics.py、physical_local.py、physical_hold_domain.py、brake_actor.py执行真实物理实验；publish_qualification.py由实际日志生成限定域资格。旧失败目录保留。

run_integration_suite.sh保留四方法与safe_handback复现入口。本轮实际完成修复前512步、修复后256步NPU训练；发现问题后停止后续调度，未中断当前训练子进程。四方法另用method_paths.py进行配对物理接线验证；actor_stream.py验证真实Actor采集和两种动态接纳，无梯度更新。16CPU环境仍按4+4+4+4初始化，主进程NPU；正式训练仍受V2资格阻挡。实际结果见VALIDATION_V2_ZH.md、ACCEPTANCE_V2.json。有界接线验证不证明学习效率提高。

## 有界训练暴露问题后的修复

首组512步中13次pregrasp保持超2mm。env07入口速度.04914m/s，预测1.574mm、真实2.131mm。最终仅在TwoPointMixin.eligible额外考虑完整50ms控制周期，预测约3.933mm即拒绝该次普通引导，允许安全RL继续。两个实际Actor动作前缀经正常初始化重放，末端复现差异为0，均拒绝不合格接管。

曾全局修改passive_hold_entry，但3例减速试验触发持物相对滑移门禁；已中止、保留brake_actor_latency_v1数据，并撤销全局修改。最终LocalRecovery和原有减速行为不变，未放宽漂移、接触力或夹持阈值。

另修复collector_cut不向Coordinator提交最后Episode的问题；修复后256步结束时worker与全局均统计16个Episode，采集截止不算完整成功。checkpoint代码哈希加入bimanual/intervention_manager/two_point，防止跨门禁版本静默恢复。
