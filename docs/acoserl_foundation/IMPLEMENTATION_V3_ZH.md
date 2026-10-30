# ACo-SERL V3 首轮实现进度

日期2026-10-30；分支codex/acoserl-v3；实现提交200e96bf6814db2e8db4e0bac701520745e858c5。任务全文保存在REQUEST_V3_ZH.md。每6小时在原对话继续执行，自动任务id=aco-serl-v3；确认正式训练真实启动并通过10项初始运行核验后才删除任务。

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
