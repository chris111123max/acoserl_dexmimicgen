# ACo-SERL：Guide冷却期的显式边界安全动作（v2）

## 根因依据

在官方SERL ResNet-10的16环境NPU有界集成中，10个PARTIAL_PHYSICS中断集中在A臂Policy/B臂Guide、episode0 frame30/31、最近参考索引15，A臂z上界超出0.030～0.387mm；当时A臂Policy冷却仍剩14～19步。16环境首次A臂Guide交还在frame11～17；10例的方向余弦均为正，因此90度正向门禁没有直接拒绝它们。

这只能说明普通Guide冷却抑制了重新接管，**并不能证明纠偏命令物理可行**。本提交提供显式、非专家的安全介入候选，等待Codex真实MuJoCo专项。正式训练仍NOT READY。

## 协议

- 名称：`stage_aware_bimanual_recovery_v2_boundary_guard`，作为控制与checkpoint不兼容性边界。
- 只在intervention_enabled且当前臂为`Policy`、`cooldown>0`时启动，不改变Guide的32个**完整Policy周期**目标。
- 风险判断使用**世界坐标**下真实末端site位置、实际site线速度、以及按现有`servo`同样坐标变换重建的Actor提出的OSC平移命令。该命令不是对下一步物理的预测。
- 与原工作空间一致：`reference_eef_position`全段xyz极值±原有`free_workspace_margin_m=0.12`，不更改工作空间。
- 进入距离：距离某轴上/下边界≤25mm，且该轴的Actor提议明显向外（绝对值>0.1mm的控制器期望平移）或真实末端速度明显向外（绝对值>5mm/s）；其他情况无介入。
- 请求的伺服目标：该风险轴向盒内侧35mm的位置；其他非风险平移轴及全部姿态维持当前实际末端位置/姿态；保持原有Actor夹爪动作、另一臂独立控制。动作仍受现有10mm位置命令上限、0.02rad角度命令上限和动作缩放边界约束。
- 不实施可疑动作的沉默剪裁：安全动作若落在原Offline ActionScaler允许域外，立即Safe_Stop并保留诊断；不回退执行高风险Actor动作。
- 不接管处于Guide、ReturnS0/P0/P1、Replay、Settle或Handoff的机械臂；不在当前臂接触对象、已进入装配接触、A/B耦合时插入单臂安全命令。疑似接触且有风险时Safe_Stop，不能假称已完成保护。
- 同一臂最多连续执行16个完整边界安全控制周期；超过且仍有风险则Safe_Stop，不允许无限掩盖Actor失控。
- **最终合法性由原ActionOnly执行器决定**：逐子步私有数据预演、80N接触力、碰撞检查、真实工作空间检查全部保持。此算法没有证明每个向内OSC命令必然可行、不会越界或不发生安全停止。

## 训练数据与控制权

- 每臂`per_arm_control=BoundaryGuard`，明确不是`Policy`或`Guide`。
- 单独标记`BoundarySafety`模式；如另一臂Guide，模式仍`Guide_AB`，但每臂控制来源、z轴边界距离、向外提议和测得速度都记录。
- 经过完整物理控制周期才可作为真实Off-policy Transition进入**Online Replay**；不完整物理动作仍经`IsolatedNStep.close()`排除。
- 按实际来源标记为`Constrained_A/B/Joint`，pose_sources=SafetyProjection=4，BoundaryGuard拥有的6D位姿不冒充Actor；两只夹爪来源不改变。
- 当任何一臂为BoundaryGuard时：`quality_validated=False`、`expert_action_mask`全零，**不能进入动态专家Demo**。同时清除同worker中可能被后续别臂Guide交还误认证的待资格候选，不添加虚假Policy handback。
- 该臂的滑动参考窗口和32步Policy冷却**只在实际完整Policy控制周期**推进；边界安全控制仅增加独立计数，不能耗尽冷却配额。
- 不使用任何MuJoCo内部物体状态作为Actor输入；安全监督器可继续读取现有特权物理测量。
- `SimpleDemo`协议升级为`simple_bimanual_demo_handback_v3_boundary_guard`，旧训练checkpoint不得无声恢复。

## 定向验证（不是冒烟测试）

先运行旧27项阶段恢复回归，加上`scripts/autoserl_baseline/test_boundary_guard_regressions.py`新增12项合同测试。新增测试包括世界坐标接近上界、向外命令/速度、无风险不干预、OSC base→world映射、真实控制来源与gripper独立、冷却不减少、专家资格隔离、不可执行动作Fail-closed等。

**然后Codex必须真实物理有界专项**：

1. 在仓库外建立只读诊断运行，以之前10失败＋6同期未失败环境的真实记录为入口。按各env原始种子和同一初始状态恢复，用真实完整Actor提议或录制的Policy提议逐步运行，不允许通过修改中途qpos/qvel来制造接近边界的状态。如果无法精确重放，应标记Not Reproduced，禁止宣称成功。
2. 在相同起点记录原行为与新BoundaryGuard物理轨迹（注意此后状态改变，不能再假定执行同一状态下的Actor输出）。逐事件验证触发frame、位置、距离、World OSC向量、实际速度、控制输出、ActionScaler允许范围、每个真实子步与预演状态。
3. 重点确认A臂风险被提前识别后，在原0.030～0.387mm越界前是否能产生**实际完成**的向内控制；若发生新的SafetyStop，必须记录是否因惯性、动作域限制、与另一臂耦合还是其他问题。
4. 核实B臂Guide未被接管，两臂夹爪原始动作不变；Guard不占用Policy滑动窗口/冷却，不生成Guide交还或新动态Demo；满足回合完整性的Guard动作在Online Replay中留有来源证据；未完成动作严格无行。
5. 必须继续遵循80N、碰撞、工作空间以及原始substep preview保护；不得扩大阈值或启用无法解释的数值回退。
6. 若连续Guard耗尽16次或无法映射到离线动作域，必须Safe_Stop且无假转移。
7. 输出真实物理完成/不完整数、触发前后z轨迹、Guide和Policy占比、非自主安全动作占比、Policy冷却持续性、Demo资格证据；只有物理专项安全可行、数据门禁无问题后再建议进入16环境NPU集成。

如果物理专项证明这种控制无法及时抵消实际z向惯性，ChatGPT需根据证据重新设计更早预测或动作约束，**不允许以取消预演或扩边界求通过**。
