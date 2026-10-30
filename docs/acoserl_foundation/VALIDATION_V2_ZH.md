# ACo-SERL V2 真实验证与启动判断

本轮实际完成代码、物理验证和NPU有界集成。总体为**限定恢复域通过、接触恢复未通过、训练接线通过、正式长训练未启动**。不能称已经解决全部问题，更不能称随机Actor已经学会装配。

## A. 已实现并通过验证

- 接触分类共享到控制决策与ActionOnly物理子步。官方示范224步严格成功，真实机器人碰撞和未验证接触链分别拒绝；未关闭碰撞、未放宽80N门禁。
- A/B工作空间、双臂同时风险、张爪接近物体、被动臂动态不合格等5类定向物理测试通过。风险动作经真实约束替换或拒绝；原建议与实际动作分开记录。
- official/safe_handback两模式，3种原始配额×3种最低正回报配额×2模式共18组合通过。原始224条、3条正回报不变；16交错流、48 Episode、141行边界契约及上游8行3-step数值对照通过。契约中的受控构造数据只用于数值验证，不冒充物理数据。
- Coordinator统一能力统计，阶段指标分离；最终collector cut的结果已补送全局，修复后worker与全局均16 Episode。能力退出合约通过，但真实Actor尚无稳定阶段完成证据。
- A/B持物纠偏和B前被动A减速为限定域真实通过，数据如下（漂移相对恢复入口固定基准）：

|模式|合格/试验|被动EEF峰值mm|被动持物峰值mm|恢复步数范围|
|---|---:|---:|---:|---|
|Carry_A|12/12|0.245|0.265|[5, 6]|
|Carry_B|12/12|0.163|0.173|[5, 6]|
|Brake_B|12/12|1.382|1.382|[8, 9]|

A/B各12次均在双臂分别正确稳定持物、示范80附近复现的实际位姿邻域，3–5mm扰动；不能外推整个轨迹。直接保持入口被动速度上界分别约2.518/3.660mm/s；B减速仅限被动A约63.793–73.024mm/s，减速3步，整体恢复8–9步。主动A减速、被动空手等域未获资格。

每次持物恢复之后5步均由真实NPU Actor以3路图像和18维状态产生动作，位置/姿态命令由Actor执行，夹爪受到安全屏蔽。**A/B两组的这120步没有未修改的完整Policy动作**，所以这是“结束恢复控制器并由Actor在安全约束下续行”，不是纯Actor无干预掌握任务。原始physical_local/hold_domain文件中seed是试验标签；实际随机流分别以20261029/20261030一次初始化后连续采样，不能解释为每trial重新设种子。脚本未来输出已明确root_seed；原始证据未改写。Brake Actor根种子20261031，同样连续流。

P0/P1保持既有物理行为：A=30→57，B=38→70；首次接触52/64、首次抓取53/65不冒充P0。窗口实际最多27/32点，按现有L=P1-P0约定，比官方闭区间点数少一点。HDF5记录229动作/229状态，独立真实回放参考含230状态；末状态没有写回原HDF5。

## B. 已实现但物理验收未通过

B第二件插入接触后竖直卸载3mm原型共12次，11次完整通过，1次（local_physics_v3/Contact_B_6.json）在交还后的第5次Actor动作执行期间触发completed_piece1_displaced；成功返回的actor_steps计数为4；第5次动作已执行，随后安全检查拒绝，不计为成功续行。该次局部servo到位并不能算任务级恢复成功。joint.enabled=false，未知真实双臂运动耦合仍停止。没有可用的Legal_Cooperative_Contact动态联合恢复域。

原局部试验被动空手组合A/B各6次均未通过交还后续行（另一臂立即重新接管等），资格继续排除，不能写成所有空手状态都物理不可恢复。

初始512步发现13次pregrasp被动保持超2mm。env07预测1.574mm、实际2.131mm。最终在接触前恢复eligible入口加完整50ms控制周期余量，计算约3.933mm时拒绝普通引导并继续安全RL。env07/env00相同实际Actor动作前缀复现差异为0，两例均通过。旧的2mm/3mm、夹持、姿态、接触力条件未放宽。

曾试图全局改变保持判据，3个已完成减速案例触发held_relative_transform_changed；试验中止，保留brake_actor_latency_v1，撤销全局变化。最终只加强pregrasp入口，原已验证LocalRecovery/Brake逻辑未改。

## C. 训练与真实采集集成

硬件实际为单张Ascend910B3，当前容器物理卡3、逻辑npu:0；驱动/npu-smi26.1.1、CANN8.0.RC2、Python3.10.20、torch2.3.1、torch_npu2.3.1、MuJoCo3.2.3。cgroup CPU22核、内存208240902144字节。环境CPU OSMesa、每worker线程1、无torch_npu上下文，模型主进程NPU。

先前a标签在容器更换时仅初始化，进程消失，旧目录完整保留，不计作训练通过。b/c均有4+4+4+4 reset屏障和all_sixteen_ready日志，再同时派发16条动作。修复前后均从相同随机Actor/Critic种子和空Online Replay开始，初始化模型hash一致。

|真实运行|有效交互|Actor/Critic各更新|Online行|Dynamic新增|Safe_Stop|含评估/保存有效步/s|末轮采集更新循环步/s|
|---|---:|---:|---:|---:|---:|---:|---:|
|v2_bound_20261029b_full_official|512|512|512|25|13|1.050|1.646|
|v2_bound_20261029c_full_official|256|256|256|0|0|0.934|1.662|

实际训练总768步/768次Actor更新/768次Critic更新，UTD=1，原始Offline各224。两次均完成16环境交互、混合batch、真实更新、pre_restore重载逐内容一致、加载后最后一次真实更新、final持久保存；未加载旧RL或Replay。Actor14、Critic15、适配层6个参数张量发生变化，三个冻结主干hash完全不变，所有loss/梯度有限。

b组动态25行来自15次Guide_B真实动作和10次Constrained_B动作；13次Guide保持失败，风险投影交还10次，其中2次之后有未修改Policy动作。c组256步均Policy，Dynamic为0，不能把未触发当恢复成功，也不能声称修复后的256步已经检验了有恢复样本的所有训练情形。

额外actor_stream_v1为**无梯度的定向真实Actor采集专项**，加载c组当前checkpoint，92条完整物理交互，0次部分子步，4次未执行拒绝（96次提议）。official与safe_handback各52行真实动态样本；每模式实际混合batch在线128/示范128，示范96原始/32动态，实际正回报4条、最低强制2条。Online 92行；两种动态缓冲及来源分别持久保存。两模式使用同一实际数据流，不是新增104条交互。

专项52次均为Constrained_A/B类风险投影的单步交还，47次同Episode之后执行了下一次Actor输出，但未修改Policy续行0次。不能把这些计为52次持物或长程恢复，不能把持续屏蔽危险方向写成Actor已经会安全探索。

### 速度与资源

|累计循环计时秒|b512|c256|
|---|---:|---:|
|inference|2.373|1.821|
|step|90.767|43.879|
|reset|4.413|0.004|
|sample|44.071|21.730|
|transfer|3.080|1.861|
|update|163.355|83.211|

b/c总墙钟分别802.2/527.5秒（含初始化）；NPU遥测AIcore均值20.36%/15.29%，峰值99%/98%，HBM峰值均5165MiB，PyTorch峰值分配1599214080字节。均值含初始化/评估/等待，不能等同更新内核利用率。每worker step均值及全部阶段计时见ACCEPTANCE_V2.json。

训练每条被接管的完整transition恰好一个50ms物理控制步；未在env.step内部隐藏多步或把重复Replay采样计为新经验。b组25/512被接管控制步（约.04883/有效步），c组0/256；额外定向采集52/92约.56522。未执行拒绝无transition。

旧ResNet18正式记录1.71有效步/s，与本轮末轮循环约1.65–1.66接近，没有明显吞吐提升；含评估/保存的本轮约1.05/.934更低。预算、冷启动和评估开销不同，不作严格硬件性能因果比较，更不能声称样本效率改善。

### 每worker诊断

b组29 Episode含最终16个采集截止；旧全局仅13个Episode的漏计已修复。c组16个Episode各16步，无安全停止，但都是采集截断，不是任务成功。

|env|b Episode|b步|b接管|b交还|b Safe_Stop|c步|c Safe_Stop|
|---|---:|---:|---:|---:|---:|---:|---:|
|00|2|32|2|1|1|16|0|
|01|2|32|1|0|1|16|0|
|02|1|32|0|0|0|16|0|
|03|2|32|2|1|1|16|0|
|04|2|32|1|0|1|16|0|
|05|2|32|1|0|1|16|0|
|06|2|32|2|1|1|16|0|
|07|2|32|3|2|1|16|0|
|08|2|32|2|1|1|16|0|
|09|1|32|0|0|0|16|0|
|10|2|32|2|1|1|16|0|
|11|2|32|2|1|1|16|0|
|12|2|32|1|0|1|16|0|
|13|1|32|0|0|0|16|0|
|14|2|32|3|2|1|16|0|
|15|2|32|1|0|1|16|0|

c组两臂approach_pregrasp分别进入/推进16次，完成/稳定完成/自主完成均0；推进指标不是装配事件成功。完整任务无成功。每worker reset/来源/失败原因和Episode长度分布在JSON完整保存。

两次各做2回合关闭干预的独立评估，b均22步后workspace_exit_B，c均13步后workspace_exit_A，成功率均0/2。同一checkpoint的两条轨迹hash完全相同，属于固定canonical初始状态的确定性重复，不是2种初始状态，更不是20种。20%/50%/80%目标均未达到，不外推收敛。

四方法均有完整配置和真实调用路径。method_paths_v2在相同38步联合前缀、40步稳定保持后做配对停滞验证：plain/window-only保持Normal，window_recovery/full自动Recover_A并安全交还（189总控制步），无状态写入。该专项用于接线，不是四组等预算RL学习效果比较；没有启动四组正式训练。

## D. 正式训练判断

1. 双臂自由空间主动干预：窗口、P0/P1、方向/停滞、风险筛选均实现并在限定域验证；不等于覆盖任意动态保持状态的完整安全能力。
2. 持物局部恢复：A/B各12次限定双持物域通过，B限定速度减速12次通过；自动carry恢复尚未在本轮RLPD短运行自然出现，Actor续行仍受夹爪约束。
3. 联合接触恢复：当前没有可启用物理域。B竖直卸载原型11/12，禁用；真正A-B动态耦合无已验证控制器。
4. 动态Demo：两模式和18采样组合通过，真实Actor专项各52行，训练b组official新增25；原始224独立，失败/截断/拒绝语义、n-step和重载验证通过。
5. 16环境NPU：已真实验证，共768有效训练步、各768次更新、两次checkpoint精确内容重载及梯度续更。
6. 是否改善随机Actor探索：只证明两个错误接管状态被排除、短程早停减少；没有阶段自主完成或装配学习效率提高证据。风险投影频繁重入仍限制自由Actor行为。
7. 是否启动正式长训练：**否**。用户已授权问题解决后直接启动，但接触恢复尚未通过、纯策略阶段进展与稳健交还证据不足，条件尚未满足。保留formal_training_approved=false，不用长训练掩盖这些失败，也不需要再次请求授权。

下一步应针对接触卸载后交还时的首件支撑/相对位姿约束，以及风险投影反复重入与Actor探索空间开展定向机制验证；不能通过放宽2mm/3mm或任务成功标准解决。

## 持久化、复现与交付

全部运行、模型、Replay和原始HDF5均保留，未执行破坏性Git清理。现有183项删除和两个旧状态文件修改不纳入本轮提交。源码实现5ec667f3c937a11234cef7fa7b6ebd97822af37c，保持入口/统计修复43073293d4d2448538a479ef2b04edff54975fd4，最终报告/复现脚本提交见Git HEAD及交付答复；未push。

核心输出：
- outputs/ascend16/v2_bound_20261029b_full_official/checkpoints/final
- outputs/ascend16/v2_bound_20261029c_full_official/checkpoints/final
- outputs/acoserl_v2/actor_stream_v1（真实Online及official/safe_handback）
- outputs/acoserl_v2/local_physics_v3、hold_domain_v1、brake_actor_v1（物理限定域）
- outputs/acoserl_v2/passive_hold_latency_v2、method_paths_v2、p0_contracts_final

checkpoint恢复只在新Episode边界进行，关闭旧合法尾部并取消物理接管，不宣称MuJoCo中间状态位级精确续训。复现脚本和实际源码清单如下，均相对项目根目录/data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen：

- configs/bimanual_autoserl/manager.json
- configs/bimanual_autoserl/methods_v2/full_acoserl.json
- configs/bimanual_autoserl/methods_v2/independent_window.json
- configs/bimanual_autoserl/methods_v2/plain_rlpd.json
- configs/bimanual_autoserl/methods_v2/window_recovery.json
- configs/bimanual_autoserl/qualification_v2.json
- docs/acoserl_foundation/ACCEPTANCE_V2.json
- docs/acoserl_foundation/IMPLEMENTATION_V2_ZH.md
- docs/acoserl_foundation/VALIDATION_V2_ZH.md
- scripts/acoserl_v2/actor_stream.py
- scripts/acoserl_v2/brake_actor.py
- scripts/acoserl_v2/brake_probe.py
- scripts/acoserl_v2/collect_results.py
- scripts/acoserl_v2/contact_trace.py
- scripts/acoserl_v2/method_paths.py
- scripts/acoserl_v2/passive_hold_latency.py
- scripts/acoserl_v2/physical_hold_domain.py
- scripts/acoserl_v2/physical_local.py
- scripts/acoserl_v2/publish_qualification.py
- scripts/acoserl_v2/risk_physics.py
- scripts/acoserl_v2/run_integration_suite.sh
- scripts/acoserl_v2/upstream_oracle.py
- scripts/acoserl_v2/validate_p0.py
- scripts/acoserl_v2/write_evidence.py
- scripts/ascend16/train.py
- src/acoserl/ascend_checkpoint.py
- src/acoserl/autoserl_env.py
- src/acoserl/bimanual_autoserl.py
- src/acoserl/contact_modes.py
- src/acoserl/dynamic_demo.py
- src/acoserl/global_capability.py
- src/acoserl/intervention_manager.py
- src/acoserl/local_recovery.py
- src/acoserl/manager_v2.py
- src/acoserl/parallel_autoserl.py
- src/acoserl/transition_semantics.py
- src/acoserl/two_point_supervisor.py
