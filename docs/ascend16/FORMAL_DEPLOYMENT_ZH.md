# Ascend 16环境正式训练部署记录（2026-10-28）

用户已明确固定UTD=1、无需进一步测试、确认环境后直接训练。本次按此执行，未把上次中断的联合压力测试记作通过。docs/ascend16/deployment_gate.json逐项记录历史分项验收通过、联合压力和联合恢复尚未验收，以及本次直接启动授权。

## 环境

通过ssh root@your-server连接新节点<REDACTED_NODE>。物理NPU编号已变为2，逻辑仍为npu:0；一张健康910B3、65536MiB HBM；cgroup实际22核、208240902144字节内存。驱动/npu-smi 26.1.1；CANN Toolkit 8.0.RC2；Python3.10.20、torch/torch_npu2.3.1、TorchRL0.4.0+b4f4daf、TensorDict0.4.0+6dc8c68、NumPy1.23.3。实际完成NPU张量计算回读2.0。

使用独立.venv_autoserl_npu，未安装修改旧环境。清除PYTHONPATH后，.venv_resfit的pip freeze与部署前完全相同；否则CANN set_env附加的包路径会产生假差异。官方兼容矩阵依据：https://gitcode.com/Ascend/pytorch/blob/bfafdcc1be36e979a4e748235a277a170ff385ea/README.md 。依赖快照见configs/ascend16/npu_runtime_freeze.txt。

## 启动与实现

正式标签formal_20261028_npu16_utd1_fastreset，北京时间09:43:31启动，正式随机初始化、不加载smoke模型。nohup+setsid持久会话、flock防重复；启动脚本scripts/ascend16/launch_formal.sh，状态读取scripts/ascend16/status.py。

4组AsyncVectorEnv依次各初始化4个并reset，全部16个ready后才计步。每轮先向4组step_async、再收取4组结果，共16环境同时采样。每worker独立种子/Episode/AutoSERL状态，OMP/BLAS/Mesa线程限制为1，无torch_npu导入；主进程集中NPU推理更新。仅父进程SpaceOnly避免额外模拟器。自动reset关闭，终止或Safe_Stop只reset对应worker。

只读demo_0、严格224条有效离线transition、18维状态/14维动作/三路84x84图像，沿用原归一化和ActionScaler。Replay保存实际执行动作，动作前拒绝不造transition；未完成物理子步单独统计不写完整transition。按env_id+episode_id隔离3-step、终止关闭bootstrap、截断保留bootstrap、短尾flush。历史nstep/offline分项验收见相应JSON。

未修改官方ResFiT网络、损失、优化器算法。10-Q及编码器在NPU；历史npu_acceptance.json记录真实更新、有限loss/梯度、参数变化、NPU优化器矩、模型和离线Replay保存加载及加载后更新。被动臂动态保持资格检查及安全阈值不变。未知/耦合状态仍安全拒绝，不支持联合恢复；特权物体状态不进入Actor。

## 配置与保存

UTD=1按预热后每条真实transition计算，batch256在线/离线各128，预热1024条。最多500000次实际交互或10小时，最后不足16步预算时提前结束，恢复动作包含计步，评估额外开销独立统计。每25000步保存完整checkpoint，每50000步关闭干预评估20Episode，仅新增1worker且暂停训练采样；到时间未完成评估标记incomplete。

模型/target/优化器/Replay/归一化/主进程及worker RNG和监督器快照持久保存，原子目录提交，无官方清理逻辑。恢复从新Episode边界开始，pending真实短尾截断；不声称MuJoCo中途位级续训。集成恢复路径尚未经联合实测。

## 结果位置及限制

outputs/ascend16/formal_20261028_npu16_utd1_fastreset：workers/startup.jsonl记录四批启动；learning_curve.jsonl记录阶段、步/s、更新/s及各环节耗时；updates.jsonl记录loss/梯度；npu_telemetry.jsonl记录真实利用率/HBM；workers/envXX.jsonl记录每环境Episode、干预/恢复/Safe_Stop原因及长度；evaluation_*.json记录无干预评估；checkpoints保存模型和Replay；summary.json在结束时汇总。

训练仍在进行，不能提前给出最终成功率、最终吞吐或完整训练效果。实际采样启动和NPU更新必须以实时日志为准，不能将进程启动等同于全部16环境ready。联合压力与完整恢复未测，已明确记录，不能宣称验收通过。

## 首次正式启动的效率修复


首次启动保留在 outputs/ascend16/formal_20261028_npu16_utd1，记录到96条真实交互、0次网络更新；重复XML重编译导致reset占用约88%，已停止该次运行并保存完整已有日志和interrupted_for_efficiency.json。当前正式标签带_fastreset后缀，从正式随机初始化重新启动。

最小修复仅作用于显式新Episode的reset：首次加载原示范XML，以后使用robosuite官方deterministic soft reset重置控制器/夹爪，复用相同MuJoCo模型和OSMesa上下文，再设置同一个demo_0/frame0初始状态。运行时逐次断言模型对象、body_pos/body_quat、完整flatten状态以及18维状态和三路图像观测完全相同；任何不一致立即报错。绝不在Episode或恢复动作中修改物理状态。不改变Safe_Stop触发、初始条件或奖励。新增SIGTERM/SIGINT请求正常循环边界退出并完整保存。

## 本轮实际学习证据（持续运行中的快照）

快照：真实交互1045条，随机预热1024条，Critic和Actor各21次更新，在线Replay已提交1042条，3条真实transition仍在n-step缓存；离线224条。当前warm-up后的有效UTD=(1045-1024)/21=1。loss和全部梯度范数均通过运行时有限值检查。首轮batch_R=0.00777344，Critic loss=0.187632，Actor loss=1.732227。未要求装配成功。

四批初始化耗时54.374、48.721、52.318、55.592秒，16个独立PID已ready。每worker常驻约1.7GiB、4个总线程（OMP/BLAS/Mesa工作线程各限制1，含辅助线程），未导入torch_npu；总内存约38GB。重置优化后早期实测2.8步/s，长一点的快照为1.852步/s；原重复XML版本约0.66步/s。当前21次更新实际累计5.102秒，即纯更新约4.12次/s；包括整个预热和采样的累计更新速率为0.0372次/s，二者不能混淆。5秒采样遥测捕获AICore峰值61%，HBM峰值5810MiB；含长预热/初始化的均值仅0.462%，不能拿该均值代表持续更新利用率。npu_smi_formal.txt保存当前实际设备进程证据。

目前16个worker的早期Episode多数仅1–2个真实控制步即因no_validated_independent_phase结束，尚无Recover_A/Recover_B和成功交还事件；这是本轮正式数据当前的真实覆盖限制，不应以历史阶段性纠偏验收冒充本轮已经发生纠偏。保留严格监督器继续训练，不以放宽阈值、改初始阶段或虚构成功掩盖。后续能否学会进入有效持物阶段必须查看实际学习曲线。formal_initial_learning.json包括逐环境Episode/Safe_Stop统计，未进行完任何20-Episode独立评估，不报告成功率。

此前第一次启动在更新前停止，仅保留已有日志和交互证据，没有创建checkpoint；当前新正式运行尚未达到25000步，故尚无周期完整checkpoint，不能宣称集成恢复已完成验证。持久化实现和历史NPU模型/离线Replay加载已验证，完整联合恢复按用户本次指示未追加测试。最终训练结束时将走完整保存路径；异常节点丢失前未保存部分可能丢失，不能宣称中途位级恢复。共享DPC文件系统残留锁曾造成重启被拒绝，确认无旧进程后防重锁迁至本节点/tmp；日志、模型和Replay仍持久保存在项目共享目录。

Implementation Git commit: `31c34f6b60ccfb6524ebab8b08892ced92db804c` (branch `codex/ascend16-autoserl`).
