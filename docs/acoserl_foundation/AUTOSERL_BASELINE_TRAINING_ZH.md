# 单任务 AutoSERL 正式训练部署与状态

系统验收A已通过，正式训练B须以真实PID、16worker就绪、NPU进程、有效步及网络更新持续增长确认；自主任务学习C仍未证明。不能将示范回放、受控P0准备、投影动作或恢复抓取写成Actor自主任务成功。

启动脚本：scripts/autoserl_baseline/launch_formal.sh。完整配置来自configs/ascend16/formal.json，并显式选择baseline监督器、resnet18_spatial84、official Dynamic接纳、原始/动态96/32、正奖励最低2。不运行普通RLPD或消融。

正式配置：16环境4+4+4+4，Ascend910B3，14维联合Actor、3路RGB84、18维低维、10Q、n=3/.99、batch256、Online/Demo各128、UTD1。预热1024真实步，之后每个有效交互步一次Actor和Critic更新。实际恢复动作和完整干预交互均计步，未执行提议不计为物理步，部分物理尝试单独记录并不伪造transition。

新训练不使用任何验收/旧RL权重或Replay。仅加载既有真实公开ResNet18预训练冻结主干，新建随机Actor/Critic/适配层及空Online/Dynamic；原始224条严格示范作为初始离线数据。正式模式不能使用--integration-pregrasp诊断准备。

预算500000实际交互或36000 s先到为止；checkpoint每25000、关闭主动干预评估每50000/20Episode，额外评估worker仅1个。评估保留同一硬安全，canonical初态重复不作为独立物理状态；阶段和严格完整成功分开，安全投影贡献不当作自主进展。

2k/5k/10k/25k早期诊断写出运动、来源、抓取/装配阶段、worker及安全原因。10k无任何任务阶段且投影或短Episode重置死锁则保存并停止；完整成功率0本身不触发停机。阶段计数中step occurrence不是Episode成功率，受控准备不进入正式训练。

checkpoint保存模型/目标网络/优化器/Replay/归一化/随机数和监督器审计信息。正常预算、操作者SIGTERM及诊断停机走完整收尾；不要SIGKILL。恢复Learner后各worker从新Episode开始，旧尾片段合法关闭；不宣称MuJoCo中途位级恢复。脚本不删除运行目录。异常日志保留；进程级故障的恢复以最后一个完整checkpoint为准。

持久化方式为nohup+setsid，当前服务器无tmux。全项目正式训练锁为/tmp/acoserl_dexmimicgen_formal.lock，避免同NPU重复训练。PID、tag、stdout在logs/autoserl_baseline。用户已授权通过必要验收后直接启动，无需再次确认。

现存183项工作树删除及两项旧状态文件修改均按现场保留，不纳入本次提交；没有同步本地E盘源码或push。

本文件末尾的运行快照必须来自真实查询，启动命令提交成功不能单独证明B完成。


## 正式运行已核验（状态B）

实现与物理/NPU验收提交为91f9e545b6ce4f364ca872b50497d8bc265fa070。正式运行使用该提交的训练源码，后续提交只保存监控工具和启动记录，不热修改训练控制代码。

运行tag：autoserl_baseline_formal_20261030_133346。
权威输出目录：/data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen/outputs/ascend16/autoserl_baseline_formal_20261030_133346。
主Learner PID112316，持久化launcher PID112311。日志位于logs/autoserl_baseline/同名tag.log。原六小时automation aco-serl-v3删除接口返回not_found，本机亦没有匹配任务；没有创建重复定时任务。

已保存两次真实查询：1120→1280有效物理步，Actor/Critic各96→256次更新，Online1072→1232条；1024步预热后UTD=1。16worker全部ready且查询时均存活，实际NPU进程匹配112316，NPU利用率快照49%→98%，HBM5164 MB。初始化记录fresh_rl=true、fresh_online_replay=true、Online0、原始Offline224，未传resume、未使用专项准备入口，也未导入验收147条Dynamic。

当前正式Dynamic仍为0、Guide/Recover接管仍为0、实际抓取和完整任务成功均为0；不能拿有界验收中的147条和一次交还冒充正式运行结果。自动干预开关保持开启，未知恢复域保持关闭，策略未获准提前关闭监督。当前1280步均有实际非零运动，仅1次workspace_exit_A安全结束，没有1–2步短Episode系统性复位或初始phase误停。

checkpoint周期仍25000，因此启动核验时尚无正式checkpoint；保存/重载/续更能力在有界验收实际通过。首次正式20Episode无干预评估应在50000步到达时进行，目前没有正式成功率证据。状态C没有达到。

新的只读状态工具：source scripts/ascend16/activate.sh后执行python scripts/autoserl_baseline/status.py。它读取当前baseline tag，避免旧status脚本读取上次实验。验收和正式运行两个目录严格分开。

完整机器启动快照保存在outputs/autoserl_baseline/formal_startup_current.json、formal_startup_confirmed.json，以及本目录AUTOSERL_BASELINE_FORMAL_STATUS.json。下一轮只监控当前PID、有效步、更新、阶段与2k/5k/10k诊断，不重复启动NPU训练，不回到旧V3高级12+4研究。

## n
上述为历史启动快照；此训练已按用户要求正常停止，最终有效步11587，Dynamic Demo为0。系统需完成新的自然探索验收后才能重新评估开训。
