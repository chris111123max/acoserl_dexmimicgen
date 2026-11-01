# 独立双臂 AutoSERL 仿真基线

## 为什么新增，而不是继续给旧控制器打补丁

本提交的父版本是 `1c235542d88e4fac1024f2cedd77b2f194d8070c`。旧版增强控制、Forward Grasp Guide、资格文件、投影器和训练入口完整保留；本基线从新的入口运行，不再继承 V2/V3/ReviewedRuntime 的控制链。

这是 AutoSERL 核心机制在双 Panda / MuJoCo 上的适配基线，不是声称对官方硬件、环境、奖励、优化器、动作坐标和成功率逐项完全复现。真实物理和Ascend效果必须由服务器验证。本代码仅面向仿真，不授权控制实体机械臂。

## 官方依据及适配边界

读取的官方实现：

- 仓库：`autoserl/AutoSERL`，提交 `d4d1f2f24e52f80ee59fb44ca5d621eccad6a732`。
- `serl_robot_infra/franka_env/envs/wrappers.py` 中的 `auto_intervention_wrapper`。
- https://github.com/autoserl/AutoSERL/blob/d4d1f2f24e52f80ee59fb44ca5d621eccad6a732/serl_robot_infra/franka_env/envs/wrappers.py

保留的机制：初始前向窗口长度 `P1-P0+1`；窗口内最近点和全局最近点处的切线方向；偏离超过2cm且方向夹角不大于90度时引导；最近参考进度连续20步近乎不动时返回P0，再执行示范恢复段；真实成功且干预控制步少于10时永久停止任务干预。

明确的任务适配：

1. 同一14维Actor，两套独立的窗口、停滞历史、Guide和恢复进度。不是两个分别训练的Actor，也不是两个独立MDP。
2. 普通Guide拥有主动臂6维位姿，夹爪仍为Policy；恢复段拥有该臂7维动作，包含真实示范的开/闭爪。另一臂继续自己的Policy或自己的干预，**没有被动臂Hold**。
3. A/B可以同时干预，不使用单个全局`active`互相抢占。各自交还要求至少一个完整、通过物理保护的Policy动作，不能把到达目标当作交还。
4. P0/P1使用仓库已有示范事件：A为30/57，B为38/70。它们是状态索引，回放`actions[P0:P1]`到达状态P1，明确采用半开区间，避免状态/动作索引混淆。
5. 恢复只在主动臂空手、无当前接触、目标物体仍在记录拾取位姿、双臂无机械耦合时开始。不回退已完成阶段，不给未知装配接触规划单臂恢复；实际Actor仍可在力/碰撞/工作空间限制下执行正常装配。
6. 增加有界恢复和实际抓取确认：到P1后检查真实抓取、闭爪、目标位姿和10步相对稳定。未抓住会失败退出，不能伪造成功。旋转、保持稳定和手交还约束是双臂抓取适配，不是声称官方逐行相同。
7. 参考坐标转换使用项目现有base-frame OSC接口。示范回放使用现有14维HDF5中对应臂的7维真实动作；不复制官方硬件坐标变换到MuJoCo。

## 唯一的新控制路径

```text
RLPD 14D Actor
  -> SimpleBimanualAutoSERL
       A: Policy / Guide / ReturnP0 / Replay / Settle / Handoff
       B: Policy / Guide / ReturnP0 / Replay / Settle / Handoff
  -> 两臂动作合并
  -> 原有动作归一化桥
  -> ActionOnly私有子步预演 + 实际子步保护
  -> 真实观测、严格奖励、Online及Demo
```

`simple_autoserl.py`不继承旧干预器；仅复用观测、Contacts和OSC变换工具。`simple_collection.py`复用旧collector的canonical reset、图像归一化和4x4并发调度，但不实例化旧的干预管理器。旧`launch_formal.sh`仍运行增强方案；新基线必须使用`launch_simple.sh`或`train_simple.py`，不要混用参数和checkpoint。

## 安全层保留什么、不再做什么

保留80N停止阈值、真实工作空间边界、非有限值检查、机器人相互碰撞/环境碰撞检查、错误抓取关系检查、已稳定完成首件的进度保护。现有ActionOnly仍在每个物理子步之前用私有MjData预演，并检查实际执行后的工作空间/接触/力；失败动作不会伪造成完整transition。

新基线不调用旧`open_gripper_unplanned_approach`的名义距离启发式，也不调用多层QP投影和静态保持证书。**开爪靠近自己的目标物体不等于实际碰撞**；是否可执行由未改变的实际物理保护决定。正常Policy接触装配物体仍允许；在耦合或装配接触中不进行未经定义的单臂干预。

这不保证每个示范动作都一定通过，更不保证实际力峰值绝无超调；若官方示范仍被实际物理保护拒绝，必须记录精确子步/几何体/力，不能为得到PASS而关闭检查。没有被动Hold意味着2mm/3mm被动漂移资格不再是该基线的前置条件，而不是扩大了一个正在执行的Hold状态域。

## 两个逻辑Buffer及终止

Online保存所有完整真实transition；Demo由原始224条和实际干预数据组成。`simple_replay.py`按真实逐臂所有权记录0/6/7/12/13/14个专家维度，另一臂不会被误记成Hold。枚举13 `Guide_AB`表示新基线干预组合，具体来源以`per_arm_control`、`pose_sources`、`gripper_sources`和`expert_action_mask`为准。

专家掩码是来源记录，不是新BC损失。RLPD仍学习完整14维动作，10-Q、UTD=1、n-step=3、128 Online+128 Demo保持。Demo内部96/32及正奖励配额2沿用本项目对照配置，不声称是官方内部配额。

每次A/B来源组合或接管编号变化都分段；n-step不跨环境、Episode、部分物理失败或不连续干预段。真实失败不入Demo；不完整执行不进Online。两臂同时干预按一次真实环境步计数，不虚增交互量。

新checkpoint协议为`simple_bimanual_collector_v1`、动态示范协议为`simple_bimanual_demo_v1`。不得直接加载旧增强方案的模型/Replay作为本次续训。恢复checkpoint时仅恢复学习器、Buffer与随机状态，所有worker从新的canonical Episode开始，不恢复中间物理状态。

## 运行方式

先使用既有Ascend环境；SSH和仓库同步流程不变。新脚本使用`PYTHONDONTWRITEBYTECODE=1`。所有新增输出必须位于Git仓库外，并先检查实际挂载。

```bash
cd /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
source scripts/ascend16/activate.sh
export PYTHONDONTWRITEBYTECODE=1
BASE=/data/home/3220251075/lerobot_workspace/training_runs/acoserl_dexmimicgen
python -u scripts/autoserl_baseline/train_simple.py \
  --mode integration --method autoserl --tag simple_seed20261032_YYYYMMDD_HHMMSS \
  --seed 20261032 --integration-steps 2048 --integration-seconds 3600 \
  --output-root "$BASE/simple/integration"
```

第二组seed为20261033，串行运行并使用唯一TAG。正式运行：

```bash
export ACOSERL_SIMPLE_METHOD=autoserl
export ACOSERL_OUTPUT_ROOT=/data/home/3220251075/lerobot_workspace/training_runs/acoserl_dexmimicgen/simple/formal
mkdir -p "$ACOSERL_OUTPUT_ROOT/launcher_logs"
TAG=simple_autoserl_$(date +%Y%m%d_%H%M%S)
nohup bash scripts/autoserl_baseline/launch_simple.sh "$TAG" \
  > "$ACOSERL_OUTPUT_ROOT/launcher_logs/${TAG}_launcher.out" 2>&1 < /dev/null &
```

普通RLPD对照使用`--method rlpd`或`ACOSERL_SIMPLE_METHOD=rlpd`，其物理安全层相同，但不执行Guide/Recovery、不采集动态Demo。两种运行均不接受旧资格文件；也不需要`--grasp-guide`或`independent_window`参数。不要静默回退为普通RLPD后称作AutoSERL。

默认16worker、1024步预热、50万实际交互或10小时上限，25k有效步/3600秒保存、50k有效步评估20回合；沿用零稳定抓取停滞诊断，不因为Demo满容量继续掩盖零任务进度。无干预评估执行相同物理保护但没有任务纠偏。保存完整模型/优化器/Buffer/RNG，并可用`--checkpoint-mirror-root`指定额外存储。路径名称和同盘副本不证明跨实例持久。

## Terra仅测试和运行

禁止改正式源码、阈值、第三方依赖、Git和旧stash。新测试代码放在仓库外，例如`$BASE/testing/simple/`。不进行冒烟测试。

1. 先阅读本文件及新控制器。确认正式实例确实是`SimpleBimanualAutoSERL`，不是旧V2/V3链。
2. 从canonical初态真实回放`demo_0`，关闭任务干预但保持同一ActionOnly保护。使用新的Simple管理器，不使用旧`intervention_enabled=False`所绑定的名义接近门禁。不许中途写qpos/qvel或伪造抓取。
3. 用真实动作验证A/B各自Guide、P0返回、7维示范抓取、稳定确认和真实Actor交还；另测两臂同时干预，另一臂Policy维度必须原样保留。接触耦合/已完成阶段不应回退。
4. 验证逐臂mask、真实完整动作入Online/Demo、16路n-step隔离、checkpoint重载并继续更新。受控准备与自然探索分开报告。
5. 前述无阻断性物理/数据缺陷后，运行两个seed的16环境NPU有界验证；不要因旧Hold证书失效阻止本基线。
6. 验证通过后按已有授权启动新TAG正式训练，报告PID、16worker、NPU、真实步数、更新数、抓取/交还/装配数据和输出路径。正常训练不要因汇报而终止；遇到代码缺陷只留证据，由ChatGPT修复。

## 2026-11-02：B臂抓取后稳定保持的有界诊断调整

在提交 `a7acf2ccec7a68cf935931b5393f3116f48a8d05` 的受控MuJoCo专项中，A臂完成Guide、ReturnP0、Replay、Settle及安全Actor交还。B臂另一条真实路径已经抓住 `piece_2`，P1位置误差约0.424mm，但最后10步相对平移变化最大约0.733mm，未达到原有0.5mm阈值，原代码在Settle第30步停止。此结果尚不足以确定永久无法稳定，也不代表安全门禁错误。

新版本只将Simple控制器Settle的独立上限由30步扩至**80步**，保留350步整体Recovery上限、真实抓取条件、连续10步相对位姿稳定、0.5mm平移限制、旋转限制和闭爪检查；仍使用相同的P1 OSC控制，不改力/碰撞/工作空间保护、不做物理状态写入，也不把未稳定的抓取当作交还。调整的是时间预算，**不是放宽稳定性判据**。

`after_step()` 的 `recovery_result.settle_diagnostics` 新增每臂真实测量：已保持步数、80步上限、是否抓住目标、夹爪闭合、连续10步窗口长度、相对平移/旋转变化、P1位姿误差及ready标志。由采集器写入真实执行日志，可直接区分接触滑移、控制位置误差和窗口不足。该字段只记录诊断信息，不输入Actor。

若B臂在80步内真实稳定并完成安全Policy交还，才可判该路径PASS；若持续滑移或触发物理保护则仍报告FAIL，并进一步依据真实状态研究抓取控制，不继续盲目延长或降低阈值。此前另一条B路径因A Guide接近装配接触被物理保护拒绝，需作为**独立的双臂干预时序问题**单独记录，不能用B稳定性修复宣称已解决。

真实MuJoCo和Ascend验收尚须在服务器执行。此小修不恢复旧checkpoint，也不授权直接跳过未完成的干预验收。

## 本地审查范围

本地执行了Python语法检查、Bash语法检查、合成状态的控制契约（双臂独立/同时Guide、P0回放、抓取确认、交还、硬限制拒绝、终止逻辑）和内存Replay替身的逐臂掩码/真实n-step边界契约。它们不是MuJoCo或TorchRL/NPU集成验收。没有报告真实机器人成功率或服务器吞吐提升。
