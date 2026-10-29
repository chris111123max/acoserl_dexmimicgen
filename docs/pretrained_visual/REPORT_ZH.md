# 预训练视觉编码器替换与全新 RL 训练报告

本次已完成真实权重加载、CPU/NPU 数值检查、离线及多环境回归、A/B 完整图像恢复测试、16 环境 NPU 有界训练与 checkpoint 重载。验收通过后，从空在线 Replay 和全新 Actor/Critic/适配层开始正式训练；只加载预训练视觉主干。装配学习收益尚未得到验证。

## 1. 保护旧实验

旧进程通过 SIGTERM 正常结束，保存完整 final checkpoint：1610 个实际交互、572 次 Actor/Critic 更新、224 条离线与1610条在线经验，pending 为0。旧目录原样保留：
`outputs/ascend16/autoserl_two_point_20261028_163354/`。
归档索引、原配置/归一化/保存元数据及指向旧目录的链接位于 `outputs/pretrained_visual/archive/`，没有搬迁、覆盖模型或导入其 RL 权重/Replay。

开发分支为 `codex/pretrained-visual-encoder`。开始时已有183个跟踪文件删除及旧部署 gate 修改，均未恢复或纳入本次提交。没有使用 GitHub main 覆盖远端，也没有操作本地 E 盘源码。

## 2. 候选与实际选择

| 候选 | 实际证据 | 结论 |
|---|---|---|
| R3M ResNet18 | 官方 MIT 仓库提供 Google Drive 权重；原链接从服务器报 Network unreachable，官方备用下载域名请求超时 | 未下载到可验证权重，不冒充加载成功 |
| DINOv2 ViT-S/14，LVD142M 无 registers | 175个张量严格加载且逐项一致；126输入产生真实9×9×384 token | CPU/NPU相对L2误差9.54%，抗锯齿缩放还发生CPU回退；拒绝正式部署，具体数值差异尚未定位到单个算子 |
| ImageNet ResNet18，224输入layer3 | 完整官方权重严格加载；CPU/NPU相对误差0.0965% | 原缩放CPU回退导致1.745秒/更新；数值等价的原生NPU上采样将其降至0.764秒，仍较慢 |
| ImageNet SqueezeNet1.1 | 52个权重张量逐项一致，CPU/NPU相对误差0.0993% | 1.137秒/更新，参数更少但本NPU上没有更快 |
| **ImageNet ResNet18，原生84输入stem–layer2** | 完整权重加载和冻结证据、数值检查、真实更新与16环境测试均通过 | **选用：0.304秒/更新，保留原生空间输入** |

最终名称是 **ResNet18 IMAGENET1K_V1 的浅层空间特征提取器**，不是 R3M，不是完整224像素分类流水线。采用原生84分辨率属于明确的分辨率迁移选择；不声称其分类精度等于官方224设置，也不声称浅层特征已证明能提高装配成功率。

官方来源与许可：
- [R3M](https://github.com/facebookresearch/r3m)，MIT。
- [DINOv2](https://github.com/facebookresearch/dinov2)，标准DINOv2代码和权重为Apache-2.0；未使用Cell/XRay变体。
- [torchvision ResNet18](https://docs.pytorch.org/vision/0.18/models/generated/torchvision.models.resnet18.html)，使用已安装torchvision 0.18.1的ResNet结构。实现为BSD-3-Clause；预训练权重及ImageNet数据权利遵循上游条件，不能将代码许可等同于数据许可。
- [SqueezeNet](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.squeezenet1_1.html)。

选用权重版本：`IMAGENET1K_V1 / resnet18-f37072fd.pth`，46,830,571字节。
来源：[官方权重文件](https://download.pytorch.org/models/resnet18-f37072fd.pth)。
SHA-256：
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

102个公开张量逐项完全相同，`strict=True` 加载成功。旧checkpoint没有20个BatchNorm批次计数器，PyTorch兼容加载为零；它们不是缺失的学习权重。三个完整主干分别加载，同样的冻结初值、独立适配层；不是将不兼容权重加载进MiniViT。构造结束及训练后继续核对整个主干state hash，防止被重新初始化或被BatchNorm训练统计改变。

## 3. 输入、空间适配及冻结

输入仍为三个RGB相机的CHW84×84图像。原环境到Replay的HWC→CHW转换不交换RGB通道；robosuite的opencv约定只调整图像纵向方向。离线/在线均沿用已有图像入口。uint8仅除255一次，浮点按已有入口的[0,1]处理。原官方RandomShiftsAug(pad=4)顺序保持不变，再按ImageNet RGB均值[0.485,0.456,0.406]、标准差[0.229,0.224,0.225]归一化。

每路执行已预训练的conv1/bn1/relu/maxpool/layer1/layer2，得到(B,128,11,11)，局部空间平均到9×9，再按固定行列顺序转成81个128维token，通过独立可训练Linear(128,128)。三路仍拼为(B,243,128)，Actor/Critic接口不变。没有全局池化后重复特征。layer3/layer4/fc完整加载、验证和冻结，但不参与所选前向。

冻结主干总参数35,068,536，其中实际执行的前缀参数2,049,216；适配层共49,536参数。主干强制eval并在no_grad下执行，适配层保持可训练。真实Adam状态审计：Actor 5,200,270，Critic 21,786,634，适配层49,536，共27,036,440个参数已产生优化状态。requires_grad标志总数58,297,522包含官方target/未使用参数，不能当作实际梯度优化参数总数；未修改官方target逻辑。

CPU/NPU检查使用真实离线agentview样本，输出(2,81,128)，最大绝对差0.00134464，相对L2误差0.000877716。三相机的完整联合入口通过后续真实NPU训练验证。微基准三路16张编码约8.53ms；同设备MiniViT约4.82ms。

## 4. 回归与16环境实测

- 224条demo_0有效transition逐一检查：当前/未来观测、图像、实际缩放动作、3-step奖励、严格终止和bootstrap均通过；Replay保存重载逐项一致。原HDF5仅只读访问。
- 16环境、48个Episode、183条交错数值测试全部通过，涵盖终止、截断、Safe_Stop前切断、短尾、无虚假transition，以及官方单环境3-step数值对照。
- 单环境完整图像恢复：A测试220个真实控制步，恢复70步后稳定抓取piece_1并交还；B测试303步，恢复125步后稳定抓取piece_2并交还，A保持piece_1。最大归一化动作1.0，所有恢复通过实际动作执行。脚本使用可重复测试控制器，不代表RL已学会抓取。
- 16环境按4+4+4+4依次完成reset后，向四组同时发动作并统一收集。所有worker均CPU仿真/OSMesa，无torch_npu上下文，独立随机种子、Episode与AutoSERL状态。
- 有界运行 `pretrained_spatial84_validation_20261028_1`：140个真实环境步、140次Critic和140次Actor更新，batch256=在线128+离线128，10Q、n-step3、UTD1。在线140/离线224；没有部分物理动作。
- env00/04/08/12各8步、独立reset2次；其余各9步、reset1次。四次Safe_Stop均为压力测试显式collector cut，其余环境没有被重启。本段仅8–9步/worker，没有触发完整自动恢复；恢复功能由上述独立A/B测试验证。
- Actor14个、Critic15个、编码器适配层6个参数张量实际改变；三个适配层末次梯度范数约0.00475/0.00493/0.00484。loss和梯度范数均有限，全部主干张量和BatchNorm统计严格不变。
- 完整checkpoint含网络/target、优化器、Replay、归一化、RNG、收集器及worker监督状态。重载后参数逐项相同，并在NPU上再做一次真实梯度更新。MuJoCo只承诺新Episode边界恢复，不声称Episode中间物理状态位级续训。

| 指标 | 既有MiniViT有界运行 | 新预训练空间前缀有界运行 |
|---|---:|---:|
| 实际交互/Actor/Critic更新 | 140/140/140 | 140/140/140 |
| 有效交互步/秒（不含初始化） | 2.482 | 1.817 |
| 纯更新累计秒 | 20.887 | 43.206 |
| 总采样训练段秒 | 56.399 | 77.061 |
| Torch NPU分配峰值字节 | 2,253,062,656 | 1,599,198,208 |

新方案有效FPS约降低26.8%，没有宣称吞吐提升。网络固定离线batch微基准已另外比较；16环境采样时策略与轨迹不同，因此端到端FPS是部署测量，并非严格相同动作的配对因果结论。原生84方案已比224预训练候选明显省时，140次真实更新未见稳定性问题，故在明确成本后启动单组正式实验。没有为提高速度改UTD、batch、网络头或安全门禁。

新有界运行的驱动采样AICore峰值98%，HBM峰值5162 MB；Torch峰值不等于整个驱动HBM。训练段77.061秒、全流程333.951秒，初始化不算正式交互步。完整时间分解、全部16个worker及利用率原始采样见acceptance和运行目录，不能拿短时峰值当持续利用率。

设备沿用Ascend910B3，逻辑npu:0（物理编号2），约22核CPU配额。Python3.10.20、torch2.3.1、torch_npu2.3.1、torchvision0.18.1；CANN目录8.0.RC2，runtime内部Version=7.3.0.1.231，驱动/npu-smi26.1.1。没有改动旧.venv/.venv_resfit或重新安装CANN；具体环境记录在runtime_versions.json。

## 5. 新正式训练

训练源码提交：`5b9b4bd6ec8bdf4a9a46083c2b58ce01a9a01e97`。
输出目录：`/data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen/outputs/ascend16/pretrained_visual_20261028_172019/`。
启动状态：as of 2026-10-28T17:30:19.057143+08:00; 16 workers initialized; actual_interactions=1245, critic_updates=207, actor_updates=207, FPS=3.897。

初始化审计确认：{"fresh_rl": true, "fresh_online_replay": true, "online_rows": 0, "offline_rows": 224}。正式命令没有--resume；在线Replay从0开始，不加载压力测试或旧RL模型。已加载的视觉主干固定，新的Actor/Critic/适配层从随机初始化开始。

训练仍为单组ResFiT RLPD+双臂AutoSERL、16环境同时采样、910B3集中推理和更新，最大500000真实交互或10小时先到即停。预热1024真实步，保持25000步checkpoint、50000步关闭干预评估20个Episode，评估只额外启动一个worker并暂停训练采样。预热/探索/恢复真实步均计入训练，评估另计。Safe_Stop只结束对应Episode，原14维ActionScaler、奖励与严格成功条件均保持。

尚无新策略的无干预评估成功率；必须等到正式评估，不能用demo回放成功或局部恢复替代策略装配成功率。有限压力测试不证明10小时长期稳定，需继续观察正式日志。冻结浅层预训练特征是否优于MiniViT也是本次实验待检验的问题。

正式launcher使用nohup+setsid及共享flock，退出SSH后保留运行，不会并行启动另一套正式NPU任务，也没有结束时清理完整运行目录的逻辑。

## 6. 复现与证据

所有操作在远端项目运行，先source scripts/ascend16/activate.sh：
1. `python scripts/pretrained_visual/prepare.py`：官方来源、固定版本、SHA检查；不覆盖已有不同来源代码。
2. `python scripts/pretrained_visual/benchmark.py resnet18_spatial84`：单独占用NPU时进行数值/更新/冻结验证。不要与正式训练并行运行此测试。
3. test_offline.py / test_nstep.py：224离线与隔离3-step验证。输出目录已存在时应使用新的测试目录，不能覆盖已验收记录。
4. `scripts/autoserl_two_point/bridge_probe.py --arm A或B --tag 唯一新标签`：真实恢复图像入口。
5. train.py --mode stress --tag 唯一新标签 --utd 1 --stress-steps 128 --visual resnet18_spatial84；checkpoint_probe.py及make_gate.py核验结果后才开放新训练gate。
6. `bash scripts/pretrained_visual/launch_formal.sh`：只在没有现有正式任务且完成验收时启动新训练。
7. `python scripts/pretrained_visual/status.py`：只读查询当前正式状态。

主证据：docs/pretrained_visual/acceptance.json、benchmark_*.json、checkpoint_acceptance.json、preservation.json及parameter_counts.json。完整运行、逐步动作/干预日志、NPU原始采样、Replay和checkpoint保存在outputs，启动/测试stdout保存在logs/pretrained_visual。原始算法文件hash与AutoSERL/配置hash均保持一致。Current bounded status snapshot: docs/pretrained_visual/formal_status.json.
