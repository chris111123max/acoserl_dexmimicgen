# 双臂 AutoSERL：统一执行、安全、数据与验收协议

## 版本范围与事实边界

本轮以 `0a44b7546a0e229d9607240493044909a2a3a9ae` 为父版本，处理已经明确的 Guide / Recovery 控制权、周期内越界、交还、动态示范、资格复验和测试输出问题。

本文件不是物理 PASS 证明。编辑环境没有 MuJoCo、robosuite、TorchRL 和 Ascend 运行环境；本地完成了 Python AST 检查、数值不变量测试和控制权契约检查。真实双臂控制、图像观测、NPU 更新和端到端速度必须由服务器验收。

已报告的 20261032 越界日志不在公开仓库，不能声称已对它逐字段复现。本轮修复了代码能够直接确认的缺口，并提供真实子步证据和复验入口。

## 唯一生产调用链

公开入口仍为 `acoserl.bimanual_autoserl.BimanualAutoSERL`。

`bimanual_core.py` 是父提交中 `bimanual_autoserl.py` 的字节级保留副本。它保留参考轨迹、P0/P1、控制器参数、任务事件和原有物理边界。新公开入口叠加 `ReviewedRuntimeMixin`，所有旧 Collector 仍导入同一生产入口；没有启用仓库里未接线的 `GuidePolicyMixin`。

Actor 仍是单个14维联合策略，包含两臂各6维位姿和1维夹爪。三路RGB84、本体18维、ResFiT 10Q、UTD=1、单条demo_0和稀疏严格成功奖励保持。

## AutoSERL闭环

官方参考：
- https://github.com/autoserl/AutoSERL/blob/main/serl_robot_infra/franka_env/envs/wrappers.py
- https://github.com/autoserl/AutoSERL/blob/main/examples/train_rlpd.py

保留滑动窗口、方向检查、停滞检测、固定P0/P1恢复和交还。普通Guide仅拥有偏离臂的6维位姿动作，夹爪及另一臂仍属于Policy。只有该臂自身动作不安全时，安全层才可以修正对应动作并记录SafetyProjection来源。Recovery继续要求合格Hold；未知接触恢复没有解锁。

Guide完成后的交还条件是：达到目标，进入Handoff_Ready，真实执行一个通过安全检查的Actor动作，再记录原takeover_id的交还。未执行、部分执行、被另一臂接管均不能算交还。完成后允许新Guide，不再强制等待20到40个随机Actor动作。

无干预评估不执行Guide、Recovery、自动夹爪保持或安全投影；危险Actor动作被拒绝并作为失败记录。

## 工作空间与物理子步

`osc_prediction.py` 不再只检查50ms周期末端。它计算名义临界阻尼轨迹的坐标极值，投影同时约束原始OSC目标、周期内轨迹、输入幅度和接近方向。允许在原空间内有限制动运动，不要求运动中的机器人瞬时反向或瞬时静止。求解失败时拒绝，不把全零动作当成成功。

名义模型不是实机或非线性接触安全证明。

`ActionOnly` 为下一物理子步建立独立MuJoCo数据副本，使用即将实际执行的ctrl进行预演。预测到工作空间超限或力阈值超限则不执行该真实子步。预测只写入独立数据，不覆盖真实qpos/qvel/act，不推进真实时间，不生成Replay。实际子步完成后刷新派生运动学，并对所有控制模式执行相同的真实边界和力检查。

MuJoCo依据： https://mujoco.readthedocs.io/en/stable/programming/simulation.html

副本预演和真实监控均保留；不能将预测PASS当作物理验收。80N仍是停止阈值，不承诺所有接触力观测绝无瞬时超调。自定义全局控制/被动力回调会导致预演拒绝，不静默执行两次回调。

每次动作只在内存保留一个控制周期的子步轨迹。异常包含真实已执行子步数、时间、末端位置/速度、力、接触和拒绝预演。第一次真实子步之前拒绝，不计为“部分物理执行”；已有真实子步后拒绝，才计部分执行。两者都不伪造完整transition。

## 两个逻辑Buffer

Online保存所有完整真实transition。Demo由原始224条和动态干预组成。RLPD仍使用完整14维动作学习；专家掩码是来源元数据，不是BC目标，也没有悄悄改变Actor损失为逐臂掩码损失。

`official`模式接纳完整、合法且未被安全修正冒充的Guide/Recovery步骤，不再要求每一个物理控制步都单调降低TCP误差。惯性、姿态纠偏或稳定保持可能不满足逐步单调，但仍是实际干预。误差变化继续记录用于分析。

`safe_handback`模式仅提交有实际安全交还证据的连续片段。缺失片段不拼接n-step；纯SafetyProjection只进入Online。动态协议号更新，旧动态Replay不能无提示导入。

Batch仍为128 Online + 128 Demo。原始/动态96/32和正奖励配额2属于本项目的示范保护选项，不是声称官方AutoSERL原样使用该内部分配。

## P0资格：测试不再必须编辑生产JSON

旧资格的源码指纹失效仍然失效；本轮没有把其passed或哈希改成新版本。

新增 `acoserl.qualification.StaticHoldTrial`：外部测试程序先用真实动作到达原已记录的静态P0状态，然后进入有限长度复验上下文。它只临时允许在同一旧位姿/抓取/速度域测试当前代码，保留2mm、3mm、姿态、接触、80N和全部子步门禁，不产生生产授权。

```python
from acoserl.qualification import StaticHoldTrial, issue_static_hold_certificate

# 真实准备结束后；manager和runner均为当前生产类。
with StaticHoldTrial(manager, active=0, output=external_receipt_path) as trial:
    # 使用既有公开eligible/start_recovery/choose接口；不改物理状态。
    # 每个真实控制后：
    trial.observe(runner, decision, after)

# A、B各至少三个当前源码下的完整真实复验收据通过后：
issue_static_hold_certificate(receipt_paths, external_certificate_path)
```

收据由执行轨迹计算结果，不允许测试者手填PASS。证书绑定全部源码、配置和ResFiT依赖状态，只保留真正覆盖的旧状态域。证书在仓库外testing目录生成；生产程序通过 `--static-hold-certificate` 读取。测试失败只生成失败收据，不能签发证书。

测试脚本不得用这个接口授权未知接触、扩大状态域、运行Learner或忽略安全失败。普通自然探索测试不依赖P0复验证书，可单独执行Guide与数据检查。

## 训练与效率

统一入口仍为 `scripts/ascend16/train.py`，实际编排在 `reviewed_learner.py`。继续复用原FourByFour、QAgent、Replay、n-step和checkpoint函数，不更换RL算法。

新增 `--output-root`；integration默认写入仓库外testing/runs。输出目录必须全新。固定16环境、两个测试seed、至多2048有效步；不支持人工P0准备冒充自然探索。正式训练保存和评估周期按有效真实步计数。

提高效率的具体措施：同一choose内部重复几何风险查询缓存；原始正奖励索引缓存；没有渲染副本环境；不每步重新构建模型；训练日志按采集轮聚合，保留worker实际动作日志。子步预演有额外物理开销，必须报告preview耗时和端到端有效步/秒，不能声称本轮必然更快。

一次成功的梯度更新仍不等于学会任务。早停、真实交互、更新、动态经验和阶段进度分别记录。预热、测试准备、部分子步和预测计算不得充当新的训练transition。

## Codex验收任务（仅测试，不修源码）

所有测试脚本、日志、模型和证书放在 `/data/home/3220251075/testing/`。禁止修改正式源码、配置或Git；禁止恢复旧stash；第三方既有修改保持并记录。不要运行冒烟测试。

1. 同版本真实A/B/双臂边界复验，核对周期内预测、私有子步预演和实际状态。重复20261032第16个动作，分别报告预测拒绝和真实越界，不混淆。
2. Guide_A+Policy_B及对称测试；保留非引导臂与两夹爪正常动作。复现20261033接触与夹爪反转，预期仍保留未经资格的接触拒绝，不能为了零停止删除门禁。
3. A/B目标到达、真实Actor交还与抢占复验；核对takeover_id、逐臂自主统计、一次物理控制只计一次干预步。
4. 通过StaticHoldTrial完成原静态P0域复验；失败不修改生产资格，也不阻断其他独立测试。
5. 用真实完整数据验证Online、Dynamic、专家掩码、n-step、128/128采样、保存加载续更。预测数据不能入库。
6. 物理安全条件允许后，运行20261032/20261033各一组16环境NPU有界验收，生成完整中文报告。不要把构造数据契约检查当作MuJoCo或NPU验收。

命令（已激活现有Ascend环境）：

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/ascend16/train.py --mode integration --tag reviewed_seed20261032 \
  --seed 20261032 --integration-steps 1024 --integration-seconds 1800 \
  --output-root /data/home/3220251075/testing/runs --visual resnet18_spatial84
```

有当前源码的外部P0证书时再添加 `--static-hold-certificate /absolute/testing/qualification.json`。未提供证书不等于允许旧证书；P0仍拒绝，Guide可独立测试。

## 正式启动门禁

Codex根据真实证据在testing目录生成 `reviewed_baseline_acceptance_v1` 报告。字段包括 `source_sha256=source_manifest()`、`dependency=dependency_manifest()`、`formal_training_approved` 和 `checks`。每项必须是 `{status: PASS/FAIL/NOT_RUN, path: 外部实际结果文件, sha256: 文件摘要}`。

必需项：workspace_projection、guide_A_ownership、guide_B_ownership、guide_A_handback、guide_B_handback、static_hold_requalified、buffer_integrity、nstep_boundaries、checkpoint_continue、npu_seed_20261032、npu_seed_20261033、strict_demo_replay。

任何物理/数据关键项未通过时不得填写批准。`require_gate`核对实际证据文件摘要和当前完整代码/依赖指纹。测试报告和证书均在仓库外，不需要改生产JSON。

正式启动通过显式设置 `ACOSERL_ACCEPTANCE_FILE` 和 `ACOSERL_HOLD_CERTIFICATE` 后调用现有 `scripts/autoserl_baseline/launch_formal.sh`。本次代码提交本身没有启动训练，也没有签发物理证书。

## 本地自审结果

- 修改Python文件的AST全部可解析。
- 200组解析轨迹极值与高密度数值采样一致。
- 120组旋转坐标系/速度/输入投影契约：119组返回可行动作、1组正确拒绝；返回动作未改其他维度，名义轨迹在既定边界内。
- 两个边界测试保留切向运动；极端制动不可行测试拒绝。
- 控制权契约覆盖原takeover归属、单次交还、Guide+投影计步不重复、非引导臂投影不计自主、非单调合法干预接纳、投影/失败排除。
- 这些是软件契约结果，不是机器人或NPU的PASS。所有物理和训练结论以服务器新证据为准。
