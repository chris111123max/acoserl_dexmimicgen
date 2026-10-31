# ACo-SERL 正式训练分级授权（2026-10-31）

本版本把**正式训练资格**与**静态 P0 / P1 Recovery 控制权**分开。适用场景是
DexMimicGen / robosuite / MuJoCo 双 Panda 仿真，**不授权实体机器人**。

| `--intervention-method` | 可用控制 | 需要当前版本静态 Hold 证书 | 动态示范 |
|---|---|---|---|
| `plain_rlpd` | 14D Policy + 保守安全检查 / 投影 | 否 | 关闭 |
| `independent_window`（正式启动默认） | Policy + 独立的 Guide_A / Guide_B | 否 | 合格 Guide 数据 |
| `full_acoserl` | 前述能力 + 有资格证书覆盖的 Recovery | **是**，还须完整验收文件 | 合格 Guide / Recovery 数据 |

## 不受资格切换影响的硬约束

所有模式保留真实 MuJoCo 子步监测、独立预演（可用时）、工作空间边界、
80N 停止阈值、接触限制、实际动作记录和失败关闭。未取得资格不能执行
静态被动臂 Hold、P0/P1 长距离 Recovery、carry-local 或 joint Recovery。
仅关闭非必要的 Recovery 干预触发，不会把越界动作自动视为安全。
Guide 仍必须通过原有方向、接触、非引导臂 Policy 和完整 14D 动作检查。
拒绝安全动作会造成训练回合失败，不代表异常被忽略。

**此分级授权不是MuJoCo物理验收结果。** 当前 Guide / Policy 是否能持续安全
采样，仍由真实环境运行、子步日志和监控判定；完整 Recovery 必须取得新的
源码及依赖绑定的外部资格证书，不能继承旧版 PASS。

## 在已激活的 Ascend 环境内启动

```bash
cd /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
source scripts/ascend16/activate.sh
python -u scripts/ascend16/train.py \
  --mode formal \
  --intervention-method independent_window \
  --tag guide_formal_001 \
  --output-root /data/home/3220251075/testing/runs/formal
```

也可以使用包装脚本：

```bash
ACOSERL_INTERVENTION_METHOD=independent_window bash scripts/autoserl_baseline/launch_formal.sh
```

独立基线：设置 `ACOSERL_INTERVENTION_METHOD=plain_rlpd`。
如需完整 Recovery：

```bash
export ACOSERL_INTERVENTION_METHOD=full_acoserl
export ACOSERL_ACCEPTANCE_FILE=/绝对路径/当前源码验收.json
export ACOSERL_HOLD_CERTIFICATE=/绝对路径/当前源码静态保持证书.json
bash scripts/autoserl_baseline/launch_formal.sh
```

完整 Recovery 模式仍校验源代码、依赖指纹、真实证据文件哈希和两臂
静态保持资格；缺失、过期或伪造证据会拒绝启动。

## 运行数据

正式训练参数仍来自 `configs/ascend16/formal.json`，
包括16个 MuJoCo worker、Ascend NPU 10Q RLPD、UTD=1、
128 Online + 128 Demo、25k 保存和50k评估周期。新建输出均要求
仓库外目录；包装脚本将运行文件写到
`/data/home/3220251075/testing/runs/formal`，将日志写到
`/data/home/3220251075/testing/logs/autoserl_baseline`。

运行时配置和 `summary.json` 显式记录
`training_tier`、`recovery_authorized`、
`acceptance_verified`，防止把无Recovery资格的训练误报为
完整 AutoSERL 认证训练。

变更仅为代码路径授权分层；尚未在实际 MuJoCo / NPU 上完成本次版本的
运行与性能验收。
