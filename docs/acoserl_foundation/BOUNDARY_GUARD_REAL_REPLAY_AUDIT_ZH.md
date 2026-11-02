# BoundaryGuard真实RGB Replay与Dynamic Demo专项验收

## 背景

指定旧基线SHA：65870666ad87b1f84063fa04d16e495677732319。此前真实MuJoCo物理专项已从canonical初态重放16组Policy提议，首次Guard之前末端轨迹误差小于3.61e-9m，26次A=BoundaryGuard/B=Guide均完成，0次被子步预演拒绝。仅有物理完成不代表Online Replay/Dynamic Demo的数据门禁通过。

旧BoundaryGuard测试脚本的fake_row缺少action_source，产生33 PASS / 1 ERROR；导入GuideProgressTests还导致重复发现旧阶段测试。本次修复后期望独立BoundaryGuard 13项，独立阶段恢复27项。

## 数据链路审计方法

新增CPU-only真实MuJoCo审计脚本：scripts/autoserl_baseline/audit_boundary_guard_real_replay.py。

该脚本不是冒烟测试，不更新Actor/Critic，不使用Torch-NPU，不修改训练代码或checkpoint；复用正式SimpleWorker、SimpleBridge、ActionOnly、make_row、IsolatedNStep、Online Replay和SimpleDemo真实方法，采集真正三路84x84 RGB观测及18维机器人状态，并用旧训练envXX日志中录制的Policy提议驱动。

逐个env在canonical Episode0开始运行，首次BoundaryGuard之前用上一轮日志的末端位置做真实误差比较。Guard之后不声称与旧状态配对，因为物理轨迹已经分叉；只评估真实已完成的动作与Replay数据是否一致。

## Terra命令

~~~bash
cd /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
source scripts/ascend16/activate.sh
export PYTHONDONTWRITEBYTECODE=1
python scripts/autoserl_baseline/test_simple_progress_regressions.py
python scripts/autoserl_baseline/test_boundary_guard_regressions.py

RUN="/data/home/3220251075/lerobot_workspace/training_runs/acoserl_dexmimicgen/simple/integration/stage_serl10_npu512_20261102_183537"
BASE="/data/home/3220251075/lerobot_workspace/training_runs/acoserl_dexmimicgen/testing/boundary_guard_v2"
TAG="real_replay_$(date +%Y%m%d_%H%M%S)"
python -u scripts/autoserl_baseline/audit_boundary_guard_real_replay.py \
  --reference-run "$RUN" \
  --output "$BASE/$TAG" \
  --env-ids $(seq 0 15) \
  --max-proposals 64
~~~

审计脚本在单进程中依次启动至多16个真正的MuJoCo Worker，绝不在Episode中途恢复qpos/qvel。运行目标是数据完整性验收，不是大规模NPU训练；不加载ResNet权重。其输出目录必须为新的仓库外目录，不得覆盖之前证据。

## 验收目标

1. 录制Policy提议为14维归一化动作，来自原完整训练Worker日志；每个旧worker的随机种子保持一致。
2. 前缀与原始末端记录误差不超过1e-5m，不符立即失败。
3. 完整控制周期获取生产SimpleWorker返回的三路uint8 RGB和18维状态，真实执行归一化动作经scaler逆变换与控制器物理动作吻合，绝对误差不超过2e-6。
4. 按实际生产路径创建row，经过n=3 IsolatedNStep后真正加入Online Replay。
5. 每条持久化Replay的起点RGB与真实执行前RGB逐像素相同；n-step next RGB必须等于最后一个已完成真实控制周期后的RGB，禁止部分物理动作的中断状态替代next_obs。
6. Replay的(env_id, episode, frame)不得重复、缺失或与执行日志不一致；所有真实完成的Guard动作必须出现在Online Replay，其来源为Constrained_A/B/Joint；Guard所拥有的六维位姿不能被标为Policy或专家控制。
7. 真实SimpleDemo.observe处理A Guard+B Guide混合控制。Guard行不能进入Dynamic Demo；要求至少发生一次真实未认证候选清除，不能只用构造数据单测证明。
8. 不完整动作只允许截断之前已经完成的n-step尾部，不得生成自身Transition。
9. 三路视觉编码器、UTD=1、10-Q和80N/碰撞/工作空间/原始substep preview均不修改。

## 结果

输出新目录中的audit_result.json、attempted_provenance.json和workers/envXX.jsonl原始记录，保留stdout和stderr。

仅当独立27+13项回归全部通过、真实Guard动作完成、Online Replay逐项入库且源/图像/n-step校验通过、真实B Guide候选被拒绝、Dynamic Demo无Guard行时，才能标记BOUNDARY_GUARD_DATA_READY。

不能将数据链路PASS当成真实16环境NPU再集成PASS，更不表示边界长期风险解除。新控制协议仍需要后续独立16环境集成和checkpoint续更后才考虑READY_FOR_FORMAL；本阶段不得启动正式训练。

任何无法真实复现的情况标记NOT REPRODUCED，任何源码问题都由ChatGPT修改；Terra只执行和报告，不修改生产.py/.json、不commit、不push。
