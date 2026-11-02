# SERL 官方预训练 ResNet-10 → PyTorch / Ascend NPU

## 目标

采用 SERL/HIL-SERL 官方 ImageNet-1K 预训练 ResNet-10，不开展 ResNet-18 与 ResNet-10 比较，也不重新预训练视觉主干。UTD=1、10-Q、16 CPU MuJoCo workers、14D Actor、RLPD在线/示范128:128 保持不变。

**网络适配而非原始SERL策略复现：**严格映射官方JAX/Flax全部四组残差块权重，保留7×7 stem、4组GroupNorm和JAX SAME卷积/池化。为保留双臂抓取的定位信息，在线取stage2的8×8×256中间特征，经训练的Linear(256,128)输出每相机64个空间token，最后stage3的权重虽然装载并校验，在线不执行。原JAX实际输入为128×128；本项目RGB84×84在NPU以bilinear放大到128×128，再采用官方ImageNet归一化。原RLPD RandomShiftsAug保持不变。

## 固定目录：Codex下载，GitHub不存储模型文件

服务器项目根目录：

/data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen

以下均位于被.gitignore忽略的固定目录：

cache/pretrained_visual/serl_resnet10_official/

- 官方原始文件：resnet10_params.pkl
- 转换后文件：serl_resnet10_torch.pt
- 转换证据：conversion_manifest.json

唯一指定官方URL：

https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl

上游发布页面：

https://github.com/rail-berkeley/serl/releases/tag/resnet10

官方资产大小21,677,915字节。官方发布元数据没有公开SHA256，因此Codex下载后的自算hash只能作复现证据，不能被误写为官方独立认证的摘要。禁止任意第三方镜像；pickle有代码执行风险，只能在明确确认官方来源后转换。

## Codex执行命令

在仓库根目录执行：

~~~bash
cd /data/home/3220251075/lerobot_workspace/acoserl_dexmimicgen
git rev-parse HEAD
bash scripts/autoserl_baseline/download_serl_resnet10.sh

source scripts/ascend16/activate.sh
F="cache/pretrained_visual/serl_resnet10_official/resnet10_params.pkl"
SHA="$(sha256sum "$F" | awk '{print $1}')"
python scripts/autoserl_baseline/prepare_serl_resnet10.py \
  --expected-sha256 "$SHA" --trust-official-pickle
~~~

下载脚本优先使用aria2c最多4连接分段续传，缺少aria2c则curl重试和续传。源文件大小不符合预期时拒绝转换；不覆盖现有完整原始/转换文件，不删除历史产物。正式训练进程不需要也不会联网下载模型。

## 加载与训练

默认视觉选项变为serl_resnet10_spatial84。模型文件缺失时在初始化16 worker前直接报错；训练使用torch.load(weights_only=True)加载转换权重，确认所有官方参数键、严格形状与源文件SHA一致。主干冻结，三路相机独立Linear适配层接受Critic梯度；Actor/Critic及10-Q照常更新。旧ResNet18训练模型不允许恢复到新视觉架构。

测试阶段先运行：

~~~bash
source scripts/ascend16/activate.sh
python scripts/autoserl_baseline/test_serl_resnet10_conversion.py
~~~

上述定向单元测试验证网络拓扑、转换映射、边界padding、冻结梯度，但不证明实际官方文件已转换或NPU可用。Codex转换真实权重后需要实际检查模型前向、所有适配层梯度有限、16环境NPU更新、Replay/n-step、checkpoint精确恢复和恢复后的一次实际梯度更新。不要进行与任务无关的冒烟测试。

## 加速训练的合理顺序

1. 优先提高真实Recovery完成率与连续Policy自主交互比例，减少无进展Guide；旧训练98%干预是数据效率问题。
2. 用现有learning_curve.jsonl中timing的inference/step/reset/sample_transfer/update真实耗时差分定位CPU物理采样还是NPU更新瓶颈。
3. 冻结SERL ResNet10主干并只执行到stage2，避免每次RL更新反向训练视觉主干；每相机64空间token减少Actor/Critic特征处理成本。
4. 如果真实更新确为瓶颈，后续才考虑三路相机共享冻结主干批量前向与优化NPU数据搬运；如果物理采样是瓶颈，再优化worker派发和重新初始化开销。不要提前更改基础采样语义。
5. 现阶段不引入未经验证的AMP、不移除随机平移增强、不缓存未增强图像特征冒充原算法训练、不修改reward或物理安全保护。

提交只含代码与配置，无官方下载权重，无真实训练结果。先由Codex验证，再决定100,000有效步正式训练。
