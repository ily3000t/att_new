# 阶段 1：clean pilot 与攻击重放记录

日期：2026-09-11。本轮完成 Catch-up 的 100,000 步 clean pilot、独立 checkout 重放和首组配对攻击评估。完整的多 seed AMB baseline 尚未完成。

后续调整：仓库所有者要求先按原 JSON 完成 10,000,000 步、20 worker、500 轮 Catch-up MAPPO 训练。后台启动与产物入口见 [完整预算训练记录](cacc_full_training.md)。此前 pilot 的 25 轮仅占完整配置环境步数的 1%、更新轮数的 5%，不用于认定完整训练已经收敛。

## 模型选择原则

按仓库所有者的最新要求，已去掉零碰撞、回报至少提升 10%、间距误差必须下降等硬性效果门槛，并写入 [AGENTS.md](../AGENTS.md)。允许 clean 存在碰撞或其他危险行为；固定 victim 后比较攻击带来的变化。

默认保存并选择最后一个训练 checkpoint，同时索引验证平均回报最高的 checkpoint。碰撞数不参与候选过滤。保留回报曲线、验证回合间标准差、后期趋势和控制行为用于分析收敛；脚本不自动判定模型通过或淘汰。协议与命令见 [cacc_clean_pilot.md](cacc_clean_pilot.md)。

## 代码与实验版本

本轮训练和攻击实际使用的干净代码提交：

`453053b3aaedbf2b71c5c6e828ef6964174da515`

该次可重放 pilot 的实验版本名为 `exp/cacc-clean-pilot-v1`，tag 指向上述实际执行提交。范围为同机、同依赖环境、一个训练 seed 的原生 CACC pilot；不代表论文统计复现、跨机器复现或全局最优策略。

| 原子提交 | 内容 |
| --- | --- |
| `474c5ca` | 默认关闭 CACC 跨回合累积的旧内存日志 |
| `f04d075` | 保存最后一次 on-policy 更新的模型 |
| `c9d6b51` | bootstrap 读取最后一次转移之后的状态 |
| `a46fc40` | 修复非默认回报分支的时间维递推和单位转换 |
| `978e120` | 记录间距/速度 RMSE 和动作计数 |
| `ebf636a` | 明确取消 clean 的安全及百分比硬门槛 |
| `4daa296` | 固定验证初态、隔离评估 RNG、保存曲线与模型索引 |
| `453053b` | 支持评估指定 snapshot，并关联原训练来源 |

修复对学习结果的影响见 [cacc_changes.md](cacc_changes.md)。原始 OVM 动力学、奖励尺度和有限回合终止行为保留。

## 验证证据及结论边界

- 完整自动化测试集 54 项通过；随后 snapshot 来源扩展的针对性测试通过，并用真实 snapshot 完成两次端到端攻击评估。
- 在当前工作目录和独立 detached checkout 各完成一次 100,000 步训练。两次均为训练 seed=1、4 个采样 worker、CPU 单线程计算。
- 两个运行的全部 70 个 checkpoint 文件（6 组 snapshot 加最终 models）SHA-256 和张量内容一致；最终配置除输出目录外一致，INI 哈希一致。
- 两次运行的 6 个验证点、每点 20 个回合记录逐字节相同，共执行 240 个验证回合。相同 seed 的重放不增加独立训练 seed 数。
- 6 万、8 万、10 万步的验证回报和控制误差形成平台。但策略主要维持原速，仍保留追赶误差；训练回报继续波动。可观察到验证行为平台，尚不能据此宣称一般性的训练收敛或控制性能优秀。
- 最终模型的验证回报低于初始模型，仍按约定保留和使用最终 checkpoint，没有因改善幅度或碰撞条件淘汰它。
- 在独立于模型选择的四个 pilot 初态上，执行 clean、zero、Gaussian、IGS，并在独立 checkout 重放，共 32 个攻击协议评估回合。四种条件的逐回合记录在两次运行间完全一致；zero 与 clean 一致，snapshot 来源和幅度约束核对通过。
- 当前 epsilon=0.05、IGS 10 次迭代、仅攻击 agent 0 的观测。非零扰动被施加，但该 agent 在所测回合一直选择动作 2，回报、碰撞率均未改变。此设置尚未观察到攻击效果，不能据此推广为模型对其他预算、对象或攻击方法鲁棒。

没有使用 SUMO 执行这些实验。本机已有 SUMO 的后续适配任务继续独立管理。

## 本地产物索引

原始权重、逐回合数据、曲线和运行日志均在被 Git 忽略的 `artifacts/`。下列索引保存查找位置和校验和；两次运行的实际目录、run ID、manifest 哈希及模型来源可从索引继续追溯。

| 内容 | 相对仓库路径 | SHA-256 |
| --- | --- | --- |
| 两次训练与全部 snapshot 重放核对 | `artifacts/cacc_pilot_validation/independent_checkout_replay.json` | `3b79deb6d9f791827b38f9b4df5aa5bc1f99fa1c4e7a8e067bed13be82a91367` |
| 两次攻击协议重放核对 | `artifacts/cacc_pilot_validation/attack_checkout_replay.json` | `2b856170def6180680c09ffcf504cd5127f9ad8b5ba13f9e58016a604788f895` |
| 首轮配对攻击指标与来源索引 | `artifacts/cacc_pilot_validation/paired_attack_report.json` | `155ddb45f9640c86b4c55a7ac0854a88fe5eece134cfc2e88b3d507e932ff785` |
| 学习曲线 | `artifacts/cacc_pilot_validation/learning_curve.png` | `50911279b2cbd72add1bc3665f93d228b62189d103497a9cbeda8fa410047894` |

独立 checkout 保留在 `artifacts/worktrees/cacc-pilot-replay/`。它固定在实验提交，不跟随主工作目录后续修改；默认输出也在该 checkout 自己的 `artifacts/` 下。

## 下一步

1. 优先完成原配置 Catch-up 训练：seed=2、10,000,000 步、20 worker、500 轮。运行期间保存训练回报与损失，等待用户在训练后发起分析，不自动扩展攻击或其他任务。
2. 根据完整训练曲线和最终 checkpoint 讨论收敛及后续评估；不设置零碰撞或固定改善比例门槛。当前 snapshot 不含完整续训状态，此次从头训练。
3. 原计划中的多 seed、Slow-down、攻击预算 / 作用对象对照及 Ours 方法开发，留待完整训练结果分析后安排。
