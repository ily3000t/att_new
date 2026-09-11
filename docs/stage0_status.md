# 阶段 0 验收与阶段 1 入口

更新日期：2026-09-11。阶段 0 的 CACC 工程修复和最小训练/攻击评估流程已完成；阶段 1 的正式 baseline 尚未完成。当前结果属于 **smoke**，不发布实验 tag。

## 本轮实现

从 `main` 的 `8ddceaa` 创建 `fix/cacc-reproducibility`，按以下原子提交开发。修复后的学习结果与导入版本分开标识，具体语义差异见 [cacc_changes.md](cacc_changes.md)。

| 提交 | 内容 |
| --- | --- |
| `1653a8c` | CACC 最小直接依赖配置；SUMO 导入延迟到 ATSC 路径 |
| `41854a3` | seed 传递到 CACC 内核；独立环境 RNG；不隐式消耗 reset |
| `2d819a1` | 修复离散观测攻击的 R_pi 记录崩溃 |
| `5bd5992` | 逐回合原始团队回报、间距、碰撞和 TTC 代理统计 |
| `72ccca9` | Git/config/INI/seed/runtime/checkpoint 来源记录与运行状态 |
| `cc9b7bf` | 保存模型的 clean、零预算、裁剪高斯噪声、原始 IGS 配对评估 |

本轮未改变原始奖励、OVM 动力学、动作定义或碰撞后的终止逻辑，也未实现 Ours。

## 已执行的验收

全部正式提交后的流程验证使用代码提交：

`cc9b7bf46b0bd591ecc3a01a3904d6163c5cbf58`

- 20 项自动化回归测试通过，覆盖无 SUMO 导入、随机性与 worker 隔离、真实 PPO 离散攻击回合、指标边界、实验来源记录。
- Catch-up 和 Slow-down 各执行两次独立进程的 1,200 步 MAPPO 训练；相同 seed=1、CPU 单线程、一个采样 worker、600 步采样片段，原始网络和优化器配置保留。
- 每个场景两次训练的 10 个 checkpoint 文件，SHA-256 和加载后的张量内容全部一致；最终配置哈希一致。
- 对四个训练产物分别评估 clean、zero、gaussian、IGS，每种条件使用环境 seeds `[500, 501]`，共 32 个完整评估回合。
- 攻击 seed 为 `[1, 2]`；只攻击 agent 0 的全部观测特征，每步可攻击；归一化 L-infinity epsilon=0.05，IGS 10 次迭代，其他动作替换概率为零。
- 两次重复评估的逐回合结果记录完全一致；zero 与 clean 的回报和安全指标一致；非零攻击确实生成非零扰动且满足幅度约束（浮点容差 1e-6）。
- 4 份训练和 16 份条件评估 manifest 均为 completed，均记录干净源码、配置、环境 INI 和 checkpoint 来源；评估加载的 checkpoint 哈希与对应训练输出一致。

这是同一台机器、同一依赖环境的重放验收，没有验证另一台机器或全新安装环境，也没有保存并逐项比较所有原始观测轨迹。32 个回合包括同 seed 重放，不能当作 32 个独立统计样本。

当前短训练模型下，随机噪声和 IGS 的回报及所记录安全指标与 clean 相同。Slow-down 的 clean 回合本身已发生间距阈值事件；Catch-up 的 TTC 代理无有效闭合样本。因此本轮不能证明攻击有效、策略收敛或策略安全，下一步应通过较长训练曲线分析 clean victim 的收敛情况。Clean 的碰撞不构成淘汰条件。

## 本地产物索引

权重和原始结果全部保存在被 Git 忽略的 `artifacts/`。下列索引随源码保存，数据本身不提交。复制产物到其他机器后，可用 `Get-FileHash -Algorithm SHA256 <path>` 校验；目录名不是 run ID，真正的 run ID 在 manifest 内。

| 产物 | 相对仓库路径 | SHA-256 |
| --- | --- | --- |
| 训练重放核对及四个训练 run ID | `artifacts/cacc_validation/training_replay.json` | `05e18e825253549a72cef32c4184d2483810c1d7a2c21fd2cefea55d55230525` |
| 评估重放核对及条件 manifest 索引 | `artifacts/cacc_validation/evaluation_replay.json` | `a8e22dc94bb5ea86db9edcd3ef4ac848a4c2bf7a29400efa7d47040ede6a4f40` |
| Catch-up 第一次配对评估 | `artifacts/cacc_evaluation_verified/20260911T020252-68233275/summary.json` | `0ab2a2a758faa15bb6e049def9ca646f623fbf98600c02955973540c2d1c9634` |
| Catch-up 第二次配对评估 | `artifacts/cacc_evaluation_verified/20260911T020421-f9fc1804/summary.json` | `ed5f938f830e784da624dac5b56e9913b4cf80186a690c2de70a1001620a49ea` |
| Slow-down 第一次配对评估 | `artifacts/cacc_evaluation_verified/20260911T020252-cec62345/summary.json` | `e7bdac582d86d59e92fc59b5c80f6c57096045a2cf79eb60311a8b3f8a624c8d` |
| Slow-down 第二次配对评估 | `artifacts/cacc_evaluation_verified/20260911T020313-9fce0a31/summary.json` | `6443ce8445a173c503e65e9806a21c2584e1cfe992d977ea83f749e27419d889` |

精确训练启动命令、平台、包版本和配置已写入各 manifest；可复用命令见 [cacc_setup.md](cacc_setup.md)。较早的 `artifacts/cacc_stage0/` 训练来自 `72ccca9`，`artifacts/cacc_evaluation/` 中存在标为 exploratory 的开发评估，不与上述验收批次混用。

## 阶段 1 的下一批原子任务

1. 核对较长训练的内存和保存行为：原始 CACCWrapper 开启的内存日志会跨回合积累；原始 runner 在保存间隔写模型，短预算未到间隔可能没有 checkpoint。以独立修改处理，验证不改变转移或奖励。
2. 单独审计 rollout 末尾 bootstrap、horizon 截断和 bad mask。当前仍保留上游逻辑；需要用明确数值案例决定是否修复，不能把本轮 smoke 当作训练算法正确性的完整证明。
3. 从 Catch-up 的一个训练 seed 做较长 clean pilot，记录训练曲线、动作分布、间距/速度跟踪情况，并在独立验证 seeds 上选择 checkpoint。训练步数是预算，不能用达到某个步数代替收敛判断。
4. 记录 clean victim 的训练曲线和收敛证据，固定 checkpoint 运行同预算 Gaussian/IGS；允许 clean 存在碰撞，比较回报和碰撞率的变化，不使用零碰撞或固定回报提升百分比作为硬门槛。记录动作改变率、攻击计算量和失败案例，再审计 targeted-observation、动作攻击与 MAD 路径。
5. 主结果采用至少 3 个独立训练 seed，预先固定独立的训练/验证/测试划分，先以每个 checkpoint 约 100 个配对测试回合估计方差，再调整数量。smoke 的 seeds 500/501 不作为未见测试集。
6. 完成干净 checkout 的正式重跑、主要 baseline 与来源核对后，才创建 annotated 实验 tag，随后进入 Ours 的威胁模型、接口、风险目标和消融开发。

原生 OVM-CACC、可选 SUMO-CACC 桥接、SUMO Ego PPO 继续分别管理。本机 SUMO 1.22.0 已确认，复用环境的 TraCI/sumolib 为 1.25.0；本轮未启动 SUMO 仿真，客户端/服务端兼容性测试留到 SUMO 阶段，不将此视为阻塞原生 CACC 的条件。
