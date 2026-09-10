# AMB → 原生 CACC 攻击研究 → SUMO 驾驶：代码核查与路线建议

日期：2026-09-10。核查起点为本地源码导入 commit `fd763573be39087f5292616e8dac080090fea4cf`。本轮未修改训练、环境或攻击源码，未训练 victim 或新攻击器。

## 结论

逐步验证路线合理：先建立可信 baseline，再检验新攻击机制，最后评估跨任务适用性。但阶段 1 应改为 **AMB 原生 CACC + 原始攻击 baseline**。本地两个 CACC 场景使用 NumPy OVM 跟驰动力学；不是 SUMO 执行的车辆控制。将其接入 SUMO 是另一个环境迁移任务。

论文正式题名及最终评测范围见 [源码来源记录](source_provenance.md)。你的“合作车辆控制鲁棒性”定位比“自动驾驶决策 benchmark”准确；现阶段还应去掉 CACC 的 “SUMO-based” 前缀。ATSC Monaco 在介绍中出现，但被排除在实际评测外。[论文附录 A.2.3](https://arxiv.org/html/2510.11824v2)

建议路线：

```text
阶段 0：来源建档、依赖隔离、运行与指标审计
  → 阶段 1：AMB 原生 OVM-CACC 的 clean + 攻击 baseline
  → 阶段 2：同预算下验证 Ours 的攻击机制与消融
  → 可选桥接：在 SUMO 中重建 CACC，检验仿真器迁移
  → 阶段 3：SUMO + Ego PPO，先一个驾驶场景，再扩展
```

如果最终论文的中心是 MARL 协作失效，可把阶段 2 作为独立成果目标；阶段 3 是外部有效性验证。如果中心是自动驾驶决策安全，阶段 3 的正式驾驶任务应作为主要证据，CACC 作为方法开发与机理验证。

## 代码的实际运行路径

| 功能 | 主要位置 | 已核实行为 |
| --- | --- | --- |
| 标准训练与攻击入口 | [single_train.py](../single_train.py)、[runner 分发](../amb/runners/__init__.py) | 选择 `single`、`perturbation`、`traitor`，加载最终配置和 victim |
| 环境工厂 | [env_utils.py](../amb/utils/env_utils.py) | 构建训练、评估环境，并调用 seed |
| 交通环境分发 | [network_env.py](../amb/envs/network/network_env.py) | `grid/net` 进入 ATSC；`catchup/slowdown` 进入 CACCWrapper |
| CACC 接口 | [CACC.py](../amb/envs/network/envs/CACC.py) | 包装 CACCEnv，局部奖励缩放，转换动作形状 |
| CACC 动力学 | [cacc_env.py](../amb/envs/network/envs/cacc_env.py) | NumPy 更新速度、间距、加速度；四个动作选择 OVM 的 alpha/beta 参数 |
| SUMO 路径 | [atsc_env.py](../amb/envs/network/envs/atsc_env.py) | 启动 SUMO 子进程并通过 TraCI 连接 |
| 观测攻击 | [igs.py](../amb/algorithms/igs.py)、[perturbation runner](../amb/runners/perturbation/base_runner.py) | 随机扰动、投影梯度符号攻击；学习型攻击通过目标动作引导 IGS |
| 动作/内鬼攻击 | [traitor runner](../amb/runners/traitor/base_runner.py) | adversary 动作替换，另有 victim 的 worst_perform 逻辑 |
| 脚本生成 | [generate.py](../experiment/generate.py) | 组合攻击别名、训练预算及评估参数 |
| 实验存档 | [config_utils.py](../amb/utils/config_utils.py) | 当前 save_config 保存配置 JSON，没有完整 Git/checkpoint 来源记录 |

默认 [network.yaml](../amb/configs/envs_cfgs/network.yaml) 是 `net`，不是 CACC。阶段 1 要显式使用 `experiment/settings/network/catchup/mappo.json` 或 `slowdown/mappo.json`，并核查其对应 INI。

本地两份 CACC INI 均为 `n_vehicle=8`，实际产生 8 个受控智能体，另有 `v0s` 给出的外部前导车轨迹。每个智能体观测为 5 维、动作数为 4；控制间隔 0.1 秒，环境 horizon 为 600 步。动作是 `(alpha,beta)` 四种组合，不是换道、超车或合流动作。

论文对 Slow-down 描述为 7 个智能体跟随前车，本地配置和实际接口为 8 个受控智能体，属于需记录的版本/描述差异。JSON 中 `episode_length=1000` 是 runner 的采样片段长度，而 INI 控制环境 600 步结束；两者不相等本身不证明错误，但终止 mask、截断与统计要验证。

## 已验证问题与待审计风险

| 优先级 | 发现 | 证据和影响 | 建议处理 |
| --- | --- | --- | --- |
| P0 | 环境 seed 未传入 CACC 内核 | `NetworkEnv.seed()` 给 wrapper 写 `seed`；调用 `seed(12345)` 后 wrapper 为 12345，内核从 INI 的 12 经 reset 变为 13 | 单独修复 seed 传递与 RNG；验证同 seed 可重放、不同 seed 不同、并行 worker 独立 |
| P0 | 离散观测攻击评估崩溃 | perturbation runner 的 `R_pi_values.append()` 仅在连续动作分支内；CACC 离散分支随后索引空列表。用真实 PPOAgent 执行原代码块已复现 `IndexError` | 修复日志分支并增加离散攻击完整回合回归验证 |
| P0 | 环境安全指标没有透传 | `NetworkEnv.step()` 返回空 `infos`，logger 只有任务名定制 | 增加原始间距/速度、最小间距、碰撞标记、TTC 代理等只读统计；验证不改变转移和奖励 |
| P1 | CACC reward 缩放被外层绕过 | wrapper 除以 100 并裁剪，但 NetworkEnv 复制原始 global_reward 给每个智能体。实测原始和外层均为 -171.5323，wrapper 缩放为 -1.7153 | 先按当前行为固定复现口径；如改变尺度，作为改变学习问题的独立实验，不静默改动 |
| P1 | 论文攻击名称不等于完整代码语义 | IGS 的离散 loss 是对目标动作的交叉熵；`PPOAgent.mad_perform()` 是另一条实现路径，由额外 flag 和概率控制 | 逐一记录实际调用、目标函数、随机初始化及预算，不将所有路径统称 KL-PGD |
| P1 | 动作攻击的对象和概率需重新核对 | traitor runner 会对所有 victim 循环调用概率性 worst_perform，然后对指定 agents 做 adversary scatter；`pert_act` 的生成 flag 与该 runner 使用的 flag 也不完全对应 | 用实际动作变化日志核对单体/全体作用域、替换概率和未攻击对象；原名称不能自动保证公平对比 |
| P1 | 恢复实验按回合计数切换 | `execute_recover = eval_episode >= recover_episode`，回合结束后环境重置 | 这不能单独证明同一车队状态遭受冲击后的物理恢复；新增同一 episode 内停止攻击的协议 |
| P1 | baseline 预算可能与论文叙述不一致 | generator 给 `pert_obs_sin` 显式 `adv_eps=0.1`，all 路径依赖默认配置；traitor YAML 的默认扰动幅度为 0.2 | 展开最终命令/配置，对照论文核实；原设置复现与重新统一预算的比较分别报告 |
| P1 | 训练环境尚未配齐 | 原依赖文件含 Linux build 和绝对路径；当前 Python 环境版本也不同 | 为 CACC 建立独立最小依赖与锁定版本，不把整份环境导出当成可移植安装说明 |

P0/P1 表示建议处理顺序，不表示所有项都已完成端到端验证。尤其动作 baseline、MAD 和恢复路径还需要独立审计。当前 `share_obs=obs`，也不应未经核对就宣称 critic 已获得完整车队全局状态。

## 本轮运行检查的边界

1. 在 base Python 3.12.7 中直接加载 CACCEnv，Catch-up 和 Slow-down 分别以固定动作 3 跑完 600 步，每个任务重复两次；观测和奖励均有限、重复轨迹哈希相同。
2. 在已有 `pytorch` 环境中导入并构建 NetworkEnv，完成 reset/step，确认 `(8,5)` 观测、`(8,)` 奖励/终止数组及空 info。
3. 复现 wrapper seed 未传递；用真实 PPOAgent 执行原始 R_pi 统计块，复现离散路径 IndexError；核对奖励尺度。
4. 该环境为 Python 3.10.16、Torch 2.5.1、Gym 0.26.1、NumPy 2.2.6、TraCI/Sumolib 1.25.0；已检查的环境缺少 tensorboardX、setproctitle、nni。系统 SUMO 路径为 1.22.0 安装目录，未在本轮启动 SUMO。

这不是训练或攻击实验结果。没有运行完整 MAPPO 训练，没有完成原攻击评估，没有复现论文统计指标，也没有生成实验发布 tag。

## 阶段 1：建立可信 baseline

先用 MAPPO 聚焦 Catch-up，再扩展到 Slow-down。MADDPG/HAPPO 作为后续受害策略泛化测试，不必一开始遍历四类环境和所有超参数。

先完成依赖锁定、上述 P0 项及 baseline 语义核对，再进行：

- **Clean**：训练或获得来源清楚的 victim；固定 checkpoint，记录训练步数和选择标准。
- **观测 baseline**：clipped Gaussian/random noise、原始 IGS、学习型 targeted-observation attack。学习型攻击不是无需成本的纯梯度 baseline，应记录训练交互预算。
- **动作 baseline**：random、greedy、learned，分别固定单智能体和全体作用域。先确认实际执行逻辑与命名一致。
- **零预算验证**：同 checkpoint、同环境初态、同动作确定性下，零攻击的轨迹和回报应与 clean 一致；额外 RNG 消耗不能改变环境初态。

建议先用 1 个训练 seed 做端到端冒烟验证；可运行后用至少 3 个独立训练 seed，主结果尽量 5 个。每个 checkpoint 先做约 100 个配对评估 episode，最终数量依据方差和碰撞稀有程度调整。不同训练 seed 才是独立训练重复，不能把同一策略的很多 episode 当成很多次独立训练。

正式比较使用同一 victim checkpoint、同一测试初态、相同可访问状态/梯度、相同修改对象与幅度、相同时间预算。报告实际预算消耗及不确定性；碰撞稀有时给出发生数/总回合数及置信区间，不能用零次碰撞直接宣称安全。

阶段通过条件：另一干净 checkout 能按记录重跑 clean 与主要 baseline；攻击确实落到指定对象；零预算一致；回报和安全指标都可追溯；不同 seed 的差异得到验证。此时才发布 `exp/cacc-mappo-baseline-v1`。

## 阶段 2：Ours 的建议研究问题

一个值得验证的假设是：**在相同观测修改预算下，针对未来安全裕度及扰动传播选择攻击目标和时机，比只增大即时动作分布差异造成更大的控制损失。**

这仍是研究假设，不是已经成立的创新。时机选择已有 strategically-timed attacks；选择脆弱智能体也已有专门的 VAI 研究，需要比较方法假设、权限、预算和机制，而不能只改 loss 名称。[IJCAI 2017 原文](https://www.ijcai.org/proceedings/2017/0525.pdf)、[VAI 原文](https://arxiv.org/abs/2509.15103)

建议优先固定为**测试时白盒观测攻击**：victim 参数冻结，攻击器修改允许的观测特征；可读取的信息、是否可查询环境模型、是否有 critic 都要写清楚。动作替换和环境动力学修改作为不同威胁模型单独评估。

从简单到复杂分解方法：

1. 固定受攻击车辆与时间表，比较原 IGS 与安全风险 surrogate，先证明目标函数有用。
2. 固定相同攻击步数，比较均匀/随机时机、策略置信度时机、安全风险时机。
3. 固定相同车辆数量，再比较固定车辆、随机车辆、考虑车队扰动传播的车辆选择。
4. 消融风险预测时域、传播项、时间平滑约束和各组成项，报告计算成本。

预算至少区分：每维幅度 epsilon、每步受攻击车辆数 k、每回合攻击步数或总 agent-step 数 B、修改特征、相邻步修改变化量、模型查询/梯度迭代预算。全体攻击与单体攻击不能只用相同 epsilon 就声称相同成本。

CACC 的 5 个观测量经过归一化，并有物理相关性。例如 `v_state` 除以目标速度 15，`h_state` 除以目标间距 20；同一归一化 epsilon 对不同特征对应不同物理尺度。若声称传感器或通信攻击，要在原始量上定义允许修改，并重新计算有关联的特征；任意修改所有归一化分量只支持抽象观测攻击的结论。

实现 Risk Loss 时，NumPy 动力学、离散动作和硬碰撞事件没有可直接沿用的梯度链。可先使用枚举离散动作的风险估计与策略动作概率构造可微期望损失，或训练可验证的短时风险预测器；真实仿真仅用于评估/标签，预测器使用的信息和查询预算也计入比较。单独添加 TTC 项而没有梯度来源和预算定义，不是完整方法。

建议新增薄接口适配现有路径：`BaseAttack`、`AttackRegistry`、`IGSAdapter`、风险攻击实现；环境指标和实验 manifest 单独模块化，每一步独立 commit。先保持原 IGS 行为可重放，再切换目标函数，避免接口重构与攻击收益混在一起。

## CACC 指标及统计口径

- **回报**：报告原始 team return 和明确的聚合方式；外层把 team reward 复制给各 agent，不可再无意求和放大 8 倍。回报往往为负，优先报告 `J_clean - J_attack`；相对值使用明确分母并说明近零处理。
- **碰撞/安全边界**：内核以 `min(hs_cur) < headway_min` 标记 collision，当前阈值为 1 米；这是模型中的间距阈值事件，不能直接当成 SUMO 车辆几何碰撞。报告每回合是否发生、首次发生时刻和最小间距。
- **TTC 代理**：仅在后车更快且间距有效时，使用 `h / (v_follower - v_leader)`；标注 h 的语义。到模型安全边界的时间 `(h-h_min)/closing_speed` 应另命名，不与 TTC 混用。没有闭合趋势时记为无穷/不适用，不记为零。
- **控制质量**：间距和速度跟踪误差、急减速事件、加速度/jerk、车队速度扰动沿车辆序列的放大。传播比值的分母接近零时注明不适用。
- **韧性**：同一 episode 内攻击一段时间后撤去，报告恢复到容差带并持续一定时间所需时长、恢复窗口累计损失、未恢复比例；碰撞或结束导致无法恢复的 episode 计入失败。

碰撞后内核保留状态并重复惩罚，直到满足 batch 边界或 horizon 才结束。首次事件之后不能把冻结状态产生的重复值当作新的独立安全事件。TTC 报告有效样本数、低分位数/每回合最小值及低于阈值的暴露比例，避免直接平均含无穷的数列。

## 阶段 3：真正的 SUMO Ego 驾驶验证

该阶段同时改变动力学、任务、观测、动作空间和多智能体结构，需要独立配置、训练 victim 和重新实现环境适配。迁移的主要对象是攻击机制和接口；新环境重新训练攻击器属于方法迁移，只有冻结攻击器直接使用并实测成功才可称为权重零样本迁移。

建议先完成一个小型 SUMO Highway 跟驰场景的 Ego PPO 控制闭环，再选一个 Merge 或 Lane Change 任务作为正式攻击场景；通过后扩展另外的场景。训练完成前先验证 PPO 输出确实影响 SUMO 执行动作，clean 成功率和安全表现足够稳定。仓库已有的 MetaDrive 路径和常见的 highway-env 都不能据名称视为 SUMO 环境。

CACC 若攻击多个受害 agent，而阶段 3 只攻击 Ego 的输入，威胁模型会变化。可迁移的核心宜放在安全风险与时机选择；车辆选择作为 MARL 扩展分别报告。如果多智能体传播本身才是主要贡献，则应增加真正的 SUMO 多车合作驾驶实验，才能保留该贡献的适用结构。

SUMO 会执行跟驰/换道安全约束；`speedMode`、`laneChangeMode`、控制间隔、碰撞判定阈值与碰撞后的 teleport/remove 行为都会影响结论。对 clean、baseline、Ours 固定并记录相同设置，分别记录策略请求动作和仿真实际执行动作；不能只对 Ours 改变安全配置来得到更多碰撞。[SUMO Safety](https://sumo.dlr.de/docs/Simulation/Safety.html)、[TraCI 车辆控制](https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html)

指标至少包括 reward、任务完成/merge success、Ego collision rate、near-miss 暴露比例、min TTC、通行时间和舒适性。普通跟驰 TTC 公式不能直接覆盖合流/交叉冲突，应使用明确的冲突几何或 SUMO SSM 定义，并处理 NA。[SUMO SSM Device](https://sumo.dlr.de/docs/Simulation/Output/SSM_Device.html)

“Reward 降低、Collision 增加、TTC 降低”是待检验的目标，三者不必同时变化。对手使 Ego 长时间停车，可能显著降低 reward，却不增加碰撞；因此主张“安全攻击”需要独立安全指标支持，也要报告失败或无效的情况。

## 下一轮建议的原子任务

1. `chore(env): add reproducible CACC dependency specification`：建立可重复的最小运行环境。
2. `fix(cacc): propagate seeds to the dynamics environment`：修复 seed，并验证 worker 隔离与配对评估。
3. `fix(eval): record divergence for discrete observation attacks`：修复并完整运行离散攻击评估。
4. `feat(metrics): expose CACC safety telemetry`：只读指标及碰撞/终止口径验证。
5. `feat(experiments): record source and checkpoint manifests`：正式实验自动绑定代码和模型。
6. 逐项核对原始攻击的实际幅度、对象、概率和时序，再形成 clean + baseline 的稳定实验版本。
7. 稳定版本通过后，按接口、风险目标、时间选择、车辆选择和消融配置拆分 Ours 开发。

当前最有价值的下一步是完成阶段 0 与一个可信的 CACC/MAPPO baseline；先获得可解释的基准，再根据失效轨迹决定 Risk Loss 的具体形式。
