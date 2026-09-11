# CACC 修复与原始实现的差异

对照起点：源码导入提交 `fd763573be39087f5292616e8dac080090fea4cf`。以下修复后的结果使用独立版本记录，不能声称与原始实现逐位一致。

## 随机性

- `NetworkEnv.seed(s)` 现在把 s 写入 CACC 内核，并且不再偷偷执行一次 reset；首次显式 reset 使用 s，之后使用 s+1、s+2。
- CACC 使用自己的 `RandomState`，保持原有单个 seed 的 NumPy 抽样算法，同时不再修改进程全局 NumPy RNG。worker 间和环境/攻击器之间不共享该 RNG。
- 每个动作空间也显式设 seed，实际 episode seed 在内核 `episode_seed` 中记录。
- 两个任务在相同内核 seed、固定动作下的 600 步轨迹和原始团队奖励，与导入版本逐步比较一致。
- 修复会改变此前失效的训练/评估 seed 分配，并消除环境重置对策略/攻击器随机性的影响，因此旧版训练产物与修复后训练产物不能混合标注为同一个实验版本。

## 离散观测攻击评估

`R_pi_values` 现在对离散和连续动作都记录分布差异，修复离散 CACC 的空列表索引错误。IGS 目标函数、幅度、作用对象和环境奖励未改变。回归测试执行真实 PPO runner 的完整短回合，检查零预算回报与配对 clean 相同，并检查单智能体攻击时其他智能体的分布未被修改。

## 安全指标

`NetworkEnv.step()` 在各 agent 的 `info['cacc']` 中透传原始间距、速度、前车速度、加速度、实际 episode seed 和碰撞状态。评估 logger 在每个完整 episode 结束时写入 `cacc_eval_XXXX.jsonl`，保存团队回报、最小间距、最小 TTC 代理、有效 TTC 样本数、低于阈值的比例和首次碰撞时间。

TTC 代理仅在间距为正且后车更快时使用 `gap/closing_speed`，其余写为 JSON null；阈值默认 1 秒，可用 `env.ttc_threshold_s` 配置。暴露比例分母为有效 TTC 的 agent-step 数。碰撞后的冻结状态不增加安全样本，原始重复惩罚仍完整计入团队回报。团队回报取外层各 agent 重复奖励的均值，避免放大 8 倍。

指标是只读附加信息。原始团队奖励尺度、600 步 horizon、碰撞后的 batch 边界终止行为均保留。当前碰撞仍是 OVM 的间距阈值事件。

## 实验来源

CACC run 现在自动记录完整 Git SHA、dirty 状态、最终配置和展开 INI 的哈希、进程与 worker seed 规则、运行平台、全部已安装包版本及输入 checkpoint 的 SHA-256。如果 checkpoint 所在训练目录有 manifest，同时关联其训练 commit/config/seed。运行完成后记录输出权重哈希；失败记录错误类型和状态。

探索运行保存 tracked 差异与 untracked 文件快照。正式启动脚本应调用 `require_clean_source()`；仅存在 manifest 不自动代表正式实验或论文复现。run 输出目录时间戳增加微秒，避免相同 seed 在同一秒启动时覆盖。

若进程被强制终止，或 runner 构造期间报错，manifest 可能保留 `running`；这表示没有完成记录，不能当作成功。metadata 不保存 RNG 中间状态，当前 checkpoint 用于评估，不是可无损续训的训练状态快照。
