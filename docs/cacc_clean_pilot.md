# 原生 CACC clean pilot 协议

本协议检查训练曲线并为攻击比较保留可追溯的 victim，不要求 clean 达到零碰撞，也不以固定比例的回报改善、间距误差下降作为淘汰条件。碰撞、回报和控制误差都是实验观测量；后续比较同一个 victim 在 clean 与攻击条件下的变化。

## 运行

从干净提交启动：

```powershell
.\.venv\Scripts\python.exe scripts/train_cacc_pilot.py
```

默认 Catch-up、MAPPO、训练 seed=1、CPU 单线程计算、4 个采样 worker、1,000 步 rollout、100,000 个环境步。网络、优化器和奖励设置继承上游对应 MAPPO JSON；使用已单独提交的训练修复。将 `--scenario` 设为 `slowdown` 可运行另一任务。

在 0、20,000、40,000、60,000、80,000、100,000 步保存模型和验证。验证固定 seeds 1,000,000–1,000,019，使用单 worker、确定性动作；评估前后恢复进程 RNG，避免改变训练抽样序列。验证 seed 范围与训练 worker/回合 seed 范围分离。后续正式测试使用另一个预先固定的 seed 集合。

训练预算与评估间隔必须是完整 rollout 批次，源码必须已提交；这些是保证实际执行与记录相符的技术约束，不是学习效果门槛。

## 收敛分析与模型选择

- 保留训练回报曲线以及每个验证点的均值、回合间标准差、碰撞发生数、间距/速度 RMSE、各车辆动作计数。
- 默认使用最后一个 checkpoint，并同时记录验证平均回报最高的 checkpoint，不按碰撞数过滤候选。可显式用 `--checkpoint-selection best-validation-return` 选择后者；规则写入 manifest。
- 保存相对初始模型的绝对回报变化，避免初始回报接近零时百分比失真。后半段曲线的斜率和范围仅供观察，不自动输出“通过/淘汰”。
- 根据后期回报是否形成稳定平台、波动和不同训练 seed 的表现判断收敛程度。回报稳定不自动代表策略优秀或安全；短预算未形成平台时继续增大训练预算，不直接进入已收敛的正式结论。
- 训练只完成设定的预算，脚本不会承诺有限步数必然收敛。`convergence_status=requires_curve_review` 表示需要结合曲线分析，而不是实验失败。

所有模型、逐回合结果和曲线数据保存在被忽略的 `artifacts/cacc_clean_pilot/`。每个 run 包含 `manifest.json`、`config.json`、`environment.ini`、`pilot_validation.jsonl`、`pilot_summary.json`、最终 `models/` 和各时刻的 `slice/<step>/`。summary 同时保存所选模型、最终模型和最高验证回报模型的哈希。当前 checkpoint 是评估权重，不包含可无损续训所需的全部优化器和 RNG 状态；增加预算时须明确新训练运行，不能伪装成连续续训。

配对攻击评估支持 `--checkpoint slice/<step>`，默认仍加载 `models/`。两者均关联训练 run 的 manifest 和该组模型的 SHA-256。示例使用独立于模型选择的 pilot 评估初态：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_cacc.py --victim-dir '<training-run-directory>' --checkpoint slice/100000 --eval-seeds 1100000 1100001 1100002 1100003 --attacks clean zero gaussian igs --epsilon 0.05 --iterations 10 --purpose pilot
```

这些 pilot 初态属于方法开发数据，不能再称为未使用过的正式测试集。测试时固定 checkpoint 和 seed，允许 clean 已有碰撞；报告攻击前后碰撞率的百分点变化和 `J_clean - J_attack`，不预设攻击必须成功。
