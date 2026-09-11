# Catch-up MAPPO 原始配置完整预算训练

2026-09-11：按仓库所有者要求，先完成 Catch-up 原始配置的完整训练，完成后由用户发起结果分析，再决定后续实验。本页为启动记录；实时状态以运行日志、进程和 `manifest.json` 为准。

## 预算与配置

直接使用 `experiment/settings/network/catchup/mappo.json`，其解析内容与原始源码导入提交一致。未使用 `train_cacc_pilot.py`，未加载 pilot checkpoint。

| 配置项 | 本次设置 |
| --- | --- |
| `num_env_steps` | 10,000,000 |
| `episode_length` | 1,000 |
| `n_rollout_threads` | 20 |
| rollout / update 轮数 | 10,000,000 / (1,000 × 20) = 500 |
| `ppo_epoch` / `critic_epoch` | 5 / 5 |
| `seed` | 2（原 JSON 值） |
| `cuda` / `torch_threads` | true / 4 |
| `hidden_sizes` / `share_param` | [128, 128] / false |
| `use_eval` | false（保留原配置） |
| `log_interval` | 5 轮，即每 100,000 环境步记录训练指标 |
| `eval_interval` | 25 轮；即使关闭评估，仍按原 runner 每 500,000 步保存模型 |
| `model_dir` | null，从头训练 |

500 指外层 rollout 和优化轮数，每轮内部还有 PPO / critic 的 epoch，不能把它当成 500 次优化器梯度更新。此前 100,000 步、4 worker 的 pilot 共 25 轮，只占这次环境步数的 1%、外层更新轮数的 5%；其短期验证平台不构成完整训练的收敛结论。

实际解析配置仅将 `log_dir` 改为独立产物目录，并指定实验名和 `purpose=full-budget-training`。原入口自动补充的 `traitor_eps=0`、`perturbation_eps=0`、`adv_all=false` 均未启用攻击。其余原有训练参数和全部环境参数逐项核对一致。

训练期按原配置记录训练回报及损失，不额外插入验证回合或攻击评估。仍不设置零碰撞或回报改善比例门槛；训练结束也不自动宣称收敛。

## 代码与后台进程

- 执行提交：`1d9f8eae8d49e9378fd90d8548d805f7125cd3b4`，干净的独立 detached checkout。
- checkout：`artifacts/worktrees/cacc-catchup-full-20260911/`。
- 这次使用原配置和当前修复版实现；随机性、bootstrap 等修复见 [cacc_changes.md](cacc_changes.md)，不称为未修改上游实现的逐字复现。
- 本机 Python：`.venv/Scripts/python.exe`；Torch 2.5.1、CUDA 12.4、RTX 4070 Laptop GPU。实际 GPU 与 4 个 Torch 线程已从运行 manifest 核对。
- 启动时间：2026-09-11 20:21:14（Asia/Shanghai）；后台 launcher PID：`47476`。PID 可能被系统复用，需要结合启动时间、日志和 manifest 判断状态。
- run ID：`4e32d93700cb467b8dc1b786584adcda`。
- 隐藏后台进程通过 Windows `Start-Process -WindowStyle Hidden` 启动，不依赖当前交互终端。电脑需要持续运行，睡眠会暂停计算。
- 启动验证已观察到首个训练日志点 `100000/10000000`，即完成 5/500 轮；标准输出持续写入，未发现异常回溯。此项仅证明任务已正常进入训练，不代表最终结果。
- 训练完成前不创建新的实验复现 tag；本次启动不等于训练成功或结果复现。

## 本地产物入口

任务目录为 `artifacts/background/cacc-catchup-full-20260911T202114/`，全部被 Git 忽略。

| 文件 | 用途 |
| --- | --- |
| `launch.json` | PID、启动时间、完整命令、执行目录、代码版本 |
| `config_audit.json` | 原 JSON 与实际配置的逐项核对、配置 / INI 哈希及 run ID |
| `console.stdout.log` | 实时标准输出；每 5 轮出现总步数、FPS 和训练回报 |
| `console.stderr.log` | 警告和异常 |
| `runs/network/catchup/single/mappo/stage1-catchup-original-10m/seed-00002-2026-09-11-20-21-16-699307/` | 本次唯一训练目录 |

训练目录保存 `config.json`、`environment.ini`、`manifest.json`、`stdout.log`、TensorBoard `logs/` 及 `models/`。正常结束时 manifest 更新为 `completed` 并写入最终权重 SHA-256；若进程被强制终止，其 manifest 可能仍为 `running`，必须结合进程和错误日志核实。现有模型文件不包含完整续训状态。

执行 checkout 中原 JSON 的 SHA-256 为 `6d7d24f994e0cfc92cda95789a2c52538fd46f4af6b5b00a207cfa29b0d75afc`。最终解析配置 SHA-256 为 `ff22a0702767687336bb17fa0a355ee5407ec4582af99294f03a2387b7c6be00`。环境 INI SHA-256 为 `28c87f95901867c9626e85d55cff3f43823bf563670d6e71384eb9fe945225bc`。

查看当前输出：

```powershell
Get-Content 'E:\adv_marl_benchmark-main\artifacts\background\cacc-catchup-full-20260911T202114\console.stdout.log' -Tail 15
```

重现实验时，在固定提交的干净 checkout 中运行以下等价训练命令，并使用新的输出目录：

```powershell
& 'E:\adv_marl_benchmark-main\.venv\Scripts\python.exe' -u single_train.py --load_config experiment/settings/network/catchup/mappo.json --exp_name stage1-catchup-original-10m --algo.log_dir artifacts/cacc_full_replay --purpose full-budget-training
```

下一次分析先核实 10,000,000 步和最终保存是否完成，再读取整段训练曲线与模型。Slow-down、多 seed 扩展和后续攻击由这次结果分析之后再安排。
