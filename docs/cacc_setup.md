# 原生 CACC 开发环境

本环境用于 AMB 原生 OVM-CACC 与 MAPPO，采用现代依赖兼容配置，不宣称逐项还原论文作者的 Linux 环境。原始 `requirements.txt`、`environment.yml` 保留用于来源对照。

推荐使用 Python 3.10.16。直接依赖固定在 `requirements/cacc.txt`，测试依赖在 `requirements/cacc-dev.txt`。这些是直接依赖约束，不是所有平台通用的完整 wheel 锁文件；每次实验还需记录完整包版本和硬件信息。

在仓库根目录创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/cacc-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

`python` 应指向已安装的 Python 3.10。CPU/GPU 运行使用对应 PyTorch wheel，并在实验记录中保存实际 Torch、CUDA/cuDNN 版本。Linux 使用 `.venv/bin/python`。

本机开发验证为了复用已有 GPU PyTorch，实际采用 `E:\Programs\EnvAnaconda3\envs\pytorch\python.exe -m venv --system-site-packages .venv`，只在项目 `.venv` 中安装缺失的 setproctitle。该方式会继承现有 Conda 包，属于本机复用方案，不能称为全新环境安装验证；正式复现宜在独立环境按配置安装。

CACC 本身不需要 SUMO。环境分发已将 SUMO 相关导入延迟到 `grid/net` 分支，测试通过主动阻止 `traci/sumolib/libsumo` 导入来验证两个 CACC 任务仍能 reset/step。ATSC 仍须另装其依赖。

执行 `python -m pytest` 仅收集新增的 `tests/`。上游根目录 `test.py` 是交互式 StarCraft 演示，不作为 CACC 自动化测试。

## 最小训练与配对评估

以下是流程验证配置，不能作为收敛的 baseline 或论文结果。原始模型网络、优化器、奖励尺度保留，采样线程缩为 1，采样片段为 600 步，训练 1,200 步，使用 CPU 单线程以检查本机重放。

```powershell
.\.venv\Scripts\python.exe -c "from amb.utils.run_manifest import require_clean_source; print(require_clean_source())"
.\.venv\Scripts\python.exe single_train.py --load_config experiment/settings/network/catchup/mappo.json --exp_name stage0-catchup --algo.num_env_steps 1200 --algo.episode_length 600 --algo.n_rollout_threads 1 --algo.n_eval_rollout_threads 1 --algo.use_eval False --algo.cuda False --algo.torch_threads 1 --algo.eval_interval 1 --algo.log_interval 1 --algo.log_dir artifacts/cacc_stage0 --algo.seed 1 --purpose smoke
```

训练日志会给出实际输出目录。将下面的占位目录替换为该目录（应包含 `config.json`、`manifest.json` 和 `models/`）：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_cacc.py --victim-dir '<training-run-directory>' --eval-seeds 500 501 --attacks clean zero gaussian igs --epsilon 0.05 --iterations 10 --purpose smoke
```

Slow-down 使用 `experiment/settings/network/slowdown/mappo.json`，其他流程相同。评估器默认要求源码已提交且干净；`--allow-dirty` 仅用于有快照的探索运行。

评估器使用原始 PPO/IGS runner，每种条件和每个 episode 显式重设相同环境 seed，并按 `attack_seed + seed_index` 重设攻击随机数。`gaussian` 为裁剪高斯观测噪声，`igs` 为原 IGS 非定向交叉熵攻击，`zero` 为零扰动校验。默认只攻击 agent 0 的观测，所有时间步允许攻击；不调用额外动作替换/MAD 路径。

输出 `summary.json`、各条件的 manifest 和逐回合安全指标，并验证零预算与 clean 完全一致、扰动幅度不超出 epsilon。`applied_agent_steps` 是获得攻击机会的选中 agent-step 数（零预算时也可大于零），`candidate_calls` 包括 runner 为未选中 agent 计算但不施加的候选扰动；不将候选计算次数冒充实际修改次数。epsilon 的单位为归一化观测 L-infinity。

## 已有 SUMO 安装

2026-09-11 已确认本机 `sumo --version` 为 **1.22.0**；所复用 Python 环境的 TraCI、sumolib 均为 **1.25.0**。本轮原生 CACC 不启动 SUMO，未修改该安装。后续 SUMO 驾驶阶段先验证这组客户端/服务端组合，必要时在单独环境匹配版本。
