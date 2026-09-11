# 开发与实验工作方式

## 一个修改，一个提交

典型过程是 `main → 工作分支 → 原子修改 → 验证 → commit → 合并 main → 实验验证后 tag`。

```powershell
git switch main
git switch -c feat/risk-aware-attack
# 完成接口，并运行针对接口的必要检查
git add amb/attacks/base.py
git commit -m "feat(attacks): add attack interface"
# 完成注册器，并验证调用关系
git add amb/attacks/registry.py
git commit -m "feat(attacks): add attack registry"
# Risk Loss 与调度策略分别实现、验证和提交
```

以上是未来开发示例，当前尚未创建这些攻击源码文件。不要为了分 commit 破坏每一步的完整性，也不要把接口、算法、环境奖励改动和结果汇总打包成一次 `update code`。

推荐类型：`feat`、`fix`、`refactor`、`test`、`docs`、`chore`。scope 使用具体模块，如 `cacc`、`attacks`、`experiments`。

```powershell
# 工作分支的必要检查通过后
git switch main
git merge --no-ff feat/risk-aware-attack
```

实验版本用 annotated tag，例如 `exp/cacc-mappo-baseline-v1`。tag 注释记录实验协议、seed 集合、产物索引与验证结论。若实验跑在功能提交而非合并提交，tag 指向实际运行的 commit，并说明它与合并版本的源码关系。没有完成复现实验时不创建此类 tag。

## 一个实验，一份可追溯记录

每个 run 至少保存以下信息。当前原生 CACC 的 `save_config()` 已自动保存 `manifest.json`、最终配置和展开后的环境 INI；`single_train.py` 结束时补充状态及输出 checkpoint 校验和。其他环境尚未接入该 manifest。

| 内容 | 必需字段 |
| --- | --- |
| 标识 | run ID、时间、任务、训练/评估/探索类型 |
| 代码 | 完整 commit SHA、branch/tag、dirty 状态、上游来源 |
| 配置 | 最终配置快照及哈希；实际加载的 INI/YAML/JSON 内容 |
| 随机性 | Python、NumPy、Torch、环境、攻击器、评估 seed；并行 worker 分配 |
| 依赖 | Python、Torch、NumPy、Gym、CUDA/cuDNN、操作系统、硬件；实际使用 SUMO 时记录其版本 |
| 模型 | victim/attacker 的训练 commit、配置、训练 seed、checkpoint SHA-256 |
| 执行 | 完整命令、工作目录、训练步数、评估回合、动作确定性、超时和终止规则 |
| 攻击 | 通道、受攻击对象、预算、时机、特征掩码、信息权限、实际修改量、查询/梯度次数 |
| 指标 | 指标定义与单位、聚合口径、原始轨迹或数据文件位置、产物校验和 |

仅记录 `seed=1` 或目录名不足以复现。CACC 外层 seed 传递及环境 RNG 隔离已修复；实际 episode seed 同时写入安全指标。详见 [CACC 变更记录](docs/cacc_changes.md)。

建议输出结构（仓库外目录或已被 Git 忽略的 `artifacts/`）：

```text
artifacts/<run_id>/
  manifest.json
  config.resolved.json
  environment.ini
  dependencies.txt
  metrics.json
  episodes.csv
  trajectories/
  checkpoints/
```

正式对比固定 victim checkpoint，clean 与各攻击共享评估场景/seed，单独区分训练 seed 和测试 episode。探索运行与论文结果分别存放。选模型、调攻击参数与最终测试使用不同数据划分。

## 当前仓库的上传

本地最初没有 `.git`。现有历史从源码导入提交开始，上游历史尚未取回，见 [来源记录](docs/source_provenance.md)。远端已配置：

- `origin`: `https://github.com/ily3000t/att_new.git`
- `upstream`: `https://github.com/BUAA-TrustworthyMARL/adv_marl_benchmark.git`

2026-09-10 当前主机到 GitHub 的 Git HTTPS 连接失败。完成本地分支合并后，可在 PowerShell 中重试：

```powershell
Set-Location 'E:\adv_marl_benchmark-main'
git status --short --branch
git remote -v
git ls-remote origin
git push -u origin main
```

`git ls-remote origin` 成功且没有输出通常表示目标仓库为空；报错不代表仓库为空。如果提示需要认证，使用本机 Git Credential Manager 或已安装的 GitHub CLI 完成登录。不要将 token 写进 remote URL 或源码。

如果 push 提示 `non-fast-forward`，先保留远端内容并检查：

```powershell
git fetch origin
git log --graph --oneline --decorate --all -20
# 如需先上传供查看，可推到一个尚不存在的独立分支
git push origin main:refs/heads/import/amb-foundation
```

不要直接 `push --force`，也不要未经检查就合并不相关历史。如果仓库尚未创建或账号没有权限，先在 GitHub 确认 `att_new` 存在且当前账号有写权限，再重试。

恢复网络后可 `git fetch upstream` 留存上游 refs。由于当前为独立源码导入历史，fetch 本身不会证明本地对应上游某个 commit；需要对比树内容、确认上游版本后再安排整合，不能盲目 `git pull upstream main`。

## 回看与回退

```powershell
# 独立目录查看已验证版本，避免打乱当前开发
git worktree add --detach '..\att_new-baseline-review' exp/cacc-mappo-baseline-v1
# 公共分支撤销某次修改：用实际提交 SHA 替换示例占位符
git revert <commit-sha>
```

示例 tag 要在真实存在后才可使用。源码快照与环境 smoke test 都不能替代学习算法、攻击 baseline 的端到端复现。
