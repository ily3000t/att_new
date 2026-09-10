# 项目开发规范

本文件记录仓库所有者于 2026-09-10 确定的长期开发要求，适用于本仓库后续任务。

## Git 与分支

- 每个完整、可独立解释和回退的原子修改单独 commit，使用 Conventional Commits。
- 开发前检查 `git status`、当前分支和相关代码；保留用户已有修改。
- 从 `main` 建立 `feat/*`、`fix/*`、`chore/*` 或 `docs/*` 分支，禁止直接在 `main` 开发。
- 在工作分支完成实现和必要验证，再提交并合并到 `main`。正常合并保留原子提交，不通过 squash 丢掉开发步骤。
- 每个经验证能够稳定复现实验的版本创建 annotated tag；仅源码导入、文档检查或 smoke test 不构成实验复现。
- 不移动已有实验 tag，不强制推送覆盖远端，不在未检查远端状态时合并未知历史。
- 回看历史优先使用独立 worktree 或 detached HEAD；公共分支回退优先 `git revert`。
- 不把 Git 提交历史、上游 commit 或实验验证状态写成未经核实的事实。

## 实验与产物

- 每次正式实验绑定完整 Git commit SHA，并保存最终解析后的配置、全部 seed、启动命令、依赖与运行平台信息。
- 正式实验从干净的代码树启动；探索运行如包含未提交修改，必须标记 dirty 并保存差异，不能作为正式对比结果。
- 记录 victim 和 attacker 各自的训练 commit、配置、checkpoint SHA-256，以及评估代码 commit。
- 每个实验使用独立 run ID 和输出目录；原始结果、权重、视频、TensorBoard/W&B 日志不提交 Git。
- Git 保存源码、配置、实验协议、产物索引或模板。大体积实验产物保存在仓库外或已忽略的 `artifacts/`、`results/`。
- `amb/models/` 是网络结构源码，必须受版本控制；不能用无路径限定的 `models/` 规则将其整体忽略。
- 科研结论区分静态检查、局部验证、端到端运行、重复实验和论文指标复现。

## 本项目的研究边界

- AMB 是合作 MARL 鲁棒性与韧性 benchmark，不能将整个上游项目描述为专门的自动驾驶 benchmark。
- 当前 `network/catchup`、`network/slowdown` 使用原生 NumPy OVM 跟驰动力学；SUMO/TraCI 用于 ATSC 路径。
- 原生 CACC、SUMO CACC、SUMO Ego 驾驶是不同实验设置，配置、指标和结论必须分别标识。
- 保留上游源码快照。修复上游问题时单独提交，记录对奖励、终止、随机性和攻击预算的影响。
- 不把原始实现的错误静默修复后仍称为“完全原样复现”；使用明确的原始版本与修复版本记录。
- 新攻击先明确威胁模型和可访问的信息，再比较同预算 baseline；不能以额外权限或更大预算冒充算法改进。
- 跨环境重新训练攻击器只能称为方法迁移；未经验证不能称为攻击权重零样本迁移。

详细工作方式见 [DEVELOPMENT.md](DEVELOPMENT.md)，当前代码分析与路线见 [docs/research_assessment.md](docs/research_assessment.md)。
