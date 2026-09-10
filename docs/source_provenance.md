# 源码来源记录

记录日期：2026-09-10。

| 项目 | 记录 |
| --- | --- |
| 用户提供的本地源码 | `E:\adv_marl_benchmark-main` |
| 用户指定的上游 | https://github.com/BUAA-TrustworthyMARL/adv_marl_benchmark |
| 用户指定的目标仓库 | https://github.com/ily3000t/att_new |
| 初始 Git 状态 | 不存在 `.git`，不是 Git checkout |
| 上游 commit SHA | **未知，尚未核实** |
| 本地源码导入 commit | `fd763573be39087f5292616e8dac080090fea4cf` |
| 导入分支 | `bootstrap/amb-import` |
| 导入规模 | 1,542 个文件 |
| 上游历史 | 未包含在本地导入历史中 |

上游网页可读取，但当前主机的 Git HTTPS 连接在普通权限及提升权限下均无法连接 GitHub 443 端口，因而未能 fetch 上游历史或确认目标仓库的 refs。网络失败不能解释为目标仓库不存在、为空或账号认证失败。

导入提交保存了现有可追踪源码、配置和上游说明资源，没有改动算法、环境或奖励实现。导入前已按已有 `.gitignore` 筛选文件，并额外排除：

- `external/demo/checkpoint/actor.pth`：上游演示权重，保留本地文件但不放进本仓库 Git 历史。
- `upload.sh`：上游机器相关上传脚本，包含凭据形式的文本；保留本地文件但不上传。

后续独立提交补充忽略规则和开发文档。`amb/models/` 是源码，未被忽略。已有资产许可证和作者说明保留；没有凭空为整个上游源码添加新的许可证声明。

这份导入记录证明本地起点，不证明它与上游某个 commit 一致，也不证明论文实验已经复现。恢复网络后应保存上游 refs、对比源码，再补充精确版本映射。

## 论文身份与范围

NeurIPS 2025 正式题名为 *Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning*，AMB 是其 benchmark/codebase 名称。正式论文研究四类应用环境中的合作 MARL 鲁棒性与韧性。[NeurIPS 正式页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e8d9bf1dd1eb9d3d9d500fb3543c87b-Abstract-Conference.html)

论文列出四种交通任务，但说明 Monaco 任务因空间异构而未纳入评测；正文按三个 Traffic 任务、总计 18 个任务描述评测。因此，上游 README 的“19 tasks”与四个交通任务介绍不能直接当成最终评测范围。[论文正文与附录 A.2.3](https://arxiv.org/html/2510.11824v2)

## 当前 Git 里没有实验发布 tag

本轮完成的是源码建档、代码分析及有限运行检查。稳定复现实验版本仍需训练与攻击评估验证，因此不创建 `exp/*` tag。
