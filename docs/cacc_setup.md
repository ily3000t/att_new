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
