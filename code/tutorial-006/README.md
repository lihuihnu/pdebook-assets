# 数值求解教程 006：波离开计算区域后，为什么又回来了？

这是《把方程算明白》第 6 篇数值求解教程的**公开读者代码、冻结参数和真实数值结果**。五幅已经独立验收的原始 PNG 见 [assets/tutorial-006](../../assets/tutorial-006/)。

本目录提供一维波动方程三层中心差分、Taylor 起步、固定端与时空平均 Mur 出流边界、同网格扩域数值参考、带符号反射投影，以及独立数学验证和单元测试。所公开的 Python 与 CSV / 逐算例 JSON 都与作者私库的冻结版本保持完全相同的 Git blob 身份；只有本 README 根据公共路径单独编写。

## 运行环境与快速检查

原始计算环境为 Linux x86_64、Python 3.14.4、NumPy 2.3.5。建议使用兼容版本（例如 Python 3.11 以上与 NumPy 2.3.5）。实验和测试不需要 Matplotlib、SciPy、PETSc 或 GPU。

从**公共仓库根目录**执行：

~~~bash
python3 -m pip install "numpy==2.3.5"
python3 -m unittest discover -s code/tutorial-006/tests -p 'test_*.py' -v
~~~

作者私人 Runner 在冻结版本上通过 15 项单元测试；这不是对读者计算环境的事先担保。

## 重新计算与独立验证

从公共仓库根目录启动，输出目录必须不存在；入口不会覆盖结果或失败现场。

~~~bash
R=code/tutorial-006
python3 "$R/run_experiments.py" \
  --config "$R/config/design.json" \
  --out /tmp/tutorial006-reader-run01

python3 "$R/src/validate.py" \
  --config "$R/config/design.json" \
  --out /tmp/tutorial006-reader-run01
~~~

一次运行包含 16 组短域问题和 4 组扩大计算区域参考问题。验证器**不导入生产求解器**，独立重算全部 20 组，并对五个最粗网格问题使用纯 Python 列表作额外复核。它在新结果目录下生成检查记录。失败或需要再次运行时，使用新的输出目录，例如 `/tmp/tutorial006-reader-run02`，不要改动公开的 `results/`。

新运行的 `run.json` 会记录当时的程序、环境和 Git 来源。更换操作系统、Python/NumPy 版本后，浮点字节级一致不一定仍成立，应依据独立验证器的误差尺度和实际指标判断。

## 冻结问题与公开结果

- PDE：无量纲一维波动方程 `u_tt=c²u_xx`，`c=1`；右行紧支撑六次幂脉冲 `g(x-t)`，正确初始速度为 `-g'(x)`。
- 短域 `[0,1]`，扩域 `[-2,3]`；终止时间 `T=1.2`。
- 四级网格 `N=200,400,800,1600`，固定 `C=0.8`；快照时刻 `t=0,0.4,0.6,0.9,1.2`。
- 短域边界为固定 Dirichlet 以及 `β=0.5,1,2` 的 Mur；扩大域用相同空间/时间步长，远端在该计算窗口内不影响短域裁剪。

`config/design.json` 是冻结参数；`src/wave.py` 是数值求解器，`run_experiments.py` 是生产入口，`src/validate.py` 是独立核验器，`tests/test_wave.py` 包含单元测试。

`results/metrics.csv` 保存 **80 行**实验指标。另有 **20 个** `results/raw/<run_id>/` 目录，每个目录包含：

- `snapshots.csv`：五个指定时刻的节点位移；扩域解只保存与短域匹配的 `[0,1]` 裁剪。
- `diagnostics.csv`：每一时间层的位移极值与离散位移范数；扩域诊断覆盖其完整计算域。
- `run.json`：本组网格、时间步、边界和输出窗口参数。

总计 **61 个原样科学结果文件**：41 个 CSV 与 20 个逐算例 `run.json`。本目录另外公开 4 个 Python 源文件、1 个配置文件与本 README，**合计 67 个读者文件**。

所有 `L^2` 和 `L^\infty` 指标衡量的是**位移误差**，不能直接称为波动能量。对于与开放域的误差，匹配 Mur 的小残余与固定端的大反射不同；边界波速失配会使连续边界模型本身产生不随网格细化消失的反射，不能只靠修改离散步长消除。

## 公开身份与保留材料

公开仓库根目录的 [manifest.json](../../manifest.json) 逐个记录公共文件与私有冻结来源文件的 Git blob SHA，除本 README 外应严格相同。该 manifest 同时记录五幅 PNG 的公私文件身份、像素尺寸和长期公开 URL。

作者私库中的全局 `results/run.json`、全局独立 `results/checks.json`、`runner-verification.json`、审稿 notes、图件源码及完整书稿**不在公共包中**。原始实验的私人 Runner 验证已通过：[真实计算与独立双次验证](https://github.com/lihuihnu/pdebook/actions/runs/38852166257)；五图生成与独立图像验收见 [图件工作流](https://github.com/lihuihnu/pdebook/actions/runs/38870225286)。这两项历史验收不能替代读者自行复算。

本实验的连续出流条件在一维常波速、合适的初值和给定时间窗内有精确意义；不直接推广为高维、变系数或任意长时间的无反射边界，也不声称完整 Mur 边界格式已有一般稳定性证明。
