# 数值求解教程 003：Crank–Nicolson 的稳定性与非光滑初值振荡

本目录是《把方程算明白》教程第 3 篇的**公开读者代码与真实实验结果**。比较同一个一维热方程上的 Backward Euler（BE）、Crank–Nicolson（CN）和 Rannacher R1 启动。

## 运行环境

- Python 3.12 或更新的兼容版本。
- NumPy、SciPy。正式实验使用过 Python 3.12.3、NumPy 2.3.5、SciPy 1.17.0。
- 本目录的 Python 数值核心不依赖 Matplotlib；五张正文图的正式绘图源码、审校证据与 SVG canonical 保留在私库，公共仓库只发布读者所需的 PNG。

代码逐字同步自私有权威仓库 `experiments/tutorial-003/src/`，配置逐字同步自 `experiments/tutorial-003/config/`；除本 README 外不维护第二套算法实现。运行脚本会从自身的 `src/` 上一级定位 `config/` 和 `results/`，可在本目录直接运行。

## 如何重算

在公共仓库根目录执行：

```bash
python -m pip install numpy scipy
python code/tutorial-003/src/run_experiments.py --experiments et003-01 et003-02 et003-03 et003-04 et003-05
```

以上指令**会覆盖本地** `results/` 中相同名称的 CSV。建议先克隆/复制仓库到自己的工作目录，再进行重算；公开的 CSV 是已经通过正式独立审计的历史权威结果，不因个人重跑自动改变其发布身份。

也可以只运行单组，如：

```bash
python code/tutorial-003/src/run_experiments.py --experiments et003-03
```

## 独立验证

独立 checker 使用 Python 标准库，不调用被测 BE/CN/R1 求解器生成参考结果。可以在本仓库根目录运行：

```bash
python code/tutorial-003/src/check_et003_01_02.py
python code/tutorial-003/src/check_et003_03.py
python code/tutorial-003/src/check_et003_04.py
python code/tutorial-003/src/check_et003_05.py
```

它们重新读取冻结的配置和 CSV，核对谱、放大因子、时间误差、总变差、反向斜率、半离散/连续参考等指标。代码计算结果的验收是容差意义下的数学/数值检查；不承诺不同操作系统、BLAS 或 SciPy 版本重新生成的 CSV 逐字节相同。

## 文件索引

- `src/heat_core.py`：Dirichlet 二阶中心差分矩阵、BE/CN/R1 步进、总变差和反向斜率。
- `src/references.py`：光滑解析参考、离散 DST-I 半离散精确时间参考及连续 Fourier 箱形参考。
- `src/run_experiments.py`：五组数值实验的正式 driver。
- `src/check_et003_*.py`：对应正式 CSV 的独立结果检查。
- `config/et003-01.json`～`et003-05.json`：冻结的全部参数。
- `results/et003-01/`：谱与解析/数值一步放大因子。
- `results/et003-02/`：光滑单模态时间阶。
- `results/et003-03/`：早期压力测试完整节点剖面、诊断和反向斜率。
- `results/et003-04/`：九点 Fourier 数扫描的完整节点剖面、总变差和反向斜率。
- `results/et003-05/`：固定终止时间的时间收敛、完整节点剖面与空间误差基线审计。

## 关键科学边界

CN 的无条件稳定性不等于高频强阻尼。对当前有限网格，(r_{\mathrm{flip}}\approx0.50000771) 是最高频模态开始变号的谱尺度，**不是**完整解出现锯齿的充分必要阈值。本例的 R1 使用两个 BE 半步跨过第一个完整时间区间，以后继续用 CN。固定空间网格下的时间阶和一般非光滑 PDE 联合网格/时间细化的理论并非同一个命题。

源代码与 CSV 的公私 Git blob 身份，以及五张公共图片的身份，由公共仓库根目录 `manifest.json` 记录。公开资源仅供学习与复现；本文 canonical 正文始终在私库维护。
