# 数值求解教程 007：同一块材料转个方向，为什么原来的导热格式就不准了？

这是《把方程算明白·数值求解教程》第 7 篇的公开读者代码、冻结参数与真实实验数据。四幅正式原始 PNG 位于 [assets/tutorial-007](../../assets/tutorial-007/)，与私库已验收的 canonical PNG 完全同字节；本目录不包含作者审稿记录或私库正文。

本篇求解单位正方形上的二维各向异性稳态导热：

\[
-\nabla\cdot(K\nabla T)=f,\qquad T|_{\partial\Omega}=g.
\]

主导热率始终为 (10,1)，只改变材料方向。M0 故意忽略交叉导热项作为对照，M1 使用保留交叉项的九点离散。比较时，**两种方法使用相同的完整连续源项与 Dirichlet 条件**，不为 M0 另改物理问题。

## 环境和轻量检查

Python 3.11 以上，依赖 NumPy 和 SciPy。先运行小规模代数测试，避免一上来重跑全部计算：

~~~bash
python3 -m pip install numpy scipy
python3 -m unittest discover -s code/tutorial-007/tests -p 'test_*.py' -v
~~~

这一步用于检查张量旋转、九点系数、边界消元和小网格通量，不代替四阶段真实数值实验或私库冻结时的独立科学验收。原始科学实验于私人 Runner 完成，独立验收记录在各阶段 `results/et007-0N/checks.json`。

## 冻结实验和文件身份

| 阶段 | 问题 | 新求解数 |
| --- | --- | ---: |
| ET007-01 | 0°，正弦制造解，N=16/32/64/128，M0/M1 | 8 |
| ET007-02 | 45°，仿射真解 T=x，N=32/64，M0/M1 | 4 |
| ET007-03 | 45°，正弦制造解，N=16/32/64/128，M0/M1 | 8 |
| ET007-04 | 15° 和 30°，N=64/128，M0/M1 | 8 |

合计 **28 个唯一真实求解**。`results/summary.csv` 冻结了这 28 行。`results/et007-04/angle_comparison.csv` 有 16 行，其中 0° 与 45° 分别原样复用 ET007-01 和 ET007-03 的结果，不能当作 16 次新的求解。

`results/et007-0N/summary.csv` 与 `checks.json` 是四阶段结果与科学检查。`raw/<case_id>/run.json` 保存 28 个案例的参数与源文件哈希。部分已指定网格还保存 `temperature.csv`、`flux_x.csv`、`flux_y.csv` 共 24 份，包含实际节点温度与共享面对偶面热流。公开结果合计 **62 个文件**。科学源代码及测试、冻结配置共 14 个原样文件，另有本公共 README。

其中 ET007-02 验证：即使两法温度误差都约为 10^-15，M0 的平均横向热流仍近零，而真实值和 M1 都是 -4.5；ET007-03 验证 M0 温度误差在约 0.076 且不随网格细化下降，M1 则近二阶收敛；角度扫描要保留 0° 两法重合以及 M1 从 30° 到 45° 的小幅误差回升。

## 在新输出目录重新求解

以下从**公共仓库根目录**运行。运行入口拒绝覆盖已有输出路径；以下命令会实际求解 PDE，可能花费计算时间。只想阅读冻结结果，无需运行。

~~~bash
R=code/tutorial-007

python3 "$R/run_experiments.py" \
  --config "$R/config/design_v01.json" --stage ET007-01 \
  --out /tmp/tutorial007-reader-et01

python3 "$R/src/validate.py" \
  --config "$R/config/design_v01.json" \
  --results /tmp/tutorial007-reader-et01 \
  --out /tmp/tutorial007-reader-et01-checks.json
~~~

其他阶段使用 `run_et007_02.py`、`run_et007_03.py`、`run_et007_04.py` 和对应的 `src/validate_et007_0N.py`。例如 ET007-04 既需要它本阶段的参数，也需要三个已冻结阶段的摘要来核对 0°/45° 的旧数据身份：

~~~bash
python3 "$R/run_et007_04.py" \
  --config "$R/config/design_v01.json" --stage ET007-04 \
  --et01-summary "$R/results/et007-01/summary.csv" \
  --et02-summary "$R/results/et007-02/summary.csv" \
  --et03-summary "$R/results/et007-03/summary.csv" \
  --out /tmp/tutorial007-reader-et04
~~~

不同 Python、NumPy、SciPy 和 BLAS 环境下重新求解，浮点结果的逐字节身份未必一致。判断科学一致性应使用独立验证器与适当容差，不应以重新输出的文件 SHA 完全相同作为先决条件。不要修改本目录的冻结 `results/`，另用新输出目录。

## 原始公私身份及范围

仓库根目录的 [manifest.json](../../manifest.json) 逐项记录公开文件路径、私库原路径和 Git blob SHA。四幅 PNG 与 76 个源码/冻结结果文件应与作者私库的原始 Git blob 相同；**本 README** 为公共目录专门编写，属于唯一允许的文字转换文件，不冒充私库 README 的字节副本。

没有公开的文件包括作者审校 notes、私人 Runner 身份与流程 JSON、阶段运行会话 `run.json`、正式书稿、字体文件以及私库 SVG。独立验证器与原始数值结果已经审阅，但这不表示每个读者的机器执行都事先通过。

四张图件的正式私库科学验收已完成。知乎文章尚需作者实际平台预览与发布确认；公共代码/图片可用不等于知乎文章已正式发布。
