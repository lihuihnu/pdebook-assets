# tutorial-002：L 形区域上的 Poisson 方程

这是《把方程算明白·数值求解教程》第 2 篇的读者配套代码与实际结果。

对应文章：

> L 形区域上的 Poisson 方程：一个凹角为什么会拖慢有限元收敛

## 依赖

数值实验使用：

- Python 3.12.3
- NumPy 2.3.5
- SciPy 1.17.0

安装：

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0
```

ET002-04 的空间累计统计只使用 Python 标准库。

## 目录

```text
code/tutorial-002/
  README.md
  config/
    et002-01.json
    ...
    et002-05.json
  src/
    fem_core.py
    error_evaluator.py
    singular_error_evaluator.py
    run_et002_01.py
    ...
    run_et002_05.py
    materialize_et002_04_inputs.py
    check_et002_*.py
  results/
    et002-01/
    et002-02/
    et002-03/
    et002-04-inputs/
    et002-04/
    et002-05/
```

三个数学核心文件 `fem_core.py`、`error_evaluator.py` 与
`singular_error_evaluator.py`，五份配置和已发布结果与私有权威源保持
Git blob 内容身份一致。driver / checker 只做了一项发布层适配：把私库中的
`experiments/tutorial-002/` 根目录改为当前公开目录 `code/tutorial-002/`；
数值公式、门槛与算法逻辑没有改变。

## 五组实验

- **ET002-01**：P1 线性斑块检验；
- **ET002-02**：光滑调和解在均匀网格上的误差收敛；
- **ET002-03**：L 形重入角奇异解的均匀网格基准；
- **ET002-04**：只读冻结 U128 sidecar，统计单元能量误差的空间集中；
- **ET002-05**：均匀 / 分级网格的 matched-DOF 比较。

主要结果与正文一致：光滑控制问题的能量误差观测阶为 1；奇异基准最后两级约
0.6541 与 0.6587；在 U128 上，按单元重心分类，约 93.3% 的奇异能量误差平方
来自 `r_K<1/8` 的单元；最细 matched-DOF 分级网格的能量误差约为均匀网格的
42.2%。

## 从头运行

从仓库根目录执行：

```bash
python code/tutorial-002/src/run_et002_01.py
python code/tutorial-002/src/run_et002_02.py
python code/tutorial-002/src/check_et002_01_02.py

python code/tutorial-002/src/run_et002_03.py
python code/tutorial-002/src/check_et002_03.py

python code/tutorial-002/src/check_et002_04_inputs.py
python code/tutorial-002/src/run_et002_04.py
python code/tutorial-002/src/check_et002_04.py

python code/tutorial-002/src/run_et002_05.py
python code/tutorial-002/src/check_et002_05.py
```

公开包已经包含 ET002-04 所需的冻结 U128 节点、三角形、数值解和逐单元误差输入。
`run_et002_04.py` 会由这些输入重新生成完整的
`results/et002-04/element_spatial.csv`。该约 4.2 MB 的纯派生文件没有再次
存入公开仓库，以避免与可重建输入重复；正文直接使用的
`summary.csv` 与 `cumulative.csv` 已发布并与私有源逐字一致。

这些 driver 默认写回本目录下的 `results/`。若希望保留仓库中发布的结果文件不变，
可先复制整个 `code/tutorial-002/` 目录到单独工作目录后运行。

## 图件

文章中的 FT002-01～05 已发布到仓库根目录：

```text
assets/tutorial-002/
```

五幅 public PNG 都是 private canonical PNG 的原始字节复制，没有重新栅格化或
重绘，因此 Git blob 与私有 canonical 完全相同。
