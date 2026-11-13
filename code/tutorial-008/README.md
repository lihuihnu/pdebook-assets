# Tutorial-008：为什么精度更高的格式，反而算出了负浓度？

这是面向读者的完整 Python 复现包。研究对象是一维定常、无源对流–扩散边值问题
\[
-Dc''+vc'=0,\qquad c(0)=0,\quad c(1)=1,
\]
其中基准物理参数为 \(L=v=1,\ D=1/60,\ P=60\)。内部未知量是节点浓度，不是单元平均；三种方法使用相同的节点中心对偶控制体和共享面通量差。

## 内容

- `src/run_et008_01.py`～`run_et008_04.py`：四阶段实际求解入口。
- `src/validate_et008_01.py`～`validate_et008_04.py`：不导入生产求解器的独立校验程序。
- `config/design_v01.json`：四阶段统一设计与 51 个唯一 case。
- `results/`：四阶段完整冻结结果，包括 summary/comparison/orders、检查报告、复用历史记录，以及全部节点、共享面通量和 run 元数据。
- `../assets/tutorial-008/`：五张正式 PNG 图件。图像与私库 canonical PNG 原字节一致。

数值核心仅使用 Python 标准库，不需要 NumPy、SciPy 或 Matplotlib。建议 Python 3.10 或更新版本。

## 从公开仓库根目录运行

所有运行命令都把新结果写入独立的 `/tmp/` 目录，不覆盖这里随包提供的冻结数据。每个输出目录必须是新的空目录。

### ET008-01：基准三方法

~~~bash
R=code/tutorial-008
OUT=/tmp/tutorial-008-et008-01

python3 "$R/src/run_et008_01.py" \
  --config "$R/config/design_v01.json" \
  --output "$OUT"

python3 "$R/src/validate_et008_01.py" \
  --config "$R/config/design_v01.json" \
  --solver-source "$R/src/run_et008_01.py" \
  --output "$OUT"
~~~

该阶段实际求解 \(N=12\) 下的 M0 中心格式、M1 迎风格式和 M2 指数拟合格式，共 3 个 case。

### ET008-02：局部 Péclet 数扫描

~~~bash
R=code/tutorial-008
OUT=/tmp/tutorial-008-et008-02

python3 "$R/src/run_et008_02.py" \
  --config "$R/config/design_v01.json" \
  --baseline "$R/results/et008-01" \
  --output "$OUT"

python3 "$R/src/validate_et008_02.py" \
  --config "$R/config/design_v01.json" \
  --baseline "$R/results/et008-01" \
  --solver-source "$R/src/run_et008_02.py" \
  --core-source "$R/src/run_et008_01.py" \
  --output "$OUT"
~~~

该阶段新增 24 个 case；\(p=5\) 的三个基准 case 从 ET008-01 只读复用，不重复求解。比较表包含 27 行。

### ET008-03：固定物理问题下的迎风数值扩散

~~~bash
R=code/tutorial-008
OUT=/tmp/tutorial-008-et008-03

python3 "$R/src/run_et008_03.py" \
  --config "$R/config/design_v01.json" \
  --baseline "$R/results/et008-01" \
  --output "$OUT"

python3 "$R/src/validate_et008_03.py" \
  --config "$R/config/design_v01.json" \
  --baseline "$R/results/et008-01" \
  --source "$R/src/run_et008_03.py" \
  --core "$R/src/run_et008_01.py" \
  --output "$OUT"
~~~

该阶段固定 \(L=v=1,\ D=1/60,\ P=60\)，新增 \(N=24,60\) 两级网格的六个 case；\(N=12\) 的三个 case 从 ET008-01 只读复用。比较表共 9 行。

### ET008-04：固定物理问题下的网格收敛

~~~bash
R=code/tutorial-008
OUT=/tmp/tutorial-008-et008-04

python3 "$R/src/run_et008_04.py" \
  --config "$R/config/design_v01.json" \
  --baseline-01 "$R/results/et008-01" \
  --baseline-03 "$R/results/et008-03" \
  --output "$OUT"

python3 "$R/src/validate_et008_04.py" \
  --config "$R/config/design_v01.json" \
  --baseline-01 "$R/results/et008-01" \
  --baseline-03 "$R/results/et008-03" \
  --output "$OUT" \
  --source "$R/src/run_et008_04.py" \
  --core "$R/src/run_et008_01.py"
~~~

该阶段固定原物理问题，只改变 \(N=12,18,24,30,48,60,96,192,384\)。新增 18 个 case；其余 9 个历史 case 从 ET008-01/03 只读复用。`comparison.csv` 有 27 行，`orders.csv` 有 M0/M1 的 16 组相邻网格观测阶，分母使用真实网格比 \(N_{\mathrm{fine}}/N_{\mathrm{coarse}}\)。M2 的节点误差处于浮点舍入地板，不对它拟合收敛阶。

## 冻结数据的含义

四阶段共包含 51 个不重复的实际求解 case（3 + 24 + 6 + 18）。每个 raw case 保存：

- `nodes.csv`：全部节点位置、实际浓度和连续解析参考；
- `flux.csv`：相邻节点间的共享面正向通量；
- `run.json`：参数、Python 版本、源码/配置 SHA-256 和原始文件 SHA-256。

各阶段的 `source_commit_sha.txt` 保留冻结来源提交身份。比较表和观测阶表保留跨阶段数据来源，避免把历史复用误认为新求解。

`results/` 中的文件是只读基准。重新运行时请始终使用新的 `/tmp` 输出目录。不同 Python 版本或浮点环境下重新求解，末位数值可能略有变化；应以独立验证器的数学判据和容差为准，不要求重新运行的 CSV 与历史文件逐字节相同。

## 公开图件

- [FT008-01：三种方法的基准节点剖面](../../assets/tutorial-008/FT008-01_baseline_profiles.png)
- [FT008-02：节点中心对偶控制体与共享面](../../assets/tutorial-008/FT008-02_dual_control_volume.png)
- [FT008-03：局部 Péclet 数的临界变化](../../assets/tutorial-008/FT008-03_peclet_threshold.png)
- [FT008-04：迎风节点与等效扩散解析解](../../assets/tutorial-008/FT008-04_fitted_diffusion.png)
- [FT008-05：固定物理问题下的网格误差与观测阶](../../assets/tutorial-008/FT008-05_mesh_convergence.png)

公开包不包含作者审图记录、私库图件 SVG、私人 CI 工作流或运行日志。五张 PNG 是经过私库审图的正式 canonical 输出。
