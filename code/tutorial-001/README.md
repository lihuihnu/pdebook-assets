# tutorial-001：复合材料中的二维稳态导热

这是《把方程算明白·数值求解教程》第 1 篇的读者配套代码与实际结果。

本目录对应文章：

> 复合材料中的二维稳态导热：导热系数突变时，热流怎样穿过界面

## 依赖

数值实验使用：

- Python 3.13
- NumPy 2.3.5
- SciPy 1.17.0

安装：

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0
```

若要重绘正式图件，还需要 Matplotlib、Pillow、svgwrite、fontTools，以及系统中的 Noto Sans CJK SC / STIX 字体。

## 目录

```text
code/tutorial-001/
  README.md
  config/
    uniform.json
    layered.json
    averaging_ablation.json
    inclusion.json
    contrast_sweep.json
  src/
    composite_heat.py
    averaging_ablation.py
    inclusion_refinement.py
    conservation_audit.py
    contrast_sweep.py
    make_FT001_01.py
    make_FT001_02.py
    make_FT001_03.py
    make_FT001_04.py
    make_FT001_05.py
  results/
    et001-01/
    et001-02/
    et001-03/
    et001-04/
    et001-05/
    et001-06/
```

`src/` 中的数值脚本、绘图脚本和 `config/` 与私有权威源逐字一致。公开 `results/` 由这些冻结源码在固定 Python / NumPy / SciPy 环境中重新执行得到，并通过与私有验收相同的解析锚点、网格细化和守恒门槛。不同 runner 的稀疏直接求解可能在约 $10^{-13}$ 量级出现舍入差异，因此公开 manifest 同时记录私有源结果 blob 与公开重跑结果 blob，不把浮点字节一致性误当成科学一致性。

## 六个实验

- **ET001-01**：均匀材料解析回归；
- **ET001-02**：两层材料串联热阻解析基准；
- **ET001-03**：调和平均与算术平均的界面处理对照；
- **ET001-04**：中央矩形嵌入体四级网格细化；
- **ET001-05**：直接读取 ET001-04 保存面通量的逐控制体守恒审计；
- **ET001-06**：固定 96×64 网格的九点导热率对比度扫描。

## 从头复现

建议把复现结果写到新的 `reproduced/` 目录，不覆盖仓库中已经发布的 `results/`。

```bash
python code/tutorial-001/src/composite_heat.py \
  --config code/tutorial-001/config/uniform.json \
  --output reproduced/et001-01

python code/tutorial-001/src/composite_heat.py \
  --config code/tutorial-001/config/layered.json \
  --output reproduced/et001-02

python code/tutorial-001/src/averaging_ablation.py \
  --config code/tutorial-001/config/averaging_ablation.json \
  --output reproduced/et001-03

python code/tutorial-001/src/inclusion_refinement.py \
  --config code/tutorial-001/config/inclusion.json \
  --output reproduced/et001-04

python code/tutorial-001/src/conservation_audit.py \
  --source reproduced/et001-04 \
  --output reproduced/et001-05

python code/tutorial-001/src/contrast_sweep.py \
  --config code/tutorial-001/config/contrast_sweep.json \
  --et004-summary reproduced/et001-04/summary.csv \
  --output reproduced/et001-06
```

ET001-04～06 会保存较大的单元/面级 CSV；它们用于流线重构、逐控制体守恒和参数扫描的独立核对。

## 图件

文章中的五幅正式图都来自私有仓库经过审校的 canonical SVG；公共 PNG 由这些最终 SVG 以 1920 px 宽度确定性栅格化，不从公开重跑结果另画一套图。

- FT001-01：复合板几何与边界；
- FT001-02：共享面与两段串联热阻；
- FT001-03：两层材料界面基准与平均方式对照；
- FT001-04：二维温度场与由保存面通量重构的热流路径；
- FT001-05：导热率对比度与等效导热系数。

公开 PNG 位于仓库根目录的 `assets/tutorial-001/`。
