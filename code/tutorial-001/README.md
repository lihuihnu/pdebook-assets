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

若要重绘文章中的正式图件，还需要 Matplotlib、Pillow、svgwrite、fontTools，以及系统中的 Noto Sans CJK SC / STIX 字体与 `rsvg-convert`。

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

`src/` 中的数值脚本和 `config/` 与私有权威实验源逐字一致；`results/` 是这些脚本实际运行得到并通过文章验收的保存结果。公共同步过程会对结果文件的 Git blob 身份逐项核对。

## 六个实验

- **ET001-01**：均匀材料解析回归；
- **ET001-02**：两层材料串联热阻解析基准；
- **ET001-03**：调和平均与算术平均的界面处理对照；
- **ET001-04**：中央矩形嵌入体四级网格细化；
- **ET001-05**：直接读取 ET001-04 保存面通量的逐控制体守恒审计；
- **ET001-06**：固定 96×64 网格的九点导热率对比度扫描。

## 从头复现

建议把复现结果写到新的 `reproduced/` 目录，不覆盖仓库中已经验收的 `results/`。

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

文章中的五幅正式图都来自本教程冻结的几何或实际计算结果：

- FT001-01：复合板几何与边界；
- FT001-02：共享面与两段串联热阻；
- FT001-03：两层材料界面基准与平均方式对照；
- FT001-04：二维温度场与由保存面通量重构的热流路径；
- FT001-05：导热率对比度与等效导热系数。

公开文章使用的 PNG 位于仓库根目录的 `assets/tutorial-001/`。
