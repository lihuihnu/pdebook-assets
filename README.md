# pdebook-assets

Public publishing assets and reader-facing companion code for **《把方程算明白》**.

The canonical manuscript, source SVG files, authoring notes, references, review materials, and development history remain in the private `lihuihnu/pdebook` repository. This public repository contains only material intended for readers and publishing platforms.

## Layout

```text
assets/
  cover/
    zhihu-column-cover.png
  pde-NNN/
    *.png

code/
  pde-NNN/
    README.md
    src/
      *.py
    results/
      *.csv

manifest.json
```

### Public figures

Zhihu publish views use public figure URLs of the form:

```text
https://raw.githubusercontent.com/lihuihnu/pdebook-assets/main/assets/pde-NNN/<figure>.png
```

Public PNG files are synchronized from reviewed canonical PNGs or rendered from vector figures in the private source repository. Conceptual illustrations and computed quantitative plots are identified in the per-article manifests. Do not edit published PNG files by hand.

The reviewed Zhihu column cover is published separately at `assets/cover/zhihu-column-cover.png`. It is mirrored byte-for-byte from the private canonical cover and is not associated with a single `pde-NNN` article.

### Public code

When an article contains executable teaching code, its reader-facing copy is published under:

```text
code/pde-NNN/
```

Python source and numerical result files are synchronized from the corresponding private `experiments/pde-NNN/` directory. Article-specific public README files may adjust paths and reader instructions for this repository, but do not maintain a second algorithm implementation.

## Publishing policy

Figures and companion code are published directly from the private `lihuihnu/pdebook` source repository by a maintainer/agent that can access both repositories.

There is no PAT-based cross-repository automation and no persistent publishing workflow in this repository.

Metadata, including source/output blob identities where applicable, is recorded in `manifest.json`.

## Current reader code

- `code/pde-007/` — 一维 Poisson 方程的第一个数值解；Python standard library only.
- `code/pde-018/` — 从局部守恒到单元平均；精确积分、中心点比较与局部守恒核对；Python standard library only.
- `code/pde-019/` — 有限体积格式与数值通量；共享通量、周期推进、波形对照与误差细化；Python standard library only.
- `code/pde-020/` — 迎风通量重新看线性平流；平移积分、重新求平均、脉冲对照与光滑细化；Python standard library only.
- `code/pde-021/` — Burgers 方程、激波与弱解；熵通量、激波与稀疏波、非熵反例、误差细化与边界输运；Python standard library only.
- `code/pde-022/` — 高阶重构与限制器的基本思想；线性重构、时间平均通量、minmod、脉冲振荡与光滑细化；Python standard library only.
- `code/pde-023/` — 从 Poisson 方程到弱形式；测试函数、分段积分、弱残差、能量与斜率误差核对；Python standard library only.
- `code/pde-024/` — 一维线性有限元与“小帐篷”基函数；非均匀帽函数、节点插值、误差与弱积分核对；Python standard library only.
