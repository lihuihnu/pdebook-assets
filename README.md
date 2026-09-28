# pdebook-assets

Public publishing assets and reader-facing companion code for **《把方程算明白》**.

The canonical manuscript, source SVG files, authoring notes, references, review materials, and development history remain in the private `lihuihnu/pdebook` repository. This public repository contains only material intended for readers and publishing platforms.

## Layout

```text
assets/
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

Generated PNG files are derived from canonical SVG figures in the private source repository. Do not edit generated PNG files by hand.

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
