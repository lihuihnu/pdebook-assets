# pdebook-assets

Public image assets for **《把方程算明白》**.

This repository contains **generated publishing assets only**. The canonical manuscript, source SVG files, notes, references, code, and review materials remain in the private `lihuihnu/pdebook` repository.

## Layout

```text
assets/
  pde-NNN/
    *.png
manifest.json
```

The public image URLs used by Zhihu publish views follow:

```text
https://raw.githubusercontent.com/lihuihnu/pdebook-assets/main/assets/pde-NNN/<figure>.png
```

Do not edit generated PNG files by hand. They are derived from the canonical SVG figures in the private book repository.


## Publishing policy

Assets are published directly from the private `lihuihnu/pdebook` source repository by a maintainer/agent that can access both repositories.

There is no PAT-based cross-repository automation and no persistent publishing workflow in this repository.

Current published article assets:

- `assets/pde-001/fig-001-function-to-field.png` — 1920 × 902
- `assets/pde-001/fig-002-heat-solution.png` — 1920 × 1110

Metadata is recorded in `manifest.json`.
