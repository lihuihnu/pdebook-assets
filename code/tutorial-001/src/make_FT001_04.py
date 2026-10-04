#!/usr/bin/env python3
"""Generate tutorial-001 FT001-04 from frozen ET001-04 cell/face data.

Panel A: raw 192x128 cell-centered temperature field (nearest/no interpolation).
Panel B: heat-flow paths reconstructed from the committed shared-face fluxes.

No PDE solve, matrix assembly, or flux recomputation from temperature is performed.
Typography follows the tutorial baseline: exact Noto Sans CJK SC + STIX/STIX Math.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from fontTools.ttLib import TTCollection
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib import rcParams
import numpy as np

CJK_REG_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_BOLD_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
STIX_REG = "/usr/share/fonts/opentype/stix-word/STIX-Regular.otf"

INK = "#171717"
GRID = "#D7D7D7"
INCLUSION = "#F4E7D8"
FLOW = "#3569B7"

rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = False
rcParams["svg.fonttype"] = "none"
rcParams["svg.hashsalt"] = "tutorial-001-ft001-04"


def family_names(ttfont):
    names = set()
    for rec in ttfont["name"].names:
        if rec.nameID not in (1, 16):
            continue
        try:
            value = rec.toUnicode().strip()
        except Exception:
            continue
        if value:
            names.add(value)
    return names


def extract_family(ttc_path: str, family: str, out_path: Path) -> int:
    collection = TTCollection(ttc_path)
    matches = [
        index for index, font in enumerate(collection.fonts)
        if family in family_names(font)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {family!r} face, got {matches}")
    index = matches[0]
    collection.fonts[index].save(out_path)
    return index


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_grid(rows, nx: int, ny: int):
    return [row for row in rows if int(row["nx"]) == nx and int(row["ny"]) == ny]


def build_cell_fields(cell_rows, nx: int, ny: int):
    temperature = np.full((ny, nx), np.nan, dtype=float)
    conductivity = np.full((ny, nx), np.nan, dtype=float)
    material = np.empty((ny, nx), dtype=object)

    for row in cell_rows:
        i = int(row["i"])
        j = int(row["j"])
        temperature[j, i] = float(row["temperature"])
        conductivity[j, i] = float(row["k"])
        material[j, i] = row["material"]

    if not np.all(np.isfinite(temperature)):
        raise RuntimeError("temperature field is incomplete")
    if not np.all(np.isfinite(conductivity)):
        raise RuntimeError("conductivity field is incomplete")

    return temperature, conductivity, material


def reconstruct_cell_center_flux(face_rows, nx: int, ny: int):
    """Reconstruct q=(qx,qy) at cell centers from saved face heat rates.

    Each saved face heat rate is divided by its face area to recover the
    positive-axis heat-flux-density component at that face. Cell-center
    components are then the arithmetic average of the two opposite face
    components. This is visualization post-processing only.
    """
    qx_w = np.full((ny, nx), np.nan, dtype=float)
    qx_e = np.full((ny, nx), np.nan, dtype=float)
    qy_s = np.full((ny, nx), np.nan, dtype=float)
    qy_n = np.full((ny, nx), np.nan, dtype=float)

    q_hot = 0.0
    q_cold = 0.0
    max_abs_internal_y_rate = 0.0

    for row in face_rows:
        kind = row["face_type"]
        i = int(row["i"])
        j = int(row["j"])
        area = float(row["area"])
        heat_rate = float(row["flux_positive_axis"])
        q = heat_rate / area if area != 0.0 else 0.0

        if kind == "internal_x":
            qx_e[j, i] = q
            qx_w[j, i + 1] = q
        elif kind == "internal_y":
            qy_n[j, i] = q
            qy_s[j + 1, i] = q
            max_abs_internal_y_rate = max(max_abs_internal_y_rate, abs(heat_rate))
        elif kind == "boundary_left":
            qx_w[j, 0] = q
            q_hot += heat_rate
        elif kind == "boundary_right":
            qx_e[j, nx - 1] = q
            q_cold += heat_rate
        elif kind == "boundary_bottom":
            qy_s[0, i] = q
        elif kind == "boundary_top":
            qy_n[ny - 1, i] = q
        else:
            raise RuntimeError(f"unknown face type {kind!r}")

    arrays = (qx_w, qx_e, qy_s, qy_n)
    if any(np.any(~np.isfinite(array)) for array in arrays):
        raise RuntimeError("face flux reconstruction left missing entries")

    qx = 0.5 * (qx_w + qx_e)
    qy = 0.5 * (qy_s + qy_n)
    return qx, qy, q_hot, q_cold, max_abs_internal_y_rate


def style_axis(ax, en_font):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(colors=INK, labelsize=10.0, width=0.8, length=3.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(en_font)
    ax.set_xlabel("x / mm", fontproperties=en_font, fontsize=11.5)
    ax.set_ylabel("y / mm", fontproperties=en_font, fontsize=11.5)
    for spine in ax.spines.values():
        spine.set_color(INK)


def make_figure(cells_csv: Path, faces_csv: Path, summary_csv: Path,
                output_dir: Path):
    nx, ny = 192, 128
    cells = select_grid(read_csv(cells_csv), nx, ny)
    faces = select_grid(read_csv(faces_csv), nx, ny)
    summary_rows = select_grid(read_csv(summary_csv), nx, ny)

    if len(cells) != nx * ny:
        raise RuntimeError(f"expected {nx*ny} cells, got {len(cells)}")
    if len(summary_rows) != 1:
        raise RuntimeError("expected one 192x128 summary row")

    summary = summary_rows[0]
    temperature, conductivity, material = build_cell_fields(cells, nx, ny)
    qx, qy, q_hot, q_cold, max_y_rate = reconstruct_cell_center_flux(
        faces, nx, ny
    )

    # Independent figure-level consistency checks against frozen ET001-04 summary.
    for field, value in (("q_hot", q_hot), ("q_cold", q_cold)):
        reference = float(summary[field])
        if abs(value - reference) > 1e-12 * max(1.0, abs(reference)):
            raise RuntimeError(f"{field} mismatch: {value} vs {reference}")
    reference_y = float(summary["max_abs_internal_y_flux"])
    if abs(max_y_rate - reference_y) > 1e-12 * max(1.0, abs(reference_y)):
        raise RuntimeError("max internal y-face heat-rate mismatch")

    dx_mm = float(summary["dx"]) * 1000.0
    dy_mm = float(summary["dy"]) * 1000.0
    x = (np.arange(nx, dtype=float) + 0.5) * dx_mm
    y = (np.arange(ny, dtype=float) + 0.5) * dy_mm

    with TemporaryDirectory(prefix="tutorial001-ft004-font-") as temp_dir:
        temp = Path(temp_dir)
        reg_otf = temp / "NotoSansCJK-SC-Regular.otf"
        bold_otf = temp / "NotoSansCJK-SC-Bold.otf"
        reg_index = extract_family(CJK_REG_TTC, "Noto Sans CJK SC", reg_otf)
        bold_index = extract_family(CJK_BOLD_TTC, "Noto Sans CJK SC", bold_otf)
        cn = FontProperties(fname=reg_otf)
        cn_bold = FontProperties(fname=bold_otf)
        en = FontProperties(fname=STIX_REG)

        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(12.2, 5.35), dpi=220, constrained_layout=True
        )

        # Panel A: exact cell-centered pixels, no smoothing/interpolation.
        image = ax_a.imshow(
            temperature,
            origin="lower",
            extent=(0, 120, 0, 80),
            interpolation="nearest",
            aspect="equal",
            cmap="cividis",
            vmin=20,
            vmax=100,
            rasterized=True,
        )
        ax_a.add_patch(
            Rectangle((45, 25), 30, 30, fill=False, edgecolor=INK,
                      linewidth=1.35, zorder=5)
        )
        style_axis(ax_a, en)
        ax_a.text(0.0, 1.04, "A", transform=ax_a.transAxes,
                  fontproperties=en, fontsize=14, fontweight="bold",
                  ha="left", va="bottom")
        ax_a.text(0.07, 1.04, "二维温度场", transform=ax_a.transAxes,
                  fontproperties=cn_bold, fontsize=12.5,
                  ha="left", va="bottom")
        cbar = fig.colorbar(image, ax=ax_a, shrink=0.83, pad=0.025)
        cbar.set_label(r"$T\,/\,^\circ\mathrm{C}$", fontsize=11.5)
        cbar.ax.tick_params(labelsize=9.5)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontproperties(en)

        # Panel B: path visualization from saved face fluxes.
        ax_b.set_facecolor("white")
        ax_b.add_patch(
            Rectangle((45, 25), 30, 30, facecolor=INCLUSION, alpha=0.36,
                      edgecolor=INK, linewidth=1.35, zorder=0)
        )
        # Seed paths from the hot-side cell centers. Streamplot interpolates the
        # reconstructed cell-center vector field only for visualization.
        seed_y = np.linspace(y[4], y[-5], 15)
        starts = np.column_stack((np.full_like(seed_y, x[0]), seed_y))
        stream = ax_b.streamplot(
            x, y, qx, qy,
            start_points=starts,
            integration_direction="forward",
            color=FLOW,
            linewidth=1.15,
            arrowsize=1.05,
            arrowstyle="-|>",
            minlength=0.08,
            maxlength=6.0,
            broken_streamlines=False,
            zorder=2,
        )
        # Preserve the material boundary over the streamlines.
        ax_b.add_patch(
            Rectangle((45, 25), 30, 30, fill=False, edgecolor=INK,
                      linewidth=1.35, zorder=4)
        )
        style_axis(ax_b, en)
        ax_b.text(0.0, 1.04, "B", transform=ax_b.transAxes,
                  fontproperties=en, fontsize=14, fontweight="bold",
                  ha="left", va="bottom")
        ax_b.text(0.07, 1.04, "二维热流路径",
                  transform=ax_b.transAxes, fontproperties=cn_bold,
                  fontsize=12.5, ha="left", va="bottom")

        output_dir.mkdir(parents=True, exist_ok=True)
        svg = output_dir / "FT001-04_temperature_and_heatflow.svg"
        png = output_dir / "FT001-04_temperature_and_heatflow.png"
        fig.savefig(
            svg, bbox_inches="tight", pad_inches=0.05,
            metadata={"Date": None, "Creator": "Matplotlib / tutorial-001 FT001-04"}
        )
        fig.savefig(
            png, bbox_inches="tight", pad_inches=0.05, dpi=320,
            metadata={"Software": "Matplotlib / tutorial-001 FT001-04"}
        )
        plt.close(fig)

        print(f"Noto Sans CJK SC regular face index: {reg_index}")
        print(f"Noto Sans CJK SC bold face index: {bold_index}")
        print(f"q_hot={q_hot:.17g}")
        print(f"q_cold={q_cold:.17g}")
        print(f"max_abs_internal_y_heat_rate={max_y_rate:.17g}")
        print(f"temperature_min={float(np.min(temperature)):.17g}")
        print(f"temperature_max={float(np.max(temperature)):.17g}")
        print(svg)
        print(png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.cells, args.faces, args.summary, args.output_dir)


if __name__ == "__main__":
    main()
