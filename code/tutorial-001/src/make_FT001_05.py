#!/usr/bin/env python3
"""Generate tutorial-001 FT001-05 from frozen ET001-06 contrast-sweep data.

The figure shows effective conductivity versus conductivity contrast chi=k_i/k_m
on a logarithmic x-axis. All nine committed parameter samples are plotted.
The connecting line is a visual guide only; it is not an analytic fit.

No PDE solve is performed.
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
from matplotlib import rcParams

CJK_REG_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_BOLD_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
STIX_REG = "/usr/share/fonts/opentype/stix-word/STIX-Regular.otf"

INK = "#171717"
GRID = "#D7D7D7"
BLUE = "#3569B7"
RED = "#C83A3A"
GRAY = "#6A6A6A"

rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = False
rcParams["svg.fonttype"] = "path"
rcParams["svg.hashsalt"] = "tutorial-001-ft001-05"


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


def make_figure(sweep_csv: Path, output_dir: Path):
    rows = read_csv(sweep_csv)
    rows = sorted(rows, key=lambda row: float(row["contrast"]))

    expected = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    contrasts = [float(row["contrast"]) for row in rows]
    if contrasts != expected:
        raise RuntimeError(f"unexpected contrast samples: {contrasts}")

    k_eff = [float(row["k_eff"]) for row in rows]
    q_hot = [float(row["q_hot"]) for row in rows]

    # Figure-level anchors already verified by ET001-06.
    chi1_index = contrasts.index(1.0)
    chi10_index = contrasts.index(10.0)
    if abs(k_eff[chi1_index] - 1.0) > 1e-9:
        raise RuntimeError("chi=1 effective conductivity anchor failed")
    if abs(q_hot[chi1_index] - 0.5333333333333333) / 0.5333333333333333 > 1e-9:
        raise RuntimeError("chi=1 heat-rate anchor failed")
    if abs(q_hot[chi10_index] - 0.6225462382844146) > 1e-12:
        raise RuntimeError("chi=10 ET001-04 anchor failed")

    with TemporaryDirectory(prefix="tutorial001-ft005-font-") as temp_dir:
        temp = Path(temp_dir)
        reg_otf = temp / "NotoSansCJK-SC-Regular.otf"
        bold_otf = temp / "NotoSansCJK-SC-Bold.otf"
        reg_index = extract_family(CJK_REG_TTC, "Noto Sans CJK SC", reg_otf)
        bold_index = extract_family(CJK_BOLD_TTC, "Noto Sans CJK SC", bold_otf)
        cn = FontProperties(fname=reg_otf)
        cn_bold = FontProperties(fname=bold_otf)
        en = FontProperties(fname=STIX_REG)

        fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=220)

        ax.set_xscale("log")
        ax.plot(
            contrasts,
            k_eff,
            color=BLUE,
            linewidth=1.7,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=BLUE,
            markeredgewidth=1.2,
            zorder=2,
        )

        # Uniform-material anchor.
        ax.axvline(1.0, color=GRAY, linewidth=1.0,
                   linestyle=(0, (4, 4)), zorder=1)
        ax.axhline(1.0, color=GRAY, linewidth=0.8,
                   linestyle=(0, (4, 4)), alpha=0.65, zorder=1)
        ax.scatter(
            [1.0], [k_eff[chi1_index]],
            s=56, marker="o", facecolor="white",
            edgecolor=RED, linewidth=1.7, zorder=4
        )
        ax.annotate(
            "均匀材料",
            xy=(1.0, k_eff[chi1_index]),
            xytext=(0.43, 1.035),
            textcoords="data",
            fontproperties=cn,
            fontsize=10.5,
            color=INK,
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8),
        )

        ax.set_xlim(0.075, 1350)
        ax.set_ylim(0.82, 1.23)
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
        ax.set_yticks([0.85, 0.95, 1.05, 1.15, 1.20])

        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(INK)
        ax.spines["bottom"].set_color(INK)
        ax.tick_params(colors=INK, labelsize=10.5, width=0.8, length=4)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(en)

        ax.set_xlabel(r"$\chi=k_i/k_m$", fontsize=13)
        ax.set_ylabel(r"$k_{\mathrm{eff}}\,/\,\mathrm{W/(m\cdot K)}$", fontsize=12.5)

        fig.subplots_adjust(left=0.14, right=0.975, bottom=0.16, top=0.88)

        output_dir.mkdir(parents=True, exist_ok=True)
        svg = output_dir / "FT001-05_contrast_sweep.svg"
        png = output_dir / "FT001-05_contrast_sweep.png"
        fig.savefig(
            svg, bbox_inches="tight", pad_inches=0.05,
            metadata={"Date": None, "Creator": "Matplotlib / tutorial-001 FT001-05"}
        )
        fig.savefig(
            png, bbox_inches="tight", pad_inches=0.05, dpi=320,
            metadata={"Software": "Matplotlib / tutorial-001 FT001-05"}
        )
        plt.close(fig)

        print(f"Noto Sans CJK SC regular face index: {reg_index}")
        print(f"Noto Sans CJK SC bold face index: {bold_index}")
        print(f"chi1_k_eff={k_eff[chi1_index]:.17g}")
        print(f"chi1_q_hot={q_hot[chi1_index]:.17g}")
        print(f"chi10_q_hot={q_hot[chi10_index]:.17g}")
        print(f"chi1000_k_eff={k_eff[-1]:.17g}")
        print(svg)
        print(png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.sweep, args.output_dir)


if __name__ == "__main__":
    main()
