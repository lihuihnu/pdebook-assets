#!/usr/bin/env python3
"""Generate tutorial-001 FT001-03 from frozen ET001-02 / ET001-03 data.

Panel A: layered-material analytic temperature profile + harmonic numerical points.
Panel B: arithmetic interface averaging heat-rate relative error for chi=10,100.\n\nPublication refinement: neutral panel titles, Chinese legend wording, and no machine-roundoff annotation inside the plotting area.

No PDE solve is performed. The script only reads committed CSV data.
Typography:
- Chinese: exact Noto Sans CJK SC face extracted from Noto CJK TTC by family name
- English/plain text: STIX
- Mathematics: Matplotlib STIX mathtext
"""

from __future__ import annotations

import argparse
import base64
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from fontTools.ttLib import TTCollection
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from PIL import Image
import svgwrite

CJK_REG_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_BOLD_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
STIX_REG = "/usr/share/fonts/opentype/stix-word/STIX-Regular.otf"

INK = "#171717"
GRID = "#D7D7D7"
LEFT_FILL = "#F4E7D8"
RIGHT_FILL = "#EAF2F8"
RED = "#C83A3A"
BLUE = "#3569B7"
GRAY = "#666666"

rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = False
rcParams["svg.fonttype"] = "none"
rcParams["svg.hashsalt"] = "tutorial-001-ft001-03"


def family_names(ttfont):
    names = set()
    for rec in ttfont["name"].names:
        if rec.nameID not in (1, 16):
            continue
        try:
            s = rec.toUnicode().strip()
        except Exception:
            continue
        if s:
            names.add(s)
    return names


def extract_family(ttc_path: str, family: str, out_path: Path) -> int:
    col = TTCollection(ttc_path)
    matches = [i for i, f in enumerate(col.fonts) if family in family_names(f)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {family!r} face, got {matches}")
    idx = matches[0]
    col.fonts[idx].save(out_path)
    return idx


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def style_axes(ax, en_font):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10.5, width=0.8, length=4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(en_font)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)


def render_panel_a(profile_csv: Path, out_svg: Path, out_png: Path, cn, cn_bold, en_font):
    rows = [r for r in read_csv(profile_csv) if int(r["nx"]) == 12 and int(r["ny"]) == 8]
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 profile rows for 12x8, got {len(rows)}")

    xs = [float(r["x"]) * 1000.0 for r in rows]
    ts = [float(r["temperature_mean_y"]) for r in rows]

    x_exact = [0.0, 60.0, 120.0]
    t_exact = [100.0, 27.27272727272727, 20.0]

    fig, ax = plt.subplots(figsize=(5.65, 4.55), dpi=220)
    style_axes(ax, en_font)

    ax.axvspan(0, 60, color=LEFT_FILL, alpha=0.55, zorder=0)
    ax.axvspan(60, 120, color=RIGHT_FILL, alpha=0.55, zorder=0)
    ax.axvline(60, color=GRAY, linestyle=(0, (4, 4)), linewidth=1.0, zorder=1)

    ax.plot(x_exact, t_exact, color=INK, linewidth=1.7, label="解析解", zorder=2)
    ax.plot(xs, ts, linestyle="none", marker="o", markersize=5.2,
            markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.25,
            label="调和平均数值点（12×8 网格）", zorder=3)
    ax.scatter([60], [27.27272727272727], s=32, color=RED, zorder=4)
    ax.annotate(r"$T_\Gamma=27.27^\circ\mathrm{C}$",
                xy=(60, 27.27272727272727), xytext=(68, 39),
                fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))

    ax.set_xlim(0, 120)
    ax.set_ylim(18, 103)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlabel("x / mm", fontproperties=en_font, fontsize=11.5)
    ax.set_ylabel(r"$T\,/\,^\circ\mathrm{C}$", fontsize=12)
    ax.text(0.0, 1.04, "A", transform=ax.transAxes, fontproperties=en_font,
            fontsize=14, fontweight="bold", ha="left", va="bottom")
    ax.text(0.07, 1.04, "两层材料温度剖面", transform=ax.transAxes,
            fontproperties=cn_bold, fontsize=12.5, ha="left", va="bottom")

    leg = ax.legend(frameon=False, loc="upper right", fontsize=9.5,
                    handlelength=2.0, borderaxespad=0.3)
    for txt in leg.get_texts():
        txt.set_fontproperties(cn)
        txt.set_fontsize(9.5)

    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.88)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.04,
                metadata={"Date": None, "Creator": "Matplotlib / tutorial-001 FT001-03A"})
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=320,
                metadata={"Software": "Matplotlib / tutorial-001 FT001-03A"})
    plt.close(fig)


def render_panel_b(ablation_csv: Path, out_svg: Path, out_png: Path, cn, cn_bold, en_font):
    rows = read_csv(ablation_csv)
    data = {}
    for r in rows:
        chi = float(r["contrast"])
        scheme = r["scheme"]
        if chi not in (10.0, 100.0):
            continue
        err_pct = float(r["q_hot_relative_error_physical"]) * 100.0
        if scheme == "arithmetic_ablation":
            data.setdefault(chi, []).append((int(r["nx"]), err_pct))

    for chi in data:
        data[chi].sort()
    if set(data) != {10.0, 100.0}:
        raise RuntimeError(f"missing arithmetic data: {sorted(data)}")

    fig, ax = plt.subplots(figsize=(5.65, 4.55), dpi=220)
    style_axes(ax, en_font)

    ax.set_xscale("log", base=2)
    x10, y10 = zip(*data[10.0])
    x100, y100 = zip(*data[100.0])
    ax.plot(x10, y10, color=BLUE, linewidth=1.7, marker="o", markersize=5.2,
            markerfacecolor="white", markeredgewidth=1.2,
            label=r"算术平均，$\chi=10$")
    ax.plot(x100, y100, color=RED, linewidth=1.7, linestyle="--", marker="s",
            markersize=5.0, markerfacecolor="white", markeredgewidth=1.2,
            label=r"算术平均，$\chi=100$")
    ax.axhline(0, color=INK, linewidth=1.0, linestyle=(0, (3, 3)))

    ax.set_xlim(5.2, 55)
    ax.set_ylim(-0.8, 21.0)
    ax.set_xticks([6, 12, 24, 48])
    ax.set_xticklabels(["6", "12", "24", "48"])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel(r"$N_x$", fontsize=12)
    ax.set_ylabel("总热流相对误差 / %", fontproperties=cn, fontsize=11.5)
    ax.text(0.0, 1.04, "B", transform=ax.transAxes, fontproperties=en_font,
            fontsize=14, fontweight="bold", ha="left", va="bottom")
    ax.text(0.07, 1.04, "算术平均的总热流误差", transform=ax.transAxes,
            fontproperties=cn_bold, fontsize=12.5, ha="left", va="bottom")

    leg = ax.legend(frameon=False, loc="upper right", fontsize=9.5,
                    handlelength=2.4, borderaxespad=0.3)
    for txt in leg.get_texts():
        txt.set_fontproperties(cn)
        txt.set_fontsize(9.5)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.15, top=0.88)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.04,
                metadata={"Date": None, "Creator": "Matplotlib / tutorial-001 FT001-03B"})
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=320,
                metadata={"Software": "Matplotlib / tutorial-001 FT001-03B"})
    plt.close(fig)


def embed_svg_data(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def combine_panels(a_svg: Path, b_svg: Path, a_png: Path, b_png: Path,
                   out_svg: Path, out_png: Path):
    ia = Image.open(a_png).convert("RGB")
    ib = Image.open(b_png).convert("RGB")
    gap = 40
    top = 16
    bottom = 16
    width = ia.width + ib.width + gap
    height = max(ia.height, ib.height) + top + bottom
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(ia, (0, top))
    canvas.paste(ib, (ia.width + gap, top))
    canvas.save(out_png, dpi=(320, 320), optimize=True)

    dwg = svgwrite.Drawing(str(out_svg), size=(f"{width}px", f"{height}px"),
                           viewBox=f"0 0 {width} {height}")
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))
    dwg.add(dwg.image(href=embed_svg_data(a_svg), insert=(0, top),
                      size=(ia.width, ia.height)))
    dwg.add(dwg.image(href=embed_svg_data(b_svg), insert=(ia.width + gap, top),
                      size=(ib.width, ib.height)))
    dwg.save(pretty=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="tutorial001-ft003-") as temp_dir:
        temp = Path(temp_dir)
        reg_otf = temp / "NotoSansCJK-SC-Regular.otf"
        bold_otf = temp / "NotoSansCJK-SC-Bold.otf"
        reg_idx = extract_family(CJK_REG_TTC, "Noto Sans CJK SC", reg_otf)
        bold_idx = extract_family(CJK_BOLD_TTC, "Noto Sans CJK SC", bold_otf)
        cn = FontProperties(fname=reg_otf)
        cn_bold = FontProperties(fname=bold_otf)
        en = FontProperties(fname=STIX_REG)

        a_svg, a_png = temp / "a.svg", temp / "a.png"
        b_svg, b_png = temp / "b.svg", temp / "b.png"
        render_panel_a(args.profile, a_svg, a_png, cn, cn_bold, en)
        render_panel_b(args.ablation, b_svg, b_png, cn, cn_bold, en)

        out_svg = args.output_dir / "FT001-03_interface_validation.svg"
        out_png = args.output_dir / "FT001-03_interface_validation.png"
        combine_panels(a_svg, b_svg, a_png, b_png, out_svg, out_png)

        print(f"Noto Sans CJK SC regular face index: {reg_idx}")
        print(f"Noto Sans CJK SC bold face index: {bold_idx}")
        print(out_svg)
        print(out_png)


if __name__ == "__main__":
    main()
