#!/usr/bin/env python3
"""Generate tutorial-001 FT001-01 using the exact Simplified Chinese glyph face.

Typography:
- Chinese: Noto Sans CJK SC selected by TTC family name (verified face index 2)
- English/plain dimensions: STIX
- Mathematics: Matplotlib STIX mathtext

Temporary extracted font files are deleted automatically and are never
committed or distributed.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
import numpy as np


from pathlib import Path
from fontTools.ttLib import TTCollection

def _family_names(ttfont):
    names = set()
    for record in ttfont["name"].names:
        if record.nameID not in (1, 16):
            continue
        try:
            value = record.toUnicode().strip()
        except Exception:
            continue
        if value:
            names.add(value)
    return names

def extract_ttc_family(ttc_path: str, target_family: str, output_path: Path) -> int:
    """Extract exactly one named family face from a TTC into a temporary OTF."""
    collection = TTCollection(ttc_path)
    matches = []
    for index, font in enumerate(collection.fonts):
        if target_family in _family_names(font):
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {target_family!r} face in {ttc_path}, got {matches}"
        )
    index = matches[0]
    collection.fonts[index].save(output_path)
    return index


HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent / "exports"
EXPORT.mkdir(parents=True, exist_ok=True)
SVG = EXPORT / "FT001-01_composite_plate_geometry.svg"
PNG = EXPORT / "FT001-01_composite_plate_geometry.png"

CJK_REG_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_BOLD_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
STIX_REG = "/usr/share/fonts/opentype/stix-word/STIX-Regular.otf"

LX, LY = 120.0, 80.0
X0, X1 = 45.0, 75.0
Y0, Y1 = 25.0, 55.0

MATRIX = "#F2F6F8"
INCLUSION = "#F4E7D8"
HOT = "#C83A3A"
COLD = "#3569B7"
GUIDE = "#777777"
INK = "#171717"

rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = False
rcParams["svg.fonttype"] = "none"
rcParams["svg.hashsalt"] = "tutorial-001-ft001-01-sc"

def add_dimension(ax, en_font, p0, p1, label, xy, rotation=0, fs=12):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="<->", mutation_scale=10.5,
            linewidth=0.95, color=INK, shrinkA=0, shrinkB=0, zorder=10,
        )
    )
    ax.text(
        *xy, label, fontproperties=en_font, fontsize=fs, rotation=rotation,
        ha="center", va="center", color=INK, zorder=11,
    )

def make_figure() -> tuple[int, int]:
    with TemporaryDirectory(prefix="tutorial001-cjk-") as temp_dir:
        temp = Path(temp_dir)
        reg_otf = temp / "NotoSansCJK-SC-Regular.otf"
        bold_otf = temp / "NotoSansCJK-SC-Bold.otf"
        reg_index = extract_ttc_family(CJK_REG_TTC, "Noto Sans CJK SC", reg_otf)
        bold_index = extract_ttc_family(CJK_BOLD_TTC, "Noto Sans CJK SC", bold_otf)

        cn = FontProperties(fname=reg_otf)
        cn_bold = FontProperties(fname=bold_otf)
        en = FontProperties(fname=STIX_REG)

        fig, ax = plt.subplots(figsize=(11.2, 7.5), dpi=200)
        ax.set_xlim(-19, 147)
        ax.set_ylim(-15, 100)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        ax.add_patch(Rectangle((0, 0), LX, LY, facecolor=MATRIX,
                               edgecolor=INK, linewidth=1.45, zorder=1))
        ax.add_patch(Rectangle((X0, Y0), X1-X0, Y1-Y0, facecolor=INCLUSION,
                               edgecolor=INK, linewidth=1.55, zorder=3))

        ax.plot([0, 0], [0, LY], color=HOT, lw=3.0, solid_capstyle="butt", zorder=5)
        ax.plot([LX, LX], [0, LY], color=COLD, lw=3.0, solid_capstyle="butt", zorder=5)

        for x in np.arange(3.0, LX - 1.0, 5.2):
            ax.plot([x - 1.7, x + 0.5], [LY, LY + 2.25], color=INK, lw=0.82, zorder=6)
            ax.plot([x - 1.7, x + 0.5], [0, -2.25], color=INK, lw=0.82, zorder=6)

        ax.plot([0, 0], [LY + 1.0, 92.0], color=INK, lw=0.75)
        ax.plot([LX, LX], [LY + 1.0, 92.0], color=INK, lw=0.75)
        add_dimension(ax, en, (0, 90), (LX, 90), "120 mm", (LX / 2, 94), fs=13)

        ax.plot([LX + 2.0, 139.0], [0, 0], color=INK, lw=0.75)
        ax.plot([LX + 2.0, 139.0], [LY, LY], color=INK, lw=0.75)
        add_dimension(ax, en, (137, 0), (137, LY), "80 mm",
                      (141.5, LY / 2), rotation=90, fs=13)

        ax.plot([X0, X0], [Y1 + 1.2, Y1 + 7.9], color=INK, lw=0.7)
        ax.plot([X1, X1], [Y1 + 1.2, Y1 + 7.9], color=INK, lw=0.7)
        add_dimension(ax, en, (X0, Y1 + 5.9), (X1, Y1 + 5.9), "30 mm",
                      ((X0 + X1) / 2, Y1 + 9.0), fs=11.5)

        ax.plot([X1 + 1.2, X1 + 7.8], [Y0, Y0], color=INK, lw=0.7)
        ax.plot([X1 + 1.2, X1 + 7.8], [Y1, Y1], color=INK, lw=0.7)
        add_dimension(ax, en, (X1 + 6.0, Y0), (X1 + 6.0, Y1), "30 mm",
                      (X1 + 9.9, (Y0 + Y1) / 2), rotation=90, fs=11.5)

        guide_kw = dict(color=GUIDE, lw=0.75, linestyle=(0, (4, 4)), zorder=2)
        ax.plot([X0, X0], [-0.8, Y0], **guide_kw)
        ax.plot([X1, X1], [-0.8, Y0], **guide_kw)
        ax.plot([35.8, X0], [Y0, Y0], **guide_kw)
        ax.plot([35.8, X0], [Y1, Y1], **guide_kw)

        ax.text(X0, -4.8, r"$x=45\ \mathrm{mm}$", fontsize=11.2,
                ha="center", va="top", color=INK)
        ax.text(X1, -4.8, r"$x=75\ \mathrm{mm}$", fontsize=11.2,
                ha="center", va="top", color=INK)
        ax.text(33.3, Y0, r"$y=25\ \mathrm{mm}$", fontsize=11.2,
                ha="right", va="center", color=INK)
        ax.text(33.3, Y1, r"$y=55\ \mathrm{mm}$", fontsize=11.2,
                ha="right", va="center", color=INK)

        ax.text(23.0, 66.0, "基体", fontproperties=cn_bold, fontsize=14.5,
                ha="center", va="center", color=INK)
        ax.text(23.0, 58.7, r"$k_m=1\ \mathrm{W/(m\cdot K)}$",
                fontsize=12.0, ha="center", va="center", color=INK)

        ax.text((X0 + X1) / 2, 43.8, "嵌入体", fontproperties=cn_bold,
                fontsize=14.5, ha="center", va="center", color=INK)
        ax.text((X0 + X1) / 2, 36.5, r"$k_i=10\ \mathrm{W/(m\cdot K)}$",
                fontsize=12.0, ha="center", va="center", color=INK)

        ax.text(-3.0, LY / 2, r"$T_H=100^\circ\mathrm{C}$",
                fontsize=13.5, ha="right", va="center", color=HOT)
        ax.text(LX + 3.0, LY / 2, r"$T_C=20^\circ\mathrm{C}$",
                fontsize=13.5, ha="left", va="center", color=COLD)
        ax.text(LX / 2, LY + 4.45, "绝热边界", fontproperties=cn,
                fontsize=11.8, ha="center", va="bottom", color=INK)
        ax.text(LX / 2, -4.45, "绝热边界", fontproperties=cn,
                fontsize=11.8, ha="center", va="top", color=INK)

        ax.scatter([0], [0], s=15, color=INK, zorder=12)
        ax.text(-1.9, -2.0, r"$O$", fontsize=12.0, ha="right", va="top", color=INK)
        ax.text(12.0, -2.4, r"$x$", fontsize=12.0, ha="center", va="top", color=INK)
        ax.text(-2.3, 12.0, r"$y$", fontsize=12.0, ha="right", va="center", color=INK)

        fig.subplots_adjust(left=0.018, right=0.986, bottom=0.035, top=0.985)
        fig.savefig(SVG, bbox_inches="tight", pad_inches=0.05,
                    metadata={"Date": None, "Creator": "Matplotlib / tutorial-001 FT001-01"})
        fig.savefig(PNG, bbox_inches="tight", pad_inches=0.05, dpi=320,
                    metadata={"Software": "Matplotlib / tutorial-001 FT001-01"})
        plt.close(fig)
        return reg_index, bold_index

if __name__ == "__main__":
    regular_index, bold_index = make_figure()
    print(f"Noto Sans CJK SC regular face index: {regular_index}")
    print(f"Noto Sans CJK SC bold face index: {bold_index}")
    print(SVG)
    print(PNG)
