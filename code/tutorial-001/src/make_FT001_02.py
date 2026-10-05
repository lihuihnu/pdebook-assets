#!/usr/bin/env python3
"""Generate tutorial-001 FT001-02 with exact Simplified Chinese glyph face.

The figure shows P -> f -> N, correct center-to-face distances d_P and d_N,
one shared heat flux F_f, and two half-cell series thermal resistances.

Typography:
- SVG text policy: all glyphs are converted to vector paths so downstream rasterizers cannot substitute CJK variants
- Chinese: Noto Sans CJK SC selected by TTC family name (verified face index 2)
- English/plain text: STIX
- Mathematics: Matplotlib STIX mathtext
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams


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
SVG = EXPORT / "FT001-02_series_resistance.svg"
PNG = EXPORT / "FT001-02_series_resistance.png"

CJK_REG_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_BOLD_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

INK = "#171717"
LEFT_FILL = "#F4E7D8"
RIGHT_FILL = "#EAF2F8"
FLUX = "#C83A3A"

rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = False
rcParams["svg.fonttype"] = "path"
rcParams["svg.hashsalt"] = "tutorial-001-ft001-02-sc"

def make_figure() -> tuple[int, int]:
    with TemporaryDirectory(prefix="tutorial001-cjk-") as temp_dir:
        temp = Path(temp_dir)
        reg_otf = temp / "NotoSansCJK-SC-Regular.otf"
        bold_otf = temp / "NotoSansCJK-SC-Bold.otf"
        reg_index = extract_ttc_family(CJK_REG_TTC, "Noto Sans CJK SC", reg_otf)
        bold_index = extract_ttc_family(CJK_BOLD_TTC, "Noto Sans CJK SC", bold_otf)
        cn_bold = FontProperties(fname=bold_otf)

        fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=220)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis("off")

        xL, xF, xR = 1.1, 7.0, 12.9
        yB, yT = 3.0, 7.8
        xP, xN = 4.05, 9.95
        yC = 5.45

        ax.add_patch(Rectangle((xL, yB), xF-xL, yT-yB,
                               facecolor=LEFT_FILL, edgecolor=INK, linewidth=1.45))
        ax.add_patch(Rectangle((xF, yB), xR-xF, yT-yB,
                               facecolor=RIGHT_FILL, edgecolor=INK, linewidth=1.45))
        ax.plot([xF, xF], [yB, yT], color=INK, lw=1.8)

        ax.annotate("", xy=(xF, yT + 0.62), xytext=(xF, yT + 1.35),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=INK))
        ax.text(xF, yT + 1.55, "共享面", fontproperties=cn_bold, fontsize=14.5,
                ha="center", va="bottom", color=INK)
        ax.text(xF, yT + 0.38, r"$f$", fontsize=17, ha="center", va="center", color=INK)

        ax.text(xL + 0.42, yT - 0.42, "控制体", fontproperties=cn_bold,
                fontsize=14.5, ha="left", va="top", color=INK)
        ax.text(xL + 1.57, yT - 0.42, r"$P$", fontsize=18, ha="left", va="top", color=INK)
        ax.text(xR - 1.96, yT - 0.42, "控制体", fontproperties=cn_bold,
                fontsize=14.5, ha="left", va="top", color=INK)
        ax.text(xR - 0.86, yT - 0.42, r"$N$", fontsize=18, ha="left", va="top", color=INK)

        ax.text((xL+xF)/2, 6.38, r"$k_P$", fontsize=24,
                ha="center", va="center", color=INK)
        ax.text((xF+xR)/2, 6.38, r"$k_N$", fontsize=24,
                ha="center", va="center", color=INK)

        ax.scatter([xP, xF, xN], [yC, yC, yC], s=[70,70,70], color=INK, zorder=5)
        ax.text(xP, yC - 0.30, r"$P$", fontsize=20, ha="center", va="top", color=INK)
        ax.text(xP, yC - 0.95, r"$T_P$", fontsize=20, ha="center", va="top", color=INK)
        ax.text(xN, yC - 0.30, r"$N$", fontsize=20, ha="center", va="top", color=INK)
        ax.text(xN, yC - 0.95, r"$T_N$", fontsize=20, ha="center", va="top", color=INK)

        ax.text(xF - 0.33, yC + 0.63, r"$T_f$", fontsize=22,
                ha="center", va="center", color=INK)
        ax.add_patch(FancyArrowPatch((xP + 0.42, yC), (xN - 0.42, yC),
                                     arrowstyle="-|>", mutation_scale=18,
                                     linewidth=2.3, color=FLUX, zorder=4))
        ax.text(xF + 0.88, yC + 0.60, r"$F_f$", fontsize=24,
                ha="center", va="center", color=FLUX)

        yDim = 3.58
        cap_h = 0.38
        for x in [xP, xF, xN]:
            ax.plot([x, x], [yDim-cap_h, yDim+cap_h], color=INK, lw=1.0)

        def dim_segment(x0, x1, label):
            ax.add_patch(FancyArrowPatch((x0, yDim), (x1, yDim),
                                         arrowstyle="<->", mutation_scale=11,
                                         linewidth=1.0, color=INK,
                                         shrinkA=0, shrinkB=0))
            ax.text((x0+x1)/2, yDim + 0.33, label, fontsize=22,
                    ha="center", va="center", color=INK)

        dim_segment(xP, xF, r"$d_P$")
        dim_segment(xF, xN, r"$d_N$")

        bar_y0, bar_y1 = 1.55, 2.05
        ax.add_patch(Rectangle((xP, bar_y0), xF-xP, bar_y1-bar_y0,
                               facecolor=LEFT_FILL, edgecolor=INK, linewidth=1.1))
        ax.add_patch(Rectangle((xF, bar_y0), xN-xF, bar_y1-bar_y0,
                               facecolor=RIGHT_FILL, edgecolor=INK, linewidth=1.1))
        for x in [xP, xF, xN]:
            ax.plot([x, x], [bar_y0 - 0.22, bar_y1 + 0.22], color=INK, lw=1.2)

        ax.text(xP, bar_y1 + 0.42, r"$P$", fontsize=16, ha="center", va="bottom", color=INK)
        ax.text(xF, bar_y1 + 0.42, r"$f$", fontsize=16, ha="center", va="bottom", color=INK)
        ax.text(xN, bar_y1 + 0.42, r"$N$", fontsize=16, ha="center", va="bottom", color=INK)

        ax.text((xP+xF)/2, 0.72, r"$R_P=\dfrac{d_P}{k_P A_f}$",
                fontsize=23, ha="center", va="center", color=INK)
        ax.text((xF+xN)/2, 0.72, r"$R_N=\dfrac{d_N}{k_N A_f}$",
                fontsize=23, ha="center", va="center", color=INK)

        fig.subplots_adjust(left=0.03, right=0.985, bottom=0.05, top=0.97)
        fig.savefig(SVG, bbox_inches="tight", pad_inches=0.06,
                    metadata={"Date": None, "Creator": "Matplotlib / tutorial-001 FT001-02"})
        fig.savefig(PNG, bbox_inches="tight", pad_inches=0.06, dpi=320,
                    metadata={"Software": "Matplotlib / tutorial-001 FT001-02"})
        plt.close(fig)
        return reg_index, bold_index

if __name__ == "__main__":
    regular_index, bold_index = make_figure()
    print(f"Noto Sans CJK SC regular face index: {regular_index}")
    print(f"Noto Sans CJK SC bold face index: {bold_index}")
    print(SVG)
    print(PNG)
