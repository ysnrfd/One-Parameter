#!/usr/bin/env python3
"""Before/after series-panel comparison figure for the audit PDF.

Crops the 'recorded series (live)' panel (top-right quadrant) from the final
saved frame of the broken 1.0.0 GUI dump and of the fixed 1.1.0 dump, and
lays them out vertically with labels. Uses only Template 07 palette colors.
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt
from PIL import Image

fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TEXT_PRIMARY = "#142840"
TEXT_MUTED = "#5a7a96"
ACCENT = "#2d7ab3"
CARD_BG = "#e4ecf5"

OUT = "/home/z/my-project/evidence/report_assets/frame_compare.png"

# series panel region (top-right quadrant, incl. title, excl. neighbour axes)
CROP = dict(x0=0.515, x1=0.995, y0=0.045, y1=0.495)


def crop_series_panel(path: str):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.crop(
        (
            int(w * CROP["x0"]),
            int(h * CROP["y0"]),
            int(w * CROP["x1"]),
            int(h * CROP["y1"]),
        )
    )


before = crop_series_panel("/home/z/my-project/evidence/frames_gui/frame_00010.png")
after = crop_series_panel(
    "/home/z/my-project/evidence/frames_gui_enterprise/frame_00010.png"
)

# equalise widths (heights are both panel height)
target_w = min(before.width, after.width)


def resize_to_width(im, target_w):
    ratio = target_w / im.width
    return im.resize((target_w, int(im.height * ratio)), Image.Resampling.LANCZOS)


before = resize_to_width(before, target_w)
after = resize_to_width(after, target_w)

fig, axes = plt.subplots(
    2, 1, figsize=(8.6, 4.1), dpi=200, constrained_layout=True
)
for ax, img, label, sub in (
    (axes[0], before, "1.0.0 - as uploaded (frame 10 of 10)",
     "data-ink 0.004% - only the legend swatches carry colour"),
    (axes[1], after, "1.1.0 - after the fix (frame 10 of 10)",
     "data-ink 3.6% - four series, marker and trail render"),
):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"{label}   -   {sub}",
        fontsize=9,
        loc="left",
        color=TEXT_PRIMARY,
        pad=4,
    )
fig.savefig(OUT, facecolor="white")
plt.close(fig)
print("written", OUT)
