#!/usr/bin/env python3
"""Objective pixel-level analysis of the series panel in every frame.

The dashboard figure is 2x2 (15x9in @110dpi), saved at dpi=90 bbox_inches=tight
(~1345x798). The 'recorded series (live)' panel is the top-right quadrant.
We measure colored (non-gray, non-white) pixel density inside that panel's plot
interior: a panel containing 4 colored series lines + purple marker has high
density; an empty panel has only the gray grid.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

def analyze(path: str) -> dict:
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    h, w, _ = im.shape
    # top-right quadrant, inset to avoid titles/legend/axis labels
    x0, x1 = int(w * 0.53), int(w * 0.985)
    y0, y1 = int(h * 0.09), int(h * 0.47)
    region = im[y0:y1, x0:x1]
    mx = region.max(axis=2)
    mn = region.min(axis=2)
    sat = mx - mn  # colorfulness
    colored = (sat > 40) & (mx > 60)          # strongly colored pixels (series lines, purple marker)
    whiteish = (mn > 200)
    dark = (mx < 120)
    total = region.shape[0] * region.shape[1]
    return {
        "frame": os.path.basename(path),
        "dir": os.path.basename(os.path.dirname(path)),
        "colored_pct": round(100.0 * colored.sum() / total, 3),
        "white_pct": round(100.0 * whiteish.sum() / total, 1),
        "dark_pct": round(100.0 * dark.sum() / total, 2),
    }

rows = []
for pattern in (
    "/home/z/my-project/evidence/original/frames/*.png",
    "/home/z/my-project/evidence/original/run_checks_frames/*.png",
    "/home/z/my-project/evidence/frames_gui/*.png",
    "/home/z/my-project/evidence/diagnose_series/*.png",
):
    for path in sorted(glob.glob(pattern)):
        rows.append(analyze(path))

print(f"{'frame':<42}{'colored%':>10}{'white%':>8}{'dark%':>7}   verdict")
print("-" * 80)
for r in rows:
    # A healthy panel with 4 series + marker typically has >0.15% strongly colored
    # pixels; an empty grid-only panel has <0.03%.
    verdict = "SERIES VISIBLE" if r["colored_pct"] >= 0.10 else "EMPTY/BROKEN"
    print(f"{r['dir'] + '/' + r['frame']:<42}{r['colored_pct']:>10}{r['white_pct']:>8}{r['dark_pct']:>7}   {verdict}")
