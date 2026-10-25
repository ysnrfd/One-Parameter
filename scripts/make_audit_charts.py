#!/usr/bin/env python3
"""Evidence charts for the audit PDF (Template 07 blue family, charts.md rules)."""
from __future__ import annotations

import os

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Template 07 body palette (fixed by cover.md)
ACCENT = "#2d7ab3"
HEADER_FILL = "#1a4a7a"
BORDER = "#c0d0e2"
TEXT_PRIMARY = "#142840"
TEXT_MUTED = "#5a7a96"
CARD_BG = "#e4ecf5"

OUT = "/home/z/my-project/evidence/report_assets"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- chart 1 ---
# Series-panel data ink, before vs after the F-01 fix (percent coloured pixels)
frames = list(range(1, 11))
before = [0.0, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004]
after = [0.000, 0.000, 2.714, 3.046, 3.361, 3.484, 3.479, 3.485, 3.548, 3.579]

fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200, constrained_layout=True)
width = 0.38
bars_a = ax.bar([f - width / 2 for f in frames], before, width,
                label="1.0.0 (as uploaded)", color=BORDER, edgecolor="none")
bars_b = ax.bar([f + width / 2 for f in frames], after, width,
                label="1.1.0 (after fix)", color=ACCENT, edgecolor="none")
for rect, value in zip(bars_b, after, strict=True):
    if value > 0:
        ax.annotate(f"{value:.1f}", xy=(rect.get_x() + rect.get_width() / 2, value),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT_PRIMARY)
ax.set_xlabel("saved frame number (headless GUI dump)", fontsize=9, color=TEXT_PRIMARY)
ax.set_ylabel("data-ink fraction (%)", fontsize=9, color=TEXT_PRIMARY)
ax.set_xticks(frames)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(BORDER)
ax.spines["bottom"].set_color(BORDER)
ax.tick_params(colors=TEXT_MUTED, labelsize=8)
ax.grid(True, axis="y", linestyle="--", alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)
ax.set_ylim(0, 4.3)
ax.axhline(0.1, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
ax.annotate("assertion threshold 0.1%", xy=(1.02, 0.1), xytext=(-4, 5),
            textcoords="offset points", fontsize=7, color=TEXT_MUTED, va="bottom")
leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False,
                fontsize=8, ncol=2, handlelength=1.2)
fig.savefig(f"{OUT}/chart_ink.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 2 ---
# Findings by severity
labels = ["critical", "high", "medium", "low"]
counts = [2, 1, 5, 7]
colors = [HEADER_FILL, ACCENT, "#6b9bc3", BORDER]

fig, ax = plt.subplots(figsize=(6.8, 2.5), dpi=200, constrained_layout=True)
bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1], height=0.62,
               edgecolor="none")
for rect, value in zip(bars, counts[::-1], strict=True):
    ax.annotate(f"{value}", xy=(value, rect.get_y() + rect.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9, color=TEXT_PRIMARY)
ax.set_xlabel("findings", fontsize=9, color=TEXT_PRIMARY)
ax.set_xlim(0, 8.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(BORDER)
ax.spines["bottom"].set_color(BORDER)
ax.tick_params(colors=TEXT_MUTED, labelsize=9)
ax.grid(True, axis="x", linestyle="--", alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)
fig.savefig(f"{OUT}/chart_findings.png")
plt.close(fig)

print("charts written to", OUT)
