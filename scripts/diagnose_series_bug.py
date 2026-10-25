#!/usr/bin/env python3
"""Diagnose the empty-series defect in the dashboard's blit/save pipeline.

Reproduce headlessly, save frames mid-run, then introspect the figure state:
artist attachment, artist data, axes view limits, and what actually lands in
the saved PNG.
"""
from __future__ import annotations

import os
import sys

ROOT = "/home/z/my-project"
sys.path.insert(0, ROOT)

from single_parameter_observatory import dashboard  # noqa: E402
from single_parameter_observatory.config import PRESETS  # noqa: E402
from single_parameter_observatory.training_engine import TrainingEngine  # noqa: E402

OUT = "/home/z/my-project/evidence/diagnose_series"
os.makedirs(OUT, exist_ok=True)

config = PRESETS["adamw"].replace(steps=40)
engine = TrainingEngine(config)
panel = dashboard.LiveDashboard(
    engine,
    interactive=False,
    save_frames_directory=OUT,
    frame_every=2,
    target_fps=60.0,
    static_hz=1.0,
    data_hz=20.0,
)
panel.controller.set_speed(20.0)

def inspect(tag: str) -> None:
    ax = panel._ax_series
    print(f"--- {tag} ---")
    print(f"  records: {len(engine.recorder)}")
    print(f"  ax.lines: {len(ax.lines)}  ids={[id(line) for line in ax.lines][:6]}")
    for key in ("series:parameter_after", "series:loss"):
        line = panel._animated.get(key)
        if line is None:
            print(f"  {key}: MISSING")
            continue
        xs, ys = line.get_data()
        attached = line.axes is ax
        print(f"  {key}: n={len(xs)} attached_to_ax={attached} "
              f"first=({xs[:1]},{ys[:1]}) last=({xs[-1:]},{ys[-1:]})")
    print(f"  xlim={ax.get_xlim()}  ylim={ax.get_ylim()}  autoscalex={ax.get_autoscalex_on()} "
          f"autoscaley={ax.get_autoscaley_on()}")
    print(f"  datalim={ax.dataLim.bounds}")

panel.build_figure()
inspect("after build_figure (pre-run)")

# monkey-patch into the run loop: inspect at drawn==6 and drawn==14
original_run = panel.run
drawn_counter = {"n": 0}
original_frame = panel._frame

def frame_with_probe(**kw):
    original_frame(**kw)
    drawn_counter["n"] += 1
    if drawn_counter["n"] in (6, 14):
        inspect(f"after drawn frame #{drawn_counter['n']}")
        before = sorted(os.listdir(OUT))
        panel._save_frame()
        after = sorted(os.listdir(OUT))
        newf = [f for f in after if f not in before]
        if newf:
            print(f"  >>> saved probe frame {newf[-1]}")

panel._frame = frame_with_probe  # type: ignore[assignment]
drawn = panel.run(max_seconds=3.0)
panel.close()

inspect("after run() end (final refresh done)")
print(f"frames drawn={drawn}; files={sorted(os.listdir(OUT))}")

# --- isolated reproduction of the suspected save path -------------------------
import matplotlib as mpl  # noqa: E402

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

print("\n=== isolated savefig-after-clear-add_line test ===")
fig, ax = plt.subplots(figsize=(6, 4))
(line,) = ax.plot([], [], label="series")
ax.clear()
ax.set_title("after clear")
ax.add_line(line)
line.set_data([0, 1, 2, 3, 4], [0.0, 0.5, -0.5, 1.0, 0.2])
fig.canvas.draw()
print(f"  after canvas.draw(): xlim={ax.get_xlim()} ylim={ax.get_ylim()}")
fig.savefig(os.path.join(OUT, "isolated.png"), dpi=90)
print(f"  saved isolated.png; xlim now={ax.get_xlim()} ylim={ax.get_ylim()}")
