#!/usr/bin/env python3
"""Reproduce and pinpoint the diverged-run export failure (audit evidence)."""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/z/my-project"
sys.path.insert(0, ROOT)

from single_parameter_observatory import exporters  # noqa: E402
from single_parameter_observatory.config import PRESETS  # noqa: E402
from single_parameter_observatory.training_engine import TrainingEngine  # noqa: E402

engine = TrainingEngine(PRESETS["diverging"])
records = engine.run()
report = engine.final_report()
print(f"diverging run: aborted={engine.aborted}, records={len(records)}")
print(f"report final_loss={report['final_loss']!r}, max_abs_gradient={report['max_abs_gradient']!r}")

# Which top-level report values are non-finite?
nonfinite = {k: v for k, v in report.items()
             if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf")))}
print(f"non-finite report fields: {nonfinite}")

# Walk each exporter individually
out = "/home/z/my-project/evidence/diverging_export_probe"
os.makedirs(out, exist_ok=True)
probes = {
    "summary.json": lambda: exporters.export_summary_json(
        engine.recorder.config, engine.recorder, report, os.path.join(out, "summary.json")),
    "trajectory.csv": lambda: exporters.export_trajectory_csv(
        engine.recorder, os.path.join(out, "trajectory.csv")),
    "trajectory.json": lambda: exporters.export_trajectory_json(
        engine.recorder, os.path.join(out, "trajectory.json"), report=report),
    "bit_history.csv": lambda: exporters.export_bit_history_csv(
        engine.recorder, os.path.join(out, "bit_history.csv")),
    "optimizer_state_history.csv": lambda: exporters.export_optimizer_state_history_csv(
        engine.recorder, os.path.join(out, "optimizer_state_history.csv")),
    "landscape.json": lambda: exporters.export_landscape_json(
        engine.recorder, os.path.join(out, "landscape.json")),
    "events.csv": lambda: exporters.export_events_csv(
        engine.recorder, os.path.join(out, "events.csv")),
    "validation.json": lambda: exporters.export_validation_json(
        engine.recorder, os.path.join(out, "validation.json")),
}
for name, fn in probes.items():
    try:
        fn()
        print(f"  OK      {name}")
    except Exception as exc:
        print(f"  FAIL    {name}: {type(exc).__name__}: {exc}")

# Demonstrate root cause: default=str does not fire for floats
try:
    json.dump({"x": float("inf")}, open(os.devnull, "w"), allow_nan=False, default=str)
except ValueError as exc:
    print(f"\nroot cause confirmed: json.dump(allow_nan=False) raises on inf even with default=str: {exc}")

# And where _jsonable IS applied (recorder canonical path) it works:
from single_parameter_observatory.recorder import _jsonable  # noqa: E402
print(f"_jsonable(inf) = {_jsonable(float('inf'))!r}  (stringified -> JSON-safe)")
