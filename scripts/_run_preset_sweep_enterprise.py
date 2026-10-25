#!/usr/bin/env python3
"""Phase 2c: run every preset end-to-end, export every artefact, and dump GUI frames.

Real execution sweep over the ORIGINAL (as-uploaded) code, producing evidence:
- one full run per preset (11 presets), each with the complete export bundle
- a headless live-dashboard session writing a multi-frame PNG dump
- a consolidated JSON summary of everything observed
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = "/home/z/my-project"
sys.path.insert(0, ROOT)

from single_parameter_observatory import exporters, visualization  # noqa: E402
from single_parameter_observatory.config import PRESETS, PRESET_DESCRIPTIONS  # noqa: E402
from single_parameter_observatory.training_engine import TrainingEngine  # noqa: E402

OUT = os.path.join(ROOT, "evidence", "preset_runs_enterprise")
FRAMES = os.path.join(ROOT, "evidence", "frames_gui_enterprise")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FRAMES, exist_ok=True)

summary: dict[str, dict] = {}

for name in sorted(PRESETS):
    config = PRESETS[name]
    started = time.time()
    entry: dict = {
        "preset": name,
        "description": PRESET_DESCRIPTIONS.get(name, ""),
        "optimizer": config.optimizer,
        "lr": config.lr,
        "steps_configured": config.steps,
    }
    try:
        engine = TrainingEngine(config)
        records = engine.run()
        report = engine.final_report()
        stability = engine.analyzer.report()
        export_dir = os.path.join(OUT, name)
        written = exporters.export_all(
            engine.recorder,
            export_dir,
            report=report,
            provenance=engine.provenance(),
            analyzer=engine.analyzer,
        )
        entry.update(
            {
                "steps_executed": len(records),
                "wall_seconds": round(time.time() - started, 2),
                "initial_parameter": report["initial_parameter"],
                "final_parameter": report["final_parameter"],
                "initial_loss": report["initial_loss"],
                "final_loss": report["final_loss"],
                "minimum_loss": report["minimum_loss"],
                "regime": stability["regime"],
                "chain_verified": report["chain_verified"],
                "max_reconstruction_ulp_error": report["max_reconstruction_ulp_error"],
                "validation_total_failures": report["validation"]["total_failures"],
                "nonfinite_steps": report["nonfinite_steps"],
                "aborted": engine.aborted,
                "abort_reason": engine.abort_reason,
                "unchanged_fp32_steps": report["unchanged_fp32_steps"],
                "below_resolution_steps": report["update_attempted_below_resolution_steps"],
                "total_ulp_movement": report["total_ulp_movement"],
                "export_files": sorted(k for k, v in written.items() if v),
                "status": "OK",
            }
        )
        # figures for two representative presets
        if name in ("adamw", "fp32_resolution", "diverging"):
            figdir = os.path.join(OUT, name, "figures")
            entry["figures"] = visualization.render_figures(
                engine.recorder, figdir, engine.analyzer
            )
    except Exception as exc:  # Evidence sweep must record failures.
        entry.update({"status": f"FAILED: {type(exc).__name__}: {exc}"})
    summary[name] = entry
    print(f"[{name}] {entry.get('status')} steps={entry.get('steps_executed')} "
          f"regime={entry.get('regime')} final_w={entry.get('final_parameter')} "
          f"loss={entry.get('final_loss')} t={entry.get('wall_seconds')}s", flush=True)

# ---- headless GUI frame dump -------------------------------------------------
from single_parameter_observatory import dashboard  # noqa: E402

gui_entry: dict = {}
try:
    config = PRESETS["adamw"].replace(steps=60)
    engine = TrainingEngine(config)
    panel = dashboard.LiveDashboard(
        engine,
        interactive=False,
        save_frames_directory=FRAMES,
        frame_every=4,          # save every 4th drawn frame -> a real dump
        target_fps=60.0,
        static_hz=2.0,
        data_hz=20.0,
    )
    panel.controller.set_speed(25.0)
    started = time.time()
    drawn = panel.run(max_seconds=12.0)
    panel.close()
    frames = sorted(os.listdir(FRAMES))
    gui_entry = {
        "frames_directory": FRAMES,
        "frames_written": len(frames),
        "frame_names": frames,
        "frames_drawn": drawn,
        "wall_seconds": round(time.time() - started, 2),
        "steps_recorded": len(engine.recorder),
        "status": "OK",
    }
except Exception as exc:
    gui_entry = {"status": f"FAILED: {type(exc).__name__}: {exc}"}
print(f"[gui] {gui_entry}", flush=True)

summary["_gui_frame_dump"] = gui_entry

with open(os.path.join(ROOT, "evidence", "preset_sweep_summary_enterprise.json"), "w") as handle:
    json.dump(summary, handle, indent=2)
print("WROTE evidence/preset_sweep_summary.json")
