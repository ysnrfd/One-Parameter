#!/usr/bin/env python3
"""Run current presets and write fresh evidence, without replacing past audits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from single_parameter_observatory import __version__, exporters  # noqa: E402
from single_parameter_observatory.config import PRESETS, PRESET_DESCRIPTIONS  # noqa: E402
from single_parameter_observatory.recorder import Recorder, canonical_json, json_safe  # noqa: E402
from single_parameter_observatory.training_engine import TrainingEngine  # noqa: E402


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "single_parameter_observatory").glob("*.py"))
    paths.extend([ROOT / "pyproject.toml", Path(__file__).resolve()])
    return {path.relative_to(ROOT).as_posix(): _hash(path) for path in paths}


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON numeric token {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _verify_exports(engine: TrainingEngine, paths: dict[str, str | None]) -> dict[str, Any]:
    required = set(exporters.EXPORT_FILES) - {"trajectory.parquet"}
    missing = sorted(name for name in required if not paths.get(name))
    if missing:
        raise RuntimeError(f"missing mandatory exports: {missing}")
    artifacts = {}
    for name, value in paths.items():
        if value is None:
            continue
        path = Path(value)
        if not path.is_file():
            raise RuntimeError(f"reported artifact does not exist: {name}: {value}")
        if path.suffix == ".json":
            _strict_json(path)
        artifacts[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _hash(path)}

    trajectory = _strict_json(Path(str(paths["trajectory.json"])))
    records = engine.recorder.records
    if len(trajectory["records"]) != len(records):
        raise RuntimeError("exported JSON record count differs from the recorder")
    predecessor = Recorder.GENESIS_HASH
    for position, record in enumerate(trajectory["records"]):
        if record["step"] != position or record["previous_hash"] != predecessor:
            raise RuntimeError(f"exported JSON chain is broken at position {position}")
        payload = {key: value for key, value in record.items() if key != "record_hash"}
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        if digest != record["record_hash"]:
            raise RuntimeError(f"exported JSON content hash differs at step {position}")
        predecessor = digest

    with Path(str(paths["trajectory.csv"])).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(records):
        raise RuntimeError("exported CSV record count differs from the recorder")
    for row, record in zip(rows, records, strict=True):
        if (float(row["parameter_after"]).hex() != record.parameter_after.hex()
                or int(row["parameter_after_uint32"]) != record.parameter_after_uint32):
            raise RuntimeError(f"exported CSV parameter differs at step {record.step}")

    import numpy as np  # noqa: PLC0415 - verify the actual saved archive

    with np.load(str(paths["trajectory.npz"]), allow_pickle=False) as archive:
        if archive["parameter_after_uint32"].tolist() != [record.parameter_after_uint32 for record in records]:
            raise RuntimeError("exported NPZ parameter words differ from the recorder")
    return artifacts


def _run_preset(name: str, destination: Path, steps: int | None, figures: bool) -> dict[str, Any]:
    preset = PRESETS[name]
    config = preset if steps is None else preset.replace(steps=steps)
    started = time.perf_counter()
    entry: dict[str, Any] = {
        "preset": name, "description": PRESET_DESCRIPTIONS.get(name, ""),
        "config": config.to_dict(), "step_override": steps,
    }
    try:
        engine = TrainingEngine(config)
        records = engine.run()
        report = engine.final_report()
        if not records or not engine.recorder.verify_chain()[0]:
            raise RuntimeError("the run did not produce a valid recorded history")
        if report["validation"]["total_failures"]:
            raise RuntimeError(f"numerical validation failed: {report['validation']}")
        if not engine.aborted and len(records) != config.steps:
            raise RuntimeError("the run stopped without completion or an explicit abort")
        if engine.aborted and not report["nonfinite_detected"]:
            raise RuntimeError("the run aborted without a recorded nonfinite observation")
        if steps is None and engine.aborted != (name == "diverging"):
            raise RuntimeError("preset abort behavior differs from its documented purpose")
        paths = exporters.export_all(
            engine.recorder, str(destination), report=report,
            provenance=engine.provenance(), analyzer=engine.analyzer,
        )
        artifacts = _verify_exports(engine, paths)
        entry.update({
            "status": "OK", "steps_executed": len(records),
            "aborted": engine.aborted, "abort_reason": engine.abort_reason,
            "report": report, "artifacts": artifacts,
            "optional_exports_unavailable": [name for name, path in paths.items() if path is None],
        })
        if figures:
            from single_parameter_observatory import visualization  # noqa: PLC0415 - opt-in plotting

            entry["figures"] = visualization.render_figures(engine.recorder, str(destination / "figures"), engine.analyzer)
    except Exception as exc:  # evidence must include every preset, even after a failure
        entry.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    entry["wall_seconds"] = time.perf_counter() - started
    return entry


def _run_gui(destination: Path, seconds: float) -> dict[str, Any]:
    from single_parameter_observatory.dashboard import LiveDashboard  # noqa: PLC0415 - opt-in UI evidence

    panel = None
    try:
        engine = TrainingEngine(PRESETS["adamw"].replace(steps=20))
        engine.run(max_steps=2)
        panel = LiveDashboard(
            engine, interactive=False, save_frames_directory=str(destination),
            frame_every=1, target_fps=30.0, static_hz=2.0, data_hz=10.0,
        )
        panel.controller.set_speed(20.0)
        drawn = panel.run(max_seconds=seconds)
        frames = sorted(destination.glob("*.png"))
        if not frames:
            raise RuntimeError("headless GUI session did not write any frames")
        return {"status": "OK", "frames_drawn": drawn, "steps_recorded": len(engine.recorder),
                "frames": [{"path": str(path), "sha256": _hash(path)} for path in frames]}
    except Exception as exc:
        return {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    finally:
        if panel is not None:
            panel.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "repair_20261023" / "preset_sweep")
    parser.add_argument("--presets", nargs="+", choices=sorted(PRESETS), default=sorted(PRESETS))
    parser.add_argument("--steps", type=_positive_int, help="explicitly override preset step counts for a smoke run")
    parser.add_argument("--figures", action="store_true", help="render figures for each preset")
    parser.add_argument("--gui", action="store_true", help="include a headless dashboard frame dump")
    parser.add_argument("--gui-seconds", type=_positive_float, default=5.0)
    args = parser.parse_args(argv)
    destination = args.output.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error(f"output must be a new or empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scope": "current checkout; full preset runs" if args.steps is None else "current checkout; smoke runs with explicit step override",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "version": __version__, "python": sys.version, "platform": platform.platform(),
        "checkout": str(ROOT), "source_hashes_start": _source_hashes(), "presets": {},
    }
    for name in dict.fromkeys(args.presets):
        entry = _run_preset(name, destination / name, args.steps, args.figures)
        summary["presets"][name] = entry
        print(f"[{name}] {entry['status']} steps={entry.get('steps_executed')} elapsed={entry['wall_seconds']:.2f}s", flush=True)
    if args.gui:
        summary["gui"] = _run_gui(destination / "frames_gui", args.gui_seconds)
        print(f"[gui] {summary['gui']['status']}", flush=True)
    summary["source_hashes_end"] = _source_hashes()
    summary["source_changed_during_run"] = summary["source_hashes_start"] != summary["source_hashes_end"]
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    passed = (all(entry["status"] == "OK" for entry in summary["presets"].values())
              and summary.get("gui", {"status": "OK"})["status"] == "OK"
              and not summary["source_changed_during_run"])
    summary["status"] = "PASS" if passed else "FAIL"
    path = destination / "summary.json"
    path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"{summary['status']}: {path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
