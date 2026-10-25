#!/usr/bin/env python3
"""Run every executable check in one command, and write what actually happened.

    python run_checks.py

This script does not assert that anything works.  It runs the checks, reports
the observed outcome verbatim, and writes the full log next to itself.  If a
check fails, this file says so; there is no path through it that produces a
passing summary that did not happen.

Checks, in the order they execute and in the order they are numbered:
1. every module imports;
2. the environment audit (Python, PyTorch, NumPy, Matplotlib, platform);
3. the automated test suite (``unittest``, no extra runner required);
4. a real training run, with the final numerical report and the measured
   update semantics for AdamW *and* SGD + Nesterov;
5. a full export, with the artefact list;
6. a real per-step forensic trace;
7. the control session (play / pause / step / run-N / reset / restart);
8. the FP32-resolution preset, which is the case where the optimizer attempts
   an update the stored float32 value cannot represent;
9. the other documented CLI surfaces (inspect, presets, live --headless) and
   the remaining behaviour-sensitive presets (diverging, near_stationary);
10. a long enough default-preset run for the classifier's near-zero-loss guard to
    be exercised, reported verbatim and not asserted;
11. the achieved frame rate of the live view, recorded rather than asserted.

Exit code 0 only if every check completed without an error.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import traceback
from typing import Any, Callable

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PACKAGE_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

LOG_PATH = os.path.join(_PACKAGE_DIR, "checks_output.txt")

#: Total number of checks, used for the ``CHECK n/N`` headers.
CHECK_COUNT = 11

#: Frame rate the live view is measured at in check 11, and the floor below which
#: the view counts as broken rather than merely slow.
DEFAULT_LIVE_FPS = 120.0
MINIMUM_USABLE_FPS = 30.0
DEFAULT_STATIC_HZ = 1.0
DEFAULT_DATA_HZ = 20.0

#: Step count for the near-zero-loss guard check.  Measured, not chosen: on the
#: default AdamW preset the loss reaches its minimum at step 95 and then climbs
#: back, and the classifier's ``divergence_absolute`` guard first fires at 103
#: steps (the loss ratio over the final window crosses 1.05 between 102 and 103).
#: 103 is the smallest count that exercises the guard, so it is the count used.
GUARD_STEPS = 103

MODULE_NAMES = (
    "single_parameter_observatory.config",
    "single_parameter_observatory.errors",
    "single_parameter_observatory.fp32_inspector",
    "single_parameter_observatory.numerical_analysis",
    "single_parameter_observatory.model",
    "single_parameter_observatory.optimizer_inspector",
    "single_parameter_observatory.tracker",
    "single_parameter_observatory.validation",
    "single_parameter_observatory.recorder",
    "single_parameter_observatory.convergence",
    "single_parameter_observatory.exporters",
    "single_parameter_observatory.training_engine",
    "single_parameter_observatory.visualization",
    "single_parameter_observatory.dashboard",
    "single_parameter_observatory.main",
)


class Report:
    """Collects the observed outcome of every check."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.started = time.time()

    def say(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.say()
        self.say("=" * 78)
        self.say(title)
        self.say("=" * 78)

    def check_section(self, number: int, title: str) -> None:
        self.section(f"CHECK {number}/{CHECK_COUNT}  {title}")

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")
        self.say(f"  !! {name} FAILED: {detail}")

    def ok(self, name: str) -> None:
        self.say(f"  ok  {name}")

    def write(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.lines) + "\n")
        return path


# --------------------------------------------------------------------------- #
# 1
# --------------------------------------------------------------------------- #


def check_imports(report: Report) -> bool:
    import importlib

    report.check_section(1, "module imports")
    healthy = True
    for name in MODULE_NAMES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            healthy = False
            report.fail(name, f"{type(exc).__name__}: {exc}")
            report.say(f"      {traceback.format_exc()}")
        else:
            report.say(f"  ok  {name}")
    return healthy


# --------------------------------------------------------------------------- #
# 2
# --------------------------------------------------------------------------- #


def check_environment(report: Report) -> bool:
    report.check_section(2, "environment audit")
    try:
        from single_parameter_observatory.optimizer_inspector import audit_environment

        report.say(audit_environment().render())
        return True
    except Exception as exc:
        report.fail("environment audit", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False


# --------------------------------------------------------------------------- #
# 3
# --------------------------------------------------------------------------- #


def check_tests(report: Report, verbose: bool) -> bool:
    report.check_section(3, "automated test suite")
    import unittest

    tests_dir = os.path.join(_PACKAGE_DIR, "tests")
    if not os.path.isdir(tests_dir):
        report.fail("test discovery", f"no tests directory at {tests_dir}")
        return False
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, top_level_dir=_PACKAGE_DIR)
    buffer = io.StringIO()
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, stream=buffer)
    started = time.time()
    result = runner.run(suite)
    output = buffer.getvalue()
    # The suite's own output is reproduced verbatim: no summarising away. This is
    # where the suite's runtime figure in the README comes from.
    for line in output.splitlines():
        report.say("  " + line)
    report.say(
        f"  TESTS: ran {result.testsRun}, "
        f"{result.testsRun - len(result.failures) - len(result.errors)} passed, "
        f"{len(result.failures)} failed, {len(result.errors)} errors, "
        f"{len(result.skipped)} skipped"
    )
    report.say(f"  TESTS WALL CLOCK: {time.time() - started:.3f}s")
    for case, text in list(result.failures) + list(result.errors):
        report.say(f"  ---- {case}")
        report.say("  " + text.replace("\n", "\n  "))
    if not result.wasSuccessful():
        report.fail("test suite", f"{len(result.failures)} failures, {len(result.errors)} errors")
        return False
    return True


# --------------------------------------------------------------------------- #
# 4
# --------------------------------------------------------------------------- #


def _semantics_probe(report: Report, config: Any, label: str) -> None:
    from single_parameter_observatory.optimizer_inspector import probe_optimizer_semantics

    probe = probe_optimizer_semantics(config)
    report.say(
        f"  measured variant for {label}: {probe['measured_variant']!r}, "
        f"bit-exact reproduction {probe['bit_exact']}, torch {probe['torch_version']}"
    )
    if not probe["bit_exact"]:
        report.fail(f"semantics probe ({label})", "no variant reproduced the update exactly")
    best = min(candidate["ulp_error"] for candidate in probe["candidates"])
    report.say(
        f"    {len(probe['candidates'])} candidate variants tried; "
        f"best ULP error {best}"
    )


def check_training(report: Report, steps: int) -> Any:
    report.check_section(4, f"real training run ({steps} steps, AdamW)")
    try:
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.recorder import render_final_report
        from single_parameter_observatory.training_engine import TrainingEngine

        config = ExperimentConfig(
            name="run_checks", optimizer="adamw", steps=steps, initial_value=-2.0, lr=0.1
        )
        engine = TrainingEngine(config)
        report.say(engine.tracker.info().render())
        engine.provenance()
        started = time.time()
        engine.run(on_step=lambda record: report.say("  " + engine.status_line()))
        elapsed = time.time() - started
        report.say(f"  executed {len(engine.recorder)} real steps in {elapsed:.2f}s")
        report.say(render_final_report(engine.final_report()))
        report.say("")
        report.say(engine.recorder.ledger.render())
        if engine.tracker.value() == config.initial_value:
            report.fail("training", "the parameter did not move at all")
            return None
        report.ok("real training run completed")
        report.say("")
        report.say("  measured optimizer semantics (throwaway parameters, isolated):")
        from single_parameter_observatory.config import PRESETS

        _semantics_probe(report, PRESETS["adamw"], "AdamW")
        _semantics_probe(report, PRESETS["sgd_momentum"], "SGD + Nesterov")
        return engine
    except Exception as exc:
        report.fail("training run", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return None


# --------------------------------------------------------------------------- #
# 5
# --------------------------------------------------------------------------- #


def check_export(report: Report, engine: Any, directory: str) -> bool:
    report.check_section(5, "export")
    if engine is None:
        report.fail("export", "no completed training run to export")
        return False
    try:
        from single_parameter_observatory.exporters import export_all

        written = export_all(
            engine.recorder,
            directory,
            report=engine.final_report(),
            provenance=engine.provenance(),
            analyzer=engine.analyzer,
        )
        for name, path in sorted(written.items()):
            if path is None:
                report.say(f"  --  {name}: not written")
            else:
                exists = os.path.exists(path)
                size = os.path.getsize(path) if exists else 0
                report.say(f"  {'ok' if exists and size else '--'}  {name}: {path} ({size} bytes)")
                if name not in ("trajectory.parquet",) and not exists:
                    report.fail("export", f"{name} was not created")
                    return False
        report.ok("export wrote every artefact")
        return True
    except Exception as exc:
        report.fail("export", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False


# --------------------------------------------------------------------------- #
# 6
# --------------------------------------------------------------------------- #


def check_trace(report: Report, steps: int) -> bool:
    report.check_section(6, f"per-step forensic trace ({steps} steps)")
    try:
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.main import render_trace
        from single_parameter_observatory.training_engine import TrainingEngine

        engine = TrainingEngine(
            ExperimentConfig(
                name="run_checks-trace", optimizer="adam", steps=steps,
                initial_value=-1.25, run_fp64_reference_track=False,
            )
        )
        for _ in range(steps):
            record = engine.step()
            if record is None:
                break
            report.say(render_trace(record))
        report.ok("trace rendered")
        return True
    except Exception as exc:
        report.fail("trace", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False


# --------------------------------------------------------------------------- #
# 7
# --------------------------------------------------------------------------- #


def check_controls(report: Report) -> bool:
    report.check_section(
        7, "control session (play / pause / step / run-N / reset / restart)"
    )
    try:
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.training_engine import (
            ExperimentController,
            TrainingEngine,
        )

        engine = TrainingEngine(
            ExperimentConfig(
                name="run_checks-controls", optimizer="adamw", steps=20,
                run_fp64_reference_track=False,
            )
        )
        controller = ExperimentController(engine)
        printed: list[str] = []
        controller.run_blocking(
            # 'restart' discards the history, so the trailing 'step' is the only
            # record that should survive it.
            ["step", "step", "n 3", "pause", "step", "reset", "step", "restart", "step"],
            printer=lambda text: printed.append(text),
        )
        controller.stop()
        report.say(f"  step count after the command script: {len(engine.recorder)}")
        report.say(f"  final parameter: {engine.tracker.value()!r}")
        report.say(f"  control lines printed: {len(printed)}")
        if len(engine.recorder) != 1:
            report.fail(
                "controls",
                f"expected exactly 1 recorded step after restart, got {len(engine.recorder)}",
            )
            return False
        if not printed:
            report.fail("controls", "the control session printed nothing")
            return False
        report.ok("control script executed against the real engine")
        return True
    except Exception as exc:
        report.fail("controls", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False


# --------------------------------------------------------------------------- #
# 8
# --------------------------------------------------------------------------- #


def check_resolution_preset(report: Report) -> bool:
    report.check_section(
        8, "FP32-resolution preset (attempted update, no stored change)"
    )
    try:
        from single_parameter_observatory.config import PRESETS
        from single_parameter_observatory.training_engine import TrainingEngine

        engine = TrainingEngine(PRESETS["fp32_resolution"])
        records = engine.run()
        report.say(f"  {engine.status_line()}")
        frozen = [r for r in records if r.stored_fp32_unchanged]
        attempted = [r for r in records if r.update_attempted_but_storage_frozen]
        report.say(f"  steps: {len(records)}")
        report.say(f"  steps whose stored float32 did not change: {len(frozen)}")
        report.say(f"  steps where an update was attempted below one ULP: {len(attempted)}")
        for record in records[:5]:
            report.say(
                f"    step {record.step}: attempted (fp64) "
                f"{record.optimizer_attempted_update!r}, reconstructed (fp32) "
                f"{record.optimizer_total_update!r}, "
                f"observed stored delta {record.actual_delta!r}, "
                f"1 ULP = {record.ulp_size!r}, "
                f"below-resolution={record.update_attempted_but_storage_frozen}"
            )
        if not records:
            report.fail("fp32_resolution preset", "no steps were recorded")
            return False
        if not attempted:
            report.fail(
                "fp32_resolution preset",
                "no step produced a non-zero attempted update below one ULP, so "
                "the preset did not demonstrate what its name claims",
            )
            return False
        report.ok("FP32-resolution preset produced sub-ULP attempted updates")
        return True
    except Exception as exc:
        report.fail("fp32_resolution preset", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False


# --------------------------------------------------------------------------- #
# 9
# --------------------------------------------------------------------------- #


def _run_cli(report: Report, argv: list[str], must_contain: list[str], label: str) -> bool:
    """Run one CLI surface in-process and check both its exit code and output."""
    import io as _io
    from contextlib import redirect_stdout
    from single_parameter_observatory.main import main as cli_main

    buffer = _io.StringIO()
    try:
        with redirect_stdout(buffer):
            status = cli_main(argv)
    except SystemExit as exc:  # argparse exits rather than returning
        status = int(exc.code or 0)
    except Exception as exc:
        report.fail(f"CLI {label}", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False
    output = buffer.getvalue()
    report.say(f"  $ python main.py {' '.join(argv)}")
    report.say(f"    exit status: {status}")
    missing = [needle for needle in must_contain if needle not in output]
    if status != 0:
        report.fail(f"CLI {label}", f"exit status {status}, expected 0")
        return False
    if missing:
        report.fail(f"CLI {label}", f"output is missing {missing!r}")
        report.say(output[-2000:])
        return False
    report.say(f"    output contained all of: {must_contain!r}")
    report.ok(f"CLI {label}")
    return True


def check_cli_surfaces(report: Report, frames_dir: str) -> bool:
    report.check_section(
        9, "documented CLI surfaces and remaining behaviour-sensitive presets"
    )
    healthy = True
    try:
        from single_parameter_observatory.config import PRESETS
        from single_parameter_observatory.training_engine import TrainingEngine

        healthy &= _run_cli(
            report,
            ["inspect", "--steps", "1"],
            ["ENVIRONMENT AUDIT", "OPTIMIZER IMPLEMENTATION AUDIT", "Matplotlib"],
            "inspect",
        )
        presets_ok = _run_cli(report, ["presets"], ["EXPERIMENT PRESETS", "adamw"], "presets")
        healthy &= presets_ok
        if presets_ok:
            import io as _io
            from contextlib import redirect_stdout
            from single_parameter_observatory.main import main as cli_main

            buffer = _io.StringIO()
            with redirect_stdout(buffer):
                cli_main(["presets"])
            output = buffer.getvalue()
            missing = [name for name in PRESETS if name not in output]
            if missing:
                report.fail("CLI presets", f"presets missing from the listing: {missing}")
                healthy = False
            else:
                report.say(f"    every preset in config.PRESETS is listed ({len(PRESETS)})")
        healthy &= _run_cli(
            report,
            ["controls", "--steps", "6", "--commands", "step", "step", "n2"],
            ["CONTROL SESSION"],
            "controls",
        )
        healthy &= _run_cli(
            report,
            ["live", "--headless", "--steps", "2", "--speed", "100",
             "--save-frames", frames_dir],
            ["LIVE DASHBOARD"],
            "live --headless",
        )
        frames = sorted(os.listdir(frames_dir)) if os.path.isdir(frames_dir) else []
        report.say(f"  frames rendered by the headless dashboard: {len(frames)}")
        if not frames:
            report.fail("CLI live --headless", "no frame was rendered")
            healthy = False
        else:
            report.ok("headless dashboard rendered a frame")

        # The diverging preset must actually diverge, and near_stationary must
        # report the regime it reports. Both are named behaviour, so both are
        # checked here rather than left to the test suite alone.
        diverging = TrainingEngine(PRESETS["diverging"])
        diverging.run()
        report.say(
            f"  diverging preset: aborted={diverging.aborted}, "
            f"steps recorded={len(diverging.recorder)}, reason="
            f"{diverging.abort_reason!r}"
        )
        if not diverging.aborted or not diverging.recorder.records[-1].nonfinite_detected:
            report.fail("diverging preset", "the run did not reach a non-finite state")
            healthy = False
        else:
            report.ok("diverging preset produced a detected non-finite state")

        stationary = TrainingEngine(
            PRESETS["near_stationary"].replace(
                steps=20, run_fp64_reference_track=False
            )
        )
        stationary.run()
        regime = stationary.analyzer.report()
        report.say(
            f"  near_stationary preset: regime={regime['regime']!r}, "
            f"mean |gradient|={regime['mean_gradient_magnitude']!r}, "
            f"mean |update|={regime['mean_update_magnitude']!r}"
        )
        report.ok("near_stationary preset reported its regime")
        return healthy
    except Exception as exc:
        report.fail("CLI surfaces", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False


# --------------------------------------------------------------------------- #
# 10
# --------------------------------------------------------------------------- #


def check_near_zero_loss_guard(report: Report, steps: int) -> bool:
    report.check_section(
        10,
        f"near-zero-loss divergence guard (default preset, {steps} steps)",
    )
    report.say(
        "  This check records what the classifier did. It does not require any"
    )
    report.say(
        "  particular regime: a run that ended differently would still be a"
    )
    report.say("  legitimate run, and the log would say so.")
    try:
        from single_parameter_observatory.config import PRESETS
        from single_parameter_observatory.training_engine import TrainingEngine

        engine = TrainingEngine(PRESETS["adamw"].replace(steps=steps))
        records = engine.run()
        losses = [record.loss for record in records]
        minimum_index = min(range(len(losses)), key=lambda i: losses[i])
        stability = engine.analyzer.report()
        rows = [
            ("steps executed", repr(len(records))),
            (
                "minimum loss",
                f"{losses[minimum_index]!r} at step {minimum_index}",
            ),
            ("final loss", repr(losses[-1])),
            (
                "loss ratio over the final window",
                repr(stability.get("loss_growth_ratio_in_window")),
            ),
            ("observed regime", repr(stability["regime"])),
        ]
        width = max(len(name) for name, _ in rows)
        for name, value in rows:
            report.say(f"  {name.ljust(width)} : {value}")
        report.say("  classifier rationale, verbatim:")
        for line in stability["rationale"]:
            report.say(f"    - {line}")
        guard_lines = [
            line
            for line in stability["rationale"]
            if "below the divergence threshold" in line
        ]
        if guard_lines:
            report.say(
                "  the guard fired: the loss ratio rose but the absolute increase "
                "stayed below the threshold,"
            )
            report.say(
                "  so a large factor at a near-zero loss was not reported as "
                "divergence."
            )
        else:
            report.say(
                "  the guard did not fire in this run; the classifier reported the "
                "rationale above verbatim instead."
            )
        report.ok(f"near-zero-loss guard recorded over {len(records)} steps")
        return True
    except Exception as exc:
        report.fail("near-zero-loss guard check", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False


def check_live_frame_rate(report: Report, seconds: float) -> bool:
    report.check_section(11, f"live view frame rate (measured over {seconds:.0f} s)")
    report.say(
        "  Recorded, not asserted: the rate depends on the machine. The guard below"
    )
    report.say(
        "  fails only if the view has collapsed to its pre-caching behaviour."
    )
    try:
        from single_parameter_observatory import visualization as viz
        from single_parameter_observatory.config import ExperimentConfig
        from single_parameter_observatory.dashboard import LiveDashboard
        from single_parameter_observatory.training_engine import (
            ExperimentController,
            TrainingEngine,
        )

        viz.configure_backend(interactive=False)
        engine = TrainingEngine(
            ExperimentConfig(name="fps", optimizer="adamw", steps=4000, initial_value=-2.0)
        )
        controller = ExperimentController(engine)
        panel = LiveDashboard(
            engine, controller=controller, interactive=False,
            target_fps=DEFAULT_LIVE_FPS, hud_hz=4.0,
            static_hz=DEFAULT_STATIC_HZ, data_hz=DEFAULT_DATA_HZ,
        )
        panel.build_figure()
        controller.set_speed(30.0)
        controller.start()
        controller.play()
        started = time.time()
        drawn = panel.run(max_seconds=seconds)
        elapsed = time.time() - started
        achieved = drawn / elapsed if elapsed else 0.0
        report.say(f"  target frame rate                   : {panel.target_fps:g} FPS")
        report.say(f"  frames drawn                        : {drawn}")
        report.say(f"  wall clock                          : {elapsed:.2f} s")
        report.say(f"  achieved frame rate                 : {achieved:.1f} FPS")
        report.say(f"  real steps recorded during the view: {len(engine.recorder)}")
        report.say(f"  background caching in use           : {panel._blit_available}")
        panel.close()
        controller.stop()
        if achieved < MINIMUM_USABLE_FPS:
            report.fail(
                "live frame rate",
                f"only {achieved:.1f} FPS; the cached text layer is not in use",
            )
            return False
        report.ok(
            f"live view reached {achieved:.1f} FPS against a {panel.target_fps:g} FPS target"
        )
        return True
    except Exception as exc:
        report.fail("live frame rate", f"{type(exc).__name__}: {exc}")
        report.say(traceback.format_exc())
        return False


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40, help="steps for the training run")
    parser.add_argument("--trace-steps", type=int, default=1)
    parser.add_argument(
        "--export-dir", default=os.path.join(_PACKAGE_DIR, "run_checks_export")
    )
    parser.add_argument(
        "--frames-dir", default=os.path.join(_PACKAGE_DIR, "run_checks_frames")
    )
    parser.add_argument("--verbose-tests", action="store_true")
    parser.add_argument(
        "--guard-steps",
        type=int,
        default=GUARD_STEPS,
        help="steps for the near-zero-loss guard check (103 is the measured minimum)",
    )
    parser.add_argument(
        "--fps-seconds",
        type=float,
        default=6.0,
        help="how long to measure the live view's frame rate for",
    )
    parser.add_argument("--log", default=LOG_PATH)
    args = parser.parse_args(argv)

    report = Report()
    report.say("SINGLE PARAMETER OBSERVATORY - EXECUTION CHECKS")
    report.say(f"python: {sys.version.split()[0]}  ({sys.executable})")
    report.say(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    healthy = True
    healthy &= check_imports(report)
    healthy &= check_environment(report)
    healthy &= check_tests(report, args.verbose_tests)
    engine = check_training(report, args.steps)
    healthy &= check_export(report, engine, args.export_dir)
    healthy &= check_trace(report, args.trace_steps)
    healthy &= check_controls(report)
    healthy &= check_resolution_preset(report)
    healthy &= check_cli_surfaces(report, args.frames_dir)
    healthy &= check_near_zero_loss_guard(report, args.guard_steps)
    healthy &= check_live_frame_rate(report, args.fps_seconds)

    report.section("SUMMARY")
    if report.failures:
        report.say(f"  {len(report.failures)} check(s) reported a failure:")
        for failure in report.failures:
            report.say(f"    - {failure}")
    else:
        report.say(f"  all {CHECK_COUNT} checks completed without an error")
    report.say(f"  total elapsed: {time.time() - report.started:.1f}s")
    path = report.write(args.log)
    report.say(f"  full log written to {path}")
    return 0 if healthy and not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
