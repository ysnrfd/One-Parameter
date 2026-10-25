#!/usr/bin/env python3
"""Command-line entry point for the single float32 parameter observatory.

Run ``python main.py`` with no arguments to execute a real training run and
print the end-of-run numerical report.  The other subcommands expose the same
real experiment in different ways; none of them is a demo mode with substituted
values.

The module lives inside the package, so it bootstraps ``sys.path`` and then
imports the package absolutely.  That is what makes ``python main.py`` work
from any working directory, which is why the imports below are not relative.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PACKAGE_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from single_parameter_observatory import dashboard, exporters, visualization  # noqa: E402
from single_parameter_observatory import numerical_analysis as na  # noqa: E402
from single_parameter_observatory import optimizer_inspector as oi  # noqa: E402
from single_parameter_observatory.config import (  # noqa: E402
    PRESET_DESCRIPTIONS,
    PRESETS,
    ExperimentConfig,
)
from single_parameter_observatory.convergence import STABILITY_DISCLAIMER  # noqa: E402
from single_parameter_observatory.errors import ObservatoryError  # noqa: E402
from single_parameter_observatory.recorder import (  # noqa: E402
    ParameterStepRecord,
    render_final_report,
)
from single_parameter_observatory.tracker import TENSOR_VS_PARAMETER_NOTES  # noqa: E402
from single_parameter_observatory.training_engine import (  # noqa: E402
    ExperimentController,
    TrainingEngine,
)

DEFAULT_EXPORT_ROOT = "runs"

# --------------------------------------------------------------------------- #
# banners
# --------------------------------------------------------------------------- #


def print_header(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_environment(engine: TrainingEngine) -> None:
    print_header("ENVIRONMENT AUDIT (observed, not assumed)")
    print(oi.audit_environment().render())


def print_parameter_identity(engine: TrainingEngine) -> None:
    print_header("TRACKED PARAMETER")
    print(engine.tracker.info().render())
    print()
    print("TENSOR vs PARAMETER")
    for note in TENSOR_VS_PARAMETER_NOTES:
        print(f"  - {note}")


def print_optimizer_audit(engine: TrainingEngine) -> None:
    print_header("OPTIMIZER IMPLEMENTATION AUDIT")
    print(engine.handle.audit.render())
    probe = engine.provenance().get("optimizer_semantics_probe") or {}
    if probe:
        print()
        print("MEASURED UPDATE SEMANTICS (one isolated real step on a throwaway parameter)")
        print(f"  measured variant        : {probe.get('measured_variant')}")
        print(f"  bit-exact reproduction  : {probe.get('bit_exact')}")
        print(f"  note                    : {probe.get('note')}")
        print("  candidates (variant -> ULP error against the real update):")
        for candidate in probe.get("candidates", [])[:24]:
            print(f"      {candidate['variant']:<44} {candidate['ulp_error']}")
        if len(probe.get("candidates", [])) > 24:
            print(f"      ... (+{len(probe['candidates']) - 24} more candidates in the export)")


# --------------------------------------------------------------------------- #
# per-step terminal trace
# --------------------------------------------------------------------------- #


def render_trace(record: ParameterStepRecord) -> str:
    """The detailed per-step forensic trace, in the required layout."""
    lines: list[str] = []
    rule = "=" * 78
    thin = "-" * 78
    lines.append(rule)
    lines.append("SINGLE PARAMETER FORENSIC TRACE")
    lines.append(rule)
    lines.append(f"STEP: {record.step}   (epoch {record.epoch}, batch {record.batch_index})")
    lines.append("")
    lines.append("PARAMETER")
    lines.append(thin)
    lines.append("dtype: float32")
    lines.append("shape: ()")
    lines.append(f"before: {record.parameter_before!r}  ({record.parameter_before_hex})")
    lines.append(f"after:  {record.parameter_after!r}  ({record.parameter_hex})")
    lines.append(
        f"delta (observed, after - before): {record.actual_delta!r}"
    )
    lines.append(
        f"  movement, in ULPs of w_before : {record.ulp_distance_same_binade:+d}"
    )
    lines.append(
        f"  displacement / ULP(w_before)  : {record.ulp_relative_movement!r}"
    )
    lines.append(
        f"  representable float32 values between before and after: "
        f"{record.ulp_distance_count}"
    )
    crossed_zero = (record.parameter_before_uint32 >> 31) != (record.parameter_after_uint32 >> 31)
    if crossed_zero or abs(record.ulp_distance_count) > 4 * max(
        1, abs(record.ulp_distance_same_binade)
    ):
        lines.append(
            "  NOTE: this step crossed a binade or sign boundary, so the "
            "representable-value count and the movement count are different "
            "quantities and diverge by orders of magnitude"
            + (" (the step crossed zero)" if crossed_zero else "")
        )
    lines.append(f"stored FP32 unchanged this step: {record.stored_fp32_unchanged}")
    lines.append(
        f"attempted displacement (float64 candidate): {record.optimizer_attempted_update!r}; "
        f"one ULP of w_before is {record.ulp_size!r}"
    )
    lines.append(
        f"reconstructed float32 update (derived, not observed): "
        f"{record.optimizer_total_update!r}"
    )
    lines.append(
        "  the observed float32 displacement is the 'delta' line in the PARAMETER "
        "block above; the two differ whenever the float32 rounding of the "
        "candidate does not reproduce the stored word"
    )
    if record.update_attempted_but_storage_frozen:
        lines.append(
            "  -> the attempted update was below the local float32 resolution, so "
            "the stored value could not move"
        )
    lines.append("")
    lines.append("SNAPSHOT CHAIN (parameter bit pattern at each labelled instant)")
    lines.append(thin)
    for label, value in zip(
        ("A before forward", "B after forward", "C after backward",
         "D before optimizer.step", "E after optimizer.step", "F after schedule step"),
        (
            record.snapshot_a_uint32, record.snapshot_b_uint32, record.snapshot_c_uint32,
            record.snapshot_d_uint32, record.snapshot_e_uint32, record.snapshot_f_uint32,
        ),
    ):
        # Snapshot F only exists when a framework operation follows the step.
        rendered = f"0x{value:08X}" if value is not None else "(no operation after the step)"
        lines.append(f"  {label:<26} {rendered}")
    lines.append(
        "  A == B == C holds by construction of the run, and the engine raises if "
        "the parameter moves during forward or backward."
    )
    lines.append("")
    lines.append("FORWARD")
    lines.append(thin)
    lines.append(f"input:        {record.input_value!r}")
    lines.append(f"target:       {record.target_value!r}")
    for snapshot in record.stages:
        if snapshot.name in ("input", "parameter", "target", "loss", "prediction"):
            continue
        lines.append(
            f"{snapshot.name:<13} {snapshot.value!r}  dtype={snapshot.dtype} "
            f"shape={list(snapshot.shape)} grad_fn={snapshot.grad_fn}"
        )
    lines.append(f"prediction:   {record.prediction!r}")
    lines.append(f"loss:         {record.loss!r}  (grad_fn={record.loss_grad_fn})")
    if record.accumulation_micro_batches > 1:
        lines.append(
            f"accumulated:  {record.accumulation_micro_batches} micro-batches, "
            f"losses={[repr(v) for v in record.accumulated_losses]}"
        )
    lines.append("")
    lines.append("BACKWARD")
    lines.append(thin)
    lines.append(f"gradient before zero_grad: {record.grad_before_zero_grad!r}")
    lines.append(
        f"gradient after zero_grad (set_to_none={record.zero_grad_set_to_none}): "
        f"{record.grad_after_zero_grad!r}"
    )
    lines.append(f"gradient before backward:  {record.grad_before_backward!r}")
    lines.append(f"gradient after backward:   {record.gradient!r}")
    lines.append(
        f"gradient dtype/shape: {record.gradient_dtype} {list(record.gradient_shape)}"
        + (f"  raw 0x{record.gradient_uint32:08X}" if record.gradient_uint32 is not None else "")
    )
    lines.append(
        f"gradient sign: {record.gradient_sign:+d}   magnitude: {record.gradient_magnitude!r}   "
        f"square: {record.gradient_square!r}"
    )
    lines.append(f"gradient finite: {record.gradient_finite}")
    lines.append("")
    lines.append("OPTIMIZER")
    lines.append(thin)
    lines.append(f"optimizer: {record.optimizer_name}")
    lines.append(
        f"step: {record.optimizer_step_before} -> {record.optimizer_step_after}"
        + ("" if record.optimizer_state_keys_after else "  (this optimizer keeps no per-parameter state)")
    )
    lines.append(
        f"state keys after step: {', '.join(record.optimizer_state_keys_after) or '(empty)'}"
    )
    lines.append(f"exp_avg before/after: {record.moment1_before!r} -> {record.moment1!r}")
    lines.append(f"exp_avg_sq before/after: {record.moment2_before!r} -> {record.moment2!r}")
    lines.append(f"bias-corrected exp_avg: {record.bias_corrected_moment1!r}")
    lines.append(f"bias-corrected exp_avg_sq: {record.bias_corrected_moment2!r}")
    lines.append(f"learning rate: {record.learning_rate!r}")
    lines.append(
        f"weight decay contribution (reconstructed, float32): "
        f"{record.optimizer_weight_decay_update!r}"
    )
    lines.append(
        f"adaptive contribution (reconstructed, float32): "
        f"{record.optimizer_adaptive_update!r}"
    )
    lines.append(
        f"total reconstructed float32 update (derived, not observed): "
        f"{record.optimizer_total_update!r}"
    )
    lines.append(
        "  the decay and adaptive contributions above are each measured as the "
        "difference between the reconstruction with weight decay enabled and the "
        "same reconstruction with it disabled, so they include float32 rounding; "
        "the observed result is the 'delta' line in the PARAMETER block"
    )
    lines.append("")
    lines.append("NUMERICAL")
    lines.append(thin)
    lines.append(f"fp64 reference candidate: {record.fp64_reconstructed_after!r}")
    lines.append(f"  (reference numerical experiment, not ground truth)")
    lines.append(f"actual FP32 stored:       {record.parameter_after!r}")
    lines.append(
        f"rounding residual: {record.fp32_rounding_error_abs!r} "
        f"({record.fp32_rounding_ulp!r} ULP, sign {record.fp32_rounding_sign:+d})"
    )
    lines.append(f"candidate rounds to stored value: {record.fp32_candidate_rounds_to_stored}")
    lines.append(f"ULP at w_before: {record.ulp_size!r}")
    lines.append(
        "movement metrics are in the PARAMETER block above; they are not "
        "repeated here"
    )
    if record.fp64_track_value is not None:
        lines.append(
            f"independent fp64 track: {record.fp64_track_value!r} "
            f"(difference {record.fp64_vs_fp32_abs!r}, {record.fp64_vs_fp32_ulp!r} ULP)"
        )
    lines.append("")
    lines.append("BITS")
    lines.append(thin)
    lines.append(f"before: {record.parameter_before_hex}  {record.parameter_before_uint32:032b}")
    lines.append(f"after:  {record.parameter_hex}  {record.parameter_bits}")
    lines.append(
        f"xor mask: 0x{record.xor_mask:08X}  changed bits: {record.changed_bit_count} "
        f"at positions {list(record.changed_bit_positions)} (0 = sign, 31 = fraction LSB)"
    )
    lines.append(
        f"sign changed: {record.sign_changed}   exponent changed: {record.exponent_changed}   "
        f"fraction changed: {record.fraction_changed}"
    )
    lines.append(
        f"after fields: sign={record.parameter_sign_bit} "
        f"exponent={record.parameter_exponent_bits} fraction={record.parameter_fraction_bits}"
    )
    if record.landscape_computed:
        lines.append("")
        lines.append("LOCAL LOSS LANDSCAPE (diagnostic slice)")
        lines.append(thin)
        lines.append(
            f"centre: {record.parameter_after!r}  loss: {record.landscape_loss_at_centre!r}"
        )
        lines.append(
            f"first derivative: {record.landscape_first_derivative!r}   "
            f"second derivative: {record.landscape_second_derivative!r}"
        )
        lines.append(f"parameter restored after the probe: {record.landscape_restoration_verified}")
        lines.append(f"{na.ONE_DIMENSIONAL_DISCLAIMER}")
    lines.append("")
    lines.append("VALIDATION")
    lines.append(thin)
    lines.append(f"gradient check (autograd vs closed form): {record.gradient_validation_status}")
    if record.gradient_analytical is not None:
        lines.append(f"  autograd:            {record.gradient!r}")
        lines.append(f"  analytical:          {record.gradient_analytical!r}")
        lines.append(f"  relative difference: {record.gradient_vs_analytical_rel!r}")
        lines.append(f"  finite difference:   {record.gradient_finite_difference!r}"
                     f"  (h = {record.gradient_finite_difference_h!r})")
        lines.append(f"  relative difference: {record.gradient_vs_finite_difference_rel!r}")
        lines.append(f"  fp64 autograd:       {record.gradient_fp64_autograd!r}")
    else:
        lines.append("  (gradient validation is disabled or not due at this step)")
    lines.append(
        f"optimizer reconstruction: {record.reconstruction_status}  "
        f"(variant {record.reconstruction_variant}, {record.reconstruction_ulp_error} ULP, "
        f"bit-exact={record.reconstruction_bit_exact})"
    )
    lines.append(f"finite: {'YES' if record.finite else 'NO'}")
    lines.append(f"events (up to the moment this record was sealed): "
                 f"{', '.join(record.events) or '(none)'}")
    lines.append(
        "  this list is deliberately partial: events that can only be decided "
        "after the record exists (STABILITY_METRIC_UPDATED, RUN_COMPLETE and, "
        "for a terminal step, RUN_ABORTED) are appended to the event log but "
        "cannot appear in the record. The event log is the complete sequence."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build a configuration from the preset plus explicit overrides."""
    if args.preset not in PRESETS:
        raise ObservatoryError(
            f"unknown preset {args.preset!r}; available: {', '.join(sorted(PRESETS))}"
        )
    config = PRESETS[args.preset]
    if getattr(args, "steps", None):
        config = config.replace(steps=int(args.steps))
    if getattr(args, "lr", None):
        config = config.replace(lr=float(args.lr))
    if getattr(args, "optimizer", None):
        if args.optimizer not in ("sgd", "adam", "adamw"):
            raise ObservatoryError(f"unknown optimizer {args.optimizer!r}")
        config = config.replace(optimizer=args.optimizer)
    if getattr(args, "weight_decay", None) is not None:
        config = config.replace(weight_decay=float(args.weight_decay))
    if getattr(args, "seed", None) is not None:
        config = config.replace(seed=int(args.seed))
    if getattr(args, "no_fp64_reference", False):
        config = config.replace(run_fp64_reference_track=False)
    if getattr(args, "no_landscape", False):
        config = config.replace(landscape=config.landscape.replace(enabled=False) if hasattr(config.landscape, "replace") else config.replace_landscape(enabled=False))
    if getattr(args, "no_validation", False):
        config = config.replace(validation=config.validation.replace(enabled=False) if hasattr(config.validation, "replace") else config.replace_validation(enabled=False))
    if getattr(args, "set_to_none", None) is not None:
        config = config.replace(zero_grad_set_to_none=bool(args.set_to_none))
    if getattr(args, "accumulate", None):
        config = config.replace(accumulation_micro_batches=int(args.accumulate))
    return config


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", default="adamw", help="named training preset")
    parser.add_argument("--steps", type=int, default=None, help="number of real steps")
    parser.add_argument("--optimizer", default=None, choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None, dest="weight_decay")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--accumulate", type=int, default=None,
                        help="micro-batches whose gradients are summed per step")
    parser.add_argument("--set-to-none", dest="set_to_none", action="store_true", default=None,
                        help="zero_grad(set_to_none=True)")
    parser.add_argument("--set-to-zero", dest="set_to_none", action="store_false", default=None,
                        help="zero_grad(set_to_none=False)")
    parser.add_argument("--no-fp64-reference", action="store_true",
                        help="disable the independent float64 reference track")
    parser.add_argument("--no-landscape", action="store_true", help="disable landscape probes")
    parser.add_argument("--no-validation", action="store_true", help="disable validation")


def _finish(
    engine: TrainingEngine,
    args: argparse.Namespace,
    *,
    quiet: bool = False,
    view_command: bool = False,
) -> int:
    """Shared tail: chain check, report, optional figures, optional export.

    ``view_command`` is set by ``live``, which is a *view* of a run rather than a
    measurement of one.  Its exit status reports whether the view ran, so a
    diverged run does not make the dashboard look broken: the divergence is
    printed in the report (``NON-FINITE DETECTED`` and ``RUN ABORTED``) and is
    exactly what the diverging preset is for.  A broken history chain or a
    validation failure still fails a view, because those are defects rather than
    observations.  ``run`` and ``trace`` keep reporting a diverged run as a
    non-zero exit, since for them the run's outcome *is* the result.
    """
    ok, bad_step = engine.recorder.verify_chain()
    if not quiet:
        print_header("HISTORY INTEGRITY")
        print(f"  hash chain verified: {ok}" + ("" if ok else f" (first bad step: {bad_step})"))
    if not ok:
        # Checked before the report is built, because `final_report` asserts the
        # chain and would raise instead of returning the documented status.
        # A broken chain means the recorded trajectory cannot be trusted, so no
        # figure derived from it is printed and the command stops here.
        if not quiet:
            print(
                f"  the recorded history is corrupt at step {bad_step}; no report "
                f"is derived from it"
            )
        return 2

    report = engine.final_report()
    if not quiet:
        print(render_final_report(report))
        print()
        print("VALIDATION LEDGER")
        print(engine.recorder.ledger.render())
        print()
        print(STABILITY_DISCLAIMER)

    export_directory = getattr(args, "export", None)
    written: dict[str, Any] = {}
    if export_directory:
        target = export_directory
        if getattr(args, "no_timestamp", False):
            os.makedirs(target, exist_ok=True)
        else:
            stamp = f"{engine.config.name}_{len(engine.recorder.records)}steps"
            target = os.path.join(target, stamp)
            os.makedirs(target, exist_ok=True)
        written = exporters.export_all(
            engine.recorder,
            target,
            report=report,
            provenance=engine.provenance(),
            analyzer=engine.analyzer,
        )
        if not quiet:
            print_header("EXPORT")
            for name, path in sorted(written.items()):
                print(f"  {name:<34} {path if path else '(not written)'}")
            print(f"  directory: {target}")

    if getattr(args, "plot", None):
        try:
            figures = visualization.render_figures(
                engine.recorder, str(args.plot), engine.analyzer
            )
            if not quiet:
                print_header("FIGURES")
                for name, path in figures.items():
                    print(f"  {name:<34} {path}")
        except ObservatoryError as exc:
            print(f"  figures unavailable: {exc}")

    # Order matters, and this is the only guard `view_command` touches.
    # A broken chain and a validation failure are *defects* in the run, so they
    # must fail a view too, and they sit outside this check.  A divergence is an
    # *observation* about the run, so it fails only the commands whose purpose is
    # to report the run's outcome -- not `live`, whose purpose is to report that
    # the view ran.  Do not move `not ok` below this line.
    if engine.aborted and not view_command:
        return 3
    failures = report.get("validation", {}).get("total_failures", 0)
    if failures:
        return 4
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    engine = TrainingEngine(config)
    print_header(f"REAL TRAINING RUN: {config.name} ({config.optimizer})")
    print_environment(engine)
    print_parameter_identity(engine)
    print_optimizer_audit(engine)
    print()
    print_header("RUNNING REAL TRAINING")
    print("  loss.backward() and optimizer.step(); no values are synthesised.")
    print("-" * 78)
    started_steps = engine.step_index
    engine.run(on_step=lambda record: print(f"  {engine.status_line()}"))
    print("-" * 78)
    print(f"  executed {engine.step_index - started_steps} real steps")
    return _finish(engine, args)


def command_trace(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    engine = TrainingEngine(config)
    print_environment(engine)
    total = min(config.steps, args.count)
    last = -1
    for _ in range(total):
        record = engine.step()
        if record is None:
            break
        if record.step >= args.from_step and (
            args.to_step < 0 or record.step <= args.to_step
        ):
            if args.every <= 1 or record.step % args.every == 0:
                print(render_trace(record))
        last = record.step
    print_header(f"TRACE COMPLETE: steps 0..{last} executed in total")
    return _finish(engine, args, quiet=not getattr(args, "verbose", False))


def command_live(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    engine = TrainingEngine(config)
    print_environment(engine)
    print_parameter_identity(engine)
    save_dir = args.save_frames or ("frames" if args.headless else None)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print_header("LIVE DASHBOARD")
    print(
        "  controls: space = play/pause, s = single step, n = run 10, r = reset live "
        "state, R = restart, +/- = speed, q = quit"
    )
    panel = dashboard.LiveDashboard(
        engine,
        interactive=not args.headless,
        save_frames_directory=save_dir or None,
        frame_every=args.frame_every,
        target_fps=args.fps,
        hud_hz=args.hud_hz,
        static_hz=args.static_hz,
        data_hz=args.data_hz,
        animate=not args.no_animation,
    )
    if args.speed is not None:
        panel.controller.set_speed(args.speed)
    try:
        panel.run(max_seconds=args.max_seconds)
    finally:
        panel.close()
    return _finish(engine, args, view_command=True)


def command_controls(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    engine = TrainingEngine(config)
    print_environment(engine)
    print_parameter_identity(engine)
    commands = args.commands
    if args.script:
        commands = [line for line in args.script.split(";") if line.strip()]
    print_header("CONTROL SESSION (real engine, no replay)")
    if args.interactive:
        print("  type a command and press enter; 'quit' ends the session")
        controller = ExperimentController(engine)
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            token = line.strip().lower()
            if token in ("quit", "q", "exit"):
                break
            if not token:
                continue
            if token in ("play", "run"):
                controller.play()
                while not engine.finished and not engine.aborted:
                    record = engine.step()
                    if record is None:
                        break
                    print(engine.status_line())
                continue
            controller.run_blocking([token], printer=print)
        controller.stop()
    else:
        dashboard.run_control_session(
            engine, commands, printer=print, show_card=args.show_card
        )
    return _finish(engine, args)


def command_inspect(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    engine = TrainingEngine(config)
    print_environment(engine)
    print_parameter_identity(engine)
    print_optimizer_audit(engine)
    provenance = engine.provenance()
    if args.json:
        print(json.dumps(provenance, indent=2, default=str))
    return 0


def command_presets(args: argparse.Namespace) -> int:
    print_header("EXPERIMENT PRESETS (each one runs real training)")
    for name in sorted(PRESETS):
        config = PRESETS[name]
        print(f"  {name}")
        print(f"      {PRESET_DESCRIPTIONS.get(name, '')}")
        print(
            f"      optimizer={config.optimizer} lr={config.lr} weight_decay={config.weight_decay} "
            f"steps={config.steps} initial={config.initial_value}"
        )
        print(
            f"      inputs [{config.dataset.input_min}, {config.dataset.input_max}] "
            f"batch={config.dataset.batch_size} walk={config.dataset.walk} "
            f"true parameter={config.dataset.target_true_parameter}"
        )
    return 0


def command_selftest(args: argparse.Namespace) -> int:
    print_header("AUTOMATED TEST SUITE (executed, not asserted)")
    import unittest  # noqa: PLC0415

    tests_dir = os.path.join(_PACKAGE_DIR, "tests")
    if not os.path.isdir(tests_dir):
        print(f"  tests directory not found at {tests_dir}")
        return 2
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, top_level_dir=_PACKAGE_DIR)
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1, stream=sys.stdout)
    result = runner.run(suite)
    print("-" * 78)
    print(
        f"  ran {result.testsRun} tests: "
        f"{result.testsRun - len(result.failures) - len(result.errors)} passed, "
        f"{len(result.failures)} failed, {len(result.errors)} errors, "
        f"{len(result.skipped)} skipped"
    )
    return 0 if result.wasSuccessful() else 1


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Forensic observatory for exactly one real trainable float32 scalar "
            "parameter. Every value displayed is read from a real PyTorch run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="execute a real training run and report it")
    _common_arguments(run)
    run.add_argument("--export", nargs="?", const=DEFAULT_EXPORT_ROOT, default=None,
                     help="write every export artefact into this directory")
    run.add_argument("--no-timestamp", action="store_true",
                     help="export into the given directory directly")
    run.add_argument("--plot", nargs="?", const="figures", default=None,
                     help="render figures into this directory")
    run.set_defaults(func=command_run)

    trace = subparsers.add_parser("trace", help="print the detailed per-step forensic trace")
    _common_arguments(trace)
    trace.add_argument("--count", type=int, default=10, help="how many real steps to execute")
    trace.add_argument("--from-step", type=int, default=0)
    trace.add_argument("--to-step", type=int, default=-1)
    trace.add_argument("--every", type=int, default=1, help="trace every Nth step")
    trace.add_argument("--verbose", action="store_true", help="also print the final report")
    trace.add_argument("--export", nargs="?", const=DEFAULT_EXPORT_ROOT, default=None)
    trace.add_argument("--no-timestamp", action="store_true")
    trace.set_defaults(func=command_trace)

    live = subparsers.add_parser("live", help="live dashboard over the real run")
    _common_arguments(live)
    live.add_argument("--headless", action="store_true",
                      help="no window: render frames to --save-frames instead")
    live.add_argument("--speed", type=float, default=None,
                      help="steps per second for the live run (also settable with +/-)")
    live.add_argument("--fps", type=float, default=120.0,
                      help="target render rate for the live view; 0 renders as fast "
                           "as the backend allows. This is a ceiling, not a promise: "
                           "the periodic re-rasterisation of the text costs about "
                           "90 ms, so the achieved rate is typically 85-90%% of it.")
    live.add_argument("--static-hz", type=float, default=1.0,
                      help="how often the text layer (state card, bit panel, causal "
                           "chain, axis labels) is re-rasterised, in hertz; lower it "
                           "to trade card freshness for frame rate")
    live.add_argument("--data-hz", type=float, default=20.0,
                      help="how often the recorded series lines are redrawn when "
                           "steps arrive faster than that")
    live.add_argument("--hud-hz", type=float, default=4.0,
                      help="how often the state card is re-rasterised, in hertz; it is "
                           "by far the most expensive text in the figure")
    live.add_argument("--no-animation", action="store_true",
                      help="do not slide the marker between recorded steps")
    live.add_argument("--save-frames", default=None,
                      help="also write each drawn frame as a PNG into this directory")
    live.add_argument("--frame-every", type=int, default=60,
                      help="save one frame in this many. Saving a frame needs a "
                           "full redraw of the text layer, so saving every frame "
                           "costs far more than drawing it; this default keeps the "
                           "dump a diagnostic rather than a throttle")
    live.add_argument("--max-seconds", type=float, default=None)
    live.add_argument("--export", nargs="?", const=DEFAULT_EXPORT_ROOT, default=None)
    live.add_argument("--no-timestamp", action="store_true")
    live.set_defaults(func=command_live)

    controls = subparsers.add_parser(
        "controls", help="headless control session (play/pause/step/run/reset/restart)"
    )
    _common_arguments(controls)
    controls.add_argument("--commands", nargs="*", default=["step", "step", "n 5", "step"],
                          help="commands to execute in order")
    controls.add_argument("--script", default=None, help="semicolon-separated command script")
    controls.add_argument("--interactive", action="store_true",
                          help="read commands from stdin instead")
    controls.add_argument("--show-card", action="store_true",
                          help="print the full state card after each step")
    controls.add_argument("--export", nargs="?", const=DEFAULT_EXPORT_ROOT, default=None)
    controls.add_argument("--no-timestamp", action="store_true")
    controls.set_defaults(func=command_controls)

    inspect = subparsers.add_parser(
        "inspect", help="environment, parameter identity and optimizer source audit"
    )
    _common_arguments(inspect)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(func=command_inspect)

    presets = subparsers.add_parser("presets", help="list the training presets")
    presets.set_defaults(func=command_presets)

    selftest = subparsers.add_parser("selftest", help="run the automated test suite")
    selftest.add_argument("--verbose", action="store_true")
    selftest.set_defaults(func=command_selftest)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        # `python main.py` with no arguments executes a real run and reports it.
        arguments = ["run"]
    parsed = parser.parse_args(arguments)
    if not hasattr(parsed, "func"):
        parser.print_help()
        return 0
    try:
        return int(parsed.func(parsed))
    except ObservatoryError as exc:
        print(f"\nERROR ({type(exc).__name__}): {exc}")
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
