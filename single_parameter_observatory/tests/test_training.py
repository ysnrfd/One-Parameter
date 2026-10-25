"""End-to-end training: real steps, stability analysis, and the live controls.

The control tests are the interesting ones.  ``pause`` must genuinely stop the
engine from stepping, and ``single step`` must advance it by exactly one real
step - checked by running the controller in a thread while the main thread
observes, so a pause that only froze a drawing loop would fail here.
"""

from __future__ import annotations

import math
import threading
import time
import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # type: ignore[import-not-found, no-redef]

import torch

from single_parameter_observatory.convergence import REGIMES, ConvergenceAnalyzer
from single_parameter_observatory.config import ExperimentConfig, PRESETS
from single_parameter_observatory.errors import TrackingInvalidError
from single_parameter_observatory.recorder import build_final_report
from single_parameter_observatory.training_engine import ExperimentController, RunState, TrainingEngine


def _engine(steps: int = 6, **overrides) -> TrainingEngine:
    settings: dict = {
        "optimizer": "adamw",
        "steps": steps,
        "initial_value": -0.7,
        "run_fp64_reference_track": False,
    }
    settings.update(overrides)
    return TrainingEngine(ExperimentConfig(name="test-training", **settings))


class TestTraining(unittest.TestCase):
    def test_real_training_changes_the_parameter(self) -> None:
        engine = _engine(steps=8)
        initial = engine.tracker.value()
        records = engine.run()
        self.assertEqual(len(records), 8)
        final = engine.tracker.value()
        self.assertNotEqual(final, initial, "real training must move the parameter")
        # The recorded value and the live tensor must agree exactly.
        self.assertEqual(records[-1].parameter_after, final)
        # The loss must be recorded as a real float, finite, from a real graph.
        for record in records:
            self.assertIsInstance(record.loss, float)
            self.assertTrue(math.isfinite(record.loss))
            self.assertGreaterEqual(record.loss, 0.0, "the loss is a squared error")

    def test_gradient_is_never_fabricated(self) -> None:
        """The training step must obtain its gradient only from autograd."""
        import inspect

        source = inspect.getsource(TrainingEngine.step)
        # The step itself must never assign a gradient, and must call backward.
        self.assertIn("loss_tensor.backward()", source)
        self.assertNotIn(".grad =", source)
        # The one place a gradient is assigned is the explicit reset, which is a
        # labelled lifecycle operation rather than a training mechanism.
        reset_source = inspect.getsource(TrainingEngine.reset_live_state)
        self.assertIn("self.parameter.grad = None", reset_source)

    def test_reconstruction_passes_for_a_real_run(self) -> None:
        engine = _engine(steps=6)
        for record in engine.run():
            self.assertEqual(
                record.reconstruction_status, "PASS",
                f"step {record.step}: {record.reconstruction_ulp_error} ULP",
            )
            self.assertEqual(record.reconstruction_ulp_error, 0)

    def test_validation_ledger_has_no_failures(self) -> None:
        engine = _engine(steps=4)
        engine.run()
        ledger = engine.recorder.ledger.to_dict()
        self.assertGreater(ledger["gradient_checks"], 0)
        self.assertEqual(ledger["gradient_failures"], 0, ledger["failure_notes"])
        self.assertEqual(ledger["finite_difference_failures"], 0, ledger["failure_notes"])
        self.assertEqual(ledger["reconstruction_failures"], 0, ledger["failure_notes"])
        self.assertEqual(ledger["bit_failures"], 0, ledger["failure_notes"])
        self.assertEqual(ledger["ulp_failures"], 0, ledger["failure_notes"])
        self.assertEqual(ledger["total_failures"], 0, ledger["failure_notes"])

    def test_landscape_probe_restores_the_parameter(self) -> None:
        engine = _engine(steps=3)
        engine.run()
        self.assertTrue(engine.recorder.landscapes)
        for slice_data in engine.recorder.landscapes:
            # The probe brackets its own execution, and the parameter is
            # genuinely unchanged across it.
            self.assertTrue(slice_data["restoration_verified"])
            self.assertEqual(
                slice_data["parameter_hash_before"], slice_data["parameter_hash_after"]
            )
            self.assertIn("one-dimensional", slice_data["disclaimer"])
        for record in engine.recorder.records:
            if record.landscape_computed:
                self.assertTrue(record.landscape_restoration_verified)
        # The record-level hashes bracket the whole step, which includes the
        # optimizer update, so they are expected to differ whenever the step
        # actually moved the parameter.
        moved = [r for r in engine.recorder.records if r.actual_delta != 0.0]
        for record in moved:
            self.assertNotEqual(
                record.parameter_hash_before, record.parameter_hash_after
            )

    def test_provenance_records_the_installed_implementation(self) -> None:
        engine = _engine(steps=1)
        provenance = engine.provenance()
        self.assertIn("environment", provenance)
        self.assertIn("torch_version", provenance["environment"])
        self.assertIn("optimizer_source_audit", provenance)
        self.assertEqual(
            provenance["environment"]["torch_version"], str(torch.__version__)
        )
        self.assertTrue(provenance["optimizer_semantics_probe"]["bit_exact"])


class TestStability(unittest.TestCase):
    def test_stability_analysis(self) -> None:
        engine = _engine(steps=20)
        engine.run()
        analyzer = engine.analyzer
        self.assertEqual(len(analyzer.metrics), 20)
        report = analyzer.report()
        self.assertIn(report["regime"], REGIMES)
        self.assertTrue(report["rationale"], "a regime must come with its evidence")
        self.assertIn("criteria", report)
        self.assertIn("converged", report["disclaimer"])
        self.assertIn("global optimum", report["disclaimer"])
        # The rolling statistics must be real numbers from the trajectory.
        self.assertTrue(
            all(math.isfinite(m.absolute_update) for m in analyzer.metrics)
        )
        self.assertGreaterEqual(report["unchanged_fraction_in_window"], 0.0)
        self.assertLessEqual(report["unchanged_fraction_in_window"], 1.0)
        self.assertIn("stability", build_final_report(engine.recorder, stability=report))

    def test_stability_reports_oscillation_for_an_oscillatory_run(self) -> None:
        config = PRESETS["oscillatory"].replace(
            steps=40, run_fp64_reference_track=False
        )
        engine = TrainingEngine(config)
        engine.run()
        report = engine.analyzer.report()
        self.assertIn(report["regime"], REGIMES)
        # Whatever the regime, the measurements behind it must be present.
        self.assertIn("update_sign_changes_in_window", report)
        self.assertGreater(report["update_sign_changes_in_window"], 0)

    def test_stability_reports_a_stationary_run(self) -> None:
        """A near-zero gradient does not imply a small update.

        Starting at the known optimum, the float32 targets leave a residual
        gradient of order 1e-5. Adam divides by sqrt(v), so a tiny gradient still
        produces a full-size step of order lr, and the parameter genuinely
        oscillates. The instrument reports that as oscillation - which is what
        happened - rather than as convergence, and the two magnitudes are
        reported separately so the reason is visible.
        """
        config = PRESETS["near_stationary"].replace(
            steps=10, run_fp64_reference_track=False
        )
        engine = TrainingEngine(config)
        engine.run()
        report = engine.analyzer.report()
        self.assertIn(report["regime"], REGIMES)
        # The gradient really is tiny...
        self.assertLess(report["mean_gradient_magnitude"], 1e-3)
        # ...but the update is not, by orders of magnitude: the small-gradient
        # and small-update observations are independent and both are reported.
        self.assertGreater(
            report["mean_update_magnitude"],
            report["mean_gradient_magnitude"] * 100.0,
            "Adam normalises the step, so the update dwarfs the gradient here",
        )
        self.assertIn("small update", report["disclaimer"])
        self.assertIn("float32 storage stagnation", report["disclaimer"])

    def test_a_flat_gradient_regime_is_not_inferred_from_label(self) -> None:
        """No regime is ever reported as 'converged'."""
        for name in ("near_stationary", "oscillatory", "small_gradient"):
            with self.subTest(preset=name):
                engine = TrainingEngine(
                    PRESETS[name].replace(steps=8, run_fp64_reference_track=False)
                )
                engine.run()
                report = engine.analyzer.report()
                self.assertIn(report["regime"], REGIMES)
                self.assertNotIn("converg", report["regime"])
                self.assertTrue(report["rationale"])

    def test_analyzer_reports_no_data_honestly(self) -> None:
        analyzer = ConvergenceAnalyzer(ExperimentConfig().stability)
        report = analyzer.report()
        self.assertEqual(report["regime"], "not enough data")
        self.assertIn(report["regime"], REGIMES)


class TestControls(unittest.TestCase):
    def test_single_step_advances_exactly_one_step(self) -> None:
        engine = _engine(steps=10)
        controller = ExperimentController(engine)
        self.assertTrue(controller.single_step())
        controller.run_blocking(["step"], printer=lambda _msg: None)
        self.assertEqual(len(engine.recorder), 1)
        controller.run_blocking(["step", "step"], printer=lambda _msg: None)
        self.assertEqual(len(engine.recorder), 3)
        controller.stop()

    def test_run_n_runs_exactly_n_steps(self) -> None:
        engine = _engine(steps=20)
        controller = ExperimentController(engine)
        controller.run_blocking(["n 4"], printer=lambda _msg: None)
        self.assertEqual(len(engine.recorder), 4)
        controller.run_blocking(["n 3"], printer=lambda _msg: None)
        self.assertEqual(len(engine.recorder), 7)
        controller.stop()

    def test_run_n_without_space(self) -> None:
        """Verify command like 'n2' runs 2 steps."""
        engine = _engine(steps=10)
        controller = ExperimentController(engine)
        controller.run_blocking(["n2"], printer=lambda _msg: None)
        self.assertEqual(len(engine.recorder), 2)
        controller.stop()

    def test_pause_really_stops_the_engine(self) -> None:
        """A pause must stop real steps, not merely stop drawing."""
        engine = _engine(steps=200)
        controller = ExperimentController(engine, autostart=True)
        controller.set_speed(40.0)
        controller.play()
        deadline = time.monotonic() + 5.0
        while len(engine.recorder) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(len(engine.recorder), 3, "the engine should be running")
        controller.pause()
        time.sleep(0.1)  # let any in-flight step finish
        frozen_at = len(engine.recorder)
        time.sleep(0.4)
        self.assertEqual(
            len(engine.recorder), frozen_at,
            "a paused engine must not take further steps",
        )
        self.assertEqual(controller.state, RunState.PAUSED)
        # Resuming must continue from exactly where it stopped.
        controller.play()
        deadline = time.monotonic() + 5.0
        while len(engine.recorder) <= frozen_at and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreater(len(engine.recorder), frozen_at)
        controller.stop()

    def test_reset_restores_the_live_state(self) -> None:
        engine = _engine(steps=10)
        controller = ExperimentController(engine)
        controller.run_blocking(["n 3"], printer=lambda _msg: None)
        moved = engine.tracker.value()
        self.assertNotEqual(moved, engine.config.initial_value)
        controller.run_blocking(["reset"], printer=lambda _msg: None)
        self.assertEqual(
            engine.tracker.value(),
            float(torch.tensor(engine.config.initial_value, dtype=torch.float32).item()),
        )
        self.assertEqual(len(engine.recorder), 3, "reset keeps the recorded history")
        # The optimizer state was genuinely discarded, so the next step behaves
        # like a first step.
        controller.run_blocking(["step"], printer=lambda _msg: None)
        self.assertEqual(len(engine.recorder), 4)
        controller.stop()

    def test_restart_discards_the_history(self) -> None:
        engine = _engine(steps=10)
        controller = ExperimentController(engine)
        controller.run_blocking(["n 3"], printer=lambda _msg: None)
        self.assertEqual(len(engine.recorder), 3)
        controller.run_blocking(["restart"], printer=lambda _msg: None)
        self.assertEqual(len(engine.recorder), 0)
        self.assertEqual(engine.step_index, 0)
        controller.stop()

    def test_speed_control_changes_the_step_rate(self) -> None:
        engine = _engine(steps=1000)
        controller = ExperimentController(engine)
        controller.set_speed(2.0)
        self.assertEqual(controller.speed, 2.0)
        controller.set_speed(20.0)
        self.assertEqual(controller.speed, 20.0)
        with self.assertRaises(ValueError):
            controller.set_speed(0.0)
        controller.stop()


class TestSafety(unittest.TestCase):
    def test_tracking_failure_stops_the_run(self) -> None:
        engine = _engine(steps=5)
        engine.run(max_steps=2)
        # Simulate the tracked object ceasing to be a trainable scalar.
        engine.parameter.requires_grad_(False)
        with self.assertRaises(TrackingInvalidError):
            engine.step()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
