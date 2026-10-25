"""Behavioral regressions for dataset, numerical, and worker lifecycle repairs."""

from __future__ import annotations

import math
import io
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any
from unittest.mock import patch

import torch

from single_parameter_observatory import fp32_inspector as f32
from single_parameter_observatory import numerical_analysis as na
from single_parameter_observatory import optimizer_inspector as oi
from single_parameter_observatory import validation as vd
from single_parameter_observatory.config import DatasetConfig, ExperimentConfig, ModelConfig, PRESETS
from single_parameter_observatory.errors import ConfigError, TrackingInvalidError
from single_parameter_observatory.model import FP64ReferenceModel, ObservatoryModel, build_dataset
from single_parameter_observatory.tracker import TrackedParameter
from single_parameter_observatory.training_engine import ExperimentController, RunState, TrainingEngine


class TestDatasetBatchCoverage(unittest.TestCase):
    def test_configured_batches_cover_remainder_once_per_epoch(self) -> None:
        for shuffled in (False, True):
            with self.subTest(shuffled=shuffled):
                dataset = build_dataset(
                    DatasetConfig(samples=5, batch_size=2, shuffle_batches=shuffled), ModelConfig()
                )
                self.assertEqual(dataset.steps_per_epoch, 3)
                batches = [dataset.batch(index) for index in range(3)]
                self.assertEqual([int(batch[0].shape[0]) for batch in batches], [2, 2, 1])
                order = dataset.shuffle_permutation or tuple(range(5))
                self.assertTrue(torch.equal(torch.cat([batch[0] for batch in batches]), dataset.inputs[list(order)]))
                self.assertTrue(torch.equal(dataset.batch(3)[0], batches[0][0]))
                self.assertEqual(dataset.batch(3)[3:], (1, 3))
                self.assertEqual(dataset.describe()["batch_size"], 2)

    def test_alternating_stream_reaches_the_negative_rows(self) -> None:
        dataset = build_dataset(DatasetConfig(samples=4, walk="alternating"), ModelConfig())
        actual = torch.cat([dataset.batch(index)[0] for index in range(4)])
        self.assertTrue(torch.equal(actual, dataset.inputs))
        self.assertLess(float(actual[1].item()), 0.0)


class TestAccumulationAndReference(unittest.TestCase):
    def test_accumulated_objective_is_validated_and_reference_uses_same_steps(self) -> None:
        for schedule in ("constant", "exponential", "cosine"):
            with self.subTest(schedule=schedule):
                config = ExperimentConfig(
                    optimizer="sgd", initial_value=0.25, steps=4, lr=0.03,
                    lr_schedule=schedule, lr_final_factor=0.7, weight_decay=0.0,
                    accumulation_micro_batches=2,
                    dataset=DatasetConfig(samples=5, batch_size=2),
                    model=ModelConfig(use_tanh=False),
                )
                engine = TrainingEngine(config)
                reference = FP64ReferenceModel(config.initial_value, config.model)
                optimizer = oi.make_optimizer(config, reference.w).optimizer
                for record in engine.run():
                    optimizer.param_groups[0]["lr"] = record.learning_rate
                    optimizer.zero_grad(set_to_none=True)
                    for micro in range(2):
                        x, target, *_ = engine.dataset.batch(record.step * 2 + micro)
                        reference.loss(x, target).backward()
                    optimizer.step()
                    assert record.fp64_track_value is not None
                    assert record.gradient_analytical is not None
                    self.assertAlmostEqual(record.fp64_track_value, float(reference.w.item()), places=14)
                    self.assertEqual(record.gradient_validation_status, "PASS")
                    self.assertEqual(record.loss, sum(record.accumulated_losses))
                    self.assertAlmostEqual(record.gradient, record.gradient_analytical, delta=3e-6)
                    if record.landscape_computed:
                        assert record.landscape_first_derivative is not None
                        self.assertAlmostEqual(record.landscape_first_derivative, record.gradient_analytical, places=9)

    def test_reset_restarts_schedule_dataset_and_reference_state(self) -> None:
        engine = TrainingEngine(ExperimentConfig(
            steps=6, lr_schedule="exponential", lr_final_factor=0.5,
            dataset=DatasetConfig(samples=5, batch_size=2), accumulation_micro_batches=2,
        ))
        first = engine.step()
        engine.step()
        engine.reset_live_state()
        reset = engine.step()
        self.assertIsNotNone(first)
        self.assertIsNotNone(reset)
        assert first is not None and reset is not None
        self.assertEqual(len(engine.recorder), 3)
        self.assertEqual(reset.learning_rate, first.learning_rate)
        self.assertEqual(reset.parameter_after_uint32, first.parameter_after_uint32)
        self.assertEqual(reset.fp64_track_value, first.fp64_track_value)
        self.assertEqual(reset.global_batch, first.global_batch)
        self.assertEqual(len(engine.analyzer.metrics), 1)

    def test_schedule_provenance_and_restart_initialization_are_truthful(self) -> None:
        engine = TrainingEngine(ExperimentConfig(steps=2, lr_schedule="cosine"))
        self.assertIn("CosineAnnealingLR", engine.provenance()["lr_schedule"])
        engine.restart()
        self.assertEqual(engine.recorder.events.counts().get("INITIALIZED"), 1)

    def test_provenance_callers_cannot_rewrite_cached_evidence(self) -> None:
        engine = TrainingEngine(ExperimentConfig(steps=1))
        first = engine.provenance()
        first["environment"]["torch_version"] = "corrupted"
        self.assertEqual(engine.provenance()["environment"]["torch_version"], str(torch.__version__))

    def test_restart_discards_the_previous_reconstruction_lock(self) -> None:
        engine = TrainingEngine(ExperimentConfig(steps=2))
        engine.step()
        self.assertTrue(engine._variant_locked)
        engine.restart()
        self.assertFalse(engine._variant_locked)
        self.assertIsNone(engine._active_variant)

    def test_building_an_observatory_preserves_the_host_rng(self) -> None:
        torch.manual_seed(1987)
        before = torch.random.get_rng_state().clone()
        TrainingEngine(ExperimentConfig(steps=1))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))


class TestNumericalEdgeCases(unittest.TestCase):
    def test_zero_gradient_zero_epsilon_records_nan_and_aborts(self) -> None:
        engine = TrainingEngine(ExperimentConfig(
            optimizer="adam", steps=3, initial_value=1.0, eps=0.0, weight_decay=0.0,
            dataset=DatasetConfig(samples=1, input_min=1.0, input_max=1.0, target_true_parameter=1.0),
        ))
        records = engine.run()
        self.assertEqual(len(records), 1)
        self.assertTrue(engine.aborted)
        self.assertTrue(math.isnan(records[0].parameter_after))
        self.assertFalse(records[0].finite)
        self.assertIn("NONFINITE", records[0].reconstruction_status)
        self.assertIn("NONFINITE_DETECTED", records[0].events)
        engine.final_report()

    def test_disabling_nonfinite_stop_continues_and_keeps_observations(self) -> None:
        engine = TrainingEngine(ExperimentConfig(
            optimizer="adam", steps=3, initial_value=1.0, eps=0.0, weight_decay=0.0,
            stop_on_nonfinite=False,
            dataset=DatasetConfig(samples=1, input_min=1.0, input_max=1.0, target_true_parameter=1.0),
        ))
        records = engine.run()
        self.assertEqual(len(records), 3)
        self.assertFalse(engine.aborted)
        self.assertTrue(all(not record.finite for record in records))
        engine.recorder.assert_chain()

    def test_divergence_can_continue_through_infinite_parameter_values(self) -> None:
        engine = TrainingEngine(PRESETS["diverging"].replace(steps=6, stop_on_nonfinite=False))
        records = engine.run()
        self.assertEqual(len(records), 6)
        self.assertTrue(any(math.isinf(record.parameter_after) for record in records))
        self.assertFalse(engine.aborted)
        self.assertTrue(engine.final_report()["nonfinite_detected"])

    def test_runtime_limits_reject_invalid_values_before_any_step(self) -> None:
        engine = TrainingEngine(ExperimentConfig(steps=1))
        for speed in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                engine.run(speed=speed)
        invalid_counts: tuple[Any, ...] = (-1, 1.5, True)
        for count in invalid_counts:
            with self.subTest(count=count), self.assertRaises(ValueError):
                engine.run(max_steps=count)
        self.assertEqual(len(engine.recorder), 0)

    def test_tracked_parameter_replacement_is_rejected(self) -> None:
        model = ObservatoryModel(0.5)
        tracker = TrackedParameter(model)
        model.w = torch.nn.Parameter(torch.tensor(0.5))
        with self.assertRaises(TrackingInvalidError):
            tracker.verify(step=0)

    def test_optimizer_state_overflow_is_detected_with_finite_loss_and_parameter(self) -> None:
        engine = TrainingEngine(ExperimentConfig(
            optimizer="adam", initial_value=1e-20, steps=3, weight_decay=0.0,
            model=ModelConfig(use_tanh=False),
            dataset=DatasetConfig(samples=1, input_min=1e21, input_max=1e21, target_true_parameter=0.0),
        ))
        records = engine.run()
        self.assertEqual(len(records), 1)
        self.assertTrue(engine.aborted)
        self.assertTrue(math.isfinite(records[0].parameter_after))
        self.assertTrue(math.isfinite(records[0].loss))
        self.assertTrue(math.isfinite(records[0].gradient))
        assert records[0].moment2 is not None
        self.assertTrue(math.isinf(records[0].moment2))
        self.assertFalse(records[0].finite)
        self.assertIn("NONFINITE", records[0].reconstruction_status)

    def test_minimum_normal_and_maximum_finite_bit_spacing(self) -> None:
        for value in (f32.SMALLEST_NORMAL_F32, -f32.SMALLEST_NORMAL_F32, torch.finfo(torch.float32).max):
            with self.subTest(value=value):
                check = vd.validate_float32_bits(torch.tensor(value, dtype=torch.float32))
                self.assertTrue(check.passed, check.status)
                self.assertTrue(vd.validate_ulp(value).passed)
        self.assertEqual(f32.ulp_size(torch.finfo(torch.float32).max), 2.0**104)

    def test_rounding_candidate_overflow_has_no_fraction_exception(self) -> None:
        result = na.analyse_rounding(1e100, torch.tensor(float("inf"), dtype=torch.float32))
        self.assertTrue(math.isinf(result.candidate_as_float32))
        self.assertFalse(result.exact_half_ulp_tie)

    def test_float64_candidate_respects_maximize_and_amsgrad_history(self) -> None:
        result = na.fp64_candidate_from_state(
            weight_before=1.0, gradient=2.0, moment1_before=0.0, moment2_before=0.0,
            step_index=1, lr=0.1, beta1=0.9, beta2=0.99, eps=0.0,
            weight_decay=0.0, optimizer="adam", maximize=True,
            amsgrad=True, max_moment2_before=100.0,
        )
        expected = 1.0 + 0.1 * 2.0 / math.sqrt(100.0 / 0.01)
        self.assertAlmostEqual(result, expected, places=14)

    def test_analytical_diagnostics_use_observed_buffer_values(self) -> None:
        model = ObservatoryModel(0.2, ModelConfig(input_scale=0.1, output_scale=0.3, use_tanh=False))
        x = torch.tensor([[1.0]])
        target = torch.tensor([[0.4]])
        with torch.no_grad():
            model.input_scale.fill_(0.7)
        candidate = FP64ReferenceModel(0.2, model.model_config)
        with torch.no_grad():
            candidate.input_scale.copy_(model.input_scale)
            candidate.output_scale.copy_(model.output_scale)
        candidate.loss(x, target).backward()
        assert candidate.w.grad is not None
        self.assertAlmostEqual(model.analytic_gradient_fp64(x, target, 0.2), float(candidate.w.grad.item()), places=14)


class TestControllerLifecycle(unittest.TestCase):
    @staticmethod
    def engine(steps: int = 100) -> TrainingEngine:
        engine = TrainingEngine(ExperimentConfig(steps=steps, run_fp64_reference_track=False))
        engine.provenance()
        return engine

    def wait_for(self, predicate, timeout: float = 4.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(predicate(), "worker did not reach the expected state")

    def test_worker_runs_exactly_n_then_parks(self) -> None:
        engine = self.engine()
        controller = ExperimentController(engine)
        controller.set_speed(500.0)
        try:
            controller.run_n(3)
            controller.start()
            self.wait_for(lambda: len(engine.recorder) >= 3)
            time.sleep(0.08)
            self.assertEqual(len(engine.recorder), 3)
            self.assertEqual(controller.state, RunState.PAUSED)
        finally:
            controller.stop()

    def test_worker_preserves_every_queued_single_step(self) -> None:
        engine = self.engine()
        controller = ExperimentController(engine)
        controller.set_speed(500.0)
        try:
            controller.single_step()
            controller.single_step()
            controller.single_step()
            controller.start()
            self.wait_for(lambda: len(engine.recorder) == 3)
            controller.pause()
        finally:
            controller.stop()

    def test_restart_after_finish_reuses_the_live_worker(self) -> None:
        engine = self.engine(steps=1)
        controller = ExperimentController(engine, autostart=True)
        try:
            controller.play()
            self.wait_for(lambda: controller.state == RunState.FINISHED)
            controller.restart_experiment()
            controller.play()
            self.wait_for(lambda: len(engine.recorder) == 1)
            self.assertEqual(engine.recorder.records[0].step, 0)
        finally:
            controller.stop()

    def test_unexpected_worker_error_is_raised_on_the_calling_thread(self) -> None:
        controller = ExperimentController(self.engine())
        try:
            with patch.object(controller.engine, "step", side_effect=RuntimeError("worker exploded")):
                controller.play()
                controller.start()
                self.wait_for(lambda: controller.state == RunState.ABORTED)
                with self.assertRaisesRegex(RuntimeError, "worker exploded"):
                    controller.raise_if_failed()
        finally:
            controller.stop()

    def test_pause_waits_for_inflight_step_and_callback(self) -> None:
        engine = self.engine()
        controller = ExperimentController(engine)
        entered = threading.Event()
        release = threading.Event()
        paused = threading.Event()
        original = engine.step

        def blocked_step():
            entered.set()
            if not release.wait(3.0):
                raise RuntimeError("test failed to release the worker")
            return original()

        controller.on_step = lambda record: None
        try:
            with patch.object(engine, "step", side_effect=blocked_step):
                controller.play()
                controller.start()
                self.assertTrue(entered.wait(3.0))
                def pause_worker() -> None:
                    controller.pause()
                    paused.set()

                pauser = threading.Thread(target=pause_worker)
                pauser.start()
                self.assertFalse(paused.wait(0.05))
                release.set()
                pauser.join(3.0)
                self.assertTrue(paused.is_set())
                self.assertEqual(len(engine.recorder), 1)
        finally:
            release.set()
            controller.stop()

    def test_failure_poll_does_not_wait_for_an_inflight_training_step(self) -> None:
        engine = self.engine()
        controller = ExperimentController(engine)
        entered = threading.Event()
        release = threading.Event()
        polled = threading.Event()
        original = engine.step

        def blocked_step():
            entered.set()
            if not release.wait(3.0):
                raise RuntimeError("test failed to release the worker")
            return original()

        def poll_failure() -> None:
            controller.raise_if_failed()
            polled.set()

        poller = threading.Thread(target=poll_failure)
        try:
            with patch.object(engine, "step", side_effect=blocked_step):
                controller.play()
                controller.start()
                self.assertTrue(entered.wait(3.0))
                poller.start()
                self.assertTrue(polled.wait(0.5), "error polling waited behind the active training step")
                self.assertFalse(release.is_set())
        finally:
            release.set()
            if poller.ident is not None:
                poller.join(3.0)
            controller.stop()

    def test_speed_rejects_nonfinite_values(self) -> None:
        controller = ExperimentController(self.engine())
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                controller.set_speed(value)

    def test_malformed_speed_commands_raise_domain_errors_before_training(self) -> None:
        controller = ExperimentController(self.engine())
        for command in ("speed nan", "speed -1", "speed nope", "sp inf", "speed 0"):
            with self.subTest(command=command), self.assertRaises(ConfigError):
                controller.run_blocking([command], printer=lambda message: None)
        self.assertEqual(len(controller.engine.recorder), 0)

    def test_cli_malformed_speed_commands_return_one_without_tracebacks(self) -> None:
        from single_parameter_observatory.main import main

        for command in ("speed nan", "speed -1", "speed nope"):
            output = io.StringIO()
            error = io.StringIO()
            with self.subTest(command=command), redirect_stdout(output), redirect_stderr(error):
                result = main(["controls", "--steps", "2", "--commands", command])
            self.assertEqual(result, 1)
            self.assertNotIn("Traceback", output.getvalue() + error.getvalue())
            self.assertIn("speed", (output.getvalue() + error.getvalue()).lower())


if __name__ == "__main__":
    unittest.main()
