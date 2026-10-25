"""Recording, immutability, integrity, export and historical replay."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unittest

try:  # when discovered as part of the `tests` package
    from . import _bootstrap
except ImportError:  # when the tests directory itself is on sys.path
    import _bootstrap  # type: ignore[import-not-found, no-redef]

import torch

from single_parameter_observatory import exporters
from single_parameter_observatory.config import ExperimentConfig, ModelConfig
from single_parameter_observatory.errors import HistoryCorruptionError, NonFiniteStateError
from single_parameter_observatory.recorder import (
    EVENT_CODES,
    Recorder,
    canonical_json,
    render_final_report,
)
from single_parameter_observatory.training_engine import TrainingEngine


def _engine(steps: int = 5, **overrides) -> TrainingEngine:
    settings: dict = {
        "optimizer": "adamw",
        "steps": steps,
        "initial_value": -0.7,
        "run_fp64_reference_track": False,
    }
    settings.update(overrides)
    return TrainingEngine(ExperimentConfig(name="test-recording", **settings))


class TestRecording(unittest.TestCase):
    def test_history_length(self) -> None:
        engine = _engine(steps=7)
        records = engine.run()
        self.assertEqual(len(records), 7)
        self.assertEqual(len(engine.recorder), 7)
        self.assertEqual([r.step for r in records], list(range(7)))
        # The history must not grow beyond the configured number of steps.
        engine.run()
        self.assertEqual(len(engine.recorder), 7)
        # A full run is a fixed point.
        self.assertIsNone(engine.step())
        self.assertEqual(len(engine.recorder), 7)

    def test_records_are_immutable(self) -> None:
        engine = _engine(steps=2)
        engine.run()
        record = engine.recorder.records[0]
        with self.assertRaises(Exception):
            record.loss = 0.0  # type: ignore[misc]
        with self.assertRaises(Exception):
            record.steps = 5  # type: ignore[misc, attr-defined]

    def test_history_hash_chain_detects_tampering(self) -> None:
        import dataclasses

        engine = _engine(steps=3)
        engine.run()
        ok, bad_step = engine.recorder.verify_chain()
        self.assertTrue(ok, f"chain should verify, first bad step {bad_step}")
        engine.recorder.assert_chain()

        # Tamper with a stored observation, as a careless editor would.
        original = engine.recorder._records[1]
        tampered = dataclasses.replace(original, loss=original.loss + 1.0)
        engine.recorder._records[1] = tampered
        ok, bad_step = engine.recorder.verify_chain()
        self.assertFalse(ok, "a modified record must break the chain")
        self.assertEqual(bad_step, 1)
        with self.assertRaises(HistoryCorruptionError):
            engine.recorder.assert_chain()

    def test_report_exposes_both_ulp_metrics(self) -> None:
        """The run aggregate must not be the sign-crossing-dominated count.

        `total_ulp_movement` is the movement metric; the exact
        representable-value count is reported beside it, along with the steps
        that crossed zero, so a reader can see which number is which.
        """
        engine = _engine(steps=12)
        engine.run()
        report = engine.final_report()
        for key in (
            "total_ulp_movement",
            "total_ulp_movement_note",
            "total_ulp_movement_steps",
            "total_representable_values_travelled",
            "total_representable_values_note",
            "sign_crossing_steps",
        ):
            self.assertIn(key, report, f"the report is missing {key}")

        # The note must say why the second number is not a movement summary.
        self.assertIn("sign crossing", report["total_representable_values_note"])
        self.assertIn("not a summary", report["total_representable_values_note"])
        self.assertIn("starting value", report["total_ulp_movement_note"])

        rendered = render_final_report(report)
        self.assertIn("TOTAL ULP MOVEMENT (in ULPs of each step's start)", rendered)
        self.assertIn("TOTAL REPRESENTABLE VALUES TRAVELLED", rendered)
        for record in engine.recorder.records:
            self.assertIn("ulp_distance_same_binade", record.to_dict())

    def test_event_log_records_real_events(self) -> None:
        engine = _engine(steps=2)
        engine.run()
        codes = set(engine.recorder.events.counts())
        self.assertIn("INITIALIZED", codes)
        self.assertIn("FORWARD_COMPLETE", codes)
        self.assertIn("BACKWARD_STARTED", codes)
        self.assertIn("GRADIENT_AVAILABLE", codes)
        self.assertIn("OPTIMIZER_STATE_READ", codes)
        self.assertIn("OPTIMIZER_STEP_COMPLETED", codes)
        self.assertIn("RUN_COMPLETE", codes)
        for code in codes:
            self.assertIn(code, EVENT_CODES, f"{code} is not a declared event code")
        first = engine.recorder.records[0]
        self.assertIn("FORWARD_COMPLETE", first.events)

    def test_gradient_lifecycle_is_recorded(self) -> None:
        for set_to_none in (True, False):
            with self.subTest(set_to_none=set_to_none):
                # Two steps: on the first, no gradient exists yet, and
                # zero_grad(set_to_none=False) has nothing to zero.
                engine = _engine(steps=2, zero_grad_set_to_none=set_to_none)
                records = engine.run()
                first, second = records
                self.assertEqual(first.zero_grad_set_to_none, set_to_none)
                self.assertIsNone(first.grad_after_zero_grad)
                self.assertIsNotNone(first.grad_after_backward)
                self.assertEqual(first.gradient, first.grad_after_backward)
                if set_to_none:
                    self.assertIsNone(second.grad_after_zero_grad)
                else:
                    # The gradient from step 0 now exists, so it is zeroed.
                    self.assertEqual(second.grad_after_zero_grad, 0.0)
                self.assertEqual(second.grad_before_backward, second.grad_after_zero_grad)

    def test_nonfinite_detection(self) -> None:
        """A diverging real run must be detected and stopped, not carried on."""
        engine = _engine(
            steps=40,
            optimizer="sgd",
            lr=1e9,
            weight_decay=0.0,
            initial_value=-5.0,
            stop_on_nonfinite=True,
            model=ModelConfig(use_tanh=False),
        )
        engine.run()
        report = engine.final_report()
        self.assertTrue(engine.aborted, "the run should have aborted")
        self.assertIsNotNone(engine.abort_reason)
        self.assertIn("non-finite", str(engine.abort_reason))
        self.assertTrue(report["nonfinite_detected"])
        self.assertLess(len(engine.recorder), 40, "the run must stop early")
        last = engine.recorder.records[-1]
        self.assertFalse(last.finite)
        self.assertIn("NONFINITE_DETECTED", last.events)
        self.assertIn("RUN_ABORTED", engine.recorder.events.counts())

    def test_tanh_saturation_prevents_divergence(self) -> None:
        """A saturated tanh kills the gradient, so a huge lr does not diverge.

        This is why the diverging preset switches the model to its linear form:
        with tanh, SGD at lr=1e9 settles at a large but finite value instead of
        producing infinities, and claiming otherwise would be a false report.
        """
        engine = _engine(
            steps=12,
            optimizer="sgd",
            lr=1e9,
            weight_decay=0.0,
            initial_value=-5.0,
        )
        engine.run()
        self.assertFalse(engine.aborted, "the saturated run should not diverge")
        self.assertTrue(math.isfinite(engine.tracker.value()))
        self.assertLess(abs(engine.tracker.value()), 1e12)

    def test_final_report_contains_measured_values(self) -> None:
        engine = _engine(steps=6)
        engine.run()
        report = engine.final_report()
        for key in (
            "initial_parameter", "final_parameter", "total_parameter_change",
            "steps", "initial_loss", "final_loss", "minimum_loss",
            "initial_gradient", "final_gradient", "max_abs_gradient",
            "max_abs_update", "mean_abs_update", "std_parameter_over_final_window",
            "gradient_sign_changes", "update_sign_changes", "unchanged_fp32_steps",
            "total_ulp_movement", "max_rounding_residual",
            "max_reconstruction_ulp_error", "chain_verified",
        ):
            self.assertIn(key, report, f"report is missing {key}")
        self.assertEqual(report["steps"], 6)
        self.assertTrue(report["chain_verified"])
        self.assertEqual(
            report["initial_parameter"],
            float(torch.tensor(-0.7, dtype=torch.float32).item()),
        )
        self.assertEqual(report["validation"]["reconstruction_failures"], 0)
        self.assertIn("COMPLETE NUMERICAL REPORT", render_final_report(report))


class TestExport(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="observatory-export-")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def test_export(self) -> None:
        engine = _engine(steps=5)
        engine.run()
        report = engine.final_report()
        written = exporters.export_all(
            engine.recorder,
            self.directory,
            report=report,
            provenance=engine.provenance(),
            analyzer=engine.analyzer,
        )
        for name in ("summary.json", "trajectory.csv", "trajectory.json",
                     "bit_history.csv", "optimizer_state_history.csv",
                     "events.csv", "validation.json", "provenance.json"):
            self.assertIn(name, written, f"{name} was not produced")
            if name in ("trajectory.parquet",):
                continue
            self.assertIsNotNone(written[name], f"{name} was not written")
            path = written[name]
            assert path is not None
            self.assertTrue(
                os.path.exists(path), f"{name} missing on disk"
            )
        self.assertTrue(written.get("trajectory.npz"))

        # The CSV must have one row per step and preserve the exact values.
        import csv as csv_module

        trajectory_csv = written["trajectory.csv"]
        assert trajectory_csv is not None
        with open(trajectory_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv_module.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertEqual([int(row["step"]) for row in rows], [0, 1, 2, 3, 4])
        first = engine.recorder.records[0]
        self.assertEqual(float(rows[0]["parameter_after"]), first.parameter_after)
        self.assertEqual(int(rows[0]["parameter_after_uint32"]), first.parameter_after_uint32)

        # A repr round-trip must reproduce the stored double exactly.
        self.assertEqual(
            float(rows[1]["parameter_after"]),
            engine.recorder.records[1].parameter_after,
            "the exported decimal must be the exact stored value",
        )

        trajectory_json = written["trajectory.json"]
        assert trajectory_json is not None
        with open(trajectory_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(len(payload["records"]), 5)
        self.assertTrue(payload["chain_verified"])
        self.assertIn("config", payload)

        summary_path = written["summary.json"]
        assert summary_path is not None
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["record_count"], 5)

        bit_history = written["bit_history.csv"]
        optimizer_history = written["optimizer_state_history.csv"]
        assert bit_history is not None and optimizer_history is not None
        self.assertTrue(os.path.exists(bit_history))
        self.assertTrue(os.path.exists(optimizer_history))

    def test_export_of_an_empty_run_is_explicit(self) -> None:
        """A run with no steps exports an explicit reason, not a silent empty file."""
        recorder = Recorder(ExperimentConfig(steps=1, run_fp64_reference_track=False))
        written = exporters.export_all(recorder, self.directory)
        self.assertIsNone(written["trajectory.csv"])
        self.assertIn("trajectory_empty", written)


class TestReplay(unittest.TestCase):
    def test_replay_uses_recorded_states(self) -> None:
        """Replay must return exactly the recorded values, in recorded order."""
        engine = _engine(steps=6)
        engine.run()
        recorded = [
            (r.step, r.parameter_after, r.parameter_after_uint32, r.loss, r.gradient)
            for r in engine.recorder.records
        ]
        replayed = []
        for record in engine.recorder.records:
            replayed.append(
                (record.step, record.parameter_after, record.parameter_after_uint32,
                 record.loss, record.gradient)
            )
        self.assertEqual(replayed, recorded)

        # Replaying many times changes nothing and produces nothing new.
        for _ in range(3):
            again = [(r.step, r.parameter_after) for r in engine.recorder.records]
            self.assertEqual(again, [(s, w) for s, w, _u, _l, _g in recorded])
        self.assertEqual(len(engine.recorder), 6)

    def test_replay_survives_a_finished_run(self) -> None:
        engine = _engine(steps=3)
        engine.run()
        first_pass = list(engine.recorder.records)
        self.assertTrue(engine.finished)
        self.assertIsNone(engine.step(), "a finished run must not step again")
        self.assertEqual(list(engine.recorder.records), first_pass)

    def test_canonical_json_is_deterministic(self) -> None:
        payload = {"b": 1, "a": [1.5, -0.0, 2], "c": (1, 2)}
        self.assertEqual(canonical_json(payload), canonical_json(dict(payload)))
        self.assertLess(canonical_json(payload).index('"a"'), canonical_json(payload).index('"b"'))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
