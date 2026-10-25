"""Regression coverage for prefix integrity and exact, failure-safe exports."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    from . import _bootstrap
except ImportError:
    import _bootstrap  # type: ignore[import-not-found, no-redef]

import numpy as np
import torch

from single_parameter_observatory import exporters
from single_parameter_observatory.config import ExperimentConfig
from single_parameter_observatory.errors import HistoryCorruptionError, NumericalConsistencyError
from single_parameter_observatory.recorder import Recorder, build_final_report
from single_parameter_observatory.training_engine import TrainingEngine


def _engine(steps: int = 2) -> TrainingEngine:
    engine = TrainingEngine(ExperimentConfig(steps=steps, run_fp64_reference_track=False))
    engine.run()
    return engine


class TestRecordPrefixIntegrity(unittest.TestCase):
    def test_changing_a_predecessor_requires_rehashing_the_suffix(self) -> None:
        engine = _engine()
        original, suffix = engine.recorder.records
        changed = dataclasses.replace(original, loss=original.loss + 1.0).with_hash(Recorder.GENESIS_HASH)
        # Rewrite just the pointer. Previously each record hashed only its own
        # observations, so this forged prefix still returned chain_verified.
        engine.recorder._records[:] = [
            changed, dataclasses.replace(suffix, previous_hash=changed.record_hash)
        ]
        self.assertEqual(engine.recorder.verify_chain(), (False, 1))

    def test_duplicate_and_skipped_steps_are_rejected(self) -> None:
        source = _engine(1).recorder.records[0]
        recorder = Recorder(ExperimentConfig(steps=3))
        recorder.record(dataclasses.replace(source, previous_hash="", record_hash=""))
        for step in (0, 2):
            with self.subTest(step=step), self.assertRaises(HistoryCorruptionError):
                recorder.record(dataclasses.replace(source, step=step, previous_hash="", record_hash=""))
        self.assertEqual(len(recorder), 1)
        self.assertTrue(recorder.verify_chain()[0])

    def test_appending_cannot_silently_reseal_an_altered_record(self) -> None:
        source = _engine(1).recorder.records[0]
        recorder = Recorder(ExperimentConfig(steps=1))
        with self.assertRaises(HistoryCorruptionError):
            recorder.record(dataclasses.replace(source, gradient=source.gradient + 1.0))
        self.assertEqual(len(recorder), 0)

    def test_frozen_records_detach_caller_owned_lists_and_snapshot_shapes(self) -> None:
        source = _engine(1).recorder.records[0]
        losses = [source.loss]
        shape = [1]
        snapshot = dataclasses.replace(source.stages[0], shape=shape)  # type: ignore[arg-type]
        stages = [snapshot]
        record = dataclasses.replace(source, stages=stages, accumulated_losses=losses)  # type: ignore[arg-type]
        sealed = record.with_hash(Recorder.GENESIS_HASH)
        losses.append(123.0)
        stages.clear()
        shape.append(999)
        self.assertEqual(sealed.accumulated_losses, (source.loss,))
        self.assertEqual(sealed.stages[0].shape, (1,))
        self.assertEqual(sealed.record_hash, sealed.compute_hash())

    def test_live_tensors_cannot_be_retained_in_a_record(self) -> None:
        source = _engine(1).recorder.records[0]
        with self.assertRaises(NumericalConsistencyError):
            dataclasses.replace(source, loss=torch.tensor(1.0, requires_grad=True))  # type: ignore[arg-type]

    def test_nonfinite_float_cannot_be_substituted_with_its_json_string(self) -> None:
        source = _engine(1).recorder.records[0]
        # The JSON encoding deliberately represents nonfinite floats as
        # strings. Runtime schema checks prevent a real string observation
        # from colliding with that numerical observation in a record hash.
        with self.assertRaises(NumericalConsistencyError):
            dataclasses.replace(source, loss="nan")  # type: ignore[arg-type]

    def test_hash_verification_detects_type_tampering_that_has_identical_json(self) -> None:
        source = _engine(1).recorder.records[0]
        recorder = Recorder(ExperimentConfig(steps=1))
        sealed = recorder.record(dataclasses.replace(source, loss=float("nan"), previous_hash="", record_hash=""))
        original_json = sealed.to_dict()
        object.__setattr__(sealed, "loss", "nan")
        self.assertEqual(sealed.to_dict(), original_json)
        self.assertEqual(recorder.verify_chain(), (False, 0))

    def test_landscapes_are_detached_on_ingestion_and_access(self) -> None:
        recorder = Recorder(ExperimentConfig(steps=1))
        source: dict[str, Any] = {"positions": [1.0, 2.0], "meta": {"hash": "original"}}
        recorder.add_landscape(source)
        source["positions"].append(3.0)
        view = recorder.landscapes
        view[0]["meta"]["hash"] = "changed"
        self.assertEqual(recorder.landscapes, [{"positions": [1.0, 2.0], "meta": {"hash": "original"}}])


class TestExactArchiveExports(unittest.TestCase):
    def test_npz_preserves_optional_observations_with_presence_masks(self) -> None:
        engine = _engine()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "numeric.data")
            self.assertEqual(exporters.export_trajectory_npz(engine.recorder, path), path)
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(path + ".npz"))
            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("moment1_before", archive.files)
                self.assertEqual(archive["moment1_before__present"].tolist(), [False, True])
                self.assertEqual(archive["moment1_before"][1], engine.recorder.records[1].moment1_before)
                self.assertEqual(archive["parameter_after"].tolist(), [r.parameter_after for r in engine.recorder.records])

    def test_npz_preserves_integer_counts_larger_than_int64(self) -> None:
        source = _engine(1).recorder.records[0]
        huge_count = 2**180 + 17
        recorder = Recorder(ExperimentConfig(steps=1))
        recorder.record(dataclasses.replace(
            source, ulp_distance_same_binade=huge_count, previous_hash="", record_hash=""
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trajectory.npz")
            exporters.export_trajectory_npz(recorder, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(int(archive["ulp_distance_same_binade"][0]), huge_count)
                metadata = json.loads(str(archive["__column_encodings__"]))
                self.assertEqual(metadata["ulp_distance_same_binade"], "arbitrary_precision_integer_decimal")

    def test_npz_preserves_observed_nan_distinct_from_missing(self) -> None:
        engine = _engine()
        recorder = Recorder(engine.config)
        for index, source in enumerate(engine.recorder.records):
            recorder.record(dataclasses.replace(
                source, moment1_before=None if index == 0 else float("nan"),
                previous_hash="", record_hash="",
            ))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trajectory.npz")
            exporters.export_trajectory_npz(recorder, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(np.isnan(archive["moment1_before"]).all())
                self.assertEqual(archive["moment1_before__present"].tolist(), [False, True])

    def test_optional_parquet_engine_absence_preserves_existing_artifact(self) -> None:
        engine = _engine(1)
        frame = types.SimpleNamespace(to_parquet=lambda *args, **kwargs: self._missing_engine())
        pandas = types.SimpleNamespace(DataFrame=lambda rows: frame)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"pandas": pandas}):
            path = Path(directory) / "trajectory.parquet"
            path.write_bytes(b"old valid archive")
            self.assertIsNone(exporters.export_parquet_if_available(engine.recorder, str(path)))
            self.assertEqual(path.read_bytes(), b"old valid archive")
            self.assertEqual([item.name for item in Path(directory).iterdir()], [path.name])

    @staticmethod
    def _missing_engine() -> None:
        raise ImportError("Unable to find a usable parquet engine")

    def test_parquet_does_not_swallow_disk_errors(self) -> None:
        engine = _engine(1)

        def fail(handle, **kwargs):
            if hasattr(handle, "write"):
                handle.write(b"partial new archive")
            else:
                Path(handle).write_bytes(b"partial new archive")
            raise OSError("disk full")

        pandas = types.SimpleNamespace(DataFrame=lambda rows: types.SimpleNamespace(to_parquet=fail))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"pandas": pandas}):
            path = Path(directory) / "trajectory.parquet"
            path.write_bytes(b"old valid archive")
            with self.assertRaisesRegex(OSError, "disk full"):
                exporters.export_parquet_if_available(engine.recorder, str(path))
            self.assertEqual(path.read_bytes(), b"old valid archive")

    def test_parquet_preserves_arbitrary_precision_integer_counts(self) -> None:
        try:
            import pandas as pd
            import pyarrow as pa
        except ImportError:
            self.skipTest("optional parquet dependencies are unavailable")
        source = _engine(1).recorder.records[0]
        count = -(2**170) + 1
        recorder = Recorder(ExperimentConfig(steps=1))
        recorder.record(dataclasses.replace(source, ulp_distance_same_binade=count, previous_hash="", record_hash=""))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trajectory.parquet")
            self.assertEqual(exporters.export_parquet_if_available(recorder, path), path)
            self.assertEqual(int(pd.read_parquet(path)["ulp_distance_same_binade"].iloc[0]), count)

    def test_failed_json_write_preserves_preexisting_file(self) -> None:
        recursive: dict[str, Any] = {}
        recursive["self"] = recursive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text('{"old": true}', encoding="utf-8")
            with self.assertRaises((RecursionError, ValueError)):
                exporters.export_provenance_json(recursive, str(path))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual([item.name for item in Path(directory).iterdir()], [path.name])

    def test_reusing_export_directory_does_not_leave_old_run_files(self) -> None:
        engine = _engine(1)
        with tempfile.TemporaryDirectory() as directory:
            exporters.export_all(engine.recorder, directory, provenance={"run": "old"}, analyzer=engine.analyzer)
            self.assertTrue((Path(directory) / "trajectory.json").is_file())
            empty = Recorder(engine.config)
            exported = exporters.export_all(empty, directory)
            self.assertIsNone(exported["trajectory.json"])
            for name in exporters.EXPORT_FILES:
                if exported.get(name) is None:
                    self.assertFalse((Path(directory) / name).exists(), name)
            self.assertEqual(json.loads((Path(directory) / "summary.json").read_text())["record_count"], 0)

    def test_empty_report_is_preserved_and_not_described_as_an_empty_run(self) -> None:
        engine = _engine(1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "summary.json")
            exporters.export_summary_json(engine.config, engine.recorder, {}, path)
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload["report"], {})
            self.assertIsNone(payload["note"])

    def test_corrupt_history_is_rejected_before_export_writes(self) -> None:
        engine = _engine(1)
        original = engine.recorder.records[0]
        engine.recorder._records[0] = dataclasses.replace(original, gradient=123.0)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new-export"
            with self.assertRaises(HistoryCorruptionError):
                exporters.export_all(engine.recorder, str(destination))
            self.assertFalse(destination.exists())
            with self.assertRaises(HistoryCorruptionError):
                exporters.export_trajectory_csv(engine.recorder, str(Path(directory) / "trajectory.csv"))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_required_npz_export_errors_propagate(self) -> None:
        engine = _engine(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            exporters, "export_trajectory_npz", side_effect=OSError("disk full")
        ), self.assertRaisesRegex(OSError, "disk full"):
            exporters.export_all(engine.recorder, directory)


class TestDivergedReportExtrema(unittest.TestCase):
    def test_infinite_observations_are_included_in_extrema(self) -> None:
        source = _engine(1).recorder.records[0]
        recorder = Recorder(ExperimentConfig(steps=1))
        recorder.record(dataclasses.replace(
            source, loss=float("inf"), gradient=float("-inf"), actual_delta=float("inf"),
            finite=False, nonfinite_detected=True, previous_hash="", record_hash="",
        ))
        report = build_final_report(recorder)
        for name in ("maximum_loss", "max_abs_gradient", "max_abs_update", "mean_abs_update"):
            self.assertEqual(report[name], float("inf"), name)
        self.assertEqual(report["minimum_loss"], float("inf"))


if __name__ == "__main__":
    unittest.main()
