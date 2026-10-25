"""Exercise real studio state transitions, exact continuation and portable evidence."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from single_parameter_observatory.studio_session import (
    PROJECT_SCHEMA, StudioStateError, StudioWorkspace, _digest,
)


class TestStudioWorkspace(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = StudioWorkspace()
        self.addCleanup(self.workspace.close)
        self.run_id = self.workspace.snapshot()["runs"][0]["id"]

    def create(self, **config):
        config.setdefault("dataset", {"samples": 13, "batch_size": 4, "shuffle_batches": True})
        config.setdefault("steps", 30)
        return self.workspace.dispatch("run.create", {"config": config})["run_id"]

    def step(self, workspace, run_id, count=1):
        return workspace.dispatch("run.step", {"run_id": run_id, "count": count})

    def reopen(self):
        result = StudioWorkspace()
        self.addCleanup(result.close)
        result.import_project(json.loads(json.dumps(self.workspace.export_project(), allow_nan=False)))
        return result

    def assert_observations_equal(self, left, right):
        left, right = dict(left), dict(right)
        for key in ("events", "previous_hash", "record_hash"):
            left.pop(key)
            right.pop(key)
        self.assertEqual(left, right)

    def test_starts_idle_and_nested_config_reaches_real_engine(self):
        snapshot = self.workspace.snapshot()
        self.assertEqual(snapshot["runs"][0]["state"], "idle")
        self.assertEqual(snapshot["runs"][0]["series"], [])
        run_id = self.create(model={"use_tanh": False}, lr=0.01)
        self.step(self.workspace, run_id)
        self.assertEqual(self.workspace.record(run_id, 0)["optimizer_name"], "adamw")
        self.assertFalse(self.workspace._runs[run_id].engine.config.model.use_tanh)
        self.assertIn("dataset", self.workspace.capabilities()["config_schema"])

    def test_json_roundtrip_continues_every_optimizer_and_schedules_exactly(self):
        for optimizer, schedule in (("sgd", "exponential"), ("adam", "cosine"), ("adamw", "constant")):
            with self.subTest(optimizer=optimizer, schedule=schedule):
                run_id = self.create(optimizer=optimizer, lr_schedule=schedule,
                                     accumulation_micro_batches=3,
                                     momentum=0.7 if optimizer == "sgd" else 0,
                                     amsgrad=optimizer != "sgd")
                self.step(self.workspace, run_id, 6)
                reopened = self.reopen()
                self.assertEqual(reopened._runs[run_id].controller.state, "paused")
                self.step(self.workspace, run_id, 4)
                self.step(reopened, run_id, 4)
                for index in range(6, 10):
                    self.assertEqual(self.workspace.record(run_id, index), reopened.record(run_id, index))

    def test_checkpoint_restore_and_fork_continue_actual_optimizer_memory(self):
        run_id = self.create(lr_schedule="cosine", accumulation_micro_batches=2)
        self.step(self.workspace, run_id, 5)
        checkpoint = self.workspace.dispatch("checkpoint.create", {"run_id": run_id, "name": "Five"})
        branch_id = self.workspace.dispatch("run.fork", {"run_id": run_id})["run_id"]
        restored_id = self.workspace.dispatch("checkpoint.restore", {
            "run_id": run_id, "checkpoint_id": checkpoint["checkpoint_id"],
        })["run_id"]
        for identity in (run_id, branch_id, restored_id):
            self.step(self.workspace, identity, 3)
        for index in range(5, 8):
            for identity in (branch_id, restored_id):
                self.assert_observations_equal(self.workspace.record(run_id, index), self.workspace.record(identity, index))
        self.assertEqual(self.workspace._runs[run_id].engine.step_index, 8)
        self.assertEqual(self.workspace._runs[branch_id].parent_id, run_id)

    def test_custom_csv_values_are_used_and_survive_restart_and_save_open(self):
        result = self.workspace.dispatch("run.dataset", {
            "run_id": self.run_id, "csv": "x,target\n-1,0.25\n0.5,0.8\n2,-0.3\n", "batch_size": 2,
        })
        run_id = result["run_id"]
        data = self.workspace.dataset(run_id)
        self.assertEqual(data["inputs"], [-1.0, 0.5, 2.0])
        self.assertEqual(data["provenance"]["kind"], "csv")
        self.step(self.workspace, run_id, 3)
        reopened = self.reopen()
        self.step(self.workspace, run_id, 2)
        self.step(reopened, run_id, 2)
        self.assertEqual(self.workspace.record(run_id, 4), reopened.record(run_id, 4))
        self.workspace.dispatch("run.restart", {"run_id": run_id})
        self.assertEqual(self.workspace.dataset(run_id)["inputs"], data["inputs"])
        self.step(self.workspace, run_id)
        self.assertEqual(self.workspace.record(run_id, 0)["target_value"], 0.25)

    def test_dataset_change_after_training_preserves_original_run_as_parent(self):
        self.step(self.workspace, self.run_id, 2)
        before = self.workspace.record(self.run_id, 1)
        branch = self.workspace.dispatch("run.dataset", {
            "run_id": self.run_id, "inputs": [1, 2], "targets": [3, 4],
        })["run_id"]
        self.assertNotEqual(branch, self.run_id)
        self.assertEqual(self.workspace.record(self.run_id, 1), before)
        self.assertEqual(self.workspace._runs[branch].parent_id, self.run_id)
        self.assertEqual(self.workspace._runs[branch].engine.step_index, 0)

    def test_changed_bits_and_optimizer_state_are_real_explicit_interventions(self):
        self.step(self.workspace, self.run_id, 2)
        self.workspace.dispatch("run.intervene", {
            "run_id": self.run_id, "parameter_hex": "3fc00001", "learning_rate": 0.025,
            "optimizer_state": {"exp_avg": 0.25, "exp_avg_sq": 0.5, "step": 2},
        })
        reopened = self.reopen()
        self.step(self.workspace, self.run_id)
        self.step(reopened, self.run_id)
        record = self.workspace.record(self.run_id, 2)
        self.assertEqual(record["parameter_before_uint32"], 0x3FC00001)
        self.assertEqual(record["moment1_before"], 0.25)
        self.assertEqual(record["moment2_before"], 0.5)
        self.assertEqual(record["learning_rate"], 0.025)
        self.assertIn("MANUAL_INTERVENTION", record["events"])
        self.assertEqual(record, reopened.record(self.run_id, 2))

    def test_manual_lr_applies_to_next_exponential_and_cosine_schedule(self):
        for schedule in ("exponential", "cosine"):
            with self.subTest(schedule=schedule):
                run_id = self.create(lr_schedule=schedule, lr_final_factor=0.8)
                self.step(self.workspace, run_id, 3)
                self.workspace.dispatch("run.intervene", {"run_id": run_id, "learning_rate": 0.02})
                self.step(self.workspace, run_id, 2)
                rate = self.workspace.record(run_id, 3)["learning_rate"]
                self.assertLessEqual(rate, 0.02)
                self.assertGreater(rate, 0.009)
                if schedule == "exponential":
                    self.assertAlmostEqual(rate, 0.02 * 0.8**3)
                reference = self.workspace._runs[run_id].engine.fp64_track
                assert reference is not None
                ref_rate = reference.handle.current_lr()
                self.assertEqual(ref_rate, self.workspace.record(run_id, 4)["learning_rate"])

    def test_live_reset_segment_analysis_and_schedule_survive_save_open(self):
        run_id = self.create(lr_schedule="cosine")
        self.step(self.workspace, run_id, 5)
        self.workspace.dispatch("run.reset", {"run_id": run_id})
        self.step(self.workspace, run_id, 2)
        reopened = self.reopen()
        self.step(self.workspace, run_id, 3)
        self.step(reopened, run_id, 3)
        self.assertEqual(self.workspace.record(run_id, 9), reopened.record(run_id, 9))
        self.assertEqual(len(reopened._runs[run_id].engine.analyzer.metrics), 5)

    def test_corrupt_engine_digest_or_chain_rejects_without_replacing_workspace(self):
        self.step(self.workspace, self.run_id, 2)
        before = self.workspace.record(self.run_id, 1)
        original = self.workspace.export_project()
        for mode in ("digest", "chain"):
            project = copy.deepcopy(original)
            captured = project["runs"][0]["engine"]
            if mode == "digest":
                captured["state_hash"] = "0" * 64
            else:
                state = dict(captured["state"]["$map"])
                state["records"][1]["fields"]["loss"] += 1
                captured["state_hash"] = _digest(captured["state"])
            with self.assertRaises((ValueError, RuntimeError)):
                self.workspace.import_project(project)
            self.assertEqual(self.workspace.record(self.run_id, 1), before)

    def test_checkpoint_corruption_rejects_whole_import(self):
        self.step(self.workspace, self.run_id, 2)
        self.workspace.dispatch("checkpoint.create", {"run_id": self.run_id})
        project = self.workspace.export_project()
        project["runs"][0]["checkpoints"][0]["state"]["state_hash"] = "bad"
        with self.assertRaises(ValueError):
            self.workspace.import_project(project)
        self.assertEqual(self.workspace._runs[self.run_id].engine.step_index, 2)

    def test_project_decoder_rejects_arbitrary_types_and_tensor_length(self):
        original = self.workspace.export_project()
        for bad in ({"$type": "eval", "fields": {}}, {"$tensor": "torch.float32", "shape": [2], "bytes": "00"}):
            project = copy.deepcopy(original)
            captured = project["runs"][0]["engine"]
            captured["state"] = bad
            captured["state_hash"] = _digest(bad)
            with self.assertRaises(ValueError):
                self.workspace.import_project(project)
        self.assertEqual(self.workspace.snapshot()["runs"][0]["id"], self.run_id)

    def test_run_export_contains_actual_portable_evidence_and_dataset(self):
        self.step(self.workspace, self.run_id, 2)
        content, name = self.workspace.export_run(self.run_id)
        self.assertTrue(name.endswith(".zip"))
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.assertIn("trajectory.csv", archive.namelist())
            self.assertIn("checkpoint.json", archive.namelist())
            trajectory = json.loads(archive.read("trajectory.json"))
            self.assertEqual(trajectory["records"][1], self.workspace.record(self.run_id, 1))
            self.assertEqual(len(json.loads(archive.read("dataset.json"))["inputs"]), 512)

    def test_multiple_live_runs_pause_independently_and_delete_stops_worker(self):
        first, second = self.create(), self.create()
        for identity in (first, second):
            self.workspace.dispatch("run.speed", {"run_id": identity, "value": 100})
            self.workspace.dispatch("run.play", {"run_id": identity})
        deadline = time.monotonic() + 10
        while min(self.workspace._runs[identity].engine.step_index for identity in (first, second)) < 2:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.workspace.dispatch("run.pause", {"run_id": first})
        count = self.workspace._runs[first].engine.step_index
        time.sleep(0.08)
        self.assertEqual(self.workspace._runs[first].engine.step_index, count)
        controller = self.workspace._runs[second].controller
        self.workspace.dispatch("run.delete", {"run_id": second})
        self.assertIsNone(controller._thread)

    def test_autosave_reopens_exact_state_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = StudioWorkspace(directory)
            run_id = saved.snapshot()["runs"][0]["id"]
            self.step(saved, run_id, 3)
            saved.close()
            self.assertTrue((Path(directory) / "autosave.json").exists())
            reopened = StudioWorkspace(directory)
            try:
                self.assertEqual(reopened._runs[run_id].engine.step_index, 3)
                self.step(reopened, run_id)
                self.assertEqual(reopened.record(run_id, 3)["step"], 3)
            finally:
                reopened.close()

    def test_invalid_inputs_leave_config_and_history_unchanged(self):
        original = self.workspace.snapshot()
        cases: list[tuple[str, dict[str, Any]]] = [
            ("run.step", {"count": -1}),
            ("run.speed", {"value": float("inf")}),
            ("run.configure", {"config": {"dataset": {"batch_size": 0}}}),
            ("run.configure", {"config": {"unrecognized": 1}}),
            ("run.dataset", {"csv": "x,target\n"}),
            ("run.dataset", {"csv": "x,target\n1,nan"}),
            ("run.intervene", {"parameter_hex": "junk"}),
            ("run.intervene", {"optimizer_state": {"exp_avg": 0.5}}),
        ]
        for action, payload in cases:
            with self.subTest(action=action, payload=payload), self.assertRaises((ValueError, TypeError)):
                self.workspace.dispatch(action, {"run_id": self.run_id, **payload})
        after = self.workspace.snapshot()
        self.assertEqual(after["runs"][0]["config"], original["runs"][0]["config"])
        self.assertEqual(after["runs"][0]["series"], [])

    def test_verify_status_and_close_lifecycle(self):
        self.step(self.workspace, self.run_id)
        result = self.workspace.dispatch("run.verify", {"run_id": self.run_id})
        self.assertEqual(result["integrity"]["status"], "verified")
        self.assertEqual(self.workspace.snapshot()["runs"][0]["integrity"]["status"], "verified")
        self.workspace.close()
        with self.assertRaises(StudioStateError):
            self.workspace.snapshot()

    def test_configuration_edit_branches_recorded_history(self):
        self.step(self.workspace, self.run_id, 2)
        branch = self.workspace.dispatch("run.configure", {"run_id": self.run_id, "config": {"lr": 0.02}})["run_id"]
        self.assertNotEqual(branch, self.run_id)
        self.assertEqual(self.workspace._runs[self.run_id].engine.step_index, 2)
        self.assertEqual(self.workspace._runs[branch].engine.config.lr, 0.02)

    def test_imported_route_ids_are_validated_without_replacing_workspace(self):
        project = self.workspace.export_project()
        project["runs"][0]["id"] = "a/b"
        project["workspace"]["active_run_id"] = "a/b"
        with self.assertRaises(ValueError):
            self.workspace.import_project(project)
        self.assertEqual(self.workspace.snapshot()["runs"][0]["id"], self.run_id)

    def test_background_failure_is_visible_and_restart_clears_it(self):
        run = self.workspace._runs[self.run_id]

        def failing_step():
            raise ValueError("actual worker failure")

        replacement = patch.object(run.engine, "step", side_effect=failing_step)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.workspace.dispatch("run.play", {"run_id": self.run_id})
        deadline = time.monotonic() + 5
        while run.controller.state != "aborted":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertIn("actual worker failure", self.workspace.snapshot()["runs"][0]["error"])
        with self.assertRaisesRegex(ValueError, "actual worker failure"):
            self.step(self.workspace, self.run_id)
        replacement.stop()
        self.workspace.dispatch("run.restart", {"run_id": self.run_id})
        self.step(self.workspace, self.run_id)
        self.assertIsNone(self.workspace.snapshot()["runs"][0]["error"])

    def test_signed_zero_bits_and_wide_seed_survive_json_roundtrip(self):
        run_id = self.create(seed=2**64 - 1, dataset={"seed": 2**64 - 1})
        self.workspace.dispatch("run.intervene", {"run_id": run_id, "parameter_hex": "80000000"})
        reopened = self.reopen()
        self.step(self.workspace, run_id)
        self.step(reopened, run_id)
        self.assertEqual(reopened.record(run_id, 0)["parameter_before_uint32"], 0x80000000)
        self.assertEqual(reopened._runs[run_id].engine.config.seed, 2**64 - 1)
        self.assertEqual(self.workspace.record(run_id, 0), reopened.record(run_id, 0))

    def test_empty_world_remains_empty_after_autosave_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = StudioWorkspace(directory)
            identity = workspace.snapshot()["runs"][0]["id"]
            workspace.dispatch("run.delete", {"run_id": identity})
            workspace.close()
            reopened = StudioWorkspace(directory)
            try:
                self.assertEqual(reopened.snapshot()["runs"], [])
            finally:
                reopened.close()

    def test_configuring_custom_data_retains_actual_rows_and_rebatches_new_optimizer(self):
        parent = self.workspace.dispatch("run.dataset", {
            "run_id": self.run_id, "csv": "x,target\n-1,0.25\n0.5,0.8\n2,-0.3\n4,1.2\n5,-0.7\n",
            "batch_size": 1,
        })["run_id"]
        self.step(self.workspace, parent, 2)
        original = self.workspace.dataset(parent)
        original_record = self.workspace.record(parent, 1)
        result = self.workspace.dispatch("run.configure", {"run_id": parent, "config": {
            "optimizer": "sgd", "momentum": 0.7, "lr": 0.03, "lr_schedule": "cosine",
            "model": {"use_tanh": False}, "dataset": {"batch_size": 2},
        }})
        child = result["run_id"]
        self.assertEqual(result["dataset_mode"], "retained_custom")
        actual = self.workspace.dataset(child)
        self.assertEqual(actual["inputs"], original["inputs"])
        self.assertEqual(actual["targets"], original["targets"])
        self.assertEqual(actual["provenance"]["sha256"], original["provenance"]["sha256"])
        self.assertEqual(actual["provenance"]["content_sha256"], original["provenance"]["content_sha256"])
        self.assertEqual(actual["description"]["batch_size"], 2)
        self.assertEqual(actual["description"]["steps_per_epoch"], 3)
        self.assertNotEqual(self.workspace._runs[parent].engine.dataset.inputs.data_ptr(),
                            self.workspace._runs[child].engine.dataset.inputs.data_ptr())
        self.step(self.workspace, child, 3)
        self.assertEqual(self.workspace.record(child, 0)["optimizer_name"], "sgd")
        self.assertAlmostEqual(self.workspace.record(child, 2)["input_value"], 5.0)
        self.assertEqual(self.workspace.record(parent, 1), original_record)
        self.assertEqual(self.workspace.dataset(parent)["description"]["batch_size"], 1)
        reopened = self.reopen()
        self.step(reopened, child)
        self.step(self.workspace, child)
        self.assertEqual(reopened.record(child, 3), self.workspace.record(child, 3))

    def test_explicit_generation_change_creates_generated_data_child(self):
        parent = self.workspace.dispatch("run.dataset", {
            "run_id": self.run_id, "inputs": [2, 3, 4], "targets": [10, 20, 30],
        })["run_id"]
        result = self.workspace.dispatch("run.configure", {
            "run_id": parent, "config": {"dataset": {"target_true_parameter": 0.7}},
        })
        self.assertEqual(result["dataset_mode"], "generated")
        self.assertEqual(self.workspace.dataset(result["run_id"])["provenance"]["kind"], "generated")
        self.assertEqual(self.workspace.dataset(parent)["targets"], [10.0, 20.0, 30.0])

    def test_browser_config_preserves_uint64_seeds_for_full_and_partial_edits(self):
        wide = 2**64 - 1
        parent = self.create(seed=wide, dataset={"samples": 13, "batch_size": 4, "seed": wide,
                                               "noise_std": 0.2, "shuffle_batches": True})
        snapshot = self.workspace.snapshot()
        browser_config = json.loads(json.dumps(next(run["config"] for run in snapshot["runs"] if run["id"] == parent)))
        self.assertEqual(browser_config["seed"], str(wide))
        self.assertEqual(browser_config["dataset"]["seed"], str(wide))
        browser_config["lr"] = 0.025
        full = self.workspace.dispatch("run.configure", {"run_id": parent, "config": browser_config})["run_id"]
        partial = self.workspace.dispatch("run.configure", {"run_id": parent, "config": {"lr": 0.025}})["run_id"]
        for child in (full, partial):
            self.assertEqual(self.workspace._runs[child].engine.config.seed, wide)
            self.assertEqual(self.workspace._runs[child].engine.config.dataset.seed, wide)
            self.assertEqual(self.workspace.dataset(child)["inputs"], self.workspace.dataset(parent)["inputs"])
            self.assertEqual(self.workspace.dataset(child)["targets"], self.workspace.dataset(parent)["targets"])
            self.assertEqual(self.workspace._runs[child].engine.dataset.shuffle_permutation,
                             self.workspace._runs[parent].engine.dataset.shuffle_permutation)
        self.step(self.workspace, full, 3)
        self.step(self.workspace, partial, 3)
        self.assertEqual(self.workspace.record(full, 2), self.workspace.record(partial, 2))
        exported = self.workspace.export_project()
        self.assertEqual(next(run["engine"]["config"]["seed"] for run in exported["runs"] if run["id"] == parent), wide)
        capabilities = self.workspace.capabilities()
        self.assertIsInstance(capabilities["default_config"]["seed"], str)
        self.assertTrue(all(isinstance(preset["config"]["dataset"]["seed"], str)
                            for preset in capabilities["presets"].values()))

    def test_browser_seed_parser_rejects_lossy_or_noninteger_strings(self):
        for value in ("18446744073709551616", "1e5", "5.0", " 5", "05", "+5", "not a seed"):
            for config in ({"seed": value}, {"dataset": {"seed": value}}):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    self.workspace.dispatch("run.configure", {"run_id": self.run_id, "config": config})


if __name__ == "__main__":
    unittest.main()
