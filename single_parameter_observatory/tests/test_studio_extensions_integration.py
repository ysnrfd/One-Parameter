"""Real transport, project, recovery, and host-operation extension regressions."""

from __future__ import annotations

import copy
import base64
import hashlib
import http.client
import io
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch

from single_parameter_observatory.studio_plugins import example_plugin
from single_parameter_observatory.studio_server import create_server
from single_parameter_observatory.studio_session import StudioStateError, StudioWorkspace


def node(identity: str, kind: str, **params) -> dict:
    return {"id": identity, "type": kind, "position": {"x": 0, "y": 0}, "params": params}


def edge(source: str, source_port: str, target: str, target_port: str) -> dict:
    return {"id": f"{source}-{source_port}-{target}-{target_port}", "source": source,
            "source_port": source_port, "target": target, "target_port": target_port}


SMALL = {"steps": 10, "dataset": {"samples": 6, "batch_size": 2},
         "validation": {"enabled": False}, "landscape": {"enabled": False},
         "run_fp64_reference_track": False}


class TestWorkspaceExtensionIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = StudioWorkspace()

    def tearDown(self) -> None:
        self.workspace.close()

    def test_atomic_custom_data_creation_does_not_publish_on_invalid_data(self) -> None:
        before = self.workspace.export_project()
        for data in ({"inputs": [0], "targets": []}, {"csv": "x,target\n1,NaN"},
                     {"inputs": [1], "targets": [2], "batch_size": 2},
                     {"inputs": [True], "targets": [1]}, {"csv": "x,target"}):
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                self.workspace.dispatch("run.create", {"config": SMALL, "dataset": data})
            after = self.workspace.export_project()
            self.assertEqual(before["runs"], after["runs"])
            self.assertEqual(before["workspace"], after["workspace"])

    def test_atomic_custom_data_creation_trains_the_supplied_actual_samples(self) -> None:
        created = self.workspace.dispatch("run.create", {"config": SMALL, "dataset": {
            "inputs": [-1, 0, 1], "targets": [-2, 0, 2], "batch_size": 2,
            "provenance": {"transform": "verified fixture"},
        }})
        identity = created["run_id"]
        self.assertEqual(len(self.workspace.snapshot()["runs"]), 2)
        self.workspace.dispatch("run.step", {"run_id": identity, "count": 2})
        data = self.workspace.dataset(identity)
        self.assertEqual(data["inputs"], [-1.0, 0.0, 1.0])
        self.assertEqual(data["targets"], [-2.0, 0.0, 2.0])
        self.assertEqual(data["provenance"]["user_provenance"]["transform"], "verified fixture")
        self.assertEqual(len(data["provenance"]["stored_tensor_sha256"]), 64)
        self.assertEqual(self.workspace.record(identity, 1)["step"], 1)

    def test_automatic_branch_positions_do_not_stack_siblings(self) -> None:
        parent = self.workspace.snapshot()["workspace"]["active_run_id"]
        for _ in range(4):
            self.workspace.dispatch("run.fork", {"run_id": parent})
        runs = self.workspace.snapshot()["runs"]
        positions = [run["position"] for run in runs]
        for index, left in enumerate(positions):
            for right in positions[index + 1:]:
                self.assertTrue(abs(left["x"] - right["x"]) >= 300
                                or abs(left["y"] - right["y"]) >= 220)
        explicit = {"x": 80, "y": 275}
        created = self.workspace.dispatch("run.create", {"position": explicit})
        actual = next(run for run in self.workspace.snapshot()["runs"] if run["id"] == created["run_id"])
        self.assertEqual(actual["position"], explicit)

    def test_reset_clears_live_intervention_indicator_but_preserves_evidence(self) -> None:
        identity = self.workspace.snapshot()["workspace"]["active_run_id"]
        self.workspace.dispatch("run.intervene", {"run_id": identity, "parameter_value": 0.25})
        self.assertTrue(self.workspace.snapshot()["runs"][0]["live_intervention"])
        self.workspace.dispatch("run.reset", {"run_id": identity})
        run = self.workspace.snapshot()["runs"][0]
        self.assertFalse(run["live_intervention"])
        self.assertEqual(run["parameter"], -2.0)
        self.assertEqual(len(run["interventions"]), 1)
        self.assertIn("MANUAL_INTERVENTION", [event["code"] for event in run["events"]])

    def test_invalid_extension_project_leaves_real_runs_and_graphs_unchanged(self) -> None:
        identity = self.workspace.snapshot()["workspace"]["active_run_id"]
        self.workspace.dispatch("run.step", {"run_id": identity, "count": 2})
        before = self.workspace.export_project()
        bad = copy.deepcopy(before)
        bad["extensions"]["graphs"].append(copy.deepcopy(bad["extensions"]["graphs"][0]))
        with self.assertRaises(ValueError):
            self.workspace.import_project(bad)
        after = self.workspace.export_project()
        self.assertEqual(before["runs"], after["runs"])
        self.assertEqual(before["extensions"], after["extensions"])

    def test_legacy_project_adds_a_nonexecuting_node_workspace(self) -> None:
        project = self.workspace.export_project()
        project.pop("extensions")
        self.workspace.import_project(project)
        state = self.workspace.graphs.snapshot()
        self.assertTrue(state["graphs"])
        self.assertFalse(state["jobs"])
        self.assertFalse(self.workspace.graphs.is_busy())
        self.assertEqual(len(self.workspace.snapshot()["runs"]), len(project["runs"]))

    def test_corrupt_autosave_is_preserved_before_recovery_is_saved(self) -> None:
        invalid_files = [b"{broken", b'{"schema_version":"spobs-studio/1","x":NaN}',
                         b'{"schema_version":"spobs-studio/1","schema_version":"spobs-studio/1"}',
                         b"\xff\xfe\x80"]
        for original in invalid_files:
            with self.subTest(original=original), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "autosave.json"
                path.write_bytes(original)
                recovered = StudioWorkspace(directory)
                try:
                    self.assertIsNotNone(recovered.snapshot()["recovery_notice"])
                    backups = list(Path(directory).glob("autosave.recovery-*.json"))
                    self.assertEqual(len(backups), 1)
                    self.assertEqual(backups[0].read_bytes(), original)
                    recovered.dispatch("workspace.update", {"notes": "new recovered project"})
                finally:
                    recovered.close()
                self.assertEqual(backups[0].read_bytes(), original)
                self.assertEqual(json.loads(path.read_text())["workspace"]["notes"], "new recovered project")

    def test_unapplied_invalid_editor_and_panel_drafts_roundtrip_without_numerical_changes(self) -> None:
        identity = self.workspace.snapshot()["workspace"]["active_run_id"]
        before = json.loads(json.dumps(self.workspace.export_project(), allow_nan=False))
        drafts = {"schema_version": "spobs-ui-drafts/1", "runs": {identity: {
            "config": {"lr": -4}, "config_json": "{ unfinished configuration",
            "config_json_dirty": True, "csv": "x,target\n1,unfinished target\n",
            "notes": "unapplied run notes", "bits": 0x7FC00001, "bit_dirty": True,
            "optimizer_json": "{ unfinished optimizer state",
        }}, "workspace_notes": "unapplied workspace notes", "plugin": "{ unfinished plugin manifest",
            "plugin_dirty": True, "node_fields": {"starter-training": {"config:value": "{ unfinished JSON"}},
            "panel_fields": {"local.example/controls": {"gain": "not a finite number", "options": "[unfinished"}}}
        self.workspace.dispatch("workspace.drafts", {"drafts": drafts})
        project = json.loads(json.dumps(self.workspace.export_project(), allow_nan=False))
        self.assertEqual(project["runs"], before["runs"])
        self.assertEqual(project["workspace"]["notes"], before["workspace"]["notes"])
        self.assertEqual(project["extensions"], before["extensions"])
        reopened = StudioWorkspace()
        try:
            reopened.import_project(project)
            self.assertEqual(reopened.ui_drafts, drafts)
            self.assertEqual(json.loads(json.dumps(reopened.export_project(), allow_nan=False))["runs"], before["runs"])
            self.assertEqual(reopened._runs[identity].engine.step_index, 0)
            self.assertFalse(reopened.graphs.is_busy())
            self.assertFalse(reopened.graphs.snapshot()["plugins"])
        finally:
            reopened.close()

    def test_invalid_draft_replacement_or_import_is_atomic(self) -> None:
        accepted = {"schema_version": "spobs-ui-drafts/1", "runs": {},
                    "panel_fields": {"local.example/controls": {"gain": "invalid but retained text"}}}
        self.workspace.dispatch("workspace.drafts", {"drafts": accepted})
        before = self.workspace.export_project()
        for invalid in (None, {"gain": 3}, {str(index): "value" for index in range(101)}):
            bad = copy.deepcopy(before)
            bad["ui_drafts"]["panel_fields"] = {"local.example/controls": invalid}
            bad["workspace"]["name"] = "must not replace current metadata"
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.workspace.import_project(bad)
            after = self.workspace.export_project()
            for key in ("workspace", "runs", "extensions", "ui_drafts"):
                self.assertEqual(after[key], before[key])
        with self.assertRaises(ValueError):
            self.workspace.dispatch("workspace.drafts", {"drafts": {**accepted, "panel_fields": []}})
        self.assertEqual(self.workspace.ui_drafts, accepted)

    def test_draft_compare_and_swap_keeps_the_first_writer_and_rejects_invalid_revision_types(self) -> None:
        revision = self.workspace._ui_drafts_revision
        first = {"schema_version": "spobs-ui-drafts/1", "runs": {}, "plugin": "first writer's unfinished text"}
        accepted = self.workspace.dispatch("workspace.drafts", {"drafts": first, "expected_revision": revision})
        self.assertEqual(accepted["revision"], revision + 1)
        for stale in (first, {"schema_version": "not a draft schema"}):
            with self.subTest(stale=stale), self.assertRaises(StudioStateError):
                self.workspace.dispatch("workspace.drafts", {"drafts": stale, "expected_revision": revision})
            self.assertEqual(self.workspace.ui_drafts, first)
            self.assertEqual(self.workspace._ui_drafts_revision, revision + 1)
        for invalid in (True, float(revision + 1), str(revision + 1), None, -1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.workspace.dispatch("workspace.drafts", {"drafts": first, "expected_revision": invalid})
            self.assertEqual(self.workspace.ui_drafts, first)
        # Direct library users that omit the optional CAS field remain compatible.
        self.workspace.dispatch("workspace.drafts", {"drafts": first})

    def test_live_parameter_bits_preserve_nonfinite_payloads_and_signed_zero_from_actual_storage(self) -> None:
        identity = self.workspace.workspace["active_run_id"]
        run = self.workspace._runs[identity]
        original = run.engine.parameter.detach().clone()
        patterns = (0x00000000, 0x80000000, 0x3F800000, 0x7F800000, 0xFF800000,
                    0x7FC12345, 0xFFC54321, 0x7FA12345)
        try:
            for bits in patterns:
                with self.subTest(bits=f"0x{bits:08x}"), self.workspace._lock, run.controller._lock, torch.no_grad():
                    # Trusted test fixture writes the storage itself, avoiding
                    # Python float conversion that can quiet or alter NaN bits.
                    signed = bits if bits < 2**31 else bits - 2**32
                    run.engine.parameter.detach().view(torch.int32).fill_(signed)
                    snapshot = self.workspace.snapshot()
                    live = next(item for item in snapshot["runs"] if item["id"] == identity)
                    self.assertEqual(live["parameter_uint32"], bits)
                    self.assertEqual(int(run.engine.parameter.detach().view(torch.int32).item()) & 0xFFFFFFFF, bits)
                    json.dumps(snapshot, allow_nan=False)
        finally:
            with self.workspace._lock, run.controller._lock, torch.no_grad():
                run.engine.parameter.copy_(original)

    def test_live_optimizer_memory_and_effective_rate_follow_manual_edit_step_and_reset(self) -> None:
        identity = self.workspace.dispatch("run.create", {"config": {
            **SMALL, "lr": 0.01, "lr_schedule": "exponential", "lr_final_factor": 0.8,
        }})["run_id"]
        self.workspace.dispatch("run.step", {"run_id": identity, "count": 2})
        recorded = self.workspace.record(identity, 1)
        self.workspace.dispatch("run.intervene", {"run_id": identity, "learning_rate": 0.02,
                                                  "optimizer_state": {"step": 2, "exp_avg": 0.25, "exp_avg_sq": 0.5}})
        live = next(item for item in self.workspace.snapshot()["runs"] if item["id"] == identity)
        self.assertTrue(live["live_optimizer_initialized"])
        self.assertEqual(live["live_optimizer_state"], {"step": 2.0, "exp_avg": 0.25, "exp_avg_sq": 0.5})
        self.assertEqual(live["live_learning_rate"], 0.02)
        self.assertEqual(self.workspace.record(identity, 1), recorded)
        self.workspace.dispatch("run.step", {"run_id": identity, "count": 1})
        live = next(item for item in self.workspace.snapshot()["runs"] if item["id"] == identity)
        # The resting optimizer group already carries the following scheduled
        # rate. The preceding immutable record describes the rate actually used
        # for that preceding step; these are different observations.
        next_rate = live["live_learning_rate"]
        self.assertNotEqual(live["live_learning_rate"], live["config"]["lr"])
        self.workspace.dispatch("run.step", {"run_id": identity, "count": 1})
        self.assertAlmostEqual(self.workspace.record(identity, 3)["learning_rate"], next_rate)
        last_record = self.workspace.record(identity, 3)
        self.workspace.dispatch("run.reset", {"run_id": identity})
        reset = next(item for item in self.workspace.snapshot()["runs"] if item["id"] == identity)
        self.assertFalse(reset["live_optimizer_initialized"])
        self.assertEqual(reset["live_optimizer_state"], {})
        self.assertAlmostEqual(reset["live_learning_rate"], 0.02)
        self.assertEqual(self.workspace.record(identity, 3), last_record)

    def test_shutdown_closes_import_and_action_admission_before_joining_graph_workers(self) -> None:
        before = self.workspace.export_project()
        entered, release = threading.Event(), threading.Event()
        errors = []
        graph_close = self.workspace.graphs.close

        def holding_graph_close():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release graph shutdown")
            graph_close()

        def close_workspace():
            try:
                self.workspace.close()
            except BaseException as exc:
                errors.append(exc)

        with patch.object(self.workspace.graphs, "close", side_effect=holding_graph_close):
            closing = threading.Thread(target=close_workspace)
            closing.start()
            try:
                self.assertTrue(entered.wait(5))
                replacement = copy.deepcopy(before)
                replacement["workspace"]["name"] = "must not partially replace during close"
                with self.assertRaises(StudioStateError):
                    self.workspace.import_project(replacement)
                with self.assertRaises(StudioStateError):
                    self.workspace.dispatch("workspace.update", {"name": "also rejected"})
                with self.assertRaises(StudioStateError):
                    self.workspace.graph_action("graph.create", {})
                self.assertEqual(self.workspace.workspace, before["workspace"])
                self.assertEqual(set(self.workspace._runs), {item["id"] for item in before["runs"]})
            finally:
                release.set()
                closing.join(timeout=5)
            self.assertFalse(closing.is_alive())
            self.assertEqual(errors, [])
        self.assertTrue(self.workspace._closed)

    def test_graph_shutdown_failure_still_stops_autosave_controllers_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = StudioWorkspace(directory)
            graph_close = workspace.graphs.close
            controller = next(iter(workspace._runs.values())).controller
            try:
                with patch.object(workspace.graphs, "close", side_effect=RuntimeError("graph shutdown fixture")), \
                        patch.object(controller, "stop", wraps=controller.stop) as stopped:
                    with self.assertRaisesRegex(RuntimeError, "graph shutdown fixture"):
                        workspace.close()
                    self.assertTrue(stopped.called)
                self.assertTrue(workspace._closed)
                self.assertTrue(workspace._stopping.is_set())
                autosave_thread = workspace._autosave_thread
                assert autosave_thread is not None
                self.assertFalse(autosave_thread.is_alive())
                self.assertEqual(json.loads((Path(directory) / "autosave.json").read_text())["workspace"]["id"], workspace.workspace["id"])
            finally:
                graph_close()
                workspace.close()

    def test_final_save_failure_does_not_leave_training_or_autosave_workers_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = StudioWorkspace(directory)
            controller = next(iter(workspace._runs.values())).controller
            try:
                with patch.object(workspace, "_save", side_effect=OSError("final save fixture")), \
                        patch.object(controller, "stop", wraps=controller.stop) as stopped:
                    with self.assertRaisesRegex(OSError, "final save fixture"):
                        workspace.close()
                    self.assertTrue(stopped.called)
                self.assertTrue(workspace._closed)
                self.assertTrue(workspace._stopping.is_set())
                autosave_thread = workspace._autosave_thread
                assert autosave_thread is not None
                self.assertFalse(autosave_thread.is_alive())
                self.assertFalse(workspace.graphs.is_busy())
            finally:
                workspace.close()

    def test_server_close_only_marks_workspace_closed_after_successful_cleanup(self) -> None:
        server = create_server(port=0)
        original_close = server.workspace.close
        calls = 0

        def initially_failing_close():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("first close fixture")
            original_close()

        try:
            with patch.object(server.workspace, "close", side_effect=initially_failing_close):
                with self.assertRaisesRegex(RuntimeError, "first close fixture"):
                    server.server_close()
                self.assertFalse(server._workspace_closed)
                self.assertEqual(server.socket.fileno(), -1)
                server.server_close()
                self.assertTrue(server._workspace_closed)
                self.assertTrue(server.workspace._closed)
            self.assertEqual(calls, 2)
        finally:
            original_close()
            server.server_close()


class TestGraphHTTPIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.server = create_server(port=0, storage_directory=self.directory.name)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.directory.cleanup()
        self.assertFalse(self.thread.is_alive())

    def request(self, route: str, payload: dict | None = None, *, headers: dict | None = None) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        try:
            connection.request("GET" if payload is None else "POST", route,
                               body=None if payload is None else json.dumps(payload),
                               headers={**({} if payload is None else {"Content-Type": "application/json"}), **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def action(self, name: str, **payload) -> dict:
        status, data = self.request("/api/graph/action", {"action": name, **payload})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["ok"])
        return data

    def create(self, nodes: list[dict], edges: list[dict] | None = None) -> str:
        return self.action("graph.create", graph={"name": "HTTP regression", "nodes": nodes,
                                                  "edges": edges or []})["result"]["graph_id"]

    def test_graph_routes_execute_real_training_and_roundtrip_project(self) -> None:
        identity = self.create([node("config", "core.config", value=SMALL),
                                node("data", "core.dataset", inputs=[-1, 0, 1], targets=[-2, 0, 2], batch_size=2),
                                node("run", "core.experiment"), node("train", "core.train", steps=2)],
                               [edge("config", "config", "run", "config"), edge("data", "dataset", "run", "dataset"),
                                edge("run", "run", "train", "run")])
        queued = self.action("graph.execute", graph_id=identity, idempotency_key="HTTP-real-training")
        job = self.server.workspace.graphs.wait_job(queued["result"]["job_id"], timeout=60)
        self.assertEqual(job["status"], "succeeded", job["error"])
        run_id = job["nodes"]["train"]["outputs"]["run"]
        self.assertEqual(self.server.workspace.record(run_id, 1)["step"], 1)
        self.assertEqual(len(self.server.workspace.snapshot()["runs"]), 2)
        status, project = self.request("/api/project")
        self.assertEqual(status, 200)
        self.assertIn("extensions", project)
        before = self.server.workspace.record(run_id, 1)
        status, reopened = self.request("/api/project", project)
        self.assertEqual(status, 200, reopened)
        self.assertEqual(before, self.server.workspace.record(run_id, 1))
        self.assertEqual(self.server.workspace.graphs.snapshot()["jobs"][0]["id"], job["id"])
        replay = self.action("graph.execute", graph_id=identity, idempotency_key="HTTP-real-training")
        self.assertTrue(replay["result"]["replayed"])
        self.assertEqual(len(self.server.workspace.snapshot()["runs"]), 2)

    def test_plugin_install_disable_enable_update_and_actual_execution(self) -> None:
        plugin = example_plugin()
        self.action("plugin.install", manifest=plugin)
        status, catalog = self.request("/api/graph/catalog")
        self.assertEqual(status, 200)
        kind = f"{plugin['id']}/scale"
        self.assertIn(kind, [item["id"] for item in catalog["types"]])
        identity = self.create([node("scale", kind, value=3, factor=4)])
        self.action("plugin.disable", plugin_id=plugin["id"])
        invalid = self.action("graph.validate", graph_id=identity)["result"]
        self.assertFalse(invalid["valid"])
        self.action("plugin.enable", plugin_id=plugin["id"])
        queued = self.action("graph.execute", graph_id=identity)
        job = self.server.workspace.graphs.wait_job(queued["result"]["job_id"], timeout=60)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertEqual(job["nodes"]["scale"]["outputs"], {"value": 12})
        changed = copy.deepcopy(plugin)
        changed["nodes"][0]["code"] = "return {value: inputs.value + params.factor};"
        status, _ = self.request("/api/graph/action", {"action": "plugin.update", "plugin_id": plugin["id"], "manifest": changed})
        self.assertEqual(status, 409)
        changed["version"] = "1.0.1"
        self.action("plugin.update", plugin_id=plugin["id"], manifest=changed)
        queued = self.action("graph.execute", graph_id=identity)
        job = self.server.workspace.graphs.wait_job(queued["result"]["job_id"], timeout=60)
        self.assertEqual(job["nodes"]["scale"]["outputs"], {"value": 7})

    def test_stale_graph_edit_returns_conflict_and_retains_current_graph(self) -> None:
        identity = self.create([node("value", "core.number", value=3)])
        graph = next(item for item in self.server.workspace.graphs.snapshot()["graphs"] if item["id"] == identity)
        changed = copy.deepcopy(graph)
        changed["name"] = "committed edit"
        self.action("graph.update", graph=changed, expected_revision=graph["revision"])
        changed["name"] = "stale edit"
        status, _ = self.request("/api/graph/action", {"action": "graph.update", "graph": changed,
                                                      "expected_revision": graph["revision"]})
        self.assertEqual(status, 409)
        current = next(item for item in self.server.workspace.graphs.snapshot()["graphs"] if item["id"] == identity)
        self.assertEqual(current["name"], "committed edit")

    def test_active_script_blocks_project_replace_until_cancel_is_acknowledged(self) -> None:
        identity = self.create([node("busy", "core.script", code="while (true) {}", timeout_ms=10000)])
        _, before = self.request("/api/project")
        queued = self.action("graph.execute", graph_id=identity)
        job_id = queued["result"]["job_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            jobs = self.server.workspace.graphs.snapshot()["jobs"]
            if next(item for item in jobs if item["id"] == job_id)["status"] == "running":
                break
            threading.Event().wait(0.01)
        status, error = self.request("/api/project", before)
        self.assertEqual(status, 409, error)
        self.action("job.cancel", job_id=job_id)
        job = self.server.workspace.graphs.wait_job(job_id, timeout=60)
        self.assertEqual(job["status"], "cancelled")
        status, result = self.request("/api/project", before)
        self.assertEqual(status, 200, result)

    def test_imported_artifact_identity_charset_routes_exact_saved_bytes(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture.txt", "exact imported workflow artifact")
        raw = output.getvalue()
        run_id = self.server.workspace.snapshot()["workspace"]["active_run_id"]
        graph_id = self.create([node("export", "core.export", run=run_id)])
        with patch.object(self.server.workspace, "export_run", return_value=(raw, "fixture.zip")):
            queued = self.action("graph.execute", graph_id=graph_id)
            job = self.server.workspace.graphs.wait_job(queued["result"]["job_id"], timeout=60)
        self.assertEqual(job["status"], "succeeded", job["error"])
        _, project = self.request("/api/project")
        artifact = project["extensions"]["artifacts"][0]
        identity = "artifact.imported:one"
        artifact["id"] = identity
        artifact["download_url"] = f"/api/graph/artifacts/{identity}"
        descriptor = project["extensions"]["jobs"][0]["nodes"]["export"]["outputs"]["artifact"]
        descriptor["id"] = identity
        descriptor["download_url"] = artifact["download_url"]
        self.assertEqual(base64.b64decode(artifact["data_base64"]), raw)
        status, reopened = self.request("/api/project", project)
        self.assertEqual(status, 200, reopened)
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        try:
            connection.request("GET", artifact["download_url"])
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Content-Type"], "application/zip")
            data = response.read()
        finally:
            connection.close()
        self.assertEqual(data, raw)
        self.assertEqual(hashlib.sha256(data).hexdigest(), artifact["sha256"])

    def test_job_download_preserves_integers_beyond_browser_precision(self) -> None:
        value = 2**60 + 1
        graph_id = self.create([node("value", "core.json", value=value)])
        queued = self.action("graph.execute", graph_id=graph_id)
        job_id = queued["result"]["job_id"]
        self.assertEqual(self.server.workspace.graphs.wait_job(job_id, timeout=60)["status"], "succeeded")
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        try:
            connection.request("GET", f"/api/graph/jobs/{job_id}")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Content-Type"], "application/json; charset=utf-8")
            self.assertIn("attachment;", response.headers["Content-Disposition"])
            raw = response.read()
        finally:
            connection.close()
        self.assertIn(f'"value":{value}'.encode(), raw)
        saved = json.loads(raw)
        self.assertEqual(saved["nodes"]["value"]["outputs"]["value"], value)
        self.assertEqual(saved["graph"]["nodes"][0]["params"]["value"], value)
        status, error = self.request("/api/graph/jobs/missing-job")
        self.assertEqual(status, 404, error)

    def test_draft_http_two_writers_receive_conflict_without_overwriting_saved_buffers(self) -> None:
        status, first_reader = self.request("/api/ui/drafts")
        self.assertEqual(status, 200)
        self.assertEqual(first_reader["drafts"]["schema_version"], "spobs-ui-drafts/1")
        _, second_reader = self.request("/api/ui/drafts")
        first = copy.deepcopy(first_reader["drafts"])
        first["plugin"] = "{ first writer's unfinished plugin"
        status, accepted = self.request("/api/action", {"action": "workspace.drafts", "drafts": first,
                                                       "expected_revision": first_reader["revision"]})
        self.assertEqual(status, 200, accepted)
        self.assertEqual(accepted["result"]["revision"], first_reader["revision"] + 1)
        second = copy.deepcopy(second_reader["drafts"])
        second["workspace_notes"] = "second writer's unrelated local notes"
        status, rejected = self.request("/api/action", {"action": "workspace.drafts", "drafts": second,
                                                       "expected_revision": second_reader["revision"]})
        self.assertEqual(status, 409, rejected)
        _, current = self.request("/api/ui/drafts")
        self.assertEqual(current["drafts"], first)
        self.assertEqual(current["revision"], accepted["result"]["revision"])

    def test_workspace_epoch_blocks_same_graph_identity_and_revision_after_project_replacement(self) -> None:
        _, state = self.request("/api/state")
        epoch = state["workspace_epoch"]
        self.assertEqual(len(epoch), 32)
        _, graph_state = self.request("/api/graph/state")
        _, drafts = self.request("/api/ui/drafts")
        self.assertEqual(graph_state["workspace_epoch"], epoch)
        self.assertEqual(drafts["workspace_epoch"], epoch)
        graph = graph_state["graphs"][0]
        _, project = self.request("/api/project")
        self.assertNotIn("workspace_epoch", project)
        self.assertNotIn("workspace_epoch", project["extensions"])
        project["workspace_epoch"] = "file-controlled-value-must-never-be-used"
        status, opened = self.request("/api/project", project, headers={"X-Studio-Epoch": epoch})
        self.assertEqual(status, 200, opened)
        current_epoch = opened["state"]["workspace_epoch"]
        self.assertNotEqual(current_epoch, epoch)
        self.assertNotEqual(current_epoch, project["workspace_epoch"])
        current_graph = next(item for item in self.server.workspace.graphs.snapshot()["graphs"] if item["id"] == graph["id"])
        self.assertEqual(current_graph["revision"], graph["revision"])
        stale = copy.deepcopy(graph)
        stale["name"] = "stale tab must not overwrite imported graph"
        status, rejected = self.request("/api/graph/action", {"action": "graph.update", "graph": stale,
                                                              "expected_revision": graph["revision"]},
                                        headers={"X-Studio-Epoch": epoch})
        self.assertEqual(status, 409, rejected)
        self.assertEqual(rejected["type"], "StudioEpochError")
        self.assertEqual(self.server.workspace.graphs.snapshot()["graphs"][0]["name"], graph["name"])
        root_id = state["workspace"]["active_run_id"]
        status, rejected = self.request("/api/action", {"action": "run.step", "run_id": root_id, "count": 1},
                                        headers={"X-Studio-Epoch": epoch})
        self.assertEqual(status, 409, rejected)
        self.assertEqual(self.server.workspace._runs[root_id].engine.step_index, 0)
        status, rejected = self.request("/api/project", project, headers={"X-Studio-Epoch": epoch})
        self.assertEqual(status, 409, rejected)
        self.assertEqual(self.server.workspace.workspace_epoch, current_epoch)
        status, invalid = self.request("/api/project", {"schema_version": "invalid"},
                                       headers={"X-Studio-Epoch": current_epoch})
        self.assertEqual(status, 400, invalid)
        self.assertEqual(self.server.workspace.workspace_epoch, current_epoch)
        stale["name"] = "current tab edit"
        status, accepted = self.request("/api/graph/action", {"action": "graph.update", "graph": stale,
                                                              "expected_revision": graph["revision"]},
                                        headers={"X-Studio-Epoch": current_epoch})
        self.assertEqual(status, 200, accepted)
        self.assertEqual(accepted["state"]["workspace_epoch"], current_epoch)
        _, exported = self.request("/api/project", headers={"X-Studio-Epoch": current_epoch})
        self.assertNotIn("workspace_epoch", exported)

    def test_stale_workspace_gets_are_rejected_before_identity_lookup_or_download(self) -> None:
        graph_id = self.create([node("value", "core.number", value=5)])
        queued = self.action("graph.execute", graph_id=graph_id)
        job_id = queued["result"]["job_id"]
        self.server.workspace.graphs.wait_job(job_id, timeout=60)
        _, project = self.request("/api/project")
        epoch = self.server.workspace.workspace_epoch
        status, opened = self.request("/api/project", project, headers={"X-Studio-Epoch": epoch})
        self.assertEqual(status, 200, opened)
        current_epoch = self.server.workspace.workspace_epoch
        run_id = project["workspace"]["active_run_id"]
        for route in ("/api/state", "/api/capabilities", "/api/project", "/api/ui/drafts", "/api/graph/state",
                      "/api/graph/catalog", f"/api/graph/jobs/{job_id}", "/api/graph/jobs/missing-job",
                      "/api/graph/artifacts/missing-artifact", f"/api/runs/{run_id}/dataset",
                      f"/api/runs/{run_id}/export", f"/api/runs/{run_id}/record?step=0"):
            with self.subTest(route=route):
                status, rejected = self.request(route, headers={"X-Studio-Epoch": epoch})
                self.assertEqual(status, 409, rejected)
                self.assertEqual(rejected["type"], "StudioEpochError")
                self.assertEqual(self.server.workspace.workspace_epoch, current_epoch)
        status, current = self.request("/api/graph/jobs/" + job_id, headers={"X-Studio-Epoch": current_epoch})
        self.assertEqual(status, 200, current)
        self.assertEqual(current["nodes"]["value"]["outputs"]["value"], 5)
        status, missing = self.request("/api/graph/artifacts/missing-artifact", headers={"X-Studio-Epoch": current_epoch})
        self.assertEqual(status, 404, missing)
        # Headerless callers retain the public API compatibility path.
        self.assertEqual(self.request("/api/state")[0], 200)
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        try:
            connection.request("GET", "/", headers={"X-Studio-Epoch": epoch})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn(b"Parameter Studio", response.read())
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
