"""Qualify portable Python exports as inert data and explicit real executions."""

from __future__ import annotations

import copy
import hashlib
import http.client
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from single_parameter_observatory import __version__
from single_parameter_observatory.studio_codegen import python_project
from single_parameter_observatory.studio_graph import _edge, _node
from single_parameter_observatory.studio_plugins import example_plugin
from single_parameter_observatory.studio_server import create_server
from single_parameter_observatory.studio_session import StudioWorkspace


def load_inert(source: bytes, directory: Path) -> types.ModuleType:
    """Import exactly the downloaded file while prohibiting runtime creation."""
    path = directory / "exported_workflow.py"
    path.write_bytes(source)
    compile(source, str(path), "exec")
    spec = importlib.util.spec_from_file_location("spobs_test_export_inert", path)
    if spec is None or spec.loader is None:
        raise AssertionError("The exported Python file could not be imported")
    module = importlib.util.module_from_spec(spec)
    with patch("single_parameter_observatory.studio_session.StudioWorkspace",
               side_effect=AssertionError("Importing an exported workflow executed the runtime")), \
            patch.object(sys, "dont_write_bytecode", True):
        spec.loader.exec_module(module)
    return module


class TestStudioPythonCodegen(unittest.TestCase):
    def test_import_is_inert_and_exact_data_cannot_become_python_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sentinel = directory / "injected.txt"
            payload = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('executed')"
            identities = ["__GRAPH_ID__", "__PROJECT_DATA__", "__PROJECT_SHA__", "__STUDIO_VERSION__",
                          "__GRAPH_ID__.__PROJECT_DATA__.__PROJECT_SHA__",
                          "quoted'\n); raise RuntimeError('graph ID executed') #"]
            for identity in identities:
                with self.subTest(graph_id=identity):
                    project = {"extensions": {"graphs": [{"id": identity, "nodes": []}], "jobs": []},
                               "wide": 2**60 + 1, "negative_wide": -(2**60 + 1), "zero": -0.0,
                               "unicode": "فارسی · λ · 🧠 · quoted '\" \\ newline\n",
                               "python_looking_text": payload,
                               "markers": ["__PROJECT_DATA__", "__PROJECT_SHA__", "__GRAPH_ID__"]}
                    source = python_project(project, identity)
                    self.assertIn(f"Requires single-parameter-observatory {__version__}".encode(), source)
                    module = load_inert(source, directory)
                    self.assertEqual(module.GRAPH_ID, identity)
                    restored = module.project_data()
                    self.assertEqual(restored, project)
                    self.assertEqual(restored["wide"], 1152921504606846977)
                    self.assertEqual(math.copysign(1.0, restored["zero"]), -1.0)
                    self.assertFalse(sentinel.exists())

    def test_embedded_project_hash_is_checked_before_project_use(self) -> None:
        project = {"extensions": {"graphs": [{"id": "graph"}], "jobs": []}, "value": 42}
        with tempfile.TemporaryDirectory() as temporary:
            module = load_inert(python_project(project, "graph"), Path(temporary))
            module.__dict__["PROJECT_SHA256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                module.project_data()

    def test_missing_workflows_active_jobs_and_nonfinite_data_reject_export(self) -> None:
        project = {"extensions": {"graphs": [{"id": "graph"}], "jobs": []}}
        with self.assertRaisesRegex(KeyError, "workflow does not exist"):
            python_project(project, "missing")
        for status in ["queued", "running", "cancelling"]:
            with self.subTest(status=status):
                busy = copy.deepcopy(project)
                busy["extensions"]["jobs"] = [{"status": status}]
                with self.assertRaisesRegex(ValueError, "Finish or cancel"):
                    python_project(busy, "graph")
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(nonfinite=value), self.assertRaises(ValueError):
                python_project({**project, "value": value}, "graph")
        for status in ["succeeded", "failed", "cancelled", "interrupted"]:
            with self.subTest(completed_status=status):
                self.assertTrue(python_project({"extensions": {
                    "graphs": [{"id": "graph"}], "jobs": [{"status": status}]}}, "graph"))

    def test_explicit_script_executes_real_graph_with_portable_plugin_checkpoint_and_artifact(self) -> None:
        workspace = StudioWorkspace()
        self.addCleanup(workspace.close)
        run_id = workspace.snapshot()["workspace"]["active_run_id"]
        checkpoint = workspace.dispatch("checkpoint.create", {"run_id": run_id, "name": "Saved λ state"})
        plugin = example_plugin()
        workspace.graph_action("plugin.install", {"manifest": plugin})
        exact = {"wide": 2**60 + 1, "zero": -0.0, "text": "فارسی 🧠 __PROJECT_DATA__"}
        graph = {"id": "__GRAPH_ID__.__PROJECT_DATA__.__PROJECT_SHA__", "name": "Portable core calculation",
                 "nodes": [_node("six", "core.number", 0, 0, value=6),
                           _node("seven", "core.number", 0, 0, value=7),
                           _node("multiply", "core.math", 300, 0, operation="multiply"),
                           _node("preview", "core.preview", 600, 0),
                           _node("exact", "core.object", 0, 300, value=exact),
                           _node("evidence", "core.export", 600, 300, run=run_id)],
                 "edges": [_edge("six", "value", "multiply", "a"),
                           _edge("seven", "value", "multiply", "b"),
                           _edge("multiply", "value", "preview", "value")]}
        graph_id = workspace.graph_action("graph.create", {"graph": graph})["graph_id"]
        project = workspace.export_project()
        # Imported projects may retain user-defined IDs; new graph creation assigns an ID.
        next(value for value in project["extensions"]["graphs"] if value["id"] == graph_id)["id"] = graph["id"]
        graph_id = str(graph["id"])
        project["extensions"]["active_graph_id"] = graph_id
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module = load_inert(python_project(project, graph_id), directory)
            self.assertEqual(workspace.graphs.snapshot()["jobs"], [])
            captured = module.project_data()
            self.assertEqual(captured["application_version"], __version__)
            self.assertEqual(captured["extensions"]["plugins"][0]["manifest"]["id"], plugin["id"])
            self.assertEqual(captured["runs"][0]["checkpoints"][0]["id"], checkpoint["checkpoint_id"])
            result_path = directory / "nested" / "job.json"
            artifact_directory = directory / "artifacts"
            completed = subprocess.run(
                [sys.executable, str(directory / "exported_workflow.py"), "--timeout", "30",
                 "--output", str(result_path), "--artifacts", str(artifact_directory)],
                cwd=directory, env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            job = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(completed.stdout), job)
            self.assertEqual(job["status"], "succeeded", job.get("error"))
            self.assertEqual(job["graph_id"], graph_id)
            self.assertEqual(job["nodes"]["preview"]["outputs"], {"value": 42})
            observed = job["nodes"]["exact"]["outputs"]["value"]
            self.assertEqual(observed["wide"], exact["wide"])
            self.assertEqual(observed["text"], exact["text"])
            self.assertEqual(math.copysign(1.0, observed["zero"]), -1.0)
            artifact = job["nodes"]["evidence"]["outputs"]["artifact"]
            path = artifact_directory / (artifact["id"] + "-" + artifact["filename"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
            self.assertEqual(path.stat().st_size, artifact["byte_length"])
            with zipfile.ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("checkpoint.json", archive.namelist())
                self.assertIn("dataset.json", archive.namelist())
            self.assertEqual(workspace.graphs.snapshot()["jobs"], [])

    def test_invalid_script_deadlines_reject_before_loading_runtime(self) -> None:
        project = {"extensions": {"graphs": [{"id": "graph"}], "jobs": []}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module = load_inert(python_project(project, "graph"), directory)
            for timeout in ["nan", "inf", "0", "-1", "86401"]:
                with self.subTest(timeout=timeout), patch.object(sys, "argv", ["workflow.py", "--timeout", timeout]), \
                        patch("single_parameter_observatory.studio_session.StudioWorkspace",
                              side_effect=AssertionError("Invalid deadline created a runtime")), \
                        patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as result:
                        module.main()
                    self.assertEqual(result.exception.code, 2)


class TestStudioPythonDownload(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.server = create_server(port=0, storage_directory=Path(self.directory.name))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02})
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.directory.cleanup()
        self.assertFalse(self.thread.is_alive())

    def request(self, path: str, epoch: str | None = None) -> tuple[int, bytes, dict[str, str]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        try:
            connection.request("GET", path, headers={} if epoch is None else {"X-Studio-Epoch": epoch})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def test_python_download_is_binary_attachment_of_current_project_and_rejects_stale_epoch(self) -> None:
        workspace = self.server.workspace
        graph_id = workspace.graph_action("graph.create", {"graph": {
            "id": "__GRAPH_ID__", "name": "HTTP λ workflow", "nodes": [], "edges": []}})["graph_id"]
        project = workspace.export_project()
        next(value for value in project["extensions"]["graphs"] if value["id"] == graph_id)["id"] = "__GRAPH_ID__"
        graph_id = "__GRAPH_ID__"
        project["extensions"]["active_graph_id"] = graph_id
        workspace.import_project(project)
        epoch = workspace.workspace_epoch
        status, health, _headers = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(health)["version"], __version__)
        status, source, headers = self.request(f"/api/graph/python/{graph_id}", epoch)
        self.assertEqual(status, 200, source)
        self.assertEqual(headers["Content-Type"], "text/x-python; charset=utf-8")
        self.assertEqual(int(headers["Content-Length"]), len(source))
        self.assertEqual(headers["Content-Disposition"], 'attachment; filename="spobs-workflow-__GRAPH_ID__.py"')
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        module = load_inert(source, Path(self.directory.name))
        self.assertEqual(module.GRAPH_ID, graph_id)
        self.assertEqual(module.project_data()["application_version"], __version__)
        self.assertEqual(workspace.graphs.snapshot()["jobs"], [])
        workspace.import_project(workspace.export_project())
        self.assertNotEqual(workspace.workspace_epoch, epoch)
        status, response, headers = self.request(f"/api/graph/python/{graph_id}", epoch)
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(response)["type"], "StudioEpochError")
        self.assertNotIn("Content-Disposition", headers)

    def test_nonexistent_and_active_job_projects_cannot_be_downloaded_as_runnable_source(self) -> None:
        status, body, headers = self.request("/api/graph/python/nonexistent")
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["type"], "KeyError")
        self.assertNotIn("Content-Disposition", headers)
        workspace = self.server.workspace
        project = workspace.export_project()
        graph_id = project["extensions"]["graphs"][0]["id"]
        project["extensions"]["jobs"] = [{"status": "running"}]
        with patch.object(workspace, "export_project", return_value=project):
            status, body, headers = self.request(f"/api/graph/python/{graph_id}")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["type"], "ValueError")
        self.assertNotIn("Content-Disposition", headers)


if __name__ == "__main__":
    unittest.main()
