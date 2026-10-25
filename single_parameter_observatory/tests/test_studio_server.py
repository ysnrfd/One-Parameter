"""Exercise the actual local HTTP transport, backend, and download boundaries."""

from __future__ import annotations

import http.client
import io
import json
import socket
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from . import _bootstrap
except ImportError:
    import _bootstrap  # type: ignore[import-not-found, no-redef]

from single_parameter_observatory import __version__
from single_parameter_observatory.studio_server import (
    MAX_BODY_BYTES, StudioRequestHandler, create_server, main, serve_studio,
)


class TestStudioServer(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.server = create_server(port=0, storage_directory=Path(self.directory.name))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.directory.cleanup()
        self.assertFalse(self.thread.is_alive())

    def request(self, path: str, method: str = "GET", payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        outgoing_headers = dict(headers or {})
        body = payload
        if isinstance(payload, dict):
            body = json.dumps(payload).encode("utf-8")
            outgoing_headers.setdefault("Content-Type", "application/json")
        try:
            connection.request(method, path, body=body, headers=outgoing_headers)
            response = connection.getresponse()
            data = response.read()
            received_headers = dict(response.getheaders())
            if "application/json" in received_headers.get("Content-Type", ""):
                data = json.loads(data, parse_constant=lambda value: self.fail(f"Non-finite JSON {value}"))
            return response.status, data, received_headers
        finally:
            connection.close()

    def action(self, action: str, **payload):
        status, data, _headers = self.request("/api/action", "POST", {"action": action, **payload})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["ok"])
        self.assertIn("state", data)
        self.assertIn("result", data)
        return data

    def test_static_assets_health_and_capabilities(self) -> None:
        for path, content_type in (("/", "text/html"), ("/studio.css", "text/css"),
                                   ("/studio.js", "text/javascript")):
            with self.subTest(path=path):
                status, data, headers = self.request(path)
                self.assertEqual(status, 200)
                self.assertGreater(len(data), 100)
                self.assertIn(content_type, headers["Content-Type"])
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        status, data, _headers = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["version"], __version__)
        for path in ("/api/state", "/api/capabilities"):
            status, data, _headers = self.request(path)
            self.assertEqual(status, 200)
            self.assertIsInstance(data, dict)

    def test_real_training_project_roundtrip_and_verified_zip(self) -> None:
        created = self.action("run.create", name="HTTP experiment", config={
            "steps": 8, "run_fp64_reference_track": False,
            "dataset": {"samples": 8, "batch_size": 2},
            "landscape": {"points": 5},
        })
        run_id = created["result"]["run_id"]
        stepped = self.action("run.step", run_id=run_id, count=2)
        self.assertTrue(stepped["ok"])
        status, record, _headers = self.request(f"/api/runs/{run_id}/record?step=1")
        self.assertEqual(status, 200, record)
        self.assertEqual(record["step"], 1)
        self.assertNotEqual(record["parameter_before"], record["parameter_after"])
        status, dataset, _headers = self.request(f"/api/runs/{run_id}/dataset")
        self.assertEqual(status, 200, dataset)
        self.assertEqual(len(dataset["inputs"]), 8)
        self.assertEqual(len(dataset["targets"]), 8)
        status, project, headers = self.request("/api/project")
        self.assertEqual(status, 200, project)
        self.assertIn("attachment", headers["Content-Disposition"])
        status, imported, _headers = self.request("/api/project", "POST", project)
        self.assertEqual(status, 200, imported)
        self.assertTrue(imported["ok"])
        status, restored, _headers = self.request(f"/api/runs/{run_id}/record?step=1")
        self.assertEqual(status, 200, restored)
        self.assertEqual(restored, record)
        status, archive, headers = self.request(f"/api/runs/{run_id}/export")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            self.assertIsNone(zipped.testzip())
            self.assertTrue(zipped.namelist())
            self.assertTrue(any(name.endswith(".json") for name in zipped.namelist()))
            self.assertTrue(all(not name.startswith(("/", "\\")) and ".." not in name.split("/")
                                for name in zipped.namelist()))

    def test_nonfinite_run_still_observes_downloads_and_restores_as_strict_json(self) -> None:
        created = self.action("run.create", preset="diverging", name="Divergence via HTTP", config={
            "steps": 6, "stop_on_nonfinite": False, "run_fp64_reference_track": False,
        })
        run_id = created["result"]["run_id"]
        stepped = self.action("run.step", run_id=run_id, count=6)
        run = next(item for item in stepped["state"]["runs"] if item["id"] == run_id)
        self.assertEqual(run["step"], 6)
        self.assertFalse(run["latest"]["finite"])
        self.assertIsInstance(run["parameter"], str)
        latest_step = run["latest"]["step"]
        for path in ("/api/state", f"/api/runs/{run_id}/dataset",
                     f"/api/runs/{run_id}/record?step={latest_step}"):
            status, data, _headers = self.request(path)
            self.assertEqual(status, 200, data)
        status, project, _headers = self.request("/api/project")
        self.assertEqual(status, 200, project)
        status, imported, _headers = self.request("/api/project", "POST", project)
        self.assertEqual(status, 200, imported)
        status, archive, _headers = self.request(f"/api/runs/{run_id}/export")
        self.assertEqual(status, 200)
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            self.assertIsNone(zipped.testzip())
            json_names = [name for name in zipped.namelist() if name.endswith(".json")]
            self.assertTrue(json_names)
            for name in json_names:
                def reject_constant(value: str, filename: str = name) -> None:
                    self.fail(f"Non-finite JSON token {value} in {filename}")
                json.loads(zipped.read(name), parse_constant=reject_constant)

    def test_rejects_host_rebinding_cross_origin_and_cross_site_changes(self) -> None:
        payload = {"action": "run.create"}
        for headers in ({"Host": f"evil.example:{self.server.server_port}"},
                        {"Host": "127.0.0.1:1"},
                        {"Origin": "https://evil.example"},
                        {"Origin": "null"},
                        {"Origin": "http://127.0.0.1:1"},
                        {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                status, data, _headers = self.request("/api/action", "POST", payload, headers)
                self.assertEqual(status, 403, data)
                self.assertFalse(data["ok"])
        status, _data, _headers = self.request("/api/state", headers={"Origin": self.server.url.rstrip("/")})
        self.assertEqual(status, 200)

    def test_fixed_routes_block_filesystem_traversal(self) -> None:
        for path in ("/../pyproject.toml", "/%2e%2e/pyproject.toml", "/%5c..%5cpyproject.toml",
                     "/studio_web/../../pyproject.toml", "/studio_server.py", "/%00",
                     "/api/runs/../export", "/api/not-found"):
            with self.subTest(path=path):
                status, data, _headers = self.request(path)
                self.assertEqual(status, 404, data)
                self.assertFalse(data["ok"])

    def test_rejects_malformed_and_oversized_json_before_mutation(self) -> None:
        for payload in (b'{"action":"run.create","value":NaN}',
                        b'{"action":"run.create","value":Infinity}',
                        b'{"action":"run.create","value":1e999}',
                        b'{"action":"run.create","action":"run.step"}',
                        b'[]', b'{', b'\xff'):
            with self.subTest(payload=payload):
                status, data, _headers = self.request("/api/action", "POST", payload,
                                                     {"Content-Type": "application/json"})
                self.assertEqual(status, 400, data)
                self.assertFalse(data["ok"])
        status, data, _headers = self.request("/api/action", "POST", b"{}",
                                             {"Content-Type": "application/json",
                                              "Content-Length": str(MAX_BODY_BYTES + 1)})
        self.assertEqual(status, 413, data)
        status, data, _headers = self.request("/api/action", "POST", b"{}")
        self.assertEqual(status, 415, data)

    def test_missing_record_validation_and_real_errors_are_visible(self) -> None:
        for path in ("/api/runs/missing/dataset", "/api/runs/missing/export",
                     "/api/runs/missing/record?step=1"):
            status, data, _headers = self.request(path)
            self.assertEqual(status, 404, data)
            self.assertFalse(data["ok"])
        for path in ("/api/runs/missing/record", "/api/runs/missing/record?step=1&step=2",
                     "/api/runs/missing/record?step=1.5"):
            status, data, _headers = self.request(path)
            self.assertEqual(status, 400, data)
        status, data, _headers = self.request("/api/action", "POST", {"action": "no.such.action"})
        self.assertEqual(status, 400, data)
        with patch.object(self.server.workspace, "snapshot", side_effect=RuntimeError("visible backend failure")):
            with self.assertLogs("single_parameter_observatory.studio_server", level="ERROR"):
                status, data, _headers = self.request("/api/state")
        self.assertEqual(status, 500, data)
        self.assertEqual(data["type"], "RuntimeError")
        self.assertIn("visible backend failure", data["error"])

    def test_workspace_lifecycle_failure_is_a_conflict(self) -> None:
        self.server.workspace.close()
        status, data, _headers = self.request("/api/state")
        self.assertEqual(status, 409, data)
        self.assertEqual(data["type"], "StudioStateError")

    def test_disallowed_methods_return_json(self) -> None:
        status, data, _headers = self.request("/api/state", "DELETE")
        self.assertEqual(status, 405, data)
        self.assertFalse(data["ok"])

    def test_server_close_closes_owned_workspace_once(self) -> None:
        self.server.shutdown()
        with patch.object(self.server.workspace, "close", wraps=self.server.workspace.close) as close:
            self.server.server_close()
            self.server.server_close()
            close.assert_called_once()

    def test_server_close_stops_real_training_and_autosave_workers(self) -> None:
        created = self.action("run.create", name="Worker shutdown", config={
            "steps": 10000, "run_fp64_reference_track": False,
            "dataset": {"samples": 8, "batch_size": 2},
            "landscape": {"enabled": False},
        })
        run_id = created["result"]["run_id"]
        self.action("run.speed", run_id=run_id, value=0.01)
        self.action("run.play", run_id=run_id)
        controller = self.server.workspace._runs[run_id].controller
        training_worker = controller._thread
        autosave_worker = self.server.workspace._autosave_thread
        self.assertIsNotNone(training_worker)
        self.assertIsNotNone(autosave_worker)
        assert training_worker is not None
        assert autosave_worker is not None
        self.server.shutdown()
        self.server.server_close()
        self.assertFalse(training_worker.is_alive())
        self.assertFalse(autosave_worker.is_alive())


class TestStudioServerLifecycle(unittest.TestCase):
    def test_browser_disconnect_during_response_write_is_not_a_backend_failure(self) -> None:
        for error in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            for failure_response in (False, True):
                with self.subTest(error=error.__name__, failure_response=failure_response):
                    handler = object.__new__(StudioRequestHandler)
                    handler.command = "GET"
                    stream = Mock()
                    stream.write.side_effect = error("Browser connection closed")
                    def write_response(current_handler: StudioRequestHandler = handler) -> None:
                        current_handler._send(b"response", "text/plain")
                    dispatch = ValueError("Invalid request") if failure_response else write_response
                    with patch.object(handler, "_validate_host"), \
                            patch.object(handler, "_get", side_effect=dispatch), \
                            patch.object(handler, "send_response") as response, \
                            patch.object(handler, "send_header"), \
                            patch.object(handler, "end_headers"), \
                            patch.object(handler, "wfile", stream, create=True), \
                            patch("single_parameter_observatory.studio_server._LOG.exception") as log:
                        handler._handle()
                        stream.write.assert_called_once()
                        response.assert_called_once_with(400 if failure_response else 200)
                        log.assert_not_called()

    def test_module_cli_persists_projects_by_default(self) -> None:
        with patch("single_parameter_observatory.studio_server.serve_studio") as serve:
            self.assertEqual(main([]), 0)
            self.assertEqual(serve.call_args.kwargs["storage_directory"], Path(".spobs-studio"))
            self.assertEqual(main(["--storage", "another-project", "--port", "9000"]), 0)
            self.assertEqual(serve.call_args.kwargs["storage_directory"], Path("another-project"))
            self.assertEqual(serve.call_args.kwargs["port"], 9000)

    def test_host_and_port_validation(self) -> None:
        for host in ("0.0.0.0", "::", "example.com", "192.168.1.5"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                create_server(host=host, port=0)
        for port in (-1, 65536, True, 1.5, "8765"):
            with self.subTest(port=port), self.assertRaises(ValueError):
                create_server(port=port)  # type: ignore[arg-type]

    def test_busy_port_fails_cleanly(self) -> None:
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            with tempfile.TemporaryDirectory() as directory, \
                    patch("single_parameter_observatory.studio_server.StudioWorkspace") as workspace:
                with self.assertRaises(OSError):
                    create_server(port=occupied.getsockname()[1], storage_directory=directory)
                workspace.assert_not_called()

    def test_second_studio_cannot_claim_a_live_studio_port(self) -> None:
        first = create_server(port=0)
        try:
            with self.assertRaises(OSError):
                second = create_server(port=first.server_port)
                second.server_close()
        finally:
            first.server_close()

    def test_interrupt_closes_workspace_and_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = create_server(port=0, storage_directory=directory)
            with patch("single_parameter_observatory.studio_server.create_server", return_value=server), \
                    patch.object(server, "serve_forever", side_effect=KeyboardInterrupt), \
                    patch.object(server.workspace, "close", wraps=server.workspace.close) as close, \
                    patch("builtins.print") as output:
                serve_studio(storage_directory=directory)
                close.assert_called_once()
                self.assertTrue(any(server.url in str(call) for call in output.call_args_list))
            self.assertEqual(server.socket.fileno(), -1)


if __name__ == "__main__":
    unittest.main()
