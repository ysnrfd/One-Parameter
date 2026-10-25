"""Loopback HTTP transport for the real, local Single Parameter Studio.

The browser is a client of :class:`StudioWorkspace`; it never evaluates Python
or chooses host filesystem paths. All training continues through the existing
PyTorch engine. No network-facing server or additional web dependency is used.
"""

from __future__ import annotations

import argparse
import copy
import ipaddress
import json
import logging
import math
import re
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__
from .studio_codegen import python_project
from .recorder import json_safe
from .studio_graph import GraphConflictError
from .studio_session import StudioStateError, StudioWorkspace

_LOG = logging.getLogger(__name__)
# A project carries full history and checkpoints, so the import boundary must
# accommodate our own downloads while still bounding a single request.
MAX_BODY_BYTES = 64 * 1024 * 1024
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/studio.css": ("studio.css", "text/css; charset=utf-8"),
    "/studio.js": ("studio.js", "text/javascript; charset=utf-8"),
    "/graph.js": ("graph.js", "text/javascript; charset=utf-8"),
    "/graph_core.js": ("graph_core.js", "text/javascript; charset=utf-8"),
    "/graph.css": ("graph.css", "text/css; charset=utf-8"),
    "/transformers.js": ("transformers.js", "text/javascript; charset=utf-8"),
    "/transformers.css": ("transformers.css", "text/css; charset=utf-8"),
}


class RequestError(ValueError):
    """A malformed or forbidden HTTP request with an explicit status."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def _loopback_host(host: str) -> str:
    if host.lower() == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Studio host must be a loopback IP address or localhost") from exc
    if not address.is_loopback:
        raise ValueError("Studio only listens on loopback addresses")
    return str(address)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value} is not accepted")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON numeric values must be finite")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class StudioHTTPServer(ThreadingHTTPServer):
    """HTTP server owning exactly one workspace and its training workers."""

    daemon_threads = True
    # Windows SO_REUSEADDR can let two live servers bind the same local port.
    # Exclusive binding keeps the printed studio URL tied to its own workspace.
    allow_reuse_address = not hasattr(socket, "SO_EXCLUSIVEADDRUSE")

    def __init__(self, host: str, port: int, storage_directory: str | Path | None = None) -> None:
        self._workspace_close_lock = threading.Lock()
        self._workspace_closed = False
        self._workspace_ready = False
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        # Claim the port before opening an autosave. A rejected second launch
        # must not touch the first studio's persistent project.
        super().__init__((host, port), StudioRequestHandler)
        try:
            self.workspace = StudioWorkspace(storage_directory=storage_directory)
            self._workspace_ready = True
        except BaseException:
            super().server_close()
            raise

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    @property
    def url(self) -> str:
        host = str(self.server_address[0])
        port = self.server_address[1]
        return f"http://{'[' + host + ']' if ':' in host else host}:{port}/"

    def server_close(self) -> None:
        try:
            with self._workspace_close_lock:
                if self._workspace_ready and not self._workspace_closed:
                    self.workspace.close()
                    self._workspace_closed = True
        finally:
            super().server_close()


class StudioRequestHandler(BaseHTTPRequestHandler):
    """Small fixed-route JSON API with no arbitrary file serving."""

    server: StudioHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "SingleParameterStudio"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format: str, *args: Any) -> None:
        _LOG.debug("%s %s", self.address_string(), format % args)

    def _send(self, data: bytes, content_type: str, status: int = HTTPStatus.OK,
              filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                         "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
                         "object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename is not None:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).name)
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, value: Any, status: int = HTTPStatus.OK,
              filename: str | None = None) -> None:
        data = json.dumps(json_safe(value), ensure_ascii=True, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
        self._send(data, "application/json; charset=utf-8", status, filename)

    def _validate_host(self) -> None:
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:
            raise RequestError("Exactly one local Host header is required", HTTPStatus.FORBIDDEN)
        try:
            parsed = urlsplit("//" + hosts[0])
            host = parsed.hostname
            port = parsed.port or 80
            if (not host or parsed.username or parsed.password or parsed.path
                    or parsed.query or parsed.fragment):
                raise ValueError("Malformed Host")
            _loopback_host(host)
        except ValueError as exc:
            raise RequestError("Host must identify this local studio", HTTPStatus.FORBIDDEN) from exc
        if port != self.server.server_address[1]:
            raise RequestError("Host port does not match this studio", HTTPStatus.FORBIDDEN)

        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1:
            raise RequestError("Only one Origin is accepted", HTTPStatus.FORBIDDEN)
        if origins:
            try:
                origin = urlsplit(origins[0])
                same_origin = (origin.scheme == "http" and origin.netloc.lower() == hosts[0].lower()
                               and not origin.path and not origin.query and not origin.fragment)
            except ValueError:
                same_origin = False
            if not same_origin:
                raise RequestError("Cross-origin studio requests are blocked", HTTPStatus.FORBIDDEN)
        if self.command != "GET" and self.command != "HEAD":
            if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
                raise RequestError("Cross-site studio changes are blocked", HTTPStatus.FORBIDDEN)

    def _request_target(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise RequestError("Only relative local request paths are accepted")
        path = unquote(parsed.path, errors="strict")
        if "\\" in path or "\x00" in path or ".." in path.split("/"):
            raise RequestError("Path traversal is not allowed", HTTPStatus.NOT_FOUND)
        return path, parse_qs(parsed.query, keep_blank_values=True, max_num_fields=16)

    def _body(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise RequestError("Transfer-Encoding is not supported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise RequestError("A valid Content-Length is required")
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            raise RequestError(f"JSON request exceeds {MAX_BODY_BYTES} bytes", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if self.headers.get_content_type() != "application/json":
            raise RequestError("Content-Type must be application/json", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise RequestError("Request body ended before Content-Length")
        try:
            value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_finite_float,
                               object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise RequestError(f"Invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RequestError("JSON request must be an object")
        return value

    def _epoch_header(self) -> str | None:
        epochs = self.headers.get_all("X-Studio-Epoch", [])
        if len(epochs) > 1:
            raise RequestError("Only one workspace epoch header is accepted")
        return epochs[0] if epochs else None

    def _get(self) -> None:
        path, query = self._request_target()
        if path in _STATIC:
            name, content_type = _STATIC[path]
            asset = resources.files("single_parameter_observatory").joinpath("studio_web").joinpath(name)
            self._send(asset.read_bytes(), content_type)
            return
        if path == "/api/health":
            self._json({"ok": True, "service": "single-parameter-studio", "version": __version__})
            return
        workspace = self.server.workspace
        with workspace._lock:
            workspace.check_workspace_epoch(self._epoch_header())
            value, response_content_type, filename = self._workspace_get(path, query)
        # The immutable bytes / copied JSON snapshot is detached from live
        # state before any potentially slow client transfer starts.
        if response_content_type is None:
            self._json(value, filename=filename)
        else:
            self._send(value, response_content_type, filename=filename)

    def _workspace_get(self, path: str, query: dict[str, list[str]]) -> tuple[Any, str | None, str | None]:
        """Read while the workspace admission lock is held; never write HTTP here."""
        workspace = self.server.workspace
        if path == "/api/state":
            return workspace.snapshot(), None, None
        if path == "/api/capabilities":
            return workspace.capabilities(), None, None
        if path == "/api/graph/state":
            return workspace.graphs.snapshot(), None, None
        if path == "/api/graph/catalog":
            return workspace.graphs.catalog(), None, None
        if path == "/api/ui/drafts":
            return {"drafts": copy.deepcopy(workspace.ui_drafts), "revision": workspace._ui_drafts_revision,
                    "workspace_epoch": workspace.workspace_epoch}, None, None
        if re.fullmatch(r"/api/graph/python/[A-Za-z0-9_.:-]+", path):
            graph_id = path.rsplit("/", 1)[-1]
            source = python_project(workspace.export_project(), graph_id)
            return source, "text/x-python; charset=utf-8", f"spobs-workflow-{graph_id[:80]}.py"
        if re.fullmatch(r"/api/graph/artifacts/[A-Za-z0-9_.:-]+", path):
            data, filename, mime = workspace.graphs.artifact(path.rsplit("/", 1)[-1])
            return data, mime, filename
        if re.fullmatch(r"/api/graph/jobs/[A-Za-z0-9_.:-]+", path):
            job_id = path.rsplit("/", 1)[-1]
            return workspace.graphs.job(job_id), None, f"spobs-job-{job_id[:80]}.json"
        if path == "/api/project":
            return workspace.export_project(), None, "single-parameter-studio.json"
        match = re.fullmatch(r"/api/runs/([A-Za-z0-9_-]+)/(record|dataset|export)", path)
        if match is None:
            raise RequestError("Route not found", HTTPStatus.NOT_FOUND)
        run_id, operation = match.groups()
        if operation == "record":
            steps = query.get("step", [])
            if len(steps) != 1 or not re.fullmatch(r"[0-9]+", steps[0]):
                raise RequestError("Record requests require one integer step query parameter")
            return workspace.record(run_id, int(steps[0])), None, None
        if operation == "dataset":
            return workspace.dataset(run_id), None, None
        data, filename = workspace.export_run(run_id)
        return data, "application/zip", filename

    def _post(self) -> None:
        path, _query = self._request_target()
        if path not in ("/api/action", "/api/project", "/api/graph/action"):
            raise RequestError("Route not found", HTTPStatus.NOT_FOUND)
        payload = self._body()
        workspace = self.server.workspace
        with workspace._lock:
            # Admission and the mutation share one gate. An import may reuse
            # graph/run IDs and revision numbers while replacing their meaning.
            workspace.check_workspace_epoch(self._epoch_header())
            if path == "/api/project":
                result = workspace.import_project(payload)
            elif path == "/api/graph/action":
                action = payload.pop("action", None)
                if not isinstance(action, str) or not action.strip():
                    raise RequestError("An action string is required")
                result = workspace.graph_action(action, payload)
                state = workspace.graphs.snapshot()
            else:
                action = payload.pop("action", None)
                if not isinstance(action, str) or not action.strip():
                    raise RequestError("An action string is required")
                result = workspace.dispatch(action, payload)
            if path != "/api/graph/action":
                state = workspace.snapshot()
        self._json({"ok": True, "state": state, "result": result})

    def _handle(self) -> None:
        try:
            self._validate_host()
            if self.command in ("GET", "HEAD"):
                self._get()
            elif self.command == "POST":
                self._post()
            else:
                raise RequestError("Method not supported", HTTPStatus.METHOD_NOT_ALLOWED)
        except ConnectionError:
            return
        except Exception as exc:
            # Stop keep-alive after rejection: unread request bodies must never
            # be interpreted as a subsequent request on the same connection.
            self.close_connection = True
            if isinstance(exc, RequestError):
                status = exc.status
            elif isinstance(exc, (KeyError, FileNotFoundError)):
                status = HTTPStatus.NOT_FOUND
            elif isinstance(exc, (ValueError, TypeError, OverflowError, RecursionError)):
                status = HTTPStatus.BAD_REQUEST
            elif isinstance(exc, (StudioStateError, GraphConflictError)):
                status = HTTPStatus.CONFLICT
            else:
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                _LOG.exception("Studio request failed")
            try:
                self._json({"ok": False, "error": str(exc), "type": type(exc).__name__}, status)
            except ConnectionError:
                pass

    do_GET = _handle
    do_HEAD = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle
    do_OPTIONS = _handle


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  storage_directory: str | Path | None = None) -> StudioHTTPServer:
    """Create a loopback server; use port 0 for an OS-selected free port.

    Binding failures are explicit and close the new workspace immediately.
    A busy requested port is never silently replaced with a different service.
    """
    host = _loopback_host(host)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Studio port must be an integer between 0 and 65535")
    return StudioHTTPServer(host, port, storage_directory)


def serve_studio(host: str = "127.0.0.1", port: int = 8765,
                 storage_directory: str | Path | None = None, open_browser: bool = False) -> None:
    """Run the studio until interrupted, then stop its real training workers."""
    server = create_server(host, port, storage_directory)
    print(f"Single Parameter Studio: {server.url}", flush=True)
    print("Press Ctrl+C to stop the studio.", flush=True)
    try:
        if open_browser:
            webbrowser.open(server.url)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Single Parameter Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--storage", "--storage-directory", dest="storage_directory", type=Path,
                        default=Path(".spobs-studio"), help="Local project directory (default: .spobs-studio)")
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)
    try:
        serve_studio(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Studio could not start: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
