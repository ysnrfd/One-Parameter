"""Validated declarative add-ons and disposable, resource-bounded JavaScript workers.

Plugins are data, never Python modules or browser scripts.  A custom node gets
JSON inputs and parameters in a new V8 isolate in a separate process.  There are
no Python callbacks, filesystem/network bindings, or inherited application state.
This is a local scripting boundary, not an operating-system virtual machine.
"""

from __future__ import annotations

import copy
import ctypes
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


MAX_CODE_BYTES = 64 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_PLUGIN_BYTES = 4 * 1024 * 1024
# JSON can encode one source character as six ASCII escape bytes.  The frame
# budget must not accidentally shrink the separately enforced source/data caps.
MAX_WORKER_REQUEST = MAX_JSON_BYTES + 6 * MAX_CODE_BYTES + 1024
WORKER_STARTUP_GRACE_MS = 30000
WORKER_EXECUTION_GRACE_MS = 3000
PORT_TYPES = frozenset({"any", "number", "integer", "string", "boolean", "object", "array",
                        "run", "config", "dataset", "tensor-expression"})
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,95}\Z", re.ASCII)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z", re.ASCII)
_RESERVED_NAMES = {"__proto__", "prototype", "constructor"}


class ScriptSandboxError(ValueError):
    """A script or manifest failed validation or a bounded worker was stopped."""


def _json_bytes(value: Any, limit: int, label: str) -> bytes:
    """Reject non-JSON values rather than letting JSON silently coerce them."""
    seen: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        if depth > 64:
            raise ScriptSandboxError(f"{label} exceeds the maximum nesting depth of 64")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ScriptSandboxError(f"{label} must contain finite numbers")
            return
        if type(item) not in (dict, list):
            raise ScriptSandboxError(f"{label} must contain only plain JSON values")
        identity = id(item)
        if identity in seen:
            raise ScriptSandboxError(f"{label} contains a circular reference")
        seen.add(identity)
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ScriptSandboxError(f"{label} object keys must be strings")
                visit(child, depth + 1)
        else:
            for child in item:
                visit(child, depth + 1)
        seen.remove(identity)

    try:
        visit(value, 0)
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        if isinstance(exc, ScriptSandboxError):
            raise
        raise ScriptSandboxError(f"Invalid {label}: {exc}") from exc
    if len(encoded) > limit:
        raise ScriptSandboxError(f"{label} exceeds {limit:,} bytes")
    return encoded


def _object(value: Any, label: str, allowed: set[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ScriptSandboxError(f"{label} must be an object")
    unexpected = set(value) - allowed
    if unexpected:
        raise ScriptSandboxError(f"{label} has unknown fields: {', '.join(sorted(unexpected))}")
    return value


def _text(value: Any, label: str, *, maximum: int = 256, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise ScriptSandboxError(f"{label} must be {'a nonempty' if not empty else 'a'} string of at most {maximum} characters")
    if "\x00" in value:
        raise ScriptSandboxError(f"{label} must not contain null characters")
    return value


def _code(value: Any, label: str) -> str:
    result = _text(value, label, maximum=MAX_CODE_BYTES)
    try:
        length = len(result.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ScriptSandboxError(f"{label} must contain valid UTF-8 text") from exc
    if length > MAX_CODE_BYTES:
        raise ScriptSandboxError(f"{label} exceeds {MAX_CODE_BYTES:,} UTF-8 bytes")
    return result


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _identifier(value: Any, label: str, *, port: bool = False) -> str:
    pattern = _NAME if port else _IDENTIFIER
    if not isinstance(value, str) or not pattern.fullmatch(value) or value in _RESERVED_NAMES:
        raise ScriptSandboxError(f"{label} must be a safe ASCII identifier")
    return value


def _items(value: Any, label: str, *, maximum: int = 64) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ScriptSandboxError(f"{label} must be a list with at most {maximum} entries")
    return value


def _typed_value(value: Any, kind: str, label: str) -> None:
    valid = {
        "any": True,
        "number": type(value) in (int, float),
        "integer": type(value) is int,
        "string": isinstance(value, str),
        "boolean": type(value) is bool,
        "object": type(value) is dict,
        "array": type(value) is list,
        "run": isinstance(value, str) and bool(value),
        "config": type(value) is dict,
        "dataset": type(value) is dict,
        "tensor-expression": type(value) is dict and value.get("kind") == "tensor-expression",
    }[kind]
    if not valid:
        raise ScriptSandboxError(f"{label} must match type {kind}")
    _json_bytes(value, MAX_JSON_BYTES, label)
    if kind == "dataset":
        if not isinstance(value.get("inputs"), list) or not isinstance(value.get("targets"), list):
            raise ScriptSandboxError(f"{label} dataset needs inputs and targets arrays")
        if len(value["inputs"]) != len(value["targets"]):
            raise ScriptSandboxError(f"{label} dataset inputs and targets lengths differ")
        for number in value["inputs"] + value["targets"]:
            if type(number) not in (int, float):
                raise ScriptSandboxError(f"{label} dataset samples must be numbers")


def _field(value: Any, label: str, *, output: bool = False, parameter: bool = False) -> dict[str, Any]:
    allowed = {"name", "type", "description"}
    if not output:
        allowed |= {"default"}
    if parameter:
        allowed |= {"min", "max", "options"}
    elif not output:
        allowed |= {"required"}
    source = _object(value, label, allowed)
    result: dict[str, Any] = {"name": _identifier(source.get("name"), f"{label}.name", port=True)}
    kind = source.get("type", "any")
    if not isinstance(kind, str) or kind not in PORT_TYPES:
        raise ScriptSandboxError(f"{label}.type is unsupported")
    result["type"] = kind
    result["description"] = _text(source.get("description", ""), f"{label}.description", maximum=4000, empty=True)
    if not parameter and not output:
        required = source.get("required", True)
        if type(required) is not bool:
            raise ScriptSandboxError(f"{label}.required must be boolean")
        result["required"] = required
    if "default" in source:
        _typed_value(source["default"], kind, f"{label}.default")
        result["default"] = copy.deepcopy(source["default"])
    for bound in ("min", "max"):
        if bound not in source:
            continue
        number = source[bound]
        if kind not in ("number", "integer") or not _finite(number):
            raise ScriptSandboxError(f"{label}.{bound} requires a finite numeric parameter")
        if kind == "integer" and type(number) is not int:
            raise ScriptSandboxError(f"{label}.{bound} must be an integer")
        result[bound] = number
    if "min" in result and "max" in result and result["min"] > result["max"]:
        raise ScriptSandboxError(f"{label}.min must not exceed max")
    if "options" in source:
        choices = _items(source["options"], f"{label}.options", maximum=256)
        if not choices:
            raise ScriptSandboxError(f"{label}.options must not be empty")
        for choice in choices:
            _typed_value(choice, kind, f"{label}.options")
        result["options"] = copy.deepcopy(choices)
    if "default" in result:
        default = result["default"]
        if (("min" in result and default < result["min"])
                or ("max" in result and default > result["max"])):
            raise ScriptSandboxError(f"{label}.default is outside its bounds")
        if "options" in result and default not in result["options"]:
            raise ScriptSandboxError(f"{label}.default must be one of its options")
    return result


def _fields(value: Any, label: str, *, output: bool = False, parameter: bool = False) -> list[dict[str, Any]]:
    result = [_field(item, f"{label}[{index}]", output=output, parameter=parameter)
              for index, item in enumerate(_items(value, label))]
    names = [item["name"] for item in result]
    if len(set(names)) != len(names):
        raise ScriptSandboxError(f"{label} contains duplicate names")
    return result


def validate_plugin(data: Any) -> dict[str, Any]:
    """Return an independent normalized plugin, rejecting unsupported capabilities.

    Templates remain declarative graph objects; the graph runtime validates their
    connections and node types when they are inserted or run.  Panels contain
    parameter fields only and never executable HTML, CSS, or browser JavaScript.
    """
    _json_bytes(data, MAX_PLUGIN_BYTES, "Plugin")
    source = _object(data, "Plugin", {"schema_version", "id", "name", "version", "description",
                                       "nodes", "templates", "panels"})
    if source.get("schema_version") != "spobs-plugin/1":
        raise ScriptSandboxError("Plugin schema_version must be spobs-plugin/1")
    result: dict[str, Any] = {
        "schema_version": "spobs-plugin/1",
        "id": _identifier(source.get("id"), "Plugin.id"),
        "name": _text(source.get("name"), "Plugin.name"),
        "version": _text(source.get("version"), "Plugin.version", maximum=64),
        "description": _text(source.get("description", ""), "Plugin.description", maximum=8000, empty=True),
        "nodes": [], "templates": [], "panels": [],
    }
    identifiers: set[str] = set()
    for index, value in enumerate(_items(source.get("nodes", []), "Plugin.nodes")):
        label = f"Plugin.nodes[{index}]"
        node = _object(value, label, {"id", "title", "category", "description", "inputs", "outputs", "parameters", "code"})
        identifier = _identifier(node.get("id"), f"{label}.id")
        if identifier in identifiers:
            raise ScriptSandboxError(f"Duplicate plugin node id: {identifier}")
        identifiers.add(identifier)
        code = _code(node.get("code"), f"{label}.code")
        result["nodes"].append({
            "id": identifier, "title": _text(node.get("title"), f"{label}.title"),
            "category": _text(node.get("category", "Custom"), f"{label}.category"),
            "description": _text(node.get("description", ""), f"{label}.description", maximum=8000, empty=True),
            "inputs": _fields(node.get("inputs", []), f"{label}.inputs"),
            "outputs": _fields(node.get("outputs", []), f"{label}.outputs", output=True),
            "parameters": _fields(node.get("parameters", []), f"{label}.parameters", parameter=True),
            "code": code,
        })
    for collection, allowed in (("templates", {"id", "name", "graph"}),
                                ("panels", {"id", "title", "description", "fields"})):
        identifiers = set()
        for index, value in enumerate(_items(source.get(collection, []), f"Plugin.{collection}", maximum=16)):
            label = f"Plugin.{collection}[{index}]"
            item = _object(value, label, allowed)
            identifier = _identifier(item.get("id"), f"{label}.id")
            if identifier in identifiers:
                raise ScriptSandboxError(f"Duplicate {collection} id: {identifier}")
            identifiers.add(identifier)
            if collection == "templates":
                if type(item.get("graph")) is not dict:
                    raise ScriptSandboxError(f"{label}.graph must be an object")
                result[collection].append({"id": identifier, "name": _text(item.get("name"), f"{label}.name"),
                                           "graph": copy.deepcopy(item["graph"])})
            else:
                result[collection].append({
                    "id": identifier, "title": _text(item.get("title"), f"{label}.title"),
                    "description": _text(item.get("description", ""), f"{label}.description", maximum=8000, empty=True),
                    "fields": _fields(item.get("fields", []), f"{label}.fields", parameter=True),
                })
    if not any(result[key] for key in ("nodes", "templates", "panels")):
        raise ScriptSandboxError("Plugin must contain at least one node, template, or panel")
    _json_bytes(result, MAX_PLUGIN_BYTES, "Normalized plugin")
    return result


def node_definitions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Build globally namespaced catalog entries from a normalized manifest."""
    plugin = validate_plugin(manifest)
    return [{**copy.deepcopy(node), "id": f"{plugin['id']}/{node['id']}",
             "type": f"{plugin['id']}/{node['id']}", "plugin_id": plugin["id"], "node_id": node["id"]}
            for node in plugin["nodes"]]


def sandbox_capabilities() -> dict[str, Any]:
    """Describe enforced scripting boundaries for the catalog and help panels."""
    return {
        "available": importlib.util.find_spec("py_mini_racer") is not None,
        "language": "javascript", "runtime": "V8 / mini-racer",
        "execution": "synchronous-function-body", "fresh_process_per_invocation": True,
        "code_limit_bytes": MAX_CODE_BYTES, "input_limit_bytes": MAX_JSON_BYTES,
        "output_limit_bytes": MAX_JSON_BYTES, "max_json_depth": 64,
        "max_output_values": 500000, "max_output_array_length": 100000,
        "timeout_ms": {"min": 10, "max": 10000, "default": 1000},
        "memory_mb": {"min": 16, "max": 64, "default": 32}, "wall_grace_ms": WORKER_STARTUP_GRACE_MS,
        "startup_timeout_ms": WORKER_STARTUP_GRACE_MS,
        "execution_wall_grace_ms": WORKER_EXECUTION_GRACE_MS,
        "worker_request_limit_bytes": MAX_WORKER_REQUEST,
        "windows_process_memory_mb": 256, "windows_process_limit": 1,
        "windows_job_active": os.name == "nt", "os_virtual_machine": False,
        "logs": {"max_entries": 100, "max_line_characters": 2048, "max_total_characters": 16384},
        "host_io": False, "python": False, "network": False, "filesystem": False,
        "dom": False, "timers": False, "webassembly": False,
        "numbers": "IEEE-754 binary64; nonexact integer inputs rejected; use decimal strings and BigInt for wide integers",
    }


def example_plugin() -> dict[str, Any]:
    """A runnable, editable plugin example with an actual dataset transform."""
    return validate_plugin({
        "schema_version": "spobs-plugin/1", "id": "local.examples", "name": "Example transformations",
        "version": "1.0.0", "description": "Copy this manifest to create your own isolated JavaScript nodes.",
        "nodes": [
            {"id": "scale", "title": "Scale a number", "category": "Examples", "description": "Multiply a value by a factor.",
             "inputs": [{"name": "value", "type": "number"}], "outputs": [{"name": "value", "type": "number"}],
             "parameters": [{"name": "factor", "type": "number", "default": 2}],
             "code": "console.log('Scaling by', params.factor); return {value: inputs.value * params.factor};"},
            {"id": "scale_targets", "title": "Scale dataset targets", "category": "Examples",
             "description": "Transform target values without changing input samples.",
             "inputs": [{"name": "data", "type": "dataset"}], "outputs": [{"name": "data", "type": "dataset"}],
             "parameters": [{"name": "factor", "type": "number", "default": 2}],
             "code": "return {data: {...inputs.data, targets: inputs.data.targets.map(v => v * params.factor), provenance: {...inputs.data.provenance, transform: 'scale_targets', factor: params.factor}}};"},
        ],
        "templates": [{"id": "scale_workflow", "name": "Example plugin: scale and observe", "graph": {
            "name": "Plugin scale playground", "description": "A real custom node transforms a typed number and previews the result.",
            "nodes": [
                {"id": "number", "type": "core.number", "position": {"x": 40, "y": 100}, "params": {"value": 21}},
                {"id": "scale", "type": "local.examples/scale", "position": {"x": 370, "y": 100}, "params": {"factor": 2}},
                {"id": "preview", "type": "core.preview", "position": {"x": 700, "y": 100}, "params": {}},
            ],
            "edges": [
                {"id": "number-scale", "source": "number", "source_port": "value", "target": "scale", "target_port": "value"},
                {"id": "scale-preview", "source": "scale", "source_port": "value", "target": "preview", "target_port": "value"},
            ], "view": {"x": 0, "y": 0, "zoom": 1},
        }}],
        "panels": [{"id": "example_controls", "title": "Example reusable controls",
                    "description": "Adjust these values and insert them as an Object node for use in your graph.",
                    "fields": [{"name": "value", "type": "number", "default": 21},
                               {"name": "factor", "type": "number", "default": 2, "min": -100, "max": 100}]}],
    })


class _WindowsJob:
    """Bound committed worker process memory and kill it if its job handle closes."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.handle: Any = None
        self.kernel: Any = None
        if os.name != "nt":
            return
        from ctypes import wintypes  # noqa: PLC0415 -- only available on Windows

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount",
                "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "Could not create script worker resource job")
        self.handle, self.kernel = job, kernel
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x0100 | 0x2000 | 0x0008  # process memory, kill on close, active count
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        limits.ProcessMemoryLimit = 256 * 1024 * 1024
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise OSError(ctypes.get_last_error(), "Could not bound script worker memory")
        process_handle = kernel.OpenProcess(0x0100 | 0x0001, False, process.pid)  # set quota, terminate
        if not process_handle:
            self.close()
            raise OSError(ctypes.get_last_error(), "Could not open script worker resource handle")
        try:
            if not kernel.AssignProcessToJobObject(job, process_handle):
                self.close()
                raise OSError(ctypes.get_last_error(), "Could not assign script worker resource job")
        finally:
            kernel.CloseHandle(process_handle)

    def close(self) -> None:
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _check_javascript_integers(value: Any) -> None:
    """Never silently round an exact Python integer on the JSON-to-V8 handoff."""
    if type(value) is int:
        try:
            number = float(value)
            exact = math.isfinite(number) and int(number) == value
        except OverflowError:
            exact = False
        if not exact:
            raise ScriptSandboxError(
                "An input integer cannot be represented exactly by JavaScript Number; "
                "use a decimal string and BigInt(value).toString() for wide integers"
            )
    elif isinstance(value, dict):
        for child in value.values():
            _check_javascript_integers(child)
    elif isinstance(value, list):
        for child in value:
            _check_javascript_integers(child)


def execute_script(code: str, inputs: dict[str, Any], params: dict[str, Any], *,
                   timeout_ms: int = 1000, memory_mb: int = 32,
                   cancel: threading.Event | None = None) -> dict[str, Any]:
    """Evaluate a synchronous JavaScript function body in a disposable worker.

    The result is a strict, finite JSON object of named outputs.  Each invocation
    starts fresh.  ``timeout_ms`` bounds V8 execution; the parent additionally
    enforces a wall deadline including startup and always kills/reaps cancellation.
    """
    code = _code(code, "Script")
    if type(inputs) is not dict or type(params) is not dict:
        raise ScriptSandboxError("Script inputs and params must be JSON objects")
    if type(timeout_ms) is not int or not 10 <= timeout_ms <= 10000:
        raise ScriptSandboxError("Script timeout_ms must be an integer between 10 and 10000")
    if type(memory_mb) is not int or not 16 <= memory_mb <= 64:
        raise ScriptSandboxError("Script memory_mb must be an integer between 16 and 64")
    _json_bytes({"inputs": inputs, "params": params}, MAX_JSON_BYTES, "Script inputs and params")
    _check_javascript_integers(inputs)
    _check_javascript_integers(params)
    payload = _json_bytes({"code": code, "inputs": inputs, "params": params,
                           "timeout_ms": timeout_ms, "memory_mb": memory_mb},
                          MAX_WORKER_REQUEST, "Worker request")
    if cancel is not None and cancel.is_set():
        raise ScriptSandboxError("Script cancelled")
    worker = Path(__file__).resolve().with_name("studio_script_worker.py")
    runtime_spec = importlib.util.find_spec("py_mini_racer")
    if runtime_spec is None or runtime_spec.origin is None:
        raise ScriptSandboxError("The mini-racer dependency is required for isolated scripts")
    runtime_directory = Path(runtime_spec.origin).resolve().parent
    # Windows virtual-environment python.exe is a launcher which starts another
    # process.  Launch the real CPython executable so the resource job owns the
    # actual worker and one-process containment never leaves a child behind.
    executable = str(getattr(sys, "_base_executable", sys.executable))
    start = time.monotonic()
    startup_deadline = start + WORKER_STARTUP_GRACE_MS / 1000
    job: _WindowsJob | None = None
    process: subprocess.Popen[bytes] | None = None
    io_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    stderr_pipe: Any = None
    stderr_tail = bytearray()
    stderr_lock = threading.Lock()
    io_done, ready = threading.Event(), threading.Event()
    exchange: dict[str, Any] = {}

    def worker_diagnostic() -> str:
        # Runtime loader diagnostics must not fill a pipe before the READY
        # handshake, or grow parent memory without bound.  Keep an 8 KiB tail
        # internally and include at most 1,000 bytes in an error message.
        with stderr_lock:
            return bytes(stderr_tail[-1000:]).decode("utf-8", errors="replace").strip()

    with tempfile.TemporaryDirectory(prefix="spobs-script-") as directory:
        env = {key: value for key in ("SystemRoot", "WINDIR", "SYSTEMDRIVE")
               if (value := os.environ.get(key)) is not None}
        env.update({"TEMP": directory, "TMP": directory, "TMPDIR": directory,
                    "PATH": str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32")
                    if os.name == "nt" else "/usr/bin:/bin"})
        try:
            process = subprocess.Popen(
                [executable, "-I", "-S", "-B", str(worker), str(runtime_directory)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory, env=env,
                close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            stderr_pipe = process.stderr
            # communicate() must never become a second reader of this pipe.
            # The dedicated reader starts before Job setup and before READY,
            # when native runtime import/initialization can emit diagnostics.
            process.stderr = None

            def drain_stderr() -> None:
                try:
                    assert stderr_pipe is not None
                    while chunk := stderr_pipe.read1(4096):
                        with stderr_lock:
                            stderr_tail.extend(chunk)
                            del stderr_tail[:-8192]
                except OSError as exc:
                    with stderr_lock:
                        stderr_tail.extend(f"\n[stderr read failed: {exc}]".encode("utf-8", errors="replace"))
                        del stderr_tail[:-8192]
                finally:
                    if stderr_pipe is not None:
                        stderr_pipe.close()

            stderr_thread = threading.Thread(target=drain_stderr, name="spobs-script-stderr", daemon=True)
            stderr_thread.start()
            job = _WindowsJob(process)
            owned_process = process

            def exchange_request() -> None:
                """Own every pipe operation; a blocked stdin write cannot block cancellation."""
                try:
                    assert owned_process.stdout is not None
                    for _ in range(8):
                        ready_line = owned_process.stdout.readline(8193)
                        if len(ready_line) > 8192:
                            raise ScriptSandboxError("Script worker returned an oversized readiness message")
                        try:
                            handshake = json.loads(ready_line)
                        except (ValueError, UnicodeError) as exc:
                            raise ScriptSandboxError("Script worker stopped before its runtime became ready") from exc
                        if isinstance(handshake, dict) and handshake.get("phase") in (
                            "worker bootstrap", "runtime import", "V8 initialization"
                        ):
                            exchange["phase"] = handshake["phase"]
                            exchange.setdefault("startup_phases_ms", {})[handshake["phase"]] = round(
                                (time.monotonic() - start) * 1000, 3
                            )
                            continue
                        break
                    else:
                        raise ScriptSandboxError("Script worker sent too many readiness messages")
                    if not isinstance(handshake, dict) or handshake.get("ready") is not True:
                        exchange["early_response"] = ready_line
                        output, _stderr = owned_process.communicate()
                        exchange.update(output=ready_line + output)
                        return
                    exchange["ready_at"] = time.monotonic()
                    exchange["initialization_ms"] = handshake.get("initialization_ms")
                    ready.set()
                    # Windows communicate writes stdin synchronously.  Keeping
                    # this entire operation in one owned thread lets the main
                    # thread enforce deadlines even when the pipe is full.
                    output, _stderr = owned_process.communicate(input=payload)
                    exchange.update(output=output)
                except BaseException as exc:
                    exchange["exception"] = exc
                finally:
                    io_done.set()

            io_thread = threading.Thread(target=exchange_request, name="spobs-script-io", daemon=True)
            io_thread.start()
            while not io_done.wait(0.05):
                if cancel is not None and cancel.is_set():
                    raise ScriptSandboxError("Script cancelled")
                if not ready.is_set() and time.monotonic() >= startup_deadline:
                    raise ScriptSandboxError(
                        f"Script runtime initialization exceeded its {WORKER_STARTUP_GRACE_MS / 1000:g}-second startup deadline "
                        f"(last phase: {exchange.get('phase', 'interpreter startup')})"
                    )
                execution_deadline = exchange.get("ready_at", time.monotonic()) + (timeout_ms + WORKER_EXECUTION_GRACE_MS) / 1000
                if ready.is_set() and time.monotonic() >= execution_deadline:
                    raise ScriptSandboxError("Script exceeded its execution/transfer wall-clock deadline")
            if "exception" in exchange:
                raise ScriptSandboxError(f"Script worker communication failed: {exchange['exception']}") from exchange["exception"]
            # communicate() has reaped the process, so the independent stderr
            # reader can finish without waiting for any user script activity.
            stderr_thread.join(timeout=5)
            output = exchange.get("output", b"").rstrip(b"\r\n")
            if process.returncode != 0:
                raise ScriptSandboxError("Script worker stopped (resource limit or runtime failure)")
            if len(output) > MAX_JSON_BYTES:
                raise ScriptSandboxError("Script output exceeds 1 MiB")
            try:
                response = json.loads(output, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite JSON")))
                _json_bytes(response, MAX_JSON_BYTES, "Script result")
            except (ValueError, UnicodeError, TypeError) as exc:
                raise ScriptSandboxError(f"Script worker returned invalid JSON: {exc}") from exc
            if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
                raise ScriptSandboxError("Script worker returned an invalid response")
            if not response["ok"]:
                raise ScriptSandboxError(str(response.get("error", "Script failed"))[:4000])
            result = response.get("result")
            if (not isinstance(result, dict) or not isinstance(result.get("outputs"), dict)
                    or not isinstance(result.get("logs"), list)
                    or not all(isinstance(line, str) for line in result["logs"])):
                raise ScriptSandboxError("Script worker returned invalid outputs")
            result["duration_ms"] = round((time.monotonic() - start) * 1000, 3)
            result["initialization_ms"] = exchange.get("initialization_ms")
            result["startup_phases_ms"] = exchange.get("startup_phases_ms", {})
            return result
        except ScriptSandboxError as exc:
            diagnostic = worker_diagnostic()
            if diagnostic:
                raise ScriptSandboxError(f"{exc}; worker stderr: {diagnostic}") from exc
            raise
        except OSError as exc:
            diagnostic = worker_diagnostic()
            raise ScriptSandboxError(f"Cannot start isolated script worker: {exc}"
                                     f"{'; worker stderr: ' + diagnostic if diagnostic else ''}") from exc
        finally:
            if job is not None:
                job.close()
            if process is not None:
                if process.poll() is None:
                    process.kill()
                if io_thread is None:
                    process.communicate()
                else:
                    io_thread.join(timeout=5)
                    if io_thread.is_alive():
                        raise ScriptSandboxError("Script worker IO did not stop after process termination")
                    process.wait(timeout=5)
                if stderr_thread is not None:
                    stderr_thread.join(timeout=5)
                    if stderr_thread.is_alive():
                        raise ScriptSandboxError("Script worker stderr reader did not stop after process termination")
                process.stderr = stderr_pipe
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()


__all__ = ["ScriptSandboxError", "validate_plugin", "node_definitions", "example_plugin", "sandbox_capabilities", "execute_script",
           "PORT_TYPES", "MAX_CODE_BYTES", "MAX_JSON_BYTES", "MAX_PLUGIN_BYTES",
           "MAX_WORKER_REQUEST", "WORKER_STARTUP_GRACE_MS", "WORKER_EXECUTION_GRACE_MS"]
