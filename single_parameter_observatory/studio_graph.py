"""Typed visual workflows, immutable queued executions, and portable extensions.

The graph is a declarative DAG. Saving/opening it never executes it. A queued
job owns a deep copy of its graph and enabled plugin manifests. Host mutations
are explicit nodes, run serially, and are never automatically retried. Completed
host effects remain visible when a later node fails or the user cancels a job.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import operator
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from . import studio_transformers

GRAPH_SCHEMA = "spobs-graphs/1"
PORT_TYPES = frozenset({"any", "number", "integer", "string", "boolean", "array", "object", "run", "config", "dataset", "tensor-expression"})
MAX_GRAPHS = 100
MAX_NODES = 500
MAX_EDGES = 2000
MAX_PLUGINS = 64
MAX_JOBS = 50
MAX_QUEUE = 16
MAX_JOB_BYTES = 8_000_000
MAX_HISTORY_BYTES = 24_000_000
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_GRAPH_REGISTRY_BYTES = 8_000_000
MAX_PLUGIN_REGISTRY_BYTES = 4_000_000
_UNSET = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


def _json(value: Any, *, limit: int = 2_000_000) -> Any:
    """Reject non-JSON, nonfinite values and oversized or excessively deep data."""
    def check(item: Any, depth: int = 0) -> None:
        if depth > 50:
            raise ValueError("JSON values may be nested at most 50 levels")
        if item is None or isinstance(item, str) or type(item) in (bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            return
        if isinstance(item, list):
            for child in item:
                check(child, depth + 1)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                check(child, depth + 1)
            return
        raise ValueError("values must be strict JSON with string object keys")
    check(value)
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("value cannot be encoded as strict JSON") from exc
    try:
        byte_length = len(encoded.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("JSON text must contain valid Unicode characters") from exc
    if byte_length > limit:
        raise ValueError(f"JSON value exceeds {limit:,} bytes")
    return json.loads(encoded)


def _text(value: Any, label: str, maximum: int = 300) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{label} must be text of at most {maximum} characters")
    return value


def _identity(value: Any, label: str = "id") -> str:
    text = _text(value, label)
    if text in {".", ".."}:
        raise ValueError(f"{label} is a reserved path segment")
    if not text or any(not (char.isascii() and (char.isalnum() or char in "_-.:")) for char in text):
        raise ValueError(f"{label} must contain ASCII letters, digits, underscores, dashes, dots or colons")
    return text


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def _number(value: Any, label: str) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _config(value: dict[str, Any]) -> Any:
    from .studio_session import config_from_dict  # noqa: PLC0415 -- the owning workspace imports this module.
    return config_from_dict(value)


def _port(name: str, kind: str = "any", *, required: bool = True, default: Any = _UNSET) -> dict[str, Any]:
    value: dict[str, Any] = {"name": name, "type": kind, "required": required}
    if default is not _UNSET:
        value["default"] = default
    return value


def _param(name: str, kind: str, default: Any, **extra: Any) -> dict[str, Any]:
    return {"name": name, "type": kind, "default": default, **extra}


def _definition(identity: str, title: str, category: str, description: str,
                inputs: list[dict[str, Any]], outputs: list[dict[str, Any]],
                parameters: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"id": identity, "title": title, "category": category, "description": description,
            "inputs": inputs, "outputs": outputs, "parameters": parameters or []}


_VALUE_OUT = [_port("value")]
BUILTINS = [
    _definition("core.number", "Number", "Values", "A finite numeric literal.", [], [_port("value", "number")],
                [_param("value", "number", 1.0)]),
    _definition("core.text", "Text", "Values", "A text literal.", [], [_port("value", "string")],
                [_param("value", "string", "Hello, studio")]),
    _definition("core.boolean", "Boolean", "Values", "A true or false literal.", [], [_port("value", "boolean")],
                [_param("value", "boolean", True)]),
    _definition("core.json", "JSON value", "Values", "Any strict JSON literal.", [], _VALUE_OUT,
                [_param("value", "any", {})]),
    _definition("core.math", "Math", "Transform", "Finite arithmetic; division by zero is an explicit failure.",
                [_port("a", "number", default=1), _port("b", "number", default=2)], [_port("value", "number")],
                [_param("operation", "string", "add", options=["add", "subtract", "multiply", "divide", "power", "min", "max", "modulo"])]),
    _definition("core.array", "Array", "Values", "An array literal or an incoming array.",
                [_port("items", "array", default=[])], [_port("value", "array")]),
    _definition("core.object", "Object", "Values", "An object literal or an incoming object.",
                [_port("value", "object", default={})], [_port("value", "object")]),
    _definition("core.get", "Get property", "Transform", "Read an object key or array index; use default when absent.",
                [_port("object"), _port("key", default="value")], _VALUE_OUT,
                [_param("default", "any", None)]),
    _definition("core.script", "JavaScript sandbox", "Scripts", "Run bounded JavaScript in an isolated V8 child process. Return an outputs object.",
                [_port("inputs", "object", default={})], [_port("value"), _port("outputs", "object")],
                [_param("code", "code", "return {value: inputs.value ?? 0};"),
                 _param("timeout_ms", "integer", 1000, min=10, max=10000),
                 _param("memory_mb", "integer", 32, min=16, max=64)]),
    _definition("core.preview", "Preview", "Observe", "Expose the exact incoming JSON value in execution results.",
                [_port("value")], _VALUE_OUT),
    _definition("core.config", "Experiment configuration", "Experiments", "Validate a partial config and produce a complete PyTorch experiment config.",
                [_port("value", "object", default={})], [_port("config", "config")]),
    _definition("core.dataset", "Custom dataset", "Experiments", "Build finite float32 input/target samples without executing training.",
                [_port("inputs", "array", default=[-1, 0, 1]), _port("targets", "array", default=[-2, 0, 2])],
                [_port("dataset", "dataset")], [_param("name", "string", "Node dataset"), _param("batch_size", "integer", 3, min=1, max=100000)]),
    _definition("core.experiment", "Create experiment", "Experiments", "Create an independent real PyTorch experiment; optionally import custom data.",
                [_port("config", "config", default={}), _port("dataset", "dataset", required=False)],
                [_port("run", "run")], [_param("name", "string", "Node experiment"), _param("preset", "string", "adamw")]),
    _definition("core.train", "Train", "Experiments", "Execute real training steps. Cancellation is checked between completed steps.",
                [_port("run", "run")], [_port("run", "run"), _port("record", "object")],
                [_param("steps", "integer", 10, min=1, max=10000)]),
    _definition("core.branch", "Branch experiment", "Experiments", "Branch exact live parameter, optimizer, scheduler, data and observation state.",
                [_port("run", "run")], [_port("run", "run")], [_param("name", "string", "Node branch")]),
    _definition("core.checkpoint", "Capture checkpoint", "Experiments", "Capture a complete real training checkpoint on the selected run.",
                [_port("run", "run")], [_port("run", "run"), _port("checkpoint", "string")],
                [_param("name", "string", "Node checkpoint")]),
    _definition("core.inspect", "Inspect experiment", "Observe", "Read an actual recorded step or the live run snapshot; optionally include data.",
                [_port("run", "run")], [_port("record", "object"), _port("dataset", "object"), _port("run", "run")],
                [_param("step", "integer", -1, min=-1, max=1000000), _param("include_dataset", "boolean", False)]),
    _definition("core.intervene", "Manual intervention", "Experiments", "Explicitly edit finite parameter/LR or supported optimizer memory on a real paused run.",
                [_port("run", "run"), _port("changes", "object", default={"parameter": 1.0})],
                [_port("run", "run"), _port("record", "object")]),
    _definition("core.verify", "Verify record chain", "Observe", "Verify the actual immutable numerical record hash chain.",
                [_port("run", "run")], [_port("run", "run"), _port("integrity", "object")]),
    _definition("core.control", "Experiment control", "Experiments", "Explicitly play, pause, reset live state, or restart an existing experiment.",
                [_port("run", "run")], [_port("run", "run"), _port("record", "object")],
                [_param("operation", "string", "pause", options=["play", "pause", "reset", "restart"])]),
    _definition("core.export", "Export evidence", "Artifacts", "Generate and retain an immutable ZIP of actual numerical evidence, bound to this execution.",
                [_port("run", "run")], [_port("run", "run"), _port("artifact", "object")]),
]
BUILTINS.extend(studio_transformers.definitions())


def _node(identity: str, kind: str, x: int, y: int, **params: Any) -> dict[str, Any]:
    return {"id": identity, "type": kind, "position": {"x": x, "y": y}, "params": params}


def _edge(source: str, source_port: str, target: str, target_port: str) -> dict[str, Any]:
    return {"id": f"{source}:{source_port}-{target}:{target_port}", "source": source, "source_port": source_port,
            "target": target, "target_port": target_port}


def _templates() -> list[dict[str, Any]]:
    training = {"name": "Training laboratory", "description": "Configuration → real experiment → training → checkpoint → observation.",
                "nodes": [_node("config", "core.config", 40, 80, value={"steps": 100}),
                          _node("experiment", "core.experiment", 370, 80), _node("train", "core.train", 700, 80, steps=10),
                          _node("checkpoint", "core.checkpoint", 1030, 80), _node("inspect", "core.inspect", 1360, 80)],
                "edges": [_edge("config", "config", "experiment", "config"), _edge("experiment", "run", "train", "run"),
                          _edge("train", "run", "checkpoint", "run"), _edge("checkpoint", "run", "inspect", "run")],
                "view": {"x": 0, "y": 0, "zoom": 0.8}}
    script = {"name": "Sandbox playground", "description": "A pure JavaScript script transforms JSON inside the bounded sandbox.",
              "nodes": [_node("input", "core.object", 40, 100, value={"value": 21}),
                        _node("script", "core.script", 370, 100, code="return {value: inputs.value * 2};"),
                        _node("preview", "core.preview", 700, 100)],
              "edges": [_edge("input", "value", "script", "inputs"), _edge("script", "value", "preview", "value")],
              "view": {"x": 0, "y": 0, "zoom": 1}}
    data = {"name": "Custom-data experiment", "description": "A typed dataset and config feed actual PyTorch training.",
            "nodes": [_node("dataset", "core.dataset", 40, 240), _node("config", "core.config", 40, 40),
                      _node("experiment", "core.experiment", 370, 120), _node("train", "core.train", 700, 120, steps=20),
                      _node("inspect", "core.inspect", 1030, 120, include_dataset=True)],
            "edges": [_edge("config", "config", "experiment", "config"), _edge("dataset", "dataset", "experiment", "dataset"),
                      _edge("experiment", "run", "train", "run"), _edge("train", "run", "inspect", "run")],
            "view": {"x": 0, "y": 0, "zoom": 0.9}}
    return [{"id": "training", "name": training["name"], "graph": training},
            {"id": "sandbox", "name": script["name"], "graph": script},
            {"id": "custom-data", "name": data["name"], "graph": data}, *studio_transformers.templates()]


def _compatible(source: str, target: str) -> bool:
    return target == "any" or source == "any" or source == target or (
        source == "integer" and target == "number"
    ) or (source in ("config", "dataset") and target == "object") or (
        source == "object" and target in ("config", "dataset")
    ) or (source == "string" and target == "run")


def _typed(value: Any, kind: str, label: str) -> None:
    if kind == "any":
        return
    ok = {
        "number": type(value) in (int, float), "integer": type(value) is int,
        "string": isinstance(value, str), "boolean": type(value) is bool,
        "array": isinstance(value, list), "object": isinstance(value, dict),
        "run": isinstance(value, str) and bool(value), "config": isinstance(value, dict),
        "dataset": isinstance(value, dict), "tensor-expression": isinstance(value, dict) and value.get("kind") == "tensor-expression",
    }.get(kind, False)
    if not ok:
        raise ValueError(f"{label} requires type {kind}")
    if kind in ("number", "integer"):
        _number(value, label)
    if kind == "config":
        _config(value)
    if kind == "dataset":
        _dataset(value)


def _dataset(value: dict[str, Any]) -> None:
    inputs, targets = value.get("inputs"), value.get("targets")
    if not isinstance(inputs, list) or not isinstance(targets, list) or len(inputs) != len(targets):
        raise ValueError("dataset requires equal-length inputs and targets arrays")
    if not 1 <= len(inputs) <= 100000:
        raise ValueError("dataset requires 1 to 100,000 samples")
    for item in inputs + targets:
        number = _number(item, "dataset sample")
        if abs(number) > 3.4028234663852886e38:
            raise ValueError("dataset samples must fit finite float32 storage")
    _integer(value.get("batch_size", min(32, len(inputs))), "batch_size", 1, len(inputs))


class _Cancelled(Exception):
    pass


class GraphConflictError(RuntimeError):
    """A stale update or active execution conflicts with the requested operation."""


class GraphStudio:
    """Thread-safe graph registry and one serial, lazily started job worker."""

    def __init__(self, workspace: Any, on_change: Callable[[], None] | None = None) -> None:
        self.workspace = workspace
        self._on_change = on_change
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._graphs: dict[str, dict[str, Any]] = {}
        self._plugins: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._queue: deque[str] = deque()
        self._worker: threading.Thread | None = None
        self._closed = False
        self._revision = 0
        self._active_graph_id: str | None = None
        self._idempotency: dict[str, str] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._context = threading.local()
        starter = self._graph({**_templates()[0]["graph"], "id": "starter-training", "revision": 1})
        self._graphs[starter["id"]] = starter
        self._active_graph_id = starter["id"]

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _ensure_open(self) -> None:
        if self._closed:
            raise GraphConflictError("node studio is closed")

    @staticmethod
    def _plugin_module() -> Any:
        from . import studio_plugins  # noqa: PLC0415 -- optional sandbox dependency is loaded on demand.
        return studio_plugins

    def _definitions(self, plugins: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        definitions = {item["id"]: copy.deepcopy(item) for item in BUILTINS}
        for plugin in (self._plugins if plugins is None else plugins).values():
            if not plugin["enabled"]:
                continue
            manifest = plugin["manifest"]
            for node in manifest["nodes"]:
                definition = copy.deepcopy(node)
                definition["id"] = f"{manifest['id']}/{node['id']}"
                definition.setdefault("title", node["id"])
                definition.setdefault("description", manifest.get("description", ""))
                definition.setdefault("category", f"Plugins · {manifest.get('name', manifest['id'])}")
                definition.setdefault("inputs", [])
                definition.setdefault("outputs", [])
                definition.setdefault("parameters", [])
                definition["plugin_id"] = manifest["id"]
                definition["plugin_version"] = manifest["version"]
                definitions[definition["id"]] = definition
        return definitions

    def catalog(self) -> dict[str, Any]:
        with self._lock:
            definitions = self._definitions()
        for definition in definitions.values():
            definition.pop("code", None)
        templates = _templates()
        with self._lock:
            for plugin in self._plugins.values():
                if plugin["enabled"]:
                    for template in plugin["manifest"].get("templates", []):
                        templates.append({**copy.deepcopy(template), "id": f"{plugin['id']}/{template['id']}"})
        return {"types": list(definitions.values()), "templates": templates, "transformers": studio_transformers.capabilities(),
                "sandbox": {**self._plugin_module().sandbox_capabilities(), "language": "javascript", "host_api": False,
                            "description": "Scripts return JSON outputs. Real experiment actions use explicit built-in nodes."},
                "limits": {"graphs": MAX_GRAPHS, "nodes_per_graph": MAX_NODES, "edges_per_graph": MAX_EDGES,
                           "plugins": MAX_PLUGINS, "queued_jobs": MAX_QUEUE, "job_history": MAX_JOBS,
                           "job_bytes": MAX_JOB_BYTES, "job_history_bytes": MAX_HISTORY_BYTES,
                           "artifact_bytes": MAX_ARTIFACT_BYTES, "graph_registry_bytes": MAX_GRAPH_REGISTRY_BYTES,
                           "plugin_registry_bytes": MAX_PLUGIN_REGISTRY_BYTES, "graph_bytes": 2_000_000}}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy({"revision": self._revision, "workspace_epoch": getattr(self.workspace, "workspace_epoch", None),
                                  "graphs": list(self._graphs.values()),
                                  "active_graph_id": self._active_graph_id, "plugins": list(self._plugins.values()),
                                  "jobs": list(self._jobs.values()),
                                  "artifacts": [self._artifact_descriptor(item) for item in self._artifacts.values()],
                                  "queue_length": sum(job["status"] == "queued" for job in self._jobs.values())})

    @staticmethod
    def _graph(value: Any) -> dict[str, Any]:
        graph = _json(value)
        if not isinstance(graph, dict):
            raise ValueError("graph must be an object")
        unknown = set(graph) - {"id", "name", "description", "nodes", "edges", "view", "revision", "created_at", "updated_at"}
        if unknown:
            raise ValueError(f"unknown graph fields: {', '.join(sorted(unknown))}")
        graph["id"] = _identity(graph.get("id", _id()), "graph id")
        graph["name"] = _text(graph.get("name", "Untitled workflow"), "graph name")
        graph["description"] = _text(graph.get("description", ""), "graph description", 100000)
        for name, maximum in (("nodes", MAX_NODES), ("edges", MAX_EDGES)):
            if not isinstance(graph.get(name, []), list) or len(graph.get(name, [])) > maximum:
                raise ValueError(f"graph {name} must be an array with at most {maximum} entries")
            graph.setdefault(name, [])
        node_ids: set[str] = set()
        for node in graph["nodes"]:
            if not isinstance(node, dict) or set(node) - {"id", "type", "title", "position", "params"}:
                raise ValueError("node must contain id, type, position, params and optional title")
            identity = _identity(node.get("id"), "node id")
            if identity in node_ids:
                raise ValueError(f"duplicate node id: {identity}")
            node_ids.add(identity)
            _text(node.get("type"), "node type")
            if "title" in node:
                _text(node["title"], "node title")
            position = node.setdefault("position", {"x": 0, "y": 0})
            if not isinstance(position, dict) or set(position) != {"x", "y"}:
                raise ValueError("node position requires x and y")
            for key, item in position.items():
                if abs(_number(item, f"node position {key}")) > 10_000_000:
                    raise ValueError("node coordinates must stay within ±10,000,000 canvas units")
            if not isinstance(node.setdefault("params", {}), dict):
                raise ValueError("node params must be an object")
        edge_ids: set[str] = set()
        occupied: set[tuple[str, str]] = set()
        for edge in graph["edges"]:
            if not isinstance(edge, dict) or set(edge) != {"id", "source", "source_port", "target", "target_port"}:
                raise ValueError("edges require id, source, source_port, target and target_port")
            identity = _identity(edge["id"], "edge id")
            if identity in edge_ids:
                raise ValueError(f"duplicate edge id: {identity}")
            edge_ids.add(identity)
            _identity(edge["source"], "edge source")
            _identity(edge["target"], "edge target")
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                raise ValueError("edge refers to a missing node")
            for key in ("source_port", "target_port"):
                _identity(edge[key], key)
            target = (edge["target"], edge["target_port"])
            if target in occupied:
                raise ValueError("an input port may have only one incoming edge")
            occupied.add(target)
        view = graph.setdefault("view", {"x": 0, "y": 0, "zoom": 1})
        if not isinstance(view, dict) or set(view) != {"x", "y", "zoom"}:
            raise ValueError("graph view requires x, y and zoom")
        for key, item in view.items():
            if abs(_number(item, f"view {key}")) > 1_000_000_000:
                raise ValueError("view coordinates must stay within ±1,000,000,000 canvas units")
        if not 0.04 <= view["zoom"] <= 4:
            raise ValueError("graph zoom must be from 0.04 to 4")
        graph["revision"] = _integer(graph.get("revision", 0), "graph revision", 0, 10**12)
        for key in ("created_at", "updated_at"):
            if key in graph:
                _text(graph[key], key)
            else:
                graph[key] = _now()
        return graph

    @staticmethod
    def _validate(graph: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        nodes = {node["id"]: node for node in graph["nodes"]}
        incoming = {(edge["target"], edge["target_port"]): edge for edge in graph["edges"]}
        dependencies: dict[str, set[str]] = {key: set() for key in nodes}
        for node in nodes.values():
            definition = definitions.get(node["type"])
            if definition is None:
                errors.append({"node_id": node["id"], "message": f"Unknown or disabled node type: {node['type']}"})
                continue
            input_defs = {item["name"]: item for item in definition["inputs"]}
            parameters = {item["name"]: item for item in definition["parameters"]}
            for name in set(node["params"]) - set(input_defs) - set(parameters):
                errors.append({"node_id": node["id"], "message": f"Unknown parameter: {name}"})
            for name, item in {**input_defs, **parameters}.items():
                if (node["id"], name) in incoming:
                    continue
                value = node["params"].get(name, item.get("default", _UNSET))
                if value is _UNSET:
                    if item.get("required", name in parameters):
                        errors.append({"node_id": node["id"], "port": name, "message": f"Missing required input: {name}"})
                    continue
                try:
                    kind = item["type"]
                    _typed(value, "string" if kind == "code" else kind, f"{node['id']}.{name}")
                    if "options" in item and value not in item["options"]:
                        raise ValueError(f"{name} must be one of {item['options']}")
                    if "min" in item and value < item["min"]:
                        raise ValueError(f"{name} must be >= {item['min']}")
                    if "max" in item and value > item["max"]:
                        raise ValueError(f"{name} must be <= {item['max']}")
                except (ValueError, TypeError) as exc:
                    errors.append({"node_id": node["id"], "port": name, "message": str(exc)})
        for edge in graph["edges"]:
            source_def, target_def = definitions.get(nodes[edge["source"]]["type"]), definitions.get(nodes[edge["target"]]["type"])
            dependencies[edge["target"]].add(edge["source"])
            if source_def is None or target_def is None:
                continue
            outputs = {item["name"]: item for item in source_def["outputs"]}
            inputs = {item["name"]: item for item in target_def["inputs"]}
            if edge["source_port"] not in outputs or edge["target_port"] not in inputs:
                errors.append({"edge_id": edge["id"], "message": "Edge refers to an unknown input or output port"})
            elif not _compatible(outputs[edge["source_port"]]["type"], inputs[edge["target_port"]]["type"]):
                errors.append({"edge_id": edge["id"], "message": "Incompatible port types"})
        ready = [key for key in nodes if not dependencies[key]]
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for key in nodes:
                if current in dependencies[key]:
                    dependencies[key].remove(current)
                    if not dependencies[key] and key not in order and key not in ready:
                        ready.append(key)
        if len(order) != len(nodes):
            errors.append({"message": "Workflow contains a cycle; execution requires a directed acyclic graph"})
        return {"valid": not errors, "errors": errors, "order": order, "node_count": len(nodes), "edge_count": len(graph["edges"])}

    def validate(self, graph: dict[str, Any]) -> dict[str, Any]:
        value = self._graph(graph)
        with self._lock:
            return self._validate(value, self._definitions())

    def is_busy(self) -> bool:
        with self._lock:
            return any(job["status"] in ("queued", "running", "cancelling") for job in self._jobs.values())

    def _get_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        identity = payload.get("graph_id", self._active_graph_id)
        if not isinstance(identity, str) or identity not in self._graphs:
            raise KeyError("workflow does not exist")
        return self._graphs[identity]

    def _get_plugin(self, payload: dict[str, Any]) -> dict[str, Any]:
        identity = payload.get("plugin_id")
        if not isinstance(identity, str) or identity not in self._plugins:
            raise KeyError("plugin does not exist")
        return self._plugins[identity]

    def dispatch(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply one registry action or enqueue an immutable workflow snapshot."""
        if not isinstance(payload, dict):
            raise ValueError("node action payload must be an object")
        with self._condition:
            self._ensure_open()
            result: dict[str, Any] = {"ok": True, "action": action}
            if action == "graph.create":
                if len(self._graphs) >= MAX_GRAPHS:
                    raise ValueError(f"the studio supports at most {MAX_GRAPHS} workflows")
                source = payload.get("graph")
                if source is None and payload.get("template_id"):
                    template = next((item for item in self.catalog()["templates"] if item["id"] == payload["template_id"]), None)
                    if template is None:
                        raise ValueError("unknown workflow template")
                    source = template["graph"]
                graph = self._graph(source if source is not None else {"name": payload.get("name", "Untitled workflow")})
                if "name" in payload:
                    graph["name"] = _text(payload["name"], "workflow name")
                graph.update(id=_id(), revision=1, created_at=_now(), updated_at=_now())
                _json([*self._graphs.values(), graph], limit=MAX_GRAPH_REGISTRY_BYTES)
                self._graphs[graph["id"]] = graph
                self._active_graph_id = graph["id"]
                result["graph_id"] = graph["id"]
            elif action == "graph.update":
                candidate = self._graph(payload.get("graph"))
                graph = self._get_graph({"graph_id": candidate["id"]})
                expected = payload.get("expected_revision", candidate["revision"])
                _integer(expected, "expected_revision", 0, 10**12)
                if expected != graph["revision"]:
                    raise GraphConflictError("workflow changed since this editor loaded it; reload before saving")
                candidate.update(revision=graph["revision"] + 1, created_at=graph["created_at"], updated_at=_now())
                _json([candidate if item["id"] == candidate["id"] else item for item in self._graphs.values()], limit=MAX_GRAPH_REGISTRY_BYTES)
                self._graphs[candidate["id"]] = candidate
                result["graph_id"] = candidate["id"]
                result["revision"] = candidate["revision"]
            elif action == "graph.select":
                graph = self._get_graph(payload)
                self._active_graph_id = graph["id"]
                result["graph_id"] = graph["id"]
            elif action == "graph.delete":
                graph = self._get_graph(payload)
                del self._graphs[graph["id"]]
                if self._active_graph_id == graph["id"]:
                    self._active_graph_id = next(iter(self._graphs), None)
                result["graph_id"] = graph["id"]
            elif action == "graph.duplicate":
                if len(self._graphs) >= MAX_GRAPHS:
                    raise ValueError(f"the studio supports at most {MAX_GRAPHS} workflows")
                graph = copy.deepcopy(self._get_graph(payload))
                graph.update(id=_id(), name=_text(payload.get("name", f"{graph['name']} · copy"), "workflow name"),
                             revision=1, created_at=_now(), updated_at=_now())
                _json([*self._graphs.values(), graph], limit=MAX_GRAPH_REGISTRY_BYTES)
                self._graphs[graph["id"]] = graph
                self._active_graph_id = graph["id"]
                result["graph_id"] = graph["id"]
            elif action == "graph.validate":
                graph = self._graph(payload["graph"]) if "graph" in payload else self._get_graph(payload)
                result.update(self._validate(graph, self._definitions()))
                return copy.deepcopy(result)
            elif action == "graph.execute":
                result.update(self._enqueue(payload))
            elif action == "job.cancel":
                job_id = payload.get("job_id")
                if not isinstance(job_id, str) or job_id not in self._jobs:
                    raise KeyError("job does not exist")
                job = self._jobs[job_id]
                if job["status"] in ("queued", "running", "cancelling"):
                    self._cancels[job_id].set()
                    if job["status"] == "queued":
                        job["status"] = "cancelled"
                        job["finished_at"] = _now()
                        for record in job["nodes"].values():
                            record["status"] = "skipped"
                        self._event(job, "cancelled", "Cancelled before execution")
                    else:
                        job["status"] = "cancelling"
                        self._event(job, "cancelling", "Cancellation requested; completed host effects remain recorded")
                result["job_id"] = job_id
            elif action in ("job.delete", "job.clear"):
                targets = [str(payload.get("job_id", ""))] if action == "job.delete" else list(self._jobs)
                if action == "job.delete" and targets[0] not in self._jobs:
                    raise KeyError("job does not exist")
                for identity in targets:
                    if self._jobs[identity]["status"] in ("queued", "running", "cancelling"):
                        if action == "job.delete":
                            raise GraphConflictError("cancel and wait for the active job before deleting it")
                        continue
                    self._jobs.pop(identity)
                    self._cancels.pop(identity, None)
                    self._prune_artifacts()
                self._idempotency = {key: value for key, value in self._idempotency.items() if value in self._jobs}
            elif action == "plugin.install":
                if len(self._plugins) >= MAX_PLUGINS:
                    raise ValueError(f"the studio supports at most {MAX_PLUGINS} plugins")
                manifest = self._plugin_module().validate_plugin(payload.get("manifest"))
                if manifest["id"] in self._plugins:
                    raise GraphConflictError("plugin is already installed; use update to replace its version")
                plugin = {"id": manifest["id"], "enabled": True, "manifest": manifest, "installed_at": _now(), "updated_at": _now()}
                staged = {**self._plugins, manifest["id"]: plugin}
                _json(list(staged.values()), limit=MAX_PLUGIN_REGISTRY_BYTES)
                self._validate_plugin_templates(manifest, staged)
                self._plugins[manifest["id"]] = plugin
                result["plugin_id"] = manifest["id"]
            elif action == "plugin.update":
                plugin = self._get_plugin(payload)
                manifest = self._plugin_module().validate_plugin(payload.get("manifest"))
                if manifest["id"] != plugin["id"]:
                    raise ValueError("updated plugin must keep the installed plugin id")
                if self._manifest_hash(manifest) == self._manifest_hash(plugin["manifest"]):
                    result["plugin_id"] = plugin["id"]
                    return result
                if manifest["version"] == plugin["manifest"]["version"]:
                    raise GraphConflictError("changed plugin content requires a new version")
                candidate = {**plugin, "manifest": manifest, "updated_at": _now()}
                _json([candidate if item["id"] == candidate["id"] else item for item in self._plugins.values()], limit=MAX_PLUGIN_REGISTRY_BYTES)
                self._validate_plugin_templates(manifest, {**self._plugins, plugin["id"]: {**candidate, "enabled": True}})
                self._plugins[plugin["id"]] = candidate
                result["plugin_id"] = plugin["id"]
            elif action in ("plugin.enable", "plugin.disable", "plugin.remove"):
                plugin = self._get_plugin(payload)
                if action == "plugin.remove":
                    del self._plugins[plugin["id"]]
                else:
                    plugin["enabled"] = action == "plugin.enable"
                    plugin["updated_at"] = _now()
                result["plugin_id"] = plugin["id"]
            else:
                raise ValueError(f"unknown node studio action: {action}")
            self._revision += 1
            self._condition.notify_all()
        self._changed()
        return copy.deepcopy(result)

    @staticmethod
    def _manifest_hash(manifest: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def _validate_plugin_templates(self, manifest: dict[str, Any], plugins: dict[str, dict[str, Any]]) -> None:
        definitions = self._definitions(plugins)
        for template in manifest.get("templates", []):
            graph = self._graph(template["graph"])
            validation = self._validate(graph, definitions)
            if not validation["valid"]:
                raise ValueError(f"plugin template {template['id']} is invalid: {validation['errors'][0]['message']}")

    def _enqueue(self, payload: dict[str, Any]) -> dict[str, Any]:
        graph = self._get_graph(payload)
        if "expected_revision" in payload:
            _integer(payload["expected_revision"], "expected_revision", 0, 10**12)
            if payload["expected_revision"] != graph["revision"]:
                raise GraphConflictError("workflow changed before execution; save or reload the current revision")
        key = payload.get("idempotency_key")
        if key is not None:
            key = _text(key, "idempotency_key", 300)
            if not key:
                raise ValueError("idempotency_key must not be empty")
            existing_id = self._idempotency.get(key)
            if existing_id is not None:
                previous = self._jobs[existing_id]
                if previous["graph_id"] != graph["id"] or previous["graph_revision"] != graph["revision"]:
                    raise GraphConflictError("idempotency_key is already bound to a different workflow revision")
                return {"job_id": existing_id, "replayed": True}
        if sum(job["status"] in ("queued", "running", "cancelling") for job in self._jobs.values()) >= MAX_QUEUE:
            raise GraphConflictError(f"at most {MAX_QUEUE} jobs may be running or queued")
        definitions = self._definitions()
        validation = self._validate(graph, definitions)
        if not validation["valid"]:
            raise ValueError(f"workflow is invalid: {validation['errors'][0]['message']}")
        if not graph["nodes"]:
            raise ValueError("add at least one node before executing a workflow")
        identity = _id()
        used_types = {node["type"] for node in graph["nodes"]}
        pinned = {key: copy.deepcopy(value) for key, value in definitions.items() if key in used_types}
        job = {"id": identity, "graph_id": graph["id"], "graph_revision": graph["revision"],
               "graph_name": graph["name"], "status": "queued", "created_at": _now(), "started_at": None,
               "finished_at": None, "graph": copy.deepcopy(graph), "definitions": pinned,
               "plugin_versions": {plugin["id"]: {"version": plugin["manifest"]["version"], "sha256": self._manifest_hash(plugin["manifest"])}
                                   for plugin in self._plugins.values() if plugin["enabled"] and any(
                                       kind.startswith(plugin["id"] + "/") for kind in used_types)},
               "order": validation["order"], "nodes": {node["id"]: {"status": "pending", "inputs": None, "outputs": None,
                                                                      "logs": [], "error": None, "duration_ms": None}
                                                      for node in graph["nodes"]},
               "events": [], "error": None, "idempotency_key": key,
               "partial_effects": "Completed experiment actions are retained if a later node fails or execution is cancelled."}
        job["snapshot_sha256"] = self._job_snapshot_hash(job)
        self._event(job, "queued", "Captured an immutable workflow and node-definition snapshot")
        _json(job, limit=MAX_JOB_BYTES - 8192)
        self._trim_jobs(extra_bytes=self._size(job) + 8192, reserve_slot=True)
        if sum(self._size(item) for item in self._jobs.values()) + self._size(job) > MAX_HISTORY_BYTES - 8192:
            raise GraphConflictError("the queued job snapshot budget is full; wait for existing jobs or clear completed history")
        self._jobs[identity] = job
        self._cancels[identity] = threading.Event()
        self._queue.append(identity)
        if key is not None:
            self._idempotency[key] = identity
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="spobs-node-worker", daemon=True)
            self._worker.start()
        return {"job_id": identity, "replayed": False}

    @staticmethod
    def _size(value: Any) -> int:
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))

    def _over_observation_budget(self, job: dict[str, Any]) -> bool:
        return self._size(job) > MAX_JOB_BYTES - 8192 or sum(self._size(item) for item in self._jobs.values()) > MAX_HISTORY_BYTES - 8192

    def _trim_jobs(self, extra_bytes: int = 0, *, reserve_slot: bool = False) -> None:
        total_bytes = sum(self._size(item) for item in self._jobs.values())
        target_count = MAX_JOBS - (1 if reserve_slot else 0)
        for identity in list(self._jobs):
            if len(self._jobs) <= target_count and total_bytes + extra_bytes <= MAX_HISTORY_BYTES:
                break
            if self._jobs[identity]["status"] not in ("queued", "running", "cancelling"):
                total_bytes -= self._size(self._jobs[identity])
                self._jobs.pop(identity)
                self._cancels.pop(identity, None)
        self._idempotency = {key: value for key, value in self._idempotency.items() if value in self._jobs}
        self._prune_artifacts()

    @staticmethod
    def _event(job: dict[str, Any], kind: str, message: str, node_id: str | None = None) -> None:
        job["events"].append({"time": _now(), "type": kind, "message": message, "node_id": node_id})
        if len(job["events"]) > 1100:
            job["events"] = job["events"][-1100:]

    @staticmethod
    def _job_snapshot_hash(job: dict[str, Any]) -> str:
        captured = {key: job[key] for key in ("graph", "definitions", "plugin_versions", "order")}
        return hashlib.sha256(json.dumps(captured, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def _work(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._queue))
                if self._closed:
                    return
                identity = self._queue.popleft()
                job = self._jobs.get(identity)
                if job is None or job["status"] == "cancelled":
                    continue
                job["status"] = "running"
                job["started_at"] = _now()
                self._event(job, "started", "Workflow execution started")
                self._revision += 1
            self._changed()
            self._run_job(identity)

    @staticmethod
    def _check_cancel(cancel: threading.Event) -> None:
        if cancel.is_set():
            raise _Cancelled("Execution cancelled; previously completed host actions are retained")

    def _run_job(self, identity: str) -> None:
        with self._lock:
            job = self._jobs[identity]
            graph = copy.deepcopy(job["graph"])
            definitions = copy.deepcopy(job["definitions"])
            order = list(job["order"])
            cancel = self._cancels[identity]
        nodes = {node["id"]: node for node in graph["nodes"]}
        incoming = {(edge["target"], edge["target_port"]): edge for edge in graph["edges"]}
        values: dict[str, dict[str, Any]] = {}
        current: str | None = None
        self._context.transformers_worker = None
        try:
            for current in order:
                self._check_cancel(cancel)
                node, definition = nodes[current], definitions[nodes[current]["type"]]
                inputs: dict[str, Any] = {}
                params = {item["name"]: copy.deepcopy(item["default"]) for item in definition["parameters"] if "default" in item}
                parameter_names = {item["name"] for item in definition["parameters"]}
                params.update({name: copy.deepcopy(value) for name, value in node["params"].items() if name in parameter_names})
                for port in definition["inputs"]:
                    edge = incoming.get((current, port["name"]))
                    if edge is not None:
                        output = values[edge["source"]]
                        if edge["source_port"] not in output:
                            raise ValueError(f"upstream node did not produce port {edge['source_port']}")
                        value = copy.deepcopy(output[edge["source_port"]])
                    elif port["name"] in node["params"]:
                        value = copy.deepcopy(node["params"][port["name"]])
                    elif "default" in port:
                        value = copy.deepcopy(port["default"])
                    elif port.get("required", True):
                        raise ValueError(f"required input {port['name']} is missing")
                    else:
                        continue
                    _typed(value, port["type"], f"{current}.{port['name']}")
                    inputs[port["name"]] = value
                with self._lock:
                    job["nodes"][current].update(status="running", inputs=copy.deepcopy(inputs), parameters=copy.deepcopy(params))
                    if self._over_observation_budget(job):
                        # Keep prior completed observations; this node has not
                        # executed any host effect and its oversized input is
                        # represented by a bounded explicit failure.
                        job["nodes"][current].update(inputs=None, parameters=None)
                        raise ValueError(f"workflow observations exceed the {MAX_JOB_BYTES:,}-byte job limit or shared history budget")
                    self._event(job, "node_started", definition["title"], current)
                    self._revision += 1
                self._changed()
                started = time.monotonic()
                self._context.job_id = identity
                self._context.node_id = current
                output, logs = self._execute_node(node["type"], inputs, params, definition, cancel)
                output = _json(output, limit=MAX_JOB_BYTES)
                if not isinstance(output, dict):
                    raise ValueError("node outputs must be an object")
                declared = {port["name"]: port for port in definition["outputs"]}
                if set(output) != set(declared):
                    raise ValueError(f"node must return exactly its declared output ports: {', '.join(declared)}")
                for name, value in output.items():
                    _typed(value, declared[name]["type"], f"output {current}.{name}")
                with self._lock:
                    # Even a cancellation racing a host effect retains the actual
                    # output as evidence, but it never reports node/job success.
                    job["nodes"][current].update(outputs=copy.deepcopy(output), logs=list(logs),
                                                 duration_ms=(time.monotonic() - started) * 1000)
                    if self._over_observation_budget(job):
                        job["nodes"][current]["outputs"] = None
                        raise ValueError(f"workflow observations exceed the {MAX_JOB_BYTES:,}-byte job limit or shared history budget; completed host effects remain in the workspace")
                self._check_cancel(cancel)
                values[current] = output
                with self._lock:
                    job["nodes"][current]["status"] = "succeeded"
                    self._event(job, "node_succeeded", "Completed with validated typed outputs", current)
                    self._revision += 1
                self._changed()
            with self._lock:
                self._check_cancel(cancel)
                job["status"] = "succeeded"
                self._event(job, "succeeded", "All workflow nodes completed")
        except _Cancelled as exc:
            with self._lock:
                job["status"] = "cancelled"
                job["error"] = str(exc)
                if current is not None and job["nodes"][current]["status"] in ("pending", "running"):
                    job["nodes"][current].update(status="cancelled", error=str(exc))
                for record in job["nodes"].values():
                    if record["status"] == "pending":
                        record["status"] = "skipped"
                self._event(job, "cancelled", str(exc), current)
        except Exception as exc:  # The worker must survive one user's malformed workflow or failed experiment.
            with self._lock:
                cancelled = cancel.is_set()
                job["status"] = "cancelled" if cancelled else "failed"
                job["error"] = str(exc)[:2000]
                if current is not None:
                    job["nodes"][current].update(status="cancelled" if cancelled else "failed", error=job["error"])
                for record in job["nodes"].values():
                    if record["status"] == "pending":
                        record["status"] = "skipped"
                self._event(job, job["status"], job["error"], current)
        finally:
            tensor_worker = getattr(self._context, "transformers_worker", None)
            if tensor_worker is not None:
                tensor_worker.close()
                self._context.transformers_worker = None
            with self._condition:
                for record in job["nodes"].values():
                    if record["status"] == "pending":
                        record["status"] = "skipped"
                job["finished_at"] = _now()
                self._trim_jobs()
                self._revision += 1
                self._condition.notify_all()
            self._changed()

    def _live_run(self, run_id: str) -> dict[str, Any]:
        snapshot = self.workspace.snapshot()
        run = next((item for item in snapshot["runs"] if item["id"] == run_id), None)
        if run is None:
            raise KeyError("experiment does not exist")
        return run

    def _tensor_progress(self, progress: dict[str, Any]) -> None:
        """Expose bounded actual worker progress while native training is running."""
        with self._lock:
            job = self._jobs.get(self._context.job_id)
            if job is None or job["status"] != "running":
                return
            record = job["nodes"].get(self._context.node_id)
            if record is None or record["status"] != "running":
                return
            record["logs"] = [f"Training step {progress['step']}/{progress['steps']}; loss={progress['loss']:.8g}; gradient norm={progress['gradient_norm']:.8g}"]
            self._revision += 1
        self._changed()

    def _execute_node(self, kind: str, inputs: dict[str, Any], params: dict[str, Any],
                      definition: dict[str, Any], cancel: threading.Event) -> tuple[dict[str, Any], list[str]]:
        self._check_cancel(cancel)
        if kind.startswith("hf."):
            tensor_worker = getattr(self._context, "transformers_worker", None)
            if tensor_worker is None:
                tensor_worker = studio_transformers.TransformersWorker(cancel, self._tensor_progress)
                self._context.transformers_worker = tensor_worker
            if kind == "hf.load":
                descriptor = inputs["artifact"]
                if not isinstance(descriptor, dict) or not isinstance(descriptor.get("id"), str):
                    raise ValueError("Load safe model requires a registered graph artifact descriptor")
                raw, _, mime = self.artifact(descriptor["id"])
                if mime != "application/zip" or descriptor.get("sha256") != hashlib.sha256(raw).hexdigest():
                    raise ValueError("Model artifact descriptor does not match the registered immutable bytes")
                inputs = {"data_base64": base64.b64encode(raw).decode("ascii")}
            output = tensor_worker.request(kind, inputs, params)
            if kind == "hf.checkpoint":
                encoded = output["artifact"]
                raw = base64.b64decode(encoded["data_base64"], validate=True)
                output["artifact"] = self._store_artifact(raw, encoded["filename"])
            return output, ["Executed actual CPU tensors in this job's isolated Transformers worker"]
        if kind in ("core.number", "core.text", "core.boolean", "core.json"):
            return {"value": params["value"]}, []
        if kind == "core.math":
            operations: dict[str, Callable[[float, float], Any]] = {
                "add": operator.add, "subtract": operator.sub, "multiply": operator.mul,
                "divide": operator.truediv, "power": operator.pow, "min": min, "max": max, "modulo": operator.mod,
            }
            result = operations[params["operation"]](_number(inputs["a"], "a"), _number(inputs["b"], "b"))
            _number(result, "arithmetic result")
            return {"value": result}, []
        if kind == "core.array":
            return {"value": inputs["items"]}, []
        if kind == "core.object":
            return {"value": inputs["value"]}, []
        if kind == "core.get":
            source, key = inputs["object"], inputs["key"]
            if isinstance(source, dict):
                value = source.get(str(key), params["default"])
            elif isinstance(source, list) and type(key) is int:
                value = source[key] if -len(source) <= key < len(source) else params["default"]
            else:
                raise ValueError("Get property expects an object or an array with an integer index")
            return {"value": value}, []
        if kind == "core.preview":
            return {"value": inputs["value"]}, []
        if kind == "core.config":
            config = json.loads(json.dumps(_config(inputs["value"]).to_dict(), allow_nan=False))
            config["seed"] = str(config["seed"])
            config["dataset"]["seed"] = str(config["dataset"]["seed"])
            return {"config": config}, []
        if kind == "core.dataset":
            value = {"inputs": inputs["inputs"], "targets": inputs["targets"],
                     "name": params["name"], "batch_size": params["batch_size"]}
            _dataset(value)
            return {"dataset": value}, []
        if kind == "core.script" or not kind.startswith("core."):
            code = params["code"] if kind == "core.script" else definition["code"]
            result = self._plugin_module().execute_script(code, inputs["inputs"] if kind == "core.script" else inputs,
                                                        params, timeout_ms=params.get("timeout_ms", 1000),
                                                        memory_mb=params.get("memory_mb", 32), cancel=cancel)
            output = result["outputs"]
            if kind == "core.script":
                return {"value": output.get("value"), "outputs": output}, result["logs"]
            return output, result["logs"]
        run_id = str(inputs.get("run", ""))
        if kind == "core.experiment":
            payload = {"name": params["name"], "preset": params["preset"], "config": inputs["config"]}
            if "dataset" in inputs:
                payload["dataset"] = {key: value for key, value in inputs["dataset"].items()
                                      if key in ("inputs", "targets", "name", "batch_size", "provenance")}
            result = self.workspace.dispatch("run.create", payload)
            return {"run": result["run_id"]}, []
        if kind == "core.branch":
            result = self.workspace.dispatch("run.fork", {"run_id": run_id, "name": params["name"]})
            return {"run": result["run_id"]}, []
        if kind == "core.train":
            before = self._live_run(run_id)["step"]
            executed = 0
            for _ in range(params["steps"]):
                self._check_cancel(cancel)
                result = self.workspace.dispatch("run.step", {"run_id": run_id, "count": 1})
                executed += result["steps_executed"]
                if not result["steps_executed"]:
                    break
            run = self._live_run(run_id)
            if run["state"] == "aborted":
                raise ValueError(f"experiment aborted after {executed} steps: {run.get('error') or (run.get('latest') or {}).get('abort_reason') or 'numerical abort'}")
            return {"run": run_id, "record": run.get("latest") or {"step": run["step"], "parameter": run["parameter"]}}, [f"Executed {executed} real training steps ({before} → {run['step']})"]
        if kind == "core.checkpoint":
            result = self.workspace.dispatch("checkpoint.create", {"run_id": run_id, "name": params["name"]})
            return {"run": run_id, "checkpoint": result["checkpoint_id"]}, []
        if kind == "core.inspect":
            record = self.workspace.record(run_id, params["step"]) if params["step"] >= 0 else self._live_run(run_id)
            data = self.workspace.dataset(run_id) if params["include_dataset"] else {}
            return {"run": run_id, "record": record, "dataset": data}, []
        if kind == "core.intervene":
            changes = copy.deepcopy(inputs["changes"])
            if set(changes) - {"parameter", "parameter_value", "parameter_hex", "lr", "learning_rate", "optimizer_state"}:
                raise ValueError("intervention changes may contain parameter_value, parameter_hex, learning_rate and optimizer_state (aliases: parameter, lr)")
            for alias, canonical in (("parameter", "parameter_value"), ("lr", "learning_rate")):
                if alias in changes:
                    if canonical in changes:
                        raise ValueError(f"supply either {alias} or {canonical}")
                    changes[canonical] = changes.pop(alias)
            self.workspace.dispatch("run.intervene", {"run_id": run_id, **changes})
            return {"run": run_id, "record": self._live_run(run_id)}, []
        if kind == "core.verify":
            result = self.workspace.dispatch("run.verify", {"run_id": run_id})
            return {"run": run_id, "integrity": result["integrity"]}, []
        if kind == "core.control":
            self.workspace.dispatch(f"run.{params['operation']}", {"run_id": run_id})
            return {"run": run_id, "record": self._live_run(run_id)}, []
        if kind == "core.export":
            self._check_cancel(cancel)
            data, filename = self.workspace.export_run(run_id)
            descriptor = self._store_artifact(data, filename)
            return {"run": run_id, "artifact": descriptor}, []
        raise ValueError(f"unsupported node type: {kind}")

    @staticmethod
    def _artifact_descriptor(item: dict[str, Any]) -> dict[str, Any]:
        return {key: item[key] for key in ("id", "job_id", "node_id", "filename", "mime", "byte_length", "sha256", "download_url")}

    def _store_artifact(self, data: bytes, filename: str) -> dict[str, Any]:
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"evidence ZIP exceeds the {MAX_ARTIFACT_BYTES:,}-byte artifact quota; use the main Artifacts view to export it")
        identity = _id()
        artifact = {"id": identity, "job_id": self._context.job_id, "node_id": self._context.node_id,
                    "filename": filename, "mime": "application/zip", "byte_length": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "download_url": f"/api/graph/artifacts/{identity}", "data": data}
        with self._lock:
            if sum(item["byte_length"] for item in self._artifacts.values()) + len(data) > MAX_ARTIFACT_BYTES:
                raise ValueError("saved node evidence fills the 8 MiB artifact quota; clear completed jobs to release their artifacts")
            self._artifacts[identity] = artifact
        return self._artifact_descriptor(artifact)

    def _prune_artifacts(self) -> None:
        self._artifacts = {key: value for key, value in self._artifacts.items() if value["job_id"] in self._jobs}

    def artifact(self, artifact_id: str) -> tuple[bytes, str, str]:
        with self._lock:
            self._ensure_open()
            if artifact_id not in self._artifacts:
                raise KeyError("workflow artifact does not exist")
            artifact = self._artifacts[artifact_id]
            return artifact["data"], artifact["filename"], artifact["mime"]

    def wait_job(self, job_id: str, timeout: float = 30) -> dict[str, Any]:
        """Bounded server/test helper; browser clients poll snapshot instead."""
        deadline = time.monotonic() + timeout
        with self._condition:
            if job_id not in self._jobs:
                raise KeyError("job does not exist")
            while self._jobs[job_id]["status"] in ("queued", "running", "cancelling"):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("workflow did not finish before the timeout")
                self._condition.wait(remaining)
            return copy.deepcopy(self._jobs[job_id])

    def job(self, job_id: str) -> dict[str, Any]:
        """Return one exact observation snapshot without browser number coercion."""
        with self._lock:
            self._ensure_open()
            if job_id not in self._jobs:
                raise KeyError("job does not exist")
            return copy.deepcopy(self._jobs[job_id])

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy({"schema_version": GRAPH_SCHEMA, "revision": self._revision,
                                  "active_graph_id": self._active_graph_id, "graphs": list(self._graphs.values()),
                                  "plugins": list(self._plugins.values()), "jobs": list(self._jobs.values()),
                                  "artifacts": [{**self._artifact_descriptor(item), "data_base64": base64.b64encode(item["data"]).decode("ascii")}
                                                for item in self._artifacts.values()]})

    @classmethod
    def validate_state(cls, value: Any) -> dict[str, Any]:
        """Stage all declarative state without starting workers or executing code."""
        if value is None:
            starter = cls._graph({**_templates()[0]["graph"], "id": "starter-training", "revision": 1})
            return {"schema_version": GRAPH_SCHEMA, "revision": 0, "active_graph_id": starter["id"],
                    "graphs": [starter], "plugins": [], "jobs": [], "artifacts": []}
        data = _json(value, limit=48_000_000)
        if not isinstance(data, dict) or data.get("schema_version") != GRAPH_SCHEMA:
            raise ValueError("unsupported node studio schema")
        if set(data) - {"schema_version", "revision", "active_graph_id", "graphs", "plugins", "jobs", "artifacts"}:
            raise ValueError("node studio state contains unknown fields")
        data["revision"] = _integer(data.get("revision", 0), "node studio revision", 0, 10**12)
        graph_list, plugin_list, jobs = data.get("graphs", []), data.get("plugins", []), data.get("jobs", [])
        if not isinstance(graph_list, list) or len(graph_list) > MAX_GRAPHS:
            raise ValueError("too many workflows in project")
        if not isinstance(plugin_list, list) or len(plugin_list) > MAX_PLUGINS:
            raise ValueError("too many plugins in project")
        if not isinstance(jobs, list) or len(jobs) > MAX_JOBS:
            raise ValueError("too many saved jobs in project")
        graphs = [cls._graph(graph) for graph in graph_list]
        _json(graphs, limit=MAX_GRAPH_REGISTRY_BYTES)
        if len({graph["id"] for graph in graphs}) != len(graphs):
            raise ValueError("duplicate workflow ids in project")
        if data.get("active_graph_id") is not None and data["active_graph_id"] not in {graph["id"] for graph in graphs}:
            raise ValueError("active workflow does not exist")
        plugins: list[dict[str, Any]] = []
        for plugin in plugin_list:
            if not isinstance(plugin, dict) or set(plugin) != {"id", "enabled", "manifest", "installed_at", "updated_at"}:
                raise ValueError("invalid saved plugin")
            if type(plugin["enabled"]) is not bool:
                raise ValueError("plugin enabled must be boolean")
            manifest = cls._plugin_module().validate_plugin(plugin["manifest"])
            for template in manifest.get("templates", []):
                cls._graph(template["graph"])
            if plugin["id"] != manifest["id"]:
                raise ValueError("saved plugin id does not match its manifest")
            for key in ("installed_at", "updated_at"):
                _text(plugin[key], key)
            plugins.append({**plugin, "manifest": manifest})
        if len({plugin["id"] for plugin in plugins}) != len(plugins):
            raise ValueError("duplicate plugin ids in project")
        _json(plugins, limit=MAX_PLUGIN_REGISTRY_BYTES)
        # Incomplete graphs and graphs depending on currently disabled/removed
        # extensions are useful editable documents, so only their structure is
        # required on import. Execution always reruns full semantic validation.
        identities: set[str] = set()
        replay_keys: set[str] = set()
        for job in jobs:
            _json(job, limit=MAX_JOB_BYTES)
            cls._validate_saved_job(job)
            if job["id"] in identities:
                raise ValueError("duplicate job ids in project")
            identities.add(job["id"])
            if job["idempotency_key"] is not None:
                if job["idempotency_key"] in replay_keys:
                    raise ValueError("duplicate saved job idempotency keys")
                replay_keys.add(job["idempotency_key"])
            if job["status"] in ("queued", "running", "cancelling"):
                job["status"] = "interrupted"
                job["finished_at"] = _now()
                job["error"] = "Project reopened; execution is never automatically resumed"
                for record in job["nodes"].values():
                    if record["status"] in ("running", "pending"):
                        record["status"] = "interrupted"
        if sum(cls._size(job) for job in jobs) > MAX_HISTORY_BYTES:
            raise ValueError("saved job history exceeds the observation byte budget")
        artifacts = data.get("artifacts", [])
        if not isinstance(artifacts, list) or len(artifacts) > MAX_JOBS * MAX_NODES:
            raise ValueError("saved artifacts must be a bounded array")
        artifact_ids: set[str] = set()
        total_bytes = 0
        jobs_by_id = {job["id"]: job for job in jobs}
        for artifact in artifacts:
            required = {"id", "job_id", "node_id", "filename", "mime", "byte_length", "sha256", "download_url", "data_base64"}
            if not isinstance(artifact, dict) or set(artifact) != required:
                raise ValueError("invalid saved workflow artifact")
            identity = _identity(artifact["id"], "artifact id")
            if identity in artifact_ids:
                raise ValueError("duplicate saved artifact ids")
            artifact_ids.add(identity)
            _identity(artifact["job_id"], "artifact job id")
            _identity(artifact["node_id"], "artifact node id")
            owner = jobs_by_id.get(artifact["job_id"])
            if owner is None or artifact["node_id"] not in owner["nodes"]:
                raise ValueError("saved artifact owner does not exist")
            filename = _text(artifact["filename"], "artifact filename", 255)
            if not filename or any(char in filename for char in "\\/\r\n\0") or filename in (".", ".."):
                raise ValueError("artifact filename must be a plain filename")
            if artifact["mime"] != "application/zip" or artifact["download_url"] != f"/api/graph/artifacts/{identity}":
                raise ValueError("invalid saved artifact MIME type or download URL")
            try:
                raw = base64.b64decode(artifact["data_base64"], validate=True)
            except (ValueError, TypeError, binascii.Error) as exc:
                raise ValueError("artifact data must be valid base64") from exc
            if type(artifact["byte_length"]) is not int or artifact["byte_length"] != len(raw) or artifact["sha256"] != hashlib.sha256(raw).hexdigest():
                raise ValueError("artifact bytes do not match their length and SHA-256")
            total_bytes += len(raw)
            if total_bytes > MAX_ARTIFACT_BYTES:
                raise ValueError("saved artifacts exceed the 8 MiB quota")
        return {"schema_version": GRAPH_SCHEMA, "revision": data["revision"], "active_graph_id": data.get("active_graph_id"),
                "graphs": graphs, "plugins": plugins, "jobs": jobs, "artifacts": artifacts}

    @classmethod
    def _validate_saved_job(cls, job: Any) -> None:
        required = {"id", "graph_id", "graph_revision", "graph_name", "status", "created_at", "started_at", "finished_at",
                    "graph", "definitions", "plugin_versions", "order", "nodes", "events", "error", "idempotency_key", "partial_effects", "snapshot_sha256"}
        if not isinstance(job, dict) or set(job) != required:
            raise ValueError("invalid saved job fields")
        _identity(job["id"], "job id")
        graph = cls._graph(job["graph"])
        if graph["id"] != job["graph_id"] or graph["revision"] != job["graph_revision"]:
            raise ValueError("saved job does not match its immutable graph snapshot")
        if job["status"] not in ("queued", "running", "cancelling", "succeeded", "failed", "cancelled", "interrupted"):
            raise ValueError("invalid job status")
        if not isinstance(job["definitions"], dict) or not isinstance(job["nodes"], dict):
            raise ValueError("invalid job definitions or observations")
        if job["snapshot_sha256"] != cls._job_snapshot_hash(job):
            raise ValueError("saved immutable workflow snapshot hash does not match its captured content")
        if set(job["definitions"]) != {node["type"] for node in graph["nodes"]}:
            raise ValueError("saved workflow definitions do not match the graph node types")
        for kind, definition in job["definitions"].items():
            if not isinstance(definition, dict) or definition.get("id") != kind:
                raise ValueError("invalid saved node definition")
            for group in ("inputs", "outputs", "parameters"):
                if not isinstance(definition.get(group), list):
                    raise ValueError("saved node definitions require input, output and parameter schemas")
                names = set()
                for port in definition[group]:
                    if not isinstance(port, dict) or not isinstance(port.get("name"), str) or port.get("type") not in PORT_TYPES | {"code"}:
                        raise ValueError("invalid saved port or parameter definition")
                    if port["name"] in names:
                        raise ValueError("duplicate saved node port or parameter name")
                    names.add(port["name"])
        if set(job["nodes"]) != {node["id"] for node in graph["nodes"]}:
            raise ValueError("saved node observations do not match the graph")
        if not isinstance(job["events"], list) or len(job["events"]) > 1100:
            raise ValueError("invalid saved job events")
        for event in job["events"]:
            if not isinstance(event, dict) or set(event) != {"time", "type", "message", "node_id"}:
                raise ValueError("invalid saved job event fields")
            _text(event["time"], "job event time")
            _text(event["type"], "job event type")
            _text(event["message"], "job event message", 2000)
            if event["node_id"] is not None:
                _identity(event["node_id"], "saved event node id")
                if event["node_id"] not in job["nodes"]:
                    raise ValueError("saved event refers to an unknown node")
        if not isinstance(job["order"], list) or len(job["order"]) != len(set(job["order"])) or set(job["order"]) != set(job["nodes"]):
            raise ValueError("invalid saved execution order")
        for record in job["nodes"].values():
            if not isinstance(record, dict) or record.get("status") not in ("pending", "running", "succeeded", "failed", "cancelled", "skipped", "interrupted"):
                raise ValueError("invalid saved node observation")
            required_record = {"status", "inputs", "outputs", "logs", "error", "duration_ms"}
            if set(record) not in (required_record, required_record | {"parameters"}):
                raise ValueError("invalid saved node observation fields")
            for field in ("inputs", "outputs", "parameters"):
                if record.get(field) is not None and not isinstance(record[field], dict):
                    raise ValueError(f"saved node {field} must be an object or null")
            if not isinstance(record["logs"], list) or len(record["logs"]) > 1000 or any(not isinstance(line, str) or len(line) > 100000 for line in record["logs"]):
                raise ValueError("saved node logs must be a bounded text array")
            if record["error"] is not None:
                _text(record["error"], "saved node error", 2000)
            if record["duration_ms"] is not None and _number(record["duration_ms"], "node duration") < 0:
                raise ValueError("saved node duration must not be negative")
            if record["status"] == "succeeded" and (record["inputs"] is None or record["outputs"] is None):
                raise ValueError("successful saved nodes require actual inputs and outputs")
            if record["status"] == "succeeded" and record["error"] is not None:
                raise ValueError("successful saved nodes cannot contain an error")
        statuses = {record["status"] for record in job["nodes"].values()}
        if job["status"] == "succeeded" and statuses != {"succeeded"}:
            raise ValueError("successful saved jobs require every node to have succeeded")
        if job["status"] == "succeeded" and job["error"] is not None:
            raise ValueError("successful saved jobs cannot contain an error")
        if job["status"] == "queued" and statuses != {"pending"}:
            raise ValueError("queued saved jobs must contain only pending nodes")
        if job["status"] in ("succeeded", "failed", "cancelled", "interrupted") and statuses & {"pending", "running"}:
            raise ValueError("terminal saved jobs cannot contain pending or running nodes")
        for key in ("created_at", "started_at", "finished_at", "error", "idempotency_key"):
            if job[key] is not None:
                _text(job[key], key, 2000)
        if job["idempotency_key"] is not None:
            if not _text(job["idempotency_key"], "idempotency key", 300):
                raise ValueError("saved idempotency keys must not be empty")
        _text(job["graph_name"], "job graph name")
        _text(job["partial_effects"], "partial effects notice", 2000)

    def import_state(self, value: Any) -> None:
        data = self.validate_state(value)
        with self._condition:
            self._ensure_open()
            if self.is_busy():
                raise GraphConflictError("cancel and wait for active workflows before opening a project")
            self._graphs = {graph["id"]: graph for graph in data["graphs"]}
            self._plugins = {plugin["id"]: plugin for plugin in data["plugins"]}
            self._jobs = {job["id"]: job for job in data["jobs"]}
            self._artifacts = {item["id"]: {**{key: value for key, value in item.items() if key != "data_base64"},
                                               "data": base64.b64decode(item["data_base64"], validate=True)}
                               for item in data["artifacts"]}
            self._cancels = {key: threading.Event() for key in self._jobs}
            self._queue.clear()
            self._idempotency = {job["idempotency_key"]: job["id"] for job in self._jobs.values() if job.get("idempotency_key")}
            self._active_graph_id = data["active_graph_id"]
            self._revision = data["revision"] + 1
        self._changed()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for identity, job in self._jobs.items():
                if job["status"] in ("queued", "running", "cancelling"):
                    self._cancels[identity].set()
                    if job["status"] == "queued":
                        job.update(status="cancelled", finished_at=_now(), error="Studio closed before execution")
                        for record in job["nodes"].values():
                            record["status"] = "skipped"
            self._condition.notify_all()
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=15)
            if worker.is_alive():
                raise RuntimeError("node worker did not stop after cancellation")
        self._changed()

