"""Persistent, independently controlled experiment worlds for the browser studio.

The studio uses the ordinary instrumented training engine. Checkpoints carry
actual tensor bytes, optimizer/scheduler/reference state and sealed observations.
The portable project format is strict JSON, with an explicit type whitelist;
opening a project never loads a pickle or executes user supplied Python.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import __version__
from . import validation as vd
from .config import (
    PRESETS, PRESET_DESCRIPTIONS, DatasetConfig, ExperimentConfig,
    LandscapeConfig, ModelConfig, StabilityCriteria, ValidationConfig,
)
from .convergence import ConvergenceAnalyzer, StepStability
from .exporters import export_all
from .model import Dataset, TensorSnapshot
from .recorder import EventEntry, ParameterStepRecord, Recorder, json_safe
from .studio_graph import GraphStudio
from .training_engine import ExperimentController, RunState, TrainingEngine

PROJECT_SCHEMA = "spobs-studio/1"


class StudioStateError(RuntimeError):
    """A studio lifecycle operation is invalid in the present state."""


class StudioEpochError(StudioStateError):
    """A browser request belongs to a replaced live workspace incarnation."""


_CONFIG_TYPES = {
    "dataset": DatasetConfig, "model": ModelConfig,
    "landscape": LandscapeConfig, "validation": ValidationConfig,
    "stability": StabilityCriteria,
}
_VALUE_TYPES = {cls.__name__: cls for cls in (
    Dataset, TensorSnapshot, ParameterStepRecord, EventEntry, StepStability,
    StabilityCriteria, vd.ValidationLedger, vd.BitValidation, vd.ULPValidation,
)}
_DTYPES = {str(value): value for value in (
    torch.float32, torch.float64, torch.int32, torch.int64, torch.bool, torch.uint8,
)}
_MISC = (
    "step_index", "_live_step_index", "aborted", "abort_reason", "_active_variant",
    "_variant_locked", "_bit_validation", "_ulp_validation", "_semantics_probe",
    "_provenance_cache", "schedule_note",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _strict_saved_json(text: str) -> Any:
    """Use the same unambiguous finite JSON boundary for disk and HTTP data."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate saved JSON key: {key}")
            output[key] = value
        return output

    def finite(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("saved JSON numbers must be finite")
        return number

    def constant(value: str) -> None:
        raise ValueError(f"non-finite saved JSON constant: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_float=finite, parse_constant=constant)


def _validate_ui_drafts(value: Any) -> dict[str, Any]:
    """Portable editing buffers are data and never implicitly applied or executed."""
    if value is None:
        return {"schema_version": "spobs-ui-drafts/1", "runs": {}, "plugin": "", "plugin_dirty": False}
    allowed = {"schema_version", "runs", "workspace_notes", "plugin", "plugin_dirty", "node_fields", "panel_fields"}
    if not isinstance(value, dict) or set(value) - allowed or value.get("schema_version") != "spobs-ui-drafts/1":
        raise ValueError("invalid studio draft schema")
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("portable editor drafts exceed 16 MiB")
    drafts = _strict_saved_json(encoded)
    runs = drafts.get("runs", {})
    if not isinstance(runs, dict) or len(runs) > 100:
        raise ValueError("editor drafts must contain at most 100 run buffers")
    fields_allowed = {"config", "config_json", "config_json_dirty", "csv", "notes", "bits", "bit_dirty", "optimizer_json"}
    for identity, run_draft in runs.items():
        _identity(identity, "draft run id")
        if not isinstance(run_draft, dict) or set(run_draft) - fields_allowed:
            raise ValueError("unknown experiment draft fields")
        if "config" in run_draft and not isinstance(run_draft["config"], dict):
            raise ValueError("configuration draft must be a JSON object")
        for key in ("config_json", "csv", "notes", "optimizer_json"):
            if key in run_draft:
                _text(run_draft[key], f"draft {key}", 10000000 if key == "csv" else 1000000)
        for key in ("config_json_dirty", "bit_dirty"):
            if key in run_draft and type(run_draft[key]) is not bool:
                raise ValueError(f"draft {key} must be boolean")
        if run_draft.get("bits") is not None:
            _integer(run_draft["bits"], "draft bits", maximum=2**32 - 1)
    for key in ("workspace_notes", "plugin"):
        if key in drafts:
            _text(drafts[key], f"draft {key}", 4 * 1024 * 1024 if key == "plugin" else 100000)
    if "plugin_dirty" in drafts and type(drafts["plugin_dirty"]) is not bool:
        raise ValueError("draft plugin_dirty must be boolean")
    node_fields = drafts.get("node_fields", {})
    if not isinstance(node_fields, dict) or len(node_fields) > 100:
        raise ValueError("node field drafts must be an object of at most 100 workflows")
    for identity, values in node_fields.items():
        _text(identity, "draft graph id", 300)
        if not isinstance(values, dict) or len(values) > 10000:
            raise ValueError("invalid workflow field draft collection")
        for key, text in values.items():
            _text(key, "draft field id", 400)
            _text(text, "draft field value", 1000000)
    panel_fields = drafts.get("panel_fields", {})
    if not isinstance(panel_fields, dict) or len(panel_fields) > 512:
        raise ValueError("add-on panel drafts must be an object of at most 512 panels")
    for identity, values in panel_fields.items():
        _text(identity, "draft panel id", 400)
        if not isinstance(values, dict) or len(values) > 100:
            raise ValueError("invalid add-on panel field draft collection")
        for key, text in values.items():
            _text(key, "draft panel field id", 400)
            _text(text, "draft panel field value", 1000000)
    return drafts


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int = 1000000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _text(value: Any, name: str, maximum: int = 100000) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{name} must be text of at most {maximum} characters")
    return value


def _identity(value: Any, name: str) -> str:
    text = _text(value, name, 300)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,300}", text):
        raise ValueError(f"{name} must contain letters, digits, underscores or hyphens")
    return text


def _seed_integer(value: Any, name: str) -> Any:
    if isinstance(value, str):
        if len(value) > 21 or not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
            raise ValueError(f"{name} must be a decimal integer without whitespace or exponent notation")
        return int(value)
    return value


def config_for_browser(config: ExperimentConfig) -> dict[str, Any]:
    """Keep 64-bit seed identities exact when JavaScript parses browser state."""
    data = config.to_dict()
    data["seed"] = str(data["seed"])
    data["dataset"]["seed"] = str(data["dataset"]["seed"])
    return data


def config_from_dict(data: dict[str, Any], base: ExperimentConfig | None = None) -> ExperimentConfig:
    """Merge complete or partial nested configuration through validated dataclasses."""
    if not isinstance(data, dict):
        raise ValueError("config must be an object")
    current = (base or PRESETS["adamw"]).to_dict()
    unknown = set(data) - set(current)
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    for name, value in data.items():
        if name in _CONFIG_TYPES:
            if not isinstance(value, dict):
                raise ValueError(f"config.{name} must be an object")
            current[name] = {**current[name], **value}
        else:
            current[name] = value
    current["seed"] = _seed_integer(current["seed"], "seed")
    current["dataset"]["seed"] = _seed_integer(current["dataset"]["seed"], "dataset.seed")
    for name, cls in _CONFIG_TYPES.items():
        current[name] = cls(**current[name])
    config = ExperimentConfig(**current)
    if config.steps > 1000000 or config.dataset.samples > 100000:
        raise ValueError("studio limits are 1,000,000 steps and 100,000 data rows per run")
    if config.landscape.points > 10001 or config.accumulation_micro_batches > 10000:
        raise ValueError("landscape points or accumulation exceed the studio resource limit")
    return config


def _prepare_dataset(config: ExperimentConfig, payload: dict[str, Any]) -> tuple[ExperimentConfig, Dataset, dict[str, Any]]:
    """Validate and materialize custom data without publishing a new experiment."""
    if not isinstance(payload, dict) or set(payload) - {"csv", "inputs", "targets", "name", "batch_size", "provenance"}:
        raise ValueError("dataset must contain CSV or numeric inputs/targets and optional name/batch_size")
    inputs: Any
    targets: Any
    provenance: dict[str, Any]
    text = payload.get("csv")
    if text is not None:
        if not isinstance(text, str) or len(text.encode("utf-8")) > 10000000:
            raise ValueError("CSV must be text of at most 10 MB")
        if "inputs" in payload or "targets" in payload:
            raise ValueError("provide either CSV or input/target arrays")
        rows = [row for row in csv.reader(io.StringIO(text)) if row and any(cell.strip() for cell in row)]
        if rows and [cell.strip().lower() for cell in rows[0]] in (["x", "target"], ["input", "target"], ["x", "y"]):
            rows = rows[1:]
        if any(len(row) != 2 for row in rows):
            raise ValueError("CSV must contain exactly two columns: x,target")
        if len(rows) > 100000:
            raise ValueError("dataset must contain 1 to 100,000 rows")
        try:
            inputs, targets = zip(*[(float(row[0]), float(row[1])) for row in rows], strict=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("CSV must contain numeric x,target rows") from exc
        provenance = {"kind": "csv", "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                      "name": _text(payload.get("name", "Imported CSV"), "dataset name", 300)}
    else:
        inputs, targets = payload.get("inputs"), payload.get("targets")
        provenance = {"kind": "arrays", "name": _text(payload.get("name", "Imported arrays"), "dataset name", 300)}
    if "provenance" in payload:
        if not isinstance(payload["provenance"], dict):
            raise ValueError("dataset provenance must be a JSON object")
        serialized = json.dumps(payload["provenance"], allow_nan=False, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > 100000:
            raise ValueError("dataset provenance exceeds 100,000 UTF-8 bytes")
        provenance["user_provenance"] = copy.deepcopy(payload["provenance"])
    if not isinstance(inputs, (list, tuple)) or not isinstance(targets, (list, tuple)) or len(inputs) != len(targets):
        raise ValueError("inputs and targets must be equal length arrays")
    if not 1 <= len(inputs) <= 100000:
        raise ValueError("dataset must contain 1 to 100,000 rows")
    xs = [_number(value, "input") for value in inputs]
    ys = [_number(value, "target") for value in targets]
    if max(abs(value) for value in xs + ys) > torch.finfo(torch.float32).max:
        raise ValueError("data must fit finite float32 storage")
    batch_size = _integer(payload.get("batch_size", min(config.dataset.batch_size, len(xs))),
                          "batch_size", minimum=1, maximum=len(xs))
    config = config.replace_dataset(samples=len(xs), batch_size=batch_size,
                                    input_min=min(xs), input_max=max(xs), shuffle_batches=False)
    dataset = Dataset(torch.tensor(xs, dtype=torch.float32).reshape(-1, 1),
                      torch.tensor(ys, dtype=torch.float32).reshape(-1, 1),
                      math.ceil(len(xs) / batch_size), "imported", None,
                      config.dataset.target_true_parameter, 0.0, batch_size)
    provenance.update({"rows": len(xs), "imported_at": _now(), "storage": "float32",
                       "content_sha256": _digest({"inputs": xs, "targets": ys}),
                       "stored_tensor_sha256": _digest(_pack(dataset))})
    return config, dataset, provenance


def _pack(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {"$tensor": str(tensor.dtype), "shape": list(tensor.shape),
                "bytes": tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex()}
    if is_dataclass(value) and not isinstance(value, type):
        if type(value).__name__ not in _VALUE_TYPES:
            raise ValueError(f"unsupported state type {type(value).__name__}")
        return {"$type": type(value).__name__, "fields": {
            item.name: _pack(getattr(value, item.name)) for item in fields(value)
        }}
    if isinstance(value, dict):
        return {"$map": [[_pack(key), _pack(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"$tuple": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": repr(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"unsupported state value {type(value).__name__}")


def _unpack(value: Any, depth: int = 0) -> Any:
    if depth > 60:
        raise ValueError("project state is nested too deeply")
    if isinstance(value, list):
        return [_unpack(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        if value is None or type(value) in (str, int, bool) or (
            type(value) is float and math.isfinite(value)
        ):
            return value
        raise ValueError("invalid JSON state scalar")
    if set(value) == {"$tensor", "shape", "bytes"}:
        dtype = _DTYPES.get(value["$tensor"])
        shape = value["shape"]
        if dtype is None or not isinstance(shape, list) or len(shape) > 4:
            raise ValueError("unsupported tensor type or shape")
        for size in shape:
            _integer(size, "tensor dimension", maximum=1000000)
        count = math.prod(shape)
        if count > 2000000 or not isinstance(value["bytes"], str):
            raise ValueError("tensor exceeds the project limit")
        raw = bytes.fromhex(value["bytes"])
        width = torch.empty((), dtype=dtype).element_size()
        if len(raw) != count * width:
            raise ValueError("tensor byte length does not match its shape")
        if not raw:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
    if set(value) == {"$float"} and value["$float"] in ("nan", "inf", "-inf"):
        return float(value["$float"])
    if set(value) == {"$tuple"} and isinstance(value["$tuple"], list):
        return tuple(_unpack(item, depth + 1) for item in value["$tuple"])
    if set(value) == {"$map"} and isinstance(value["$map"], list):
        result = {}
        for pair in value["$map"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid state mapping")
            key, item = (_unpack(member, depth + 1) for member in pair)
            if type(key) not in (str, int) or key in result:
                raise ValueError("invalid or duplicate state mapping key")
            result[key] = item
        return result
    if set(value) == {"$type", "fields"}:
        cls = _VALUE_TYPES.get(value["$type"])
        if cls is None or not isinstance(value["fields"], dict):
            raise ValueError("unsupported project state type")
        if set(value["fields"]) != {item.name for item in fields(cls)}:
            raise ValueError("state fields do not match the declared type")
        return cls(**{key: _unpack(item, depth + 1) for key, item in value["fields"].items()})
    raise ValueError("invalid project state encoding")


@dataclass
class _Run:
    id: str
    name: str
    engine: TrainingEngine
    controller: ExperimentController
    parent_id: str | None
    position: dict[str, float]
    notes: str
    checkpoints: list[dict[str, Any]]
    dataset_provenance: dict[str, Any]
    interventions: list[dict[str, Any]]
    created_at: str
    error: str | None = None
    verified_hash: str = Recorder.GENESIS_HASH


class StudioWorkspace:
    """One local studio with concurrent, source-bound real training runs."""

    def __init__(self, storage_directory: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._dirty = threading.Event()
        self._stopping = threading.Event()
        self._closed = False
        self._closing = False
        self._close_lock = threading.Lock()
        self.workspace_epoch = _id()
        self._autosave_thread: threading.Thread | None = None
        self.storage_directory = Path(storage_directory).resolve() if storage_directory is not None else None
        self.autosave_error: str | None = None
        self.recovery_notice: str | None = None
        self._autosave_blocked = False
        self.ui_drafts = _validate_ui_drafts(None)
        self._ui_drafts_revision = 0
        self.workspace: dict[str, Any] = {
            "id": _id(), "name": "Untitled studio", "notes": "", "active_run_id": None,
            "view": {"x": 0.0, "y": 0.0, "zoom": 1.0}, "revision": 0,
        }
        self._runs: dict[str, _Run] = {}
        self.graphs = GraphStudio(self, on_change=self._dirty.set)
        restored = False
        if self.storage_directory is not None:
            self.storage_directory.mkdir(parents=True, exist_ok=True)
            path = self.storage_directory / "autosave.json"
            if path.exists():
                try:
                    self.import_project(_strict_saved_json(path.read_text(encoding="utf-8")))
                    restored = True
                except (ValueError, TypeError, KeyError, RuntimeError, UnicodeError) as exc:
                    recovery = path.with_name(f"autosave.recovery-{uuid.uuid4().hex}.json")
                    try:
                        with recovery.open("xb") as backup:
                            backup.write(path.read_bytes())
                            backup.flush()
                            os.fsync(backup.fileno())
                        self.recovery_notice = f"Previous autosave could not be opened; original bytes preserved at {recovery.name}: {exc}"
                    except OSError as backup_error:
                        self._autosave_blocked = True
                        self.recovery_notice = f"Previous autosave could not be opened; automatic replacement is blocked because its recovery copy failed: {backup_error}"
                    self.autosave_error = self.recovery_notice
        if not self._runs and not restored:
            self._create(PRESETS["adamw"], "AdamW exploration", {"x": 80.0, "y": 275.0})
        if self.storage_directory is not None:
            self._autosave_thread = threading.Thread(
                target=self._autosave_loop, name="studio-autosave", daemon=True,
            )
            self._autosave_thread.start()

    def _create(
        self, config: ExperimentConfig, name: str, position: dict[str, float],
        *, parent_id: str | None = None, run_id: str | None = None, arrange: bool = False,
    ) -> _Run:
        if len(self._runs) >= 100:
            raise ValueError("a studio can contain at most 100 runs")
        if arrange:
            position = dict(position)
            while any(abs(position["x"] - existing.position["x"]) < 300
                      and abs(position["y"] - existing.position["y"]) < 220 for existing in self._runs.values()):
                position["y"] += 240
        engine = TrainingEngine(config)
        controller = ExperimentController(engine)
        controller.on_step = lambda _record: self._dirty.set()
        run = _Run(run_id or _id(), _text(name, "run name", 300), engine, controller,
                   parent_id, position, "", [], {"kind": "generated", "config": config.dataset.__dict__},
                   [], _now())
        self._runs[run.id] = run
        self.workspace["active_run_id"] = run.id
        return run

    def _get(self, payload: dict[str, Any] | str) -> _Run:
        run_id = payload if isinstance(payload, str) else payload.get("run_id", self.workspace["active_run_id"])
        if run_id not in self._runs:
            raise KeyError("the requested run does not exist")
        return self._runs[run_id]

    def _changed(self) -> None:
        self.workspace["revision"] += 1
        self._dirty.set()

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise StudioStateError("the studio is closing" if self._closing and not self._closed else "the studio is closed")

    def check_workspace_epoch(self, expected: str | None) -> None:
        """Reject stale browser admission tokens; caller holds _lock through mutation."""
        with self._lock:
            self._ensure_open()
            if expected is not None and expected != self.workspace_epoch:
                raise StudioEpochError("This workspace was replaced in another tab. Reload the current workspace before continuing.")

    def capabilities(self) -> dict[str, Any]:
        defaults = config_for_browser(PRESETS["adamw"])
        schema = {}
        for item in fields(ExperimentConfig):
            value = defaults[item.name]
            schema[item.name] = {"type": str(item.type), "default": value}
            if item.name in _CONFIG_TYPES:
                schema[item.name]["fields"] = {
                    member.name: {"type": str(member.type), "default": value[member.name]}
                    for member in fields(_CONFIG_TYPES[item.name])
                }
        return json_safe({
            "version": __version__, "torch_version": torch.__version__,
            "project_schema": PROJECT_SCHEMA,
            "presets": {name: {"description": PRESET_DESCRIPTIONS[name], "config": config_for_browser(config)}
                        for name, config in PRESETS.items()},
            "default_config": defaults, "config_schema": schema,
            "limits": {"runs": 100, "steps": 1000000, "dataset_rows": 100000},
        })

    def _run_snapshot(self, run: _Run) -> dict[str, Any]:
        with run.controller._lock:
            engine = run.engine
            try:
                run.controller.raise_if_failed()
            except BaseException as exc:
                run.error = f"{type(exc).__name__}: {exc}"
            records = engine.recorder.records
            if len(records) > 2000:
                indices = sorted({round(index * (len(records) - 1) / 1999) for index in range(2000)})
                displayed_records = [records[index] for index in indices]
            else:
                displayed_records = list(records)
            latest = records[-1] if records else None
            state = run.controller.state
            if engine.finished:
                state = RunState.ABORTED if engine.aborted else RunState.FINISHED
            live_intervention = False
            for event in reversed(engine.recorder.events.entries):
                if event.code == "INITIALIZED":
                    break
                if event.code == "MANUAL_INTERVENTION":
                    live_intervention = event.step == engine.step_index
                    break
            optimizer_state = engine.handle.optimizer.state.get(engine.parameter, {})
            live_optimizer_state = {
                key: value.detach().cpu().item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value
                for key, value in optimizer_state.items()
            }
            return {
                "id": run.id, "name": run.name, "parent_id": run.parent_id,
                "position": dict(run.position), "config": config_for_browser(engine.config),
                "state": state, "step": engine.step_index,
                "parameter": float(engine.parameter.detach().item()),
                "parameter_uint32": int(engine.parameter.detach().view(torch.int32).item()) & 0xFFFFFFFF,
                "loss": latest.loss if latest else None, "gradient": latest.gradient if latest else None,
                "speed": run.controller.speed, "latest": latest.to_dict() if latest else None,
                "series": [{"step": record.step, "w": record.parameter_after, "loss": record.loss,
                            "gradient": record.gradient, "delta": record.actual_delta,
                            "moment1": record.moment1, "moment2": record.moment2,
                            "lr": record.learning_rate, "fp64": record.fp64_track_value}
                           for record in displayed_records],
                "series_total": len(records), "series_sampled": len(displayed_records) < len(records),
                "events": engine.recorder.events.to_list()[-100:],
                "checkpoints": [{key: item[key] for key in ("id", "name", "step", "created_at")}
                                for item in run.checkpoints],
                "notes": run.notes, "error": run.error or engine.abort_reason,
                "dataset_provenance": run.dataset_provenance,
                "interventions": copy.deepcopy(run.interventions),
                "live_intervention": live_intervention,
                "live_optimizer_state": live_optimizer_state,
                "live_optimizer_initialized": bool(optimizer_state),
                "live_learning_rate": float(engine.handle.optimizer.param_groups[0]["lr"]),
                "integrity": {"status": "verified" if latest is None or latest.record_hash == run.verified_hash else "unverified",
                              "hash": latest.record_hash if latest else Recorder.GENESIS_HASH},
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            return json_safe({"workspace": copy.deepcopy(self.workspace),
                              "workspace_epoch": self.workspace_epoch,
                              "runs": [self._run_snapshot(run) for run in self._runs.values()],
                              "autosave_error": self.autosave_error, "recovery_notice": self.recovery_notice,
                              "ui_drafts_revision": self._ui_drafts_revision})

    def graph_action(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Serialize graph admission with project replacement; workers run outside this lock."""
        with self._lock:
            self._ensure_open()
            result = self.graphs.dispatch(action, payload)
            self._changed()
            return result

    def dispatch(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("action payload must be an object")
        with self._lock:
            self._ensure_open()
            result: dict[str, Any] = {"ok": True, "action": action}
            if action == "workspace.drafts":
                if "expected_revision" in payload:
                    expected = _integer(payload["expected_revision"], "expected draft revision", maximum=2**53 - 1)
                    if expected != self._ui_drafts_revision:
                        raise StudioStateError("Editor drafts changed in another tab. Keep your local draft or reload the saved drafts before replacing them.")
                self.ui_drafts = _validate_ui_drafts(payload.get("drafts"))
                self._ui_drafts_revision += 1
                result["revision"] = self._ui_drafts_revision
            elif action == "workspace.update":
                updated = copy.deepcopy(self.workspace)
                for key in ("name", "notes"):
                    if key in payload:
                        updated[key] = _text(payload[key], key, 300 if key == "name" else 100000)
                if "view" in payload:
                    view = payload["view"]
                    if not isinstance(view, dict) or set(view) - {"x", "y", "zoom"}:
                        raise ValueError("view must contain x, y and/or zoom")
                    updated["view"] = {**updated["view"], **{
                        key: _number(value, f"view.{key}") for key, value in view.items()
                    }}
                    if not 0.05 <= updated["view"]["zoom"] <= 10:
                        raise ValueError("view.zoom must be from 0.05 to 10")
                self.workspace = updated
            elif action == "run.create":
                preset = payload.get("preset", "adamw")
                if preset not in PRESETS:
                    raise ValueError("unknown preset")
                config = config_from_dict(payload.get("config", {}), PRESETS[preset])
                custom = _prepare_dataset(config, payload["dataset"]) if "dataset" in payload else None
                if custom is not None:
                    config = custom[0]
                run = self._create(config, payload.get("name", f"{preset} exploration"),
                                   self._position(payload.get("position", {"x": 100, "y": 275})),
                                   arrange="position" not in payload)
                if custom is not None:
                    _, run.engine.dataset, run.dataset_provenance = custom
                    run.engine.recorder.emit(0, "DATASET_IMPORTED", json.dumps(run.dataset_provenance))
                result["run_id"] = run.id
            else:
                run = self._get(payload)
                result["run_id"] = run.id
                if action == "run.select":
                    self.workspace["active_run_id"] = run.id
                elif action == "run.update":
                    name = _text(payload.get("name", run.name), "run name", 300)
                    notes = _text(payload.get("notes", run.notes), "run notes")
                    position = self._position(payload.get("position", run.position))
                    run.name, run.notes, run.position = name, notes, position
                elif action == "run.delete":
                    run.controller.stop()
                    del self._runs[run.id]
                    if self.workspace["active_run_id"] == run.id:
                        self.workspace["active_run_id"] = next(iter(self._runs), None)
                elif action == "run.play":
                    run.controller.start()
                    run.controller.play()
                elif action == "run.pause":
                    run.controller.pause()
                elif action == "run.speed":
                    speed = _number(payload.get("value", payload.get("speed")), "speed", minimum=0.01)
                    if speed > 10000:
                        raise ValueError("speed must be <= 10000 steps/s")
                    run.controller.set_speed(speed)
                elif action == "run.step":
                    count = _integer(payload.get("count", 1), "count", minimum=1, maximum=10000)
                    run.controller.pause()
                    with run.controller._lock:
                        run.controller.raise_if_failed()
                        before = run.engine.step_index
                        for _ in range(count):
                            if run.engine.step() is None:
                                break
                        run.controller._set_state(
                            RunState.ABORTED if run.engine.aborted else (
                                RunState.FINISHED if run.engine.finished else RunState.PAUSED
                            )
                        )
                        result["steps_executed"] = run.engine.step_index - before
                elif action in ("run.reset", "run.restart"):
                    run.controller.pause()
                    with run.controller._lock:
                        if action == "run.reset":
                            run.controller.reset()
                        else:
                            retained_data = run.engine.dataset if run.dataset_provenance.get("kind") != "generated" else None
                            run.controller.restart_experiment()
                            if retained_data is not None:
                                run.engine.dataset = retained_data
                                run.engine.recorder.emit(0, "DATASET_IMPORTED", json.dumps(run.dataset_provenance))
                            run.interventions = []
                            run.verified_hash = Recorder.GENESIS_HASH
                        run.error = None
                elif action == "run.configure":
                    config = config_from_dict(payload.get("config", {}), run.engine.config)
                    run.controller.pause()
                    previous_data_config = run.engine.config.to_dict()["dataset"]
                    updated_data_config = config.to_dict()["dataset"]
                    retain_custom_data = run.dataset_provenance.get("kind") != "generated" and all(
                        previous_data_config[key] == value
                        for key, value in updated_data_config.items() if key != "batch_size"
                    )
                    branch = self._create(config, payload.get("name", f"{run.name} · configured"),
                                          {"x": run.position["x"] + 360, "y": run.position["y"]},
                                          parent_id=run.id, arrange=True)
                    if retain_custom_data:
                        with run.controller._lock:
                            source = run.engine.dataset
                            branch.engine.dataset = Dataset(
                                source.inputs.clone(), source.targets.clone(),
                                math.ceil(len(source.inputs) / config.dataset.batch_size),
                                source.walk, source.shuffle_permutation, source.target_true_parameter,
                                source.noise_std, config.dataset.batch_size,
                            )
                        branch.dataset_provenance = copy.deepcopy(run.dataset_provenance)
                        branch.dataset_provenance["reused_from_run"] = run.id
                        branch.engine.recorder.emit(
                            0, "DATASET_IMPORTED",
                            f"Retained actual custom data from run {run.id}; batch_size={config.dataset.batch_size}",
                        )
                    result["run_id"] = branch.id
                    result["dataset_mode"] = "retained_custom" if retain_custom_data else "generated"
                elif action in ("run.fork", "checkpoint.restore"):
                    run.controller.pause()
                    checkpoint_id = payload.get("checkpoint_id")
                    if action == "checkpoint.restore" and not checkpoint_id:
                        raise ValueError("checkpoint_id is required")
                    if checkpoint_id:
                        checkpoint = self._checkpoint(run, checkpoint_id)
                        captured = checkpoint["state"]
                    else:
                        captured = self._capture(run)
                    branch = self._branch(run, captured, payload.get("name", f"{run.name} · branch"))
                    result["run_id"] = branch.id
                elif action == "checkpoint.create":
                    run.controller.pause()
                    if len(run.checkpoints) >= 100:
                        raise ValueError("a run can contain at most 100 checkpoints")
                    checkpoint = {"id": _id(), "name": _text(payload.get("name", f"Step {run.engine.step_index}"), "checkpoint name", 300),
                                  "step": run.engine.step_index, "created_at": _now()}
                    run.engine.recorder.emit(run.engine.step_index, "CHECKPOINT_CREATED", checkpoint["name"])
                    checkpoint["state"] = self._capture(run)
                    run.checkpoints.append(checkpoint)
                    result["checkpoint_id"] = checkpoint["id"]
                elif action == "checkpoint.delete":
                    checkpoint = self._checkpoint(run, payload.get("checkpoint_id"))
                    run.checkpoints.remove(checkpoint)
                elif action == "run.intervene":
                    self._intervene(run, payload)
                elif action == "run.dataset":
                    result["run_id"] = self._import_dataset(run, payload).id
                elif action == "run.verify":
                    with run.controller._lock:
                        ok, bad_step = run.engine.recorder.verify_chain()
                        if ok:
                            run.verified_hash = run.engine.recorder.records[-1].record_hash if len(run.engine.recorder) else Recorder.GENESIS_HASH
                        result["integrity"] = {"status": "verified" if ok else "corrupt", "first_bad_step": bad_step,
                                               "hash": run.engine.recorder.records[-1].record_hash if len(run.engine.recorder) else Recorder.GENESIS_HASH}
                else:
                    raise ValueError(f"unknown studio action: {action}")
            self._changed()
            return json_safe(result)

    @staticmethod
    def _position(value: Any) -> dict[str, float]:
        if not isinstance(value, dict) or set(value) != {"x", "y"}:
            raise ValueError("position must contain x and y")
        return {key: _number(value[key], f"position.{key}") for key in ("x", "y")}

    @staticmethod
    def _checkpoint(run: _Run, checkpoint_id: Any) -> dict[str, Any]:
        for item in run.checkpoints:
            if item["id"] == checkpoint_id:
                return item
        raise ValueError("the requested checkpoint does not exist")

    def _capture(self, run: _Run) -> dict[str, Any]:
        with run.controller._lock:
            engine = run.engine
            engine.recorder.assert_chain()
            state = {
                "model": engine.model.state_dict(), "gradient": engine.parameter.grad,
                "optimizer": engine.handle.optimizer.state_dict(),
                "scheduler": engine._lr_scheduler.state_dict() if engine._lr_scheduler is not None else None,
                "reference": None if engine.fp64_track is None else {
                    "model": engine.fp64_track.model.state_dict(),
                    "optimizer": engine.fp64_track.optimizer.state_dict(),
                    "gradient": engine.fp64_track.parameter.grad,
                },
                "dataset": engine.dataset,
                "records": list(engine.recorder.records), "events": list(engine.recorder.events.entries),
                "landscapes": engine.recorder.landscapes, "ledger": engine.recorder.ledger,
                "recorder_provenance": engine.recorder.provenance,
                "analyzer": vars(engine.analyzer),
                "misc": {key: getattr(engine, key) for key in _MISC},
            }
            packed = _pack(state)
            return {"config": engine.config.to_dict(), "state": packed,
                    "state_hash": _digest(packed), "dataset_provenance": copy.deepcopy(run.dataset_provenance),
                    "interventions": copy.deepcopy(run.interventions), "speed": run.controller.speed}

    def _restore(self, captured: dict[str, Any]) -> tuple[TrainingEngine, dict[str, Any]]:
        if not isinstance(captured, dict) or set(captured) != {
            "config", "state", "state_hash", "dataset_provenance", "interventions", "speed",
        }:
            raise ValueError("saved run state is incomplete")
        if not isinstance(captured["dataset_provenance"], dict) or not isinstance(captured["interventions"], list):
            raise ValueError("invalid dataset or intervention provenance")
        if captured["dataset_provenance"].get("kind") not in ("generated", "csv", "arrays"):
            raise ValueError("invalid dataset provenance kind")
        for entry in captured["interventions"]:
            if not isinstance(entry, dict) or set(entry) != {
                "at_step", "created_at", "before", "changes", "parameter_after", "learning_rate_after",
            }:
                raise ValueError("invalid intervention provenance")
            _integer(entry["at_step"], "intervention step")
            _text(entry["created_at"], "intervention timestamp", 100)
            if not isinstance(entry["before"], dict) or not isinstance(entry["changes"], dict):
                raise ValueError("invalid intervention detail")
            _number(entry["parameter_after"], "intervention parameter")
            _number(entry["learning_rate_after"], "intervention rate", minimum=0)
        config = config_from_dict(captured["config"])
        if captured.get("state_hash") != _digest(captured["state"]):
            raise ValueError("saved engine state hash does not match")
        state = _unpack(captured["state"])
        if not isinstance(state, dict) or set(state) != {
            "model", "gradient", "optimizer", "scheduler", "reference", "dataset", "records",
            "events", "landscapes", "ledger", "recorder_provenance", "analyzer", "misc",
        }:
            raise ValueError("saved engine state is incomplete")
        engine = TrainingEngine(config)
        records = state["records"]
        if not isinstance(records, list) or len(records) > config.steps:
            raise ValueError("saved history length does not match the configuration")
        recorder = Recorder(config)
        for record in records:
            if not isinstance(record, ParameterStepRecord):
                raise ValueError("saved history contains an invalid record")
            if not record.record_hash or record.record_hash != record.compute_hash():
                raise ValueError(f"saved record {record.step} failed its hash")
            recorder.record(record)
        recorder.assert_chain()
        if not isinstance(state["events"], list) or any(not isinstance(item, EventEntry) for item in state["events"]):
            raise ValueError("invalid saved events")
        recorder.events._entries = state["events"]
        if not isinstance(state["ledger"], vd.ValidationLedger):
            raise ValueError("invalid validation ledger")
        recorder.ledger = state["ledger"]
        recorder._landscapes = state["landscapes"]
        recorder.provenance = state["recorder_provenance"]
        misc = state["misc"]
        if not isinstance(misc, dict) or set(misc) != set(_MISC):
            raise ValueError("invalid engine counters")
        if type(misc["step_index"]) is not int or misc["step_index"] != len(records):
            raise ValueError("saved engine step does not match its sealed history")
        _integer(misc["_live_step_index"], "live step", maximum=config.steps)
        if misc["_live_step_index"] > misc["step_index"]:
            raise ValueError("live step exceeds recorded step count")
        if type(misc["aborted"]) is not bool or type(misc["_variant_locked"]) is not bool:
            raise ValueError("invalid saved engine flags")
        for key in ("abort_reason", "_active_variant"):
            if misc[key] is not None and not isinstance(misc[key], str):
                raise ValueError(f"invalid saved {key}")
        for key in ("_semantics_probe", "_provenance_cache"):
            if misc[key] is not None and not isinstance(misc[key], dict):
                raise ValueError(f"invalid saved {key}")
        if misc["_bit_validation"] is not None and not isinstance(misc["_bit_validation"], vd.BitValidation):
            raise ValueError("invalid bit validation state")
        if misc["_ulp_validation"] is not None and not isinstance(misc["_ulp_validation"], vd.ULPValidation):
            raise ValueError("invalid ULP validation state")
        dataset = state["dataset"]
        if not isinstance(dataset, Dataset):
            raise ValueError("invalid saved dataset")
        if (dataset.inputs.dtype != torch.float32 or dataset.targets.dtype != torch.float32
                or dataset.inputs.shape != (config.dataset.samples, 1)
                or dataset.targets.shape != dataset.inputs.shape
                or not torch.isfinite(dataset.inputs).all() or not torch.isfinite(dataset.targets).all()
                or dataset.batch_size != config.dataset.batch_size
                or dataset.steps_per_epoch != math.ceil(config.dataset.samples / dataset.batch_size)):
            raise ValueError("saved dataset does not match its configuration")
        if dataset.shuffle_permutation is not None and sorted(dataset.shuffle_permutation) != list(range(config.dataset.samples)):
            raise ValueError("invalid saved data permutation")
        self._load_model(engine.model, state["model"])
        engine.parameter.grad = state["gradient"]
        self._load_optimizer(engine.handle.optimizer, state["optimizer"])
        if engine._lr_scheduler is not None:
            if not isinstance(state["scheduler"], dict):
                raise ValueError("missing scheduler state")
            engine._lr_scheduler.load_state_dict(state["scheduler"])
        elif state["scheduler"] is not None:
            raise ValueError("unexpected scheduler state")
        reference = state["reference"]
        if engine.fp64_track is not None:
            if not isinstance(reference, dict) or set(reference) != {"model", "optimizer", "gradient"}:
                raise ValueError("missing reference track state")
            self._load_model(engine.fp64_track.model, reference["model"])
            engine.fp64_track.parameter.grad = reference["gradient"]
            self._load_optimizer(engine.fp64_track.optimizer, reference["optimizer"])
        elif reference is not None:
            raise ValueError("unexpected reference track")
        engine.dataset = dataset
        engine.recorder = recorder
        for key, value in misc.items():
            setattr(engine, key, value)
        recorder.aborted_reason = engine.abort_reason
        analyzer = ConvergenceAnalyzer(config.stability)
        if not isinstance(state["analyzer"], dict) or set(state["analyzer"]) != set(vars(analyzer)):
            raise ValueError("invalid saved analysis state")
        analysis = state["analyzer"]
        if analysis["criteria"] != config.stability or not isinstance(analysis["metrics"], list):
            raise ValueError("saved analysis criteria or metrics are invalid")
        if any(not isinstance(item, StepStability) for item in analysis["metrics"]) or len(analysis["metrics"]) > len(records):
            raise ValueError("invalid saved stability metrics")
        for key in ("_updates", "_gradients", "_losses", "_parameters"):
            if (not isinstance(analysis[key], list) or len(analysis[key]) != len(analysis["metrics"])
                    or any(type(item) not in (int, float) for item in analysis[key])):
                raise ValueError("invalid saved analysis observations")
        if not isinstance(analysis["_nonfinite_steps"], list) or any(
            type(item) is not int or not 0 <= item < len(records) for item in analysis["_nonfinite_steps"]
        ):
            raise ValueError("invalid nonfinite observation indices")
        for key in ("_last_gradient_sign", "_last_update_sign"):
            if type(analysis[key]) is not int or analysis[key] not in (-1, 0, 1):
                raise ValueError("invalid saved analysis sign")
        for key in ("_gradient_sign_changes", "_update_sign_changes"):
            _integer(analysis[key], key, maximum=len(records))
        analyzer.__dict__.update(state["analyzer"])
        engine.analyzer = analyzer
        return engine, captured

    @staticmethod
    def _load_model(model: torch.nn.Module, state: Any) -> None:
        expected = model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("model state does not match the experiment")
        for key, tensor in state.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape
                    or tensor.dtype != expected[key].dtype):
                raise ValueError("model tensor shape or dtype does not match the experiment")
        for key in set(state) - {"w"}:
            if not torch.equal(state[key], expected[key]):
                raise ValueError("model buffers do not match the configuration")
        model.load_state_dict(state, strict=True)

    @staticmethod
    def _load_optimizer(optimizer: torch.optim.Optimizer, state: Any) -> None:
        if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
            raise ValueError("invalid optimizer state")
        if not isinstance(state["param_groups"], list) or len(state["param_groups"]) != 1:
            raise ValueError("optimizer must have one parameter group")
        group = state["param_groups"][0]
        if not isinstance(group, dict) or group.get("params") != [0]:
            raise ValueError("optimizer must refer to the one tracked parameter")
        expected = optimizer.state_dict()["param_groups"][0]
        if set(group) != set(expected):
            raise ValueError("optimizer options do not match the selected optimizer")
        for key, value in group.items():
            if key in ("params", "initial_lr"):
                continue
            if key == "lr":
                _number(value, "saved learning rate", minimum=0)
            elif value != expected[key] or type(value) is not type(expected[key]):
                raise ValueError(f"optimizer option {key} disagrees with the configuration")
        if not isinstance(state["state"], dict) or set(state["state"]) - {0}:
            raise ValueError("invalid optimizer parameter state")
        for values in state["state"].values():
            if not isinstance(values, dict) or set(values) - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}:
                raise ValueError("invalid optimizer state fields")
            if any(not isinstance(value, torch.Tensor) or value.shape != torch.Size([]) for value in values.values()):
                raise ValueError("optimizer state values must be scalar tensors")
            required = {"momentum_buffer"} if isinstance(optimizer, torch.optim.SGD) else {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad"):
                required.add("max_exp_avg_sq")
            if set(values) != required:
                raise ValueError("optimizer memory is incomplete")
        optimizer.load_state_dict(state)

    def _branch(self, parent: _Run, captured: dict[str, Any], name: str) -> _Run:
        engine, metadata = self._restore(captured)
        run = self._create(engine.config, name,
                           {"x": parent.position["x"] + 360, "y": parent.position["y"] + 70},
                           parent_id=parent.id, arrange=True)
        run.engine = run.controller.engine = engine
        run.controller.set_speed(metadata["speed"])
        run.controller._set_state(RunState.PAUSED)
        run.dataset_provenance = copy.deepcopy(metadata["dataset_provenance"])
        run.interventions = copy.deepcopy(metadata["interventions"])
        run.engine.recorder.emit(run.engine.step_index, "BRANCH_CREATED", f"Exact state continued from run {parent.id}")
        return run

    def _intervene(self, run: _Run, payload: dict[str, Any]) -> None:
        allowed = {"parameter_value", "parameter_hex", "learning_rate", "optimizer_state"}
        if not allowed.intersection(payload):
            raise ValueError("supply parameter_value, parameter_hex, learning_rate or optimizer_state")
        if "parameter_value" in payload and "parameter_hex" in payload:
            raise ValueError("supply one parameter representation")
        parameter = None
        if "parameter_hex" in payload:
            word = payload["parameter_hex"]
            if not isinstance(word, str) or not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{8}", word):
                raise ValueError("parameter_hex must contain exactly 8 hexadecimal digits")
            parameter = struct.unpack("!f", struct.pack("!I", int(word, 16)))[0]
            if not math.isfinite(parameter):
                raise ValueError("manual parameter must be finite")
        elif "parameter_value" in payload:
            parameter = _number(payload["parameter_value"], "parameter value")
            if abs(parameter) > torch.finfo(torch.float32).max:
                raise ValueError("parameter must fit finite float32 storage")
        rate = None
        if "learning_rate" in payload:
            rate = _number(payload["learning_rate"], "learning rate", minimum=1e-45)
            run.engine.config.replace(lr=rate)
        edits = payload.get("optimizer_state", {})
        if not isinstance(edits, dict):
            raise ValueError("optimizer_state must be an object")
        supported = {"momentum_buffer"} if run.engine.config.optimizer == "sgd" else {"step", "exp_avg", "exp_avg_sq"}
        if run.engine.config.amsgrad:
            supported.add("max_exp_avg_sq")
        if set(edits) - supported:
            raise ValueError("unknown state field for this optimizer")
        edits = {key: _number(value, f"optimizer_state.{key}", minimum=0 if key in ("step", "exp_avg_sq", "max_exp_avg_sq") else None)
                 for key, value in edits.items()}
        if "step" in edits and not edits["step"].is_integer():
            raise ValueError("optimizer step must be an integer")
        if any(abs(value) > torch.finfo(torch.float32).max for value in edits.values()):
            raise ValueError("optimizer state values must fit float32")
        run.controller.pause()
        with run.controller._lock:
            engine = run.engine
            if edits and engine.config.optimizer != "sgd" and not engine.handle.optimizer.state:
                required = {"step", "exp_avg", "exp_avg_sq"}
                if engine.config.amsgrad:
                    required.add("max_exp_avg_sq")
                if set(edits) != required:
                    raise ValueError("initializing Adam state requires every optimizer state field")
            before = {"parameter": float(engine.parameter.detach().item()), "lr": engine.handle.current_lr()}
            with torch.no_grad():
                if parameter is not None:
                    engine.parameter.fill_(parameter)
                    engine.parameter.grad = None
                    if engine.fp64_track is not None:
                        engine.fp64_track.parameter.fill_(float(engine.parameter.item()))
                        engine.fp64_track.parameter.grad = None
                if rate is not None:
                    engine.config = engine.config.replace(lr=rate)
                    engine.recorder.config = engine.config
                    engine.handle.set_lr(rate)
                    if engine.fp64_track is not None:
                        engine.fp64_track.handle.set_lr(rate)
                    if engine._lr_scheduler is not None:
                        engine._lr_scheduler.base_lrs = [rate]
                        engine._lr_scheduler.eta_min = rate * engine.config.lr_final_factor
                        engine._lr_scheduler._last_lr = [rate]
                for track_parameter, optimizer in [(engine.parameter, engine.handle.optimizer)] + (
                    [(engine.fp64_track.parameter, engine.fp64_track.optimizer)] if engine.fp64_track is not None else []
                ):
                    if edits:
                        state = optimizer.state[track_parameter]
                        for key, value in edits.items():
                            dtype = torch.float32 if key == "step" else track_parameter.dtype
                            state[key] = torch.tensor(value, dtype=dtype)
            entry = {"at_step": engine.step_index, "created_at": _now(), "before": before,
                     "changes": {key: payload[key] for key in allowed if key in payload},
                     "parameter_after": float(engine.parameter.item()), "learning_rate_after": engine.handle.current_lr()}
            run.interventions.append(entry)
            engine._provenance_cache = None
            engine.recorder.emit(engine.step_index, "MANUAL_INTERVENTION", json.dumps(entry, allow_nan=False))

    def _import_dataset(self, run: _Run, payload: dict[str, Any]) -> _Run:
        config, dataset, provenance = _prepare_dataset(run.engine.config, {
            key: value for key, value in payload.items() if key != "run_id"
        })
        run.controller.pause()
        run = self._create(config, f"{run.name} · custom data",
                           {"x": run.position["x"] + 360, "y": run.position["y"] + 70},
                           parent_id=run.id, arrange=True)
        with run.controller._lock:
            run.engine.dataset = dataset
            run.dataset_provenance = provenance
            run.engine.recorder.emit(0, "DATASET_IMPORTED", json.dumps(provenance))
        return run

    def record(self, run_id: str, step: int) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            run = self._get(run_id)
            _integer(step, "step")
            with run.controller._lock:
                return run.engine.recorder.record_at(step).to_dict()

    def dataset(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            run = self._get(run_id)
            with run.controller._lock, torch.no_grad():
                engine = run.engine
                data = engine.dataset
                description = dict(data.describe())
                if run.dataset_provenance.get("kind") != "generated":
                    description.update({"target_true_parameter": None, "noise_std": None,
                                        "target_origin": "user supplied targets; no known optimum is asserted"})
                predictions = engine.model(data.inputs).reshape(-1).tolist()
                centre = float(engine.parameter.item())
                if not math.isfinite(centre):
                    centre = engine.config.initial_value
                radius = max(2.0, abs(centre - engine.config.dataset.target_true_parameter) * 1.2)
                probes = torch.linspace(centre - radius, centre + radius, 121, dtype=torch.float64)
                losses = [engine.model.loss_fp64_python(data.inputs, data.targets, float(value)) for value in probes]
                return json_safe({"inputs": data.inputs.reshape(-1).tolist(), "targets": data.targets.reshape(-1).tolist(),
                                  "predictions": predictions, "provenance": copy.deepcopy(run.dataset_provenance),
                                  "description": description,
                                  "landscape": {"parameters": probes.tolist(), "losses": losses,
                                                "objective": "full dataset mean squared error, float64 diagnostic"}})

    def export_project(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            return self._project_data()

    def _project_data(self) -> dict[str, Any]:
        """Capture under _lock, including the final save after admission closes."""
        return {"schema_version": PROJECT_SCHEMA, "application_version": __version__, "saved_at": _now(),
                "extensions": self.graphs.export_state(),
                "ui_drafts": copy.deepcopy(self.ui_drafts),
                "workspace": copy.deepcopy(self.workspace), "runs": [{
                    "id": run.id, "name": run.name, "parent_id": run.parent_id,
                    "position": dict(run.position), "notes": run.notes,
                    "created_at": run.created_at, "error": run.error,
                    "engine": self._capture(run), "checkpoints": copy.deepcopy(run.checkpoints),
                } for run in self._runs.values()]}

    def import_project(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            if not isinstance(data, dict) or data.get("schema_version") != PROJECT_SCHEMA:
                raise ValueError(f"project schema must be {PROJECT_SCHEMA}")
            extension_state = GraphStudio.validate_state(data.get("extensions"))
            editor_drafts = _validate_ui_drafts(data.get("ui_drafts"))
            if self.graphs.is_busy():
                raise StudioStateError("Cancel or finish the active node-graph jobs before opening another project")
            workspace = copy.deepcopy(data.get("workspace"))
            if not isinstance(workspace, dict) or set(workspace) != {"id", "name", "notes", "active_run_id", "view", "revision"}:
                raise ValueError("invalid workspace metadata")
            for key in ("id", "name", "notes"):
                _text(workspace[key], f"workspace.{key}", 100000 if key == "notes" else 300)
            _integer(workspace["revision"], "revision", maximum=2**53 - 1)
            view = workspace["view"]
            if not isinstance(view, dict) or set(view) != {"x", "y", "zoom"}:
                raise ValueError("invalid workspace view")
            for key in view:
                _number(view[key], f"view.{key}")
            if not 0.05 <= view["zoom"] <= 10:
                raise ValueError("invalid workspace zoom")
            incoming = data.get("runs")
            if not isinstance(incoming, list) or len(incoming) > 100:
                raise ValueError("project must contain at most 100 runs")
            staged: dict[str, _Run] = {}
            for item in incoming:
                if not isinstance(item, dict) or set(item) != {
                    "id", "name", "parent_id", "position", "notes", "created_at", "error", "engine", "checkpoints",
                }:
                    raise ValueError("invalid run metadata")
                run_id = _identity(item["id"], "run id")
                if not run_id or run_id in staged:
                    raise ValueError("run IDs must be nonempty and unique")
                name = _text(item["name"], "run name", 300)
                notes = _text(item["notes"], "run notes")
                position = self._position(item["position"])
                parent_id = item["parent_id"]
                if parent_id is not None:
                    _identity(parent_id, "parent id")
                engine, captured = self._restore(item["engine"])
                checkpoints = copy.deepcopy(item["checkpoints"])
                if not isinstance(checkpoints, list) or len(checkpoints) > 100:
                    raise ValueError("invalid checkpoint collection")
                checkpoint_ids = set()
                for checkpoint in checkpoints:
                    if not isinstance(checkpoint, dict) or set(checkpoint) != {"id", "name", "step", "created_at", "state"}:
                        raise ValueError("invalid checkpoint metadata")
                    identity = _identity(checkpoint["id"], "checkpoint id")
                    if identity in checkpoint_ids:
                        raise ValueError("checkpoint IDs must be unique")
                    checkpoint_ids.add(identity)
                    _text(checkpoint["name"], "checkpoint name", 300)
                    _text(checkpoint["created_at"], "checkpoint date", 100)
                    checkpoint_engine, _metadata = self._restore(checkpoint["state"])
                    if checkpoint["step"] != checkpoint_engine.step_index:
                        raise ValueError("checkpoint step disagrees with its state")
                controller = ExperimentController(engine)
                speed = _number(captured["speed"], "saved speed", minimum=0.01)
                if speed > 10000:
                    raise ValueError("saved speed exceeds the studio limit")
                controller.set_speed(speed)
                controller._set_state(RunState.PAUSED)
                controller.on_step = lambda _record: self._dirty.set()
                created_at = _text(item["created_at"], "creation date", 100)
                if item.get("error") is not None:
                    _text(item["error"], "saved error")
                staged[run_id] = _Run(run_id, name, engine, controller, parent_id, position, notes, checkpoints,
                                      copy.deepcopy(captured["dataset_provenance"]),
                                      copy.deepcopy(captured["interventions"]), created_at, item.get("error"))
            if workspace["active_run_id"] is not None and workspace["active_run_id"] not in staged:
                raise ValueError("active run does not exist in project")
            for old in self._runs.values():
                old.controller.stop()
            self._runs = staged
            self.workspace = workspace
            self.graphs.import_state(extension_state)
            self.ui_drafts = editor_drafts
            self._ui_drafts_revision += 1
            # This admission token belongs to this live workspace incarnation,
            # not its portable identity. Even reopening the same file invalidates
            # requests that were prepared before the replacement.
            self.workspace_epoch = _id()
            self._dirty.set()
            return {"ok": True, "runs_imported": len(staged), "active_run_id": workspace["active_run_id"]}

    def export_run(self, run_id: str) -> tuple[bytes, str]:
        with self._lock:
            self._ensure_open()
            run = self._get(run_id)
            with run.controller._lock:
                run.engine.recorder.assert_chain()
                with tempfile.TemporaryDirectory(prefix="spobs-studio-export-") as temporary:
                    directory = Path(temporary)
                    provenance = run.engine.provenance()
                    if run.dataset_provenance.get("kind") != "generated":
                        provenance["dataset"] = self.dataset(run_id)["description"]
                    provenance["studio"] = {"dataset": run.dataset_provenance, "interventions": run.interventions,
                                            "manual_edits_are_optimizer_steps": False}
                    if run.interventions:
                        provenance["notes"] = ["This studio run includes explicit manual edits between optimizer steps; see studio.interventions and MANUAL_INTERVENTION events."]
                    export_all(run.engine.recorder, str(directory), provenance=provenance, analyzer=run.engine.analyzer)
                    (directory / "checkpoint.json").write_text(json.dumps(self._capture(run), allow_nan=False), encoding="utf-8")
                    (directory / "dataset.json").write_text(json.dumps(self.dataset(run_id), allow_nan=False), encoding="utf-8")
                    output = io.BytesIO()
                    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                        for path in sorted(directory.iterdir()):
                            if path.is_file():
                                archive.write(path, path.name)
                    return output.getvalue(), f"spobs-run-{run.id[:12]}.zip"

    def _save(self) -> None:
        if self.storage_directory is None:
            return
        if self._autosave_blocked:
            raise StudioStateError("Autosave replacement is blocked to preserve the unreadable original; download a project copy")
        with self._lock:
            if self._closed:
                raise StudioStateError("the studio is closed")
            data = self._project_data()
            content = json.dumps(data, allow_nan=False, separators=(",", ":"))
            descriptor, temporary = tempfile.mkstemp(prefix=".autosave-", suffix=".json", dir=self.storage_directory)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.storage_directory / "autosave.json")
                self.autosave_error = None
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def _autosave_loop(self) -> None:
        while not self._stopping.wait(2.0):
            if self._dirty.is_set():
                self._dirty.clear()
                try:
                    self._save()
                except Exception as exc:
                    self.autosave_error = f"Autosave failed: {exc}"
                    self._dirty.set()

    def close(self) -> None:
        with self._close_lock:
            # Serialize admission with any project replacement already holding
            # the workspace lock. Graph workers need that lock for host calls,
            # so joining them must happen after releasing this admission gate.
            with self._lock:
                if self._closed:
                    return
                self._closing = True
                self._stopping.set()
            failures: list[BaseException] = []
            try:
                self.graphs.close()
            except BaseException as exc:
                failures.append(exc)
            if self._autosave_thread is not None and self._autosave_thread is not threading.current_thread():
                self._autosave_thread.join(timeout=5)
                if self._autosave_thread.is_alive():
                    failures.append(RuntimeError("the autosave worker did not stop within 5 seconds"))
            with self._lock:
                # A graph shutdown or one failed controller must not skip the
                # other controllers, final persistence, or the closed marker.
                for run in self._runs.values():
                    try:
                        run.controller.stop()
                    except BaseException as exc:
                        failures.append(exc)
                try:
                    self._save()
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    self._closed = True
            if failures:
                raise failures[0]


__all__ = ["StudioWorkspace", "StudioStateError", "StudioEpochError", "PROJECT_SCHEMA", "config_from_dict", "config_for_browser"]
