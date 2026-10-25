"""The trajectory recorder: immutable per-step records, event log, final report.

Design rules, all of which exist to make the record trustworthy:
* records are **frozen**; once a step has been recorded, nothing can rewrite it;
* raw observations and derived analysis are **separate fields**, so a derived
  number never replaces the raw one it came from;
* each record carries a **hash of the previous record**, forming a chain.  If a
  stored history is edited after the fact, :func:`Recorder.verify_chain` reports
  exactly where the chain stops matching;
* every value is a Python ``float``/``int``/``str``/``bool``, never a live
  tensor, so a recorded history cannot be affected by later training.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, get_args, get_origin, get_type_hints
from collections.abc import Iterable, Mapping, Sequence

from .config import ExperimentConfig
from .errors import HistoryCorruptionError, NumericalConsistencyError
from .model import TensorSnapshot
from .validation import ValidationLedger

#: Event codes emitted from real program state.  Each is raised by the engine at
#: the moment the corresponding condition is actually observed.
EVENT_CODES: tuple[str, ...] = (
    "INITIALIZED",
    "FORWARD_COMPLETE",
    "LOSS_COMPUTED",
    "BACKWARD_STARTED",
    "GRADIENT_AVAILABLE",
    "GRADIENT_ACCUMULATED",
    "OPTIMIZER_STATE_READ",
    "OPTIMIZER_STEP_STARTED",
    "OPTIMIZER_STEP_COMPLETED",
    "PARAMETER_CHANGED",
    "PARAMETER_UNCHANGED",
    "UPDATE_ATTEMPTED_BELOW_FP32_RESOLUTION",
    "BIT_PATTERN_CHANGED",
    "BIT_PATTERN_UNCHANGED",
    "FP32_ROUNDING_OBSERVED",
    "LANDSCAPE_PROBED",
    "VALIDATION_FAILED",
    "STABILITY_METRIC_UPDATED",
    "NONFINITE_DETECTED",
    "RUN_COMPLETE",
    "RUN_ABORTED",
    "MANUAL_INTERVENTION",
    "DATASET_IMPORTED",
    "CHECKPOINT_CREATED",
    "BRANCH_CREATED",
)

#: Snapshot labels, in the order the specification requires them.
SNAPSHOT_LABELS: tuple[str, ...] = (
    "A: before forward",
    "B: after forward",
    "C: after backward",
    "D: before optimizer.step",
    "E: after optimizer.step",
    "F: after subsequent framework operation",
)


def _jsonable(value: Any) -> Any:
    # Finite floats pass through unchanged - the JSON export is meant to carry
    # real numbers (``export_trajectory_json`` says so on its face), and the
    # original fall-through to ``repr`` silently stringified every finite float
    # in the trajectory export while the summary export kept real numbers.
    # Non-finite floats are stringified, so a diverged run still exports as
    # valid JSON instead of a bare `NaN` token.
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return repr(value)


def json_safe(value: Any) -> Any:
    """Public form of the sanitiser every JSON export must pass through.

    ``json.dump(..., allow_nan=False, default=str)`` raises ``ValueError`` on a
    non-finite float: the ``default`` hook only fires for objects json does not
    recognise, and a float is recognised.  Any payload that can contain a raw
    ``inf``/``nan`` - the final report of a diverged run, a validation relative
    error against a zero reference - must therefore be sanitised *before* the
    dump, with non-finite floats rendered as their ``repr`` strings.  Records
    already go through this via ``to_dict``; the report and other plain dicts
    previously bypassed it, which made every diverged run unexportable.
    """
    return _jsonable(value)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """A deterministic string form, used for hashing and for the JSON export."""
    return json.dumps(_jsonable(dict(payload)), sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EventEntry:
    """One thing that actually happened, at one step."""

    step: int
    code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if type(self.step) is not int or not isinstance(self.detail, str):
            raise NumericalConsistencyError("events require an integer step and string detail")
        if self.code not in EVENT_CODES:
            raise NumericalConsistencyError(f"unknown event code: {self.code!r}")


class EventLog:
    """Append-only log of events generated from real program state."""

    def __init__(self) -> None:
        self._entries: list[EventEntry] = []

    def emit(self, step: int, code: str, detail: str) -> EventEntry:
        entry = EventEntry(step=step, code=code, detail=detail)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[EventEntry, ...]:
        return tuple(self._entries)

    def codes_for_step(self, step: int) -> tuple[str, ...]:
        return tuple(entry.code for entry in self._entries if entry.step == step)

    def to_list(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.code] = counts.get(entry.code, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._entries)


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParameterStepRecord:
    """One immutable record of one real training step.

    Fields are grouped by the causal stage they belong to, and raw observations
    are never overwritten by the analysis derived from them.

    Two fields need a word of explanation.  ``events`` lists the event codes
    emitted for this step *up to the moment the record was sealed*; the event log
    itself remains the complete, authoritative sequence.  And when
    ``accumulation_micro_batches`` is greater than one, ``loss`` is the sum of
    the micro-batch losses and ``gradient`` is the corresponding sum autograd
    accumulated. ``accumulated_losses`` lists those losses; ``prediction``,
    input/target and ``stages`` describe the final micro-batch's forward pass.
    """

    # identity
    step: int
    epoch: int
    batch_index: int
    global_batch: int

    # forward (raw)
    input_value: float
    target_value: float
    prediction: float
    loss: float
    loss_dtype: str
    loss_grad_fn: str | None
    stages: tuple[TensorSnapshot, ...]

    # gradient lifecycle (raw observations of p.grad at each moment)
    zero_grad_set_to_none: bool
    accumulation_micro_batches: int
    accumulated_losses: tuple[float, ...]
    grad_before_zero_grad: float | None
    grad_after_zero_grad: float | None
    grad_before_backward: float | None
    grad_after_backward: float | None
    gradient: float
    gradient_uint32: int | None
    gradient_dtype: str
    gradient_shape: tuple[int, ...]
    gradient_sign: int
    gradient_magnitude: float
    gradient_square: float
    gradient_finite: bool

    # parameter (raw)
    parameter_before: float
    parameter_after: float
    parameter_before_uint32: int
    parameter_after_uint32: int
    parameter_before_hex: str
    parameter_hex: str
    parameter_bits: str
    parameter_exponent_bits: int
    parameter_fraction_bits: int
    parameter_sign_bit: int

    # parameter (derived, never replacing the raw values above)
    actual_delta: float
    absolute_delta: float
    relative_delta: float
    direction: str
    xor_mask: int
    changed_bit_count: int
    changed_bit_positions: tuple[int, ...]
    sign_changed: bool
    exponent_changed: bool
    fraction_changed: bool
    ulp_previous_distance: float
    ulp_next_distance: float
    ulp_size: float
    ulp_relative_movement: float
    #: Representable float32 values strictly between before and after.  Exact,
    #: but dominated by the subnormal range when a step crosses zero, so it is
    #: not summed into the run's headline movement figure.
    ulp_distance_count: int
    #: Signed movement in ULPs of the *starting* value.  This is the per-step
    #: movement metric and the one the run aggregate uses.
    ulp_distance_same_binade: int
    stored_fp32_unchanged: bool
    update_attempted_but_storage_frozen: bool

    # snapshot chain ( 19): raw uint32 of the parameter at each labelled instant
    snapshot_a_uint32: int
    snapshot_b_uint32: int
    snapshot_c_uint32: int
    snapshot_d_uint32: int
    snapshot_e_uint32: int
    snapshot_f_uint32: int | None

    # optimizer
    optimizer_name: str
    learning_rate: float
    optimizer_step_before: int | None
    optimizer_step_after: int | None
    optimizer_state_present_before: bool
    optimizer_state_keys_before: tuple[str, ...]
    optimizer_state_keys_after: tuple[str, ...]
    moment1_before: float | None
    moment2_before: float | None
    moment1: float | None
    moment2: float | None
    bias_corrected_moment1: float | None
    bias_corrected_moment2: float | None
    optimizer_total_update: float
    optimizer_attempted_update: float
    optimizer_adaptive_update: float
    optimizer_weight_decay_update: float

    # independent reconstruction
    reconstruction_variant: str
    reconstruction_status: str
    reconstruction_bit_exact: bool
    reconstruction_ulp_error: int
    reconstruction_absolute_error: float
    reconstruction_relative_error: float
    reconstructed_after: float
    reconstructed_uint32: int
    fp64_reconstructed_after: float

    # floating point
    fp32_candidate: float
    fp32_rounding_error_abs: float
    fp32_rounding_error_rel: float
    fp32_rounding_sign: int
    fp32_rounding_ulp: float
    fp32_candidate_rounds_to_stored: bool
    fp64_track_value: float | None
    fp64_track_loss: float | None
    fp64_vs_fp32_abs: float | None
    fp64_vs_fp32_ulp: float | None

    # gradient validation
    gradient_analytical: float | None
    gradient_finite_difference: float | None
    gradient_finite_difference_h: float | None
    gradient_fp64_autograd: float | None
    gradient_vs_analytical_rel: float | None
    gradient_vs_analytical_abs: float | None
    gradient_vs_finite_difference_rel: float | None
    gradient_vs_finite_difference_abs: float | None
    gradient_tolerance_abs: float | None
    finite_difference_tolerance_abs: float | None
    gradient_validation_status: str

    # landscape (diagnostic, never part of the trajectory)
    landscape_computed: bool
    landscape_radius: float
    landscape_points: int
    landscape_loss_at_centre: float | None
    landscape_minimum_loss: float | None
    landscape_first_derivative: float | None
    landscape_second_derivative: float | None
    landscape_restoration_verified: bool

    # run-level bookkeeping for this step
    finite: bool
    nonfinite_detected: bool
    parameter_hash_before: str
    parameter_hash_after: str
    events: tuple[str, ...]
    previous_hash: str
    record_hash: str

    def __post_init__(self) -> None:
        # frozen=True prevents assignment, but does not freeze a caller's list.
        # Own every sequence, including shapes inside frozen TensorSnapshots,
        # and reject live tensors/objects instead of hashing their repr strings.
        annotations = _observation_types(type(self))
        for field in fields(self):
            value = getattr(self, field.name)
            frozen = _freeze_observation(value)
            if not _matches_observation_type(frozen, annotations[field.name]):
                raise NumericalConsistencyError(
                    f"record field {field.name!r} received {type(value).__name__}, expected {field.type}"
                )
            object.__setattr__(self, field.name, frozen)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_row(self) -> dict[str, Any]:
        """A flat mapping for CSV: sequences are JSON-encoded, never truncated."""
        row: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (list, tuple)):
                row[f.name] = json.dumps(_jsonable(list(value)), separators=(",", ":"))
            elif isinstance(value, float):
                # repr round-trips exactly; %.17g would too, but repr is the
                # shortest string that returns the same double.
                row[f.name] = repr(value)
            else:
                row[f.name] = value
        return row

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    def stage_values(self) -> dict[str, float]:
        return {snapshot.name: snapshot.value for snapshot in self.stages}

    def compute_hash(self) -> str:
        """Hash the observations and predecessor, excluding only this hash.

        Including the predecessor binds the entire prefix. Otherwise a suffix
        could be relinked after a different record without changing its hash.
        """
        annotations = _observation_types(type(self))
        if any(not _matches_observation_type(getattr(self, field.name), annotations[field.name])
               for field in fields(self)):
            raise NumericalConsistencyError("record fields no longer match the observation schema")
        payload = self.to_dict()
        payload.pop("record_hash", None)
        return _hash_payload(payload)

    def with_hash(self, previous_hash: str) -> ParameterStepRecord:
        """Return a copy carrying its chain hash (dataclasses.replace)."""
        unlinked = replace(self, record_hash="", previous_hash=previous_hash)
        linked = replace(unlinked, record_hash=unlinked.compute_hash())
        return linked


def _freeze_observation(value: Any) -> Any:
    """Detach the supported observation tree from caller-owned containers."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_observation(item) for item in value)
    if isinstance(value, TensorSnapshot):
        snapshot = replace(value, **{
            field.name: _freeze_observation(getattr(value, field.name))
            for field in fields(value)
        })
        annotations = _observation_types(type(snapshot))
        if any(not _matches_observation_type(getattr(snapshot, field.name), annotations[field.name])
               for field in fields(snapshot)):
            raise NumericalConsistencyError("TensorSnapshot contains an observation of the wrong type")
        return snapshot
    raise NumericalConsistencyError(
        f"record observations must be detached Python scalars, tuples or TensorSnapshots; "
        f"received {type(value).__name__}"
    )


_OBSERVATION_TYPES: dict[type, dict[str, Any]] = {}


def _observation_types(cls: type) -> dict[str, Any]:
    if cls not in _OBSERVATION_TYPES:
        _OBSERVATION_TYPES[cls] = get_type_hints(cls)
    return _OBSERVATION_TYPES[cls]


def _matches_observation_type(value: Any, expected: Any) -> bool:
    if expected is float:
        return type(value) in (int, float)
    if expected in (int, bool, str):
        return type(value) is expected
    if expected is type(None):
        return value is None
    if expected is TensorSnapshot:
        if not isinstance(value, TensorSnapshot):
            return False
        annotations = _observation_types(type(value))
        return all(_matches_observation_type(getattr(value, field.name), annotations[field.name])
                   for field in fields(value))
    arguments = get_args(expected)
    if get_origin(expected) is tuple:
        return isinstance(value, tuple) and all(
            _matches_observation_type(item, arguments[0]) for item in value
        )
    return bool(arguments) and any(_matches_observation_type(value, member) for member in arguments)


# --------------------------------------------------------------------------- #
# recorder
# --------------------------------------------------------------------------- #


class Recorder:
    """Collects records and events, and keeps the hash chain intact."""

    GENESIS_HASH: str = "0" * 64

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.events = EventLog()
        self._records: list[ParameterStepRecord] = []
        self.ledger = ValidationLedger()
        self.provenance: dict[str, Any] = {}
        self._landscapes: list[Mapping[str, Any]] = []
        self.aborted_reason: str | None = None

    # -- writing ------------------------------------------------------------
    def record(self, draft: ParameterStepRecord) -> ParameterStepRecord:
        """Link a draft record into the chain and store it."""
        expected_step = self._records[-1].step + 1 if self._records else 0
        if type(draft.step) is not int or draft.step != expected_step:
            raise HistoryCorruptionError(
                f"expected step {expected_step}, received {draft.step!r}"
            )
        if draft.record_hash and draft.record_hash != draft.compute_hash():
            raise HistoryCorruptionError(f"record for step {draft.step} has an invalid record_hash")
        previous = self._records[-1].record_hash if self._records else self.GENESIS_HASH
        if draft.previous_hash not in ("", previous):
            raise HistoryCorruptionError(
                f"record for step {draft.step} declares previous_hash "
                f"{draft.previous_hash[:12]}... but the chain head is {previous[:12]}..."
            )
        linked = draft.with_hash(previous)
        self._records.append(linked)
        return linked

    def emit(self, step: int, code: str, detail: str) -> None:
        self.events.emit(step, code, detail)

    def add_landscape(self, slice_data: Mapping[str, Any]) -> None:
        self._landscapes.append(deepcopy(dict(slice_data)))

    # -- reading ------------------------------------------------------------
    @property
    def records(self) -> tuple[ParameterStepRecord, ...]:
        return tuple(self._records)

    @property
    def landscapes(self) -> list[Mapping[str, Any]]:
        """Detached diagnostic snapshots; editing a view cannot rewrite them."""
        return deepcopy(self._landscapes)

    def __len__(self) -> int:
        return len(self._records)

    def record_at(self, step: int) -> ParameterStepRecord:
        for record in self._records:
            if record.step == step:
                return record
        raise KeyError(f"no record for step {step}")

    def values(self, attribute: str) -> tuple[float, ...]:
        """One recorded series, in step order."""
        return tuple(float(getattr(r, attribute)) for r in self._records)

    def ints(self, attribute: str) -> tuple[int, ...]:
        return tuple(int(getattr(r, attribute)) for r in self._records)

    def optional_values(self, attribute: str) -> tuple[float | None, ...]:
        return tuple(
            (None if getattr(r, attribute) is None else float(getattr(r, attribute)))
            for r in self._records
        )

    def verify_chain(self) -> tuple[bool, int | None]:
        """Recompute every hash; return ``(ok, first_bad_step)``."""
        previous = self.GENESIS_HASH
        for expected_step, record in enumerate(self._records):
            if record.step != expected_step:
                return False, record.step
            if record.previous_hash != previous:
                return False, record.step
            try:
                computed = record.compute_hash()
            except NumericalConsistencyError:
                return False, record.step
            if computed != record.record_hash:
                return False, record.step
            previous = record.record_hash
        return True, None

    def assert_chain(self) -> None:
        ok, bad_step = self.verify_chain()
        if not ok:
            raise HistoryCorruptionError(
                f"the recorded history failed its hash chain at step {bad_step}"
            )

    # -- export helpers -----------------------------------------------------
    def rows(self) -> list[dict[str, Any]]:
        return [record.to_row() for record in self._records]

    def field_names(self) -> list[str]:
        return list(self._records[0].field_names) if self._records else []


# --------------------------------------------------------------------------- #
# final report
# --------------------------------------------------------------------------- #


def _safe(values: Iterable[float | None], default: float = float("nan")) -> float:
    present = [v for v in values if v is not None and math.isfinite(v)]
    return present[0] if present else default


def _max_abs(values: Iterable[float | None]) -> float:
    present = [abs(v) for v in values if v is not None and not math.isnan(v)]
    return max(present) if present else float("nan")


def _min_present(values: Iterable[float | None]) -> float:
    """Smallest defined observation, retaining infinities; NaN if absent."""
    present = [v for v in values if v is not None and not math.isnan(v)]
    return min(present) if present else float("nan")


def _mean_abs(values: Iterable[float | None]) -> float:
    present = [abs(v) for v in values if v is not None and not math.isnan(v)]
    if not present:
        return float("nan")
    if any(math.isinf(value) for value in present):
        return float("inf")
    scale = max(present)
    return (sum(value / scale for value in present) / len(present)) * scale if scale else 0.0


def _std(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    scale = max(abs(value) for value in finite)
    if scale == 0.0:
        return 0.0
    scaled = [value / scale for value in finite]
    mean = math.fsum(scaled) / len(scaled)
    variance = math.fsum((value - mean) ** 2 for value in scaled) / (len(scaled) - 1)
    return math.sqrt(variance) * scale


def _sign_changes(values: Sequence[float | None]) -> int:
    signs = [
        (1 if v > 0 else -1 if v < 0 else 0)
        for v in values
        if v is not None and math.isfinite(v) and v != 0.0
    ]
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)


def build_final_report(
    recorder: Recorder,
    *,
    stability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The end-of-run numerical report, measured entirely from the records."""
    records = recorder.records
    if not records:
        raise NumericalConsistencyError("cannot build a final report from zero records")
    first, last = records[0], records[-1]
    losses = [r.loss for r in records]
    gradients = [r.gradient for r in records]
    updates = [r.actual_delta for r in records]
    window = max(1, min(len(records), recorder.config.stability.window))
    tail_parameters = [r.parameter_after for r in records[-window:]]
    nonfinite_steps = [r.step for r in records if r.nonfinite_detected]
    unchanged = [r.step for r in records if r.stored_fp32_unchanged]
    below_resolution = [
        r.step for r in records if r.update_attempted_but_storage_frozen
    ]

    report: dict[str, Any] = {
        "run_name": recorder.config.name,
        "optimizer": first.optimizer_name,
        "steps": len(records),
        "initial_parameter": first.parameter_before,
        "initial_parameter_hex": first.parameter_before_hex,
        "initial_loss": first.loss,
        "initial_gradient": first.gradient,
        "final_parameter": last.parameter_after,
        "final_parameter_hex": last.parameter_hex,
        "final_loss": last.loss,
        "final_gradient": last.gradient,
        "total_parameter_change": last.parameter_after - first.parameter_before,
        "minimum_loss": _min_present(losses),
        "maximum_loss": _max_abs(losses),
        "max_abs_gradient": _max_abs(gradients),
        "max_abs_update": _max_abs(updates),
        "mean_abs_update": _mean_abs(updates),
        "std_parameter_over_final_window": _std(tail_parameters),
        "final_window_size": window,
        "gradient_sign_changes": _sign_changes(gradients),
        "update_sign_changes": _sign_changes(updates),
        "unchanged_fp32_steps": len(unchanged),
        "unchanged_fp32_step_indices": unchanged,
        "update_attempted_below_resolution_steps": len(below_resolution),
        "update_attempted_below_resolution_indices": below_resolution,
        "total_ulp_movement": sum(abs(r.ulp_distance_same_binade) for r in records),
        "total_ulp_movement_note": (
            "Sum of |movement| over steps, measured in ULPs of each step's "
            "starting value. This is the per-step movement metric and does not "
            "blow up when a step crosses a binade boundary or zero."
        ),
        "total_ulp_movement_steps": sum(
            1 for r in records if r.ulp_distance_same_binade != 0
        ),
        "total_representable_values_travelled": sum(
            abs(r.ulp_distance_count) for r in records
        ),
        "total_representable_values_note": (
            "Sum of the exact count of representable float32 values between "
            "before and after each step. Correct, but at a sign crossing a "
            "single step contributes roughly 2.1 billion, so it is not a "
            "summary of how far the parameter travelled; see "
            "'total_ulp_movement' for that."
        ),
        "sign_crossing_steps": [
            r.step
            for r in records
            if r.parameter_before_uint32 >> 31 != r.parameter_after_uint32 >> 31
        ],
        "max_rounding_residual": _max_abs(r.fp32_rounding_error_abs for r in records),
        "max_reconstruction_ulp_error": max(
            (r.reconstruction_ulp_error for r in records), default=0
        ),
        "reconstruction_checks": sum(
            1 for r in records if r.reconstruction_variant not in ("", None)
        ),
        "reconstruction_failures": sum(
            1 for r in records if r.reconstruction_status not in ("PASS", "")
        ),
        "max_gradient_validation_relative_error": _max_abs(
            r.gradient_vs_analytical_rel for r in records
        ),
        "max_gradient_validation_absolute_error": _max_abs(
            r.gradient_vs_analytical_abs for r in records
        ),
        "max_finite_difference_relative_error": _max_abs(
            r.gradient_vs_finite_difference_rel for r in records
        ),
        "max_finite_difference_absolute_error": _max_abs(
            r.gradient_vs_finite_difference_abs for r in records
        ),
        # The tightest bound actually applied to any step, so a reader can check
        # the observed error against the criterion that produced PASS/FAIL
        # without recomputing it. The rule itself is in
        # validation.gradient_tolerance.
        "min_gradient_tolerance_absolute": _min_present(
            r.gradient_tolerance_abs for r in records
        ),
        "min_finite_difference_tolerance_absolute": _min_present(
            r.finite_difference_tolerance_abs for r in records
        ),
        "nonfinite_steps": nonfinite_steps,
        "nonfinite_detected": bool(nonfinite_steps),
        "max_fp32_vs_fp64_difference": _max_abs(r.fp64_vs_fp32_abs for r in records),
        "chain_verified": recorder.verify_chain()[0],
        "event_counts": recorder.events.counts(),
        "validation": recorder.ledger.to_dict(),
        "aborted_reason": recorder.aborted_reason,
    }
    if stability is not None:
        report["stability"] = dict(stability)
    return report


REPORT_LABELS: tuple[tuple[str, str], ...] = (
    ("initial_parameter", "INITIAL PARAMETER"),
    ("final_parameter", "FINAL PARAMETER"),
    ("total_parameter_change", "TOTAL PARAMETER CHANGE"),
    ("steps", "NUMBER OF STEPS"),
    ("initial_loss", "INITIAL LOSS"),
    ("final_loss", "FINAL LOSS"),
    ("minimum_loss", "MINIMUM LOSS"),
    ("initial_gradient", "INITIAL GRADIENT"),
    ("final_gradient", "FINAL GRADIENT"),
    ("max_abs_gradient", "MAX ABS GRADIENT"),
    ("max_abs_update", "MAX ABS UPDATE"),
    ("mean_abs_update", "MEAN ABS UPDATE"),
    ("std_parameter_over_final_window", "STD PARAMETER OVER FINAL WINDOW"),
    ("gradient_sign_changes", "GRADIENT SIGN CHANGES"),
    ("update_sign_changes", "UPDATE SIGN CHANGES"),
    ("unchanged_fp32_steps", "UNCHANGED FP32 STEPS"),
    ("total_ulp_movement", "TOTAL ULP MOVEMENT (in ULPs of each step's start)"),
    ("total_ulp_movement_steps", "  ... contributed by this many steps"),
    ("total_representable_values_travelled", "TOTAL REPRESENTABLE VALUES TRAVELLED"),
    ("sign_crossing_steps", "  ... steps that crossed zero"),
    ("max_rounding_residual", "MAX ROUNDING RESIDUAL"),
    ("max_reconstruction_ulp_error", "MAX RECONSTRUCTION ERROR (ULP)"),
    ("reconstruction_failures", "RECONSTRUCTION FAILURES"),
    ("max_gradient_validation_relative_error", "MAX GRADIENT VALIDATION RELATIVE ERROR"),
    ("max_gradient_validation_absolute_error", "MAX GRADIENT VALIDATION ABSOLUTE ERROR"),
    ("min_gradient_tolerance_absolute", "TIGHTEST GRADIENT TOLERANCE (absolute)"),
    ("max_finite_difference_relative_error", "MAX FINITE-DIFFERENCE RELATIVE ERROR"),
    ("max_finite_difference_absolute_error", "MAX FINITE-DIFFERENCE ABSOLUTE ERROR"),
    ("min_finite_difference_tolerance_absolute", "TIGHTEST FINITE-DIFF TOLERANCE (absolute)"),
    ("max_fp32_vs_fp64_difference", "MAX FP32 vs FP64 REFERENCE DIFFERENCE"),
    ("nonfinite_detected", "NON-FINITE DETECTED"),
    ("chain_verified", "HISTORY HASH CHAIN VERIFIED"),
)


def render_final_report(report: Mapping[str, Any]) -> str:
    """Human-readable end-of-run report."""
    width = max(len(label) for _key, label in REPORT_LABELS)
    lines = [
        "=" * 78,
        "COMPLETE NUMERICAL REPORT (all values measured from this run)",
        "=" * 78,
    ]
    for key, label in REPORT_LABELS:
        if key not in report:
            continue
        value = report[key]
        rendered = str(value) if isinstance(value, (int, bool, str)) else repr(value)
        lines.append(f"{label.ljust(width)} : {rendered}")

    if report.get("unchanged_fp32_step_indices"):
        indices = report["unchanged_fp32_step_indices"]
        preview = ", ".join(str(i) for i in indices[:20])
        if len(indices) > 20:
            preview += f", ... (+{len(indices) - 20} more)"
        lines.append(f"{'UNCHANGED FP32 STEP INDICES'.ljust(width)} : {preview}")

    if report.get("update_attempted_below_resolution_indices"):
        indices = report["update_attempted_below_resolution_indices"]
        preview = ", ".join(str(i) for i in indices[:20])
        if len(indices) > 20:
            preview += f", ... (+{len(indices) - 20} more)"
        lines.append(
            f"{'UPDATE ATTEMPTED BELOW ULP'.ljust(width)} : {preview}"
        )

    for key in ("total_ulp_movement_note", "total_representable_values_note"):
        if report.get(key):
            lines.append("")
            lines.append(f"  {report[key]}")

    if report.get("sign_crossing_steps"):
        lines.append(
            f"  Note: steps {report['sign_crossing_steps']} crossed zero, which is "
            "what inflates TOTAL REPRESENTABLE VALUES TRAVELLED."
        )

    stability = report.get("stability")
    if stability:
        lines.append("-" * 78)
        lines.append("STABILITY / REGIME (observed behaviour, not a convergence claim)")
        for key in (
            "regime",
            "rationale",
            "final_window",
            "mean_update_magnitude",
            "std_update_magnitude",
            "mean_gradient_magnitude",
            "gradient_sign_changes_in_window",
            "update_sign_changes_in_window",
            "unchanged_fraction_in_window",
            "total_ulp_movement_in_window",
        ):
            if key in stability:
                lines.append(f"{key.ljust(width)} : {stability[key]!r}")

    validation = report.get("validation")
    if validation:
        lines.append("-" * 78)
        lines.append("VALIDATION LEDGER")
        for key in (
            "gradient_checks",
            "gradient_failures",
            "finite_difference_failures",
            "reconstruction_checks",
            "reconstruction_failures",
            "bit_checks",
            "bit_failures",
            "ulp_checks",
            "ulp_failures",
            "total_failures",
        ):
            lines.append(f"{key.ljust(width)} : {validation.get(key)}")
        for note in validation.get("failure_notes", [])[:10]:
            lines.append(f"    ! {note}")

    if report.get("aborted_reason"):
        lines.append("-" * 78)
        lines.append(f"RUN ABORTED: {report['aborted_reason']}")

    lines.append("=" * 78)
    return "\n".join(lines)


__all__ = [
    "EVENT_CODES",
    "SNAPSHOT_LABELS",
    "EventEntry",
    "EventLog",
    "ParameterStepRecord",
    "Recorder",
    "build_final_report",
    "render_final_report",
    "canonical_json",
    "json_safe",
]
