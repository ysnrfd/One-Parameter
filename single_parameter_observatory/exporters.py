"""Export of recorded observations.

Everything written here comes from :class:`~.recorder.Recorder`.  Raw values are
exported as they were recorded: floats use ``repr``, which is the shortest string
that returns the identical double, so a CSV cell round-trips to the same value
rather than to a rounded display form.  Nothing is smoothed, resampled or rounded
on the way out.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import fields

from . import numerical_analysis as na
from .config import ExperimentConfig
from .errors import ConfigError
from .recorder import ParameterStepRecord, Recorder, json_safe

#: Files written by :func:`export_all`, in the order they are produced.
EXPORT_FILES: tuple[str, ...] = (
    "summary.json",
    "trajectory.csv",
    "trajectory.json",
    "trajectory.npz",
    "bit_history.csv",
    "optimizer_state_history.csv",
    "stability.csv",
    "landscape.json",
    "events.csv",
    "validation.json",
    "provenance.json",
    "trajectory.parquet",
)


def _ensure_directory(directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    return directory


@contextmanager
def _atomic_output(path: str, *, binary: bool = False) -> Iterator[Any]:
    """Replace a completed artifact, preserving an older file if writing fails."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(os.path.abspath(path))
    )
    try:
        stream = os.fdopen(descriptor, "wb") if binary else os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        )
        with stream as handle:
            yield handle
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with _atomic_output(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: Any) -> None:
    with _atomic_output(path) as handle:
        # The payload is sanitised *before* the dump.  ``allow_nan=False`` plus
        # ``default=str`` is not sufficient on its own: ``default`` never fires
        # for a float, so a raw inf/nan in the final report of a diverged run
        # raised ValueError and made the whole export fail.  ``json_safe``
        # renders non-finite floats as their repr strings, so the file stays
        # strict, parseable JSON and the divergence is still legible.
        json.dump(
            json_safe(payload), handle, indent=2, default=str, allow_nan=False
        )


# --------------------------------------------------------------------------- #
# individual artefacts
# --------------------------------------------------------------------------- #


def export_trajectory_csv(recorder: Recorder, path: str) -> str:
    """One row per recorded step, with every field, unrounded."""
    recorder.assert_chain()
    _write_csv(path, recorder.rows(), recorder.field_names())
    return path


def export_trajectory_json(
    recorder: Recorder, path: str, *, report: Mapping[str, Any] | None = None
) -> str:
    recorder.assert_chain()
    payload = {
        "config": recorder.config.to_dict(),
        "records": [record.to_dict() for record in recorder.records],
        "events": recorder.events.to_list(),
        "landscapes": list(recorder.landscapes),
        "report": dict(report) if report is not None else None,
        "chain_verified": recorder.verify_chain()[0],
        "note": (
            "Values are exactly as recorded. Floats are Python floats (IEEE-754 "
            "double) holding the exact value of the observed float32; hex and raw "
            "uint32 fields carry the stored binary32 word itself."
        ),
    }
    _write_json(path, payload)
    return path


def export_trajectory_npz(recorder: Recorder, path: str) -> str:
    """Numeric observations without pickle or precision loss.

    Nullable columns include a ``<column>__present`` boolean mask, so an absent
    observation remains distinct from an observed NaN. Python integers outside
    int64 are decimal Unicode arrays, described by ``__column_encodings__``.
    """
    import numpy as np  # noqa: PLC0415 - required dependency, used only here

    recorder.assert_chain()
    if not recorder.records:
        raise ConfigError("nothing to export: the run recorded zero steps")
    columns: dict[str, Any] = {}
    encodings: dict[str, str] = {}
    annotations = {field.name: field.type for field in fields(ParameterStepRecord)}
    for name in recorder.field_names():
        values = [getattr(record, name) for record in recorder.records]
        present = [value for value in values if value is not None]
        if not present and annotations[name] not in ("float | None", "int | None"):
            continue
        if present and all(isinstance(value, bool) for value in present):
            columns[name] = np.asarray([False if value is None else value for value in values], dtype=bool)
        elif (present and all(type(value) is int for value in present)) or (
            not present and annotations[name] == "int | None"
        ):
            integers = [0 if value is None else value for value in values]
            if all(-(2**63) <= value < 2**63 for value in integers):
                columns[name] = np.asarray(integers, dtype=np.int64)
            else:
                columns[name] = np.asarray([str(value) for value in integers])
                encodings[name] = "arbitrary_precision_integer_decimal"
        elif not present or all(type(value) in (int, float) for value in present):
            columns[name] = np.asarray(
                [float("nan") if value is None else float(value) for value in values], dtype=np.float64
            )
        else:
            continue
        if any(value is None for value in values):
            columns[f"{name}__present"] = np.asarray([value is not None for value in values], dtype=bool)
    if encodings:
        columns["__column_encodings__"] = np.asarray(json.dumps(encodings, sort_keys=True))
    # Passing a handle prevents numpy from silently appending '.npz' to a
    # caller's path and returning a path that was never actually written.
    with _atomic_output(path, binary=True) as handle:
        np.savez_compressed(handle, **columns)
    return path


def export_bit_history_csv(recorder: Recorder, path: str) -> str:
    """The bit-level history: every stored word and every bit that changed."""
    recorder.assert_chain()
    rows: list[dict[str, Any]] = []
    for record in recorder.records:
        rows.append(
            {
                "step": record.step,
                "before_uint32": record.parameter_before_uint32,
                "before_hex": record.parameter_before_hex,
                "before_bits": f"{record.parameter_before_uint32:032b}",
                "after_uint32": record.parameter_after_uint32,
                "after_hex": record.parameter_hex,
                "after_bits": record.parameter_bits,
                "xor_mask": f"0x{record.xor_mask:08X}",
                "changed_bit_count": record.changed_bit_count,
                "changed_bit_positions": json.dumps(list(record.changed_bit_positions)),
                "sign_changed": record.sign_changed,
                "exponent_changed": record.exponent_changed,
                "fraction_changed": record.fraction_changed,
                "exponent_bits": record.parameter_exponent_bits,
                "fraction_bits": record.parameter_fraction_bits,
                "sign_bit": record.parameter_sign_bit,
                "ulp_distance": record.ulp_distance_count,
                "ulp_size": repr(record.ulp_size),
                "ulp_relative_movement": repr(record.ulp_relative_movement),
                "stored_fp32_unchanged": record.stored_fp32_unchanged,
                "update_attempted_but_storage_frozen": record.update_attempted_but_storage_frozen,
            }
        )
    _write_csv(
        path,
        rows,
        [
            "step", "before_uint32", "before_hex", "before_bits", "after_uint32",
            "after_hex", "after_bits", "xor_mask", "changed_bit_count",
            "changed_bit_positions", "sign_changed", "exponent_changed",
            "fraction_changed", "exponent_bits", "fraction_bits", "sign_bit",
            "ulp_distance", "ulp_size", "ulp_relative_movement",
            "stored_fp32_unchanged", "update_attempted_but_storage_frozen",
        ],
    )
    return path


def export_optimizer_state_history_csv(recorder: Recorder, path: str) -> str:
    """The optimizer's own state, step by step, as read from ``optimizer.state``."""
    recorder.assert_chain()
    rows: list[dict[str, Any]] = []
    for record in recorder.records:
        rows.append(
            {
                "step": record.step,
                "optimizer": record.optimizer_name,
                "learning_rate": repr(record.learning_rate),
                "step_before": record.optimizer_step_before,
                "step_after": record.optimizer_step_after,
                "state_present_before": record.optimizer_state_present_before,
                "keys_before": json.dumps(list(record.optimizer_state_keys_before)),
                "keys_after": json.dumps(list(record.optimizer_state_keys_after)),
                "moment1_before": _num(record.moment1_before),
                "moment1": _num(record.moment1),
                "moment2_before": _num(record.moment2_before),
                "moment2": _num(record.moment2),
                "bias_corrected_moment1": _num(record.bias_corrected_moment1),
                "bias_corrected_moment2": _num(record.bias_corrected_moment2),
                "gradient": repr(record.gradient),
                # Named for what they are: the first is derived from the
                # reconstruction, the second from the float64 candidate. The
                # observed stored delta is the `actual_delta` column of
                # trajectory.csv, and is a different quantity from both.
                "reconstructed_optimizer_total_update_fp32": repr(
                    record.optimizer_total_update
                ),
                "optimizer_attempted_update_fp64": repr(record.optimizer_attempted_update),
                "optimizer_adaptive_update": repr(record.optimizer_adaptive_update),
                "optimizer_weight_decay_update": repr(record.optimizer_weight_decay_update),
                "reconstruction_variant": record.reconstruction_variant,
                "reconstruction_status": record.reconstruction_status,
                "reconstruction_ulp_error": record.reconstruction_ulp_error,
                "fp64_reconstructed_after": repr(record.fp64_reconstructed_after),
            }
        )
    _write_csv(
        path,
        rows,
        [
            "step", "optimizer", "learning_rate", "step_before", "step_after",
            "state_present_before", "keys_before", "keys_after", "moment1_before",
            "moment1", "moment2_before", "moment2", "bias_corrected_moment1",
            "bias_corrected_moment2", "gradient",
            "reconstructed_optimizer_total_update_fp32",
            "optimizer_adaptive_update", "optimizer_weight_decay_update",
            "optimizer_attempted_update_fp64",
            "reconstruction_variant", "reconstruction_status",
            "reconstruction_ulp_error", "fp64_reconstructed_after",
        ],
    )
    return path


def export_stability_csv(analyzer: Any, path: str) -> str:
    """Per-step stability measurements (kept separate from the raw trajectory)."""
    rows = [metric.to_row() for metric in analyzer.metrics]
    if rows:
        _write_csv(path, rows, list(rows[0].keys()))
    else:
        _write_csv(path, [], ["step"])
    return path


def _num(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def export_landscape_json(recorder: Recorder, path: str) -> str:
    recorder.assert_chain()
    _write_json(
        path,
        {
            "slices": list(recorder.landscapes),
            "disclaimer": na.ONE_DIMENSIONAL_DISCLAIMER,
            "note": (
                "These probes are diagnostics. They are evaluated through a pure "
                "loss function that builds a throwaway tensor, so they cannot "
                "perturb the recorded trajectory; each slice records the hash of "
                "the parameter before and after to prove it."
            ),
        },
    )
    return path


def export_events_csv(recorder: Recorder, path: str) -> str:
    recorder.assert_chain()
    rows = [entry.to_dict() for entry in recorder.events.entries]
    _write_csv(path, rows, ["step", "code", "detail"])
    return path


def export_validation_json(recorder: Recorder, path: str) -> str:
    recorder.assert_chain()
    _write_json(
        path,
        {
            "ledger": recorder.ledger.to_dict(),
            "per_step": [
                {
                    "step": record.step,
                    "gradient_analytical": record.gradient_analytical,
                    "gradient_finite_difference": record.gradient_finite_difference,
                    "gradient_finite_difference_h": record.gradient_finite_difference_h,
                    "gradient_fp64_autograd": record.gradient_fp64_autograd,
                    "gradient_vs_analytical_rel": record.gradient_vs_analytical_rel,
                    "gradient_vs_finite_difference_rel": record.gradient_vs_finite_difference_rel,
                    "status": record.gradient_validation_status,
                    "reconstruction_status": record.reconstruction_status,
                    "reconstruction_ulp_error": record.reconstruction_ulp_error,
                    "reconstruction_bit_exact": record.reconstruction_bit_exact,
                }
                for record in recorder.records
            ],
        },
    )
    return path


def export_provenance_json(provenance: Mapping[str, Any], path: str) -> str:
    _write_json(path, dict(provenance))
    return path


def export_summary_json(
    config: ExperimentConfig,
    recorder: Recorder,
    report: Mapping[str, Any] | None,
    path: str,
) -> str:
    recorder.assert_chain()
    _write_json(
        path,
        {
            "run": config.to_dict(),
            "report": dict(report) if report is not None else None,
            "events": recorder.events.counts(),
            "chain_verified": recorder.verify_chain()[0],
            "record_count": len(recorder.records),
            "note": (
                None
                if report is not None
                else (
                    "the run recorded zero steps, so there is no trajectory to summarise"
                    if not recorder.records else "no final report was supplied"
                )
            ),
        },
    )
    return path


def export_parquet_if_available(
    recorder: Recorder, path: str
) -> str | None:
    """Write Parquet when pandas and a Parquet engine are available."""
    recorder.assert_chain()
    if not recorder.records:
        return None
    try:
        import pandas as pd  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None
    rows = recorder.rows()
    for name in recorder.field_names():
        values = [row[name] for row in rows]
        if any(type(value) is int and not -(2**63) <= value < 2**63 for value in values):
            for row in rows:
                if row[name] is not None:
                    row[name] = str(row[name])
    frame = pd.DataFrame(rows)
    try:
        with _atomic_output(path, binary=True) as handle:
            frame.to_parquet(handle, index=False)
    except ImportError:
        # pandas alone cannot write Parquet. Its optional engine import occurs
        # in to_parquet, rather than when pandas itself is imported.
        return None
    return path


# --------------------------------------------------------------------------- #
# the whole export
# --------------------------------------------------------------------------- #


def export_all(
    recorder: Recorder,
    directory: str,
    *,
    report: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    analyzer: Any = None,
) -> dict[str, str | None]:
    """Write every artefact and return ``{artefact name: path or None}``."""
    recorder.assert_chain()
    _ensure_directory(directory)
    written: dict[str, str | None] = {}
    if report is None and recorder.records:
        from .recorder import build_final_report  # noqa: PLC0415 - avoids a cycle at import time

        report = build_final_report(recorder)

    written["summary.json"] = export_summary_json(
        recorder.config, recorder, report, os.path.join(directory, "summary.json")
    )
    if recorder.records:
        written["trajectory.csv"] = export_trajectory_csv(
            recorder, os.path.join(directory, "trajectory.csv")
        )
        written["trajectory.json"] = export_trajectory_json(
            recorder, os.path.join(directory, "trajectory.json"), report=report
        )
        written["trajectory.npz"] = export_trajectory_npz(
            recorder, os.path.join(directory, "trajectory.npz")
        )
        written["bit_history.csv"] = export_bit_history_csv(
            recorder, os.path.join(directory, "bit_history.csv")
        )
        written["optimizer_state_history.csv"] = export_optimizer_state_history_csv(
            recorder, os.path.join(directory, "optimizer_state_history.csv")
        )
        written["landscape.json"] = export_landscape_json(
            recorder, os.path.join(directory, "landscape.json")
        )
    else:
        for name in (
            "trajectory.csv",
            "trajectory.json",
            "trajectory.npz",
            "bit_history.csv",
            "optimizer_state_history.csv",
            "landscape.json",
        ):
            written[name] = None
        written["trajectory_empty"] = "the run recorded zero steps"

    if analyzer is not None:
        written["stability.csv"] = export_stability_csv(
            analyzer, os.path.join(directory, "stability.csv")
        )
    else:
        written["stability.csv"] = None

    written["events.csv"] = export_events_csv(recorder, os.path.join(directory, "events.csv"))
    written["validation.json"] = export_validation_json(
        recorder, os.path.join(directory, "validation.json")
    )
    if provenance is not None:
        written["provenance.json"] = export_provenance_json(
            provenance, os.path.join(directory, "provenance.json")
        )
    else:
        written["provenance.json"] = None

    written["trajectory.parquet"] = export_parquet_if_available(
        recorder, os.path.join(directory, "trajectory.parquet")
    )
    # Reusing an export directory must not leave an older run's trajectory or
    # optional artifacts beside the current summary when this run skips them.
    for name in EXPORT_FILES:
        if written.get(name) is None:
            stale_path = os.path.join(directory, name)
            if os.path.isfile(stale_path):
                os.unlink(stale_path)
    return written


__all__ = [
    "EXPORT_FILES",
    "export_all",
    "export_trajectory_csv",
    "export_trajectory_json",
    "export_trajectory_npz",
    "export_bit_history_csv",
    "export_optimizer_state_history_csv",
    "export_stability_csv",
    "export_landscape_json",
    "export_events_csv",
    "export_validation_json",
    "export_provenance_json",
    "export_summary_json",
    "export_parquet_if_available",
]
