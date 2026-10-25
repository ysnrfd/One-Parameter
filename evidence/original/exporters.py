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
import math
import os
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from . import numerical_analysis as na
from .config import ExperimentConfig
from .errors import ConfigError
from .recorder import Recorder, canonical_json

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
)


def _ensure_directory(directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    return directory


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        # allow_nan=False would reject a diverged run outright; the recorder
        # already renders non-finite floats as strings, so this stays valid JSON.
        json.dump(payload, handle, indent=2, default=str, allow_nan=False)


# --------------------------------------------------------------------------- #
# individual artefacts
# --------------------------------------------------------------------------- #


def export_trajectory_csv(recorder: Recorder, path: str) -> str:
    """One row per recorded step, with every field, unrounded."""
    _write_csv(path, recorder.rows(), recorder.field_names())
    return path


def export_trajectory_json(
    recorder: Recorder, path: str, *, report: Mapping[str, Any] | None = None
) -> str:
    payload = {
        "config": recorder.config.to_dict(),
        "records": [record.to_dict() for record in recorder.records],
        "events": recorder.events.to_list(),
        "landscapes": list(recorder.landscapes),
        "report": dict(report) if report else None,
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
    """Numeric columns as numpy arrays, for plotting or further analysis."""
    import numpy as np  # local import: numpy is required, but only here

    if not recorder.records:
        raise ConfigError("nothing to export: the run recorded zero steps")
    columns: dict[str, Any] = {}
    for name in recorder.field_names():
        values = [getattr(record, name) for record in recorder.records]
        if all(isinstance(v, bool) for v in values):
            columns[name] = np.asarray(values, dtype=bool)
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            columns[name] = np.asarray(values, dtype=np.int64)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            columns[name] = np.asarray([float(v) for v in values], dtype=np.float64)
        else:
            continue
    np.savez_compressed(path, **columns)
    return path


def export_bit_history_csv(recorder: Recorder, path: str) -> str:
    """The bit-level history: every stored word and every bit that changed."""
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
    rows = [entry.to_dict() for entry in recorder.events.entries]
    _write_csv(path, rows, ["step", "code", "detail"])
    return path


def export_validation_json(recorder: Recorder, path: str) -> str:
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
    _write_json(
        path,
        {
            "run": config.to_dict(),
            "report": dict(report) if report else None,
            "events": recorder.events.counts(),
            "chain_verified": recorder.verify_chain()[0],
            "record_count": len(recorder.records),
            "note": (
                None
                if report
                else "the run recorded zero steps, so there is no trajectory to summarise"
            ),
        },
    )
    return path


def export_parquet_if_available(
    recorder: Recorder, path: str
) -> str | None:
    """Write Parquet when pandas is installed; otherwise say so, do not guess."""
    if not recorder.records:
        return None
    try:
        import pandas as pd  # noqa: PLC0415 - optional dependency
    except Exception:
        return None
    frame = pd.DataFrame(recorder.rows())
    frame.to_parquet(path, index=False)
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
    _ensure_directory(directory)
    written: dict[str, str | None] = {}
    if report is None and recorder.records:
        from .recorder import build_final_report

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
        try:
            written["trajectory.npz"] = export_trajectory_npz(
                recorder, os.path.join(directory, "trajectory.npz")
            )
        except Exception as exc:  # pragma: no cover - never silently skip
            written["trajectory.npz"] = None
            written["trajectory.npz_error"] = f"{type(exc).__name__}: {exc}"
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
