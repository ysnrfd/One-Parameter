"""Figures built from recorded data only.

Every point on every plot is a value that was read out of a real run.  Where a
derived quantity is shown (rolling means, error bars), it is derived from the
recorded series and labelled as such; nothing is interpolated to fill gaps, and
missing values are simply not drawn.

The module works with a non-interactive Matplotlib backend as well, so a run on
a machine with no display still produces the same figures.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Iterable, Mapping, Sequence

import torch

from . import numerical_analysis as na
from .config import ExperimentConfig
from .convergence import STABILITY_DISCLAIMER
from .errors import DependencyMissingError
from .recorder import ParameterStepRecord, Recorder

#: The causal chain, in execution order.  The dashboard highlights the active one.
CAUSAL_CHAIN: tuple[str, ...] = (
    "INPUT",
    "TRACKED PARAMETER w",
    "FORWARD OPERATION",
    "INTERMEDIATE",
    "OUTPUT",
    "LOSS",
    "AUTOGRAD",
    "GRADIENT",
    "OPTIMIZER STATE",
    "UPDATE",
    "FP32 STORAGE",
    "NEW PARAMETER",
)

#: The phase currently being executed, used to highlight the causal diagram.
STEP_PHASES: tuple[str, ...] = (
    "STATE A (read)",
    "FORWARD",
    "LOSS",
    "BACKWARD",
    "STATE C (read)",
    "DIAGNOSTICS",
    "STATE D (read)",
    "OPTIMIZER",
    "STATE E (read)",
    "RECONSTRUCTION",
)

#: The static view's series list, as (recorded attribute, panel title) pairs.
#: The live view's list is `dashboard.DASHBOARD_SERIES`; both name the same
#: recorded attributes of the same records and deliberately draw different subsets
#: of them, so the two are allowed to differ in content.  What they must not do is
#: draw a series differently: both go through `draw_recorded_points`, which
#: filters non-finite values and marks every recorded point.
OVERVIEW_PANELS: tuple[tuple[str, str], ...] = (
    ("parameter_after", "A. parameter w_t (stored float32)"),
    ("actual_delta", "B. parameter delta"),
    ("gradient", "C. gradient"),
    ("loss", "D. loss"),
    ("moment1", "E. first moment m_t"),
    ("moment2", "F. second moment v_t"),
    ("bias_corrected_moment1", "G. bias-corrected m-hat"),
    ("bias_corrected_moment2", "H. bias-corrected v-hat"),
    ("learning_rate", "I. learning rate"),
    ("optimizer_weight_decay_update", "J. weight decay contribution"),
    ("fp32_rounding_error_abs", "K. fp32 rounding residual"),
    ("ulp_relative_movement", "L. ULP-relative movement"),
    ("changed_bit_count", "M. bit-change history"),
)


def configure_backend(interactive: bool) -> str:
    """Choose a Matplotlib backend explicitly and report which one was chosen."""
    import matplotlib  # noqa: PLC0415 - optional dependency

    if interactive:
        for candidate in ("TkAgg", "QtAgg", "MacOSX", "WebAgg"):
            try:
                matplotlib.use(candidate, force=True)
                return candidate
            except Exception:
                continue
    matplotlib.use("Agg", force=True)
    return "Agg"


def _require_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415 - optional dependency
    except Exception as exc:  # pragma: no cover - matplotlib is in requirements
        raise DependencyMissingError(
            f"matplotlib is required for figures but could not be imported: {exc}"
        ) from exc
    return plt


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _steps(records: Sequence[ParameterStepRecord]) -> list[int]:
    return [record.step for record in records]


def recorded_series(
    records: Sequence[ParameterStepRecord], attribute: str
) -> tuple[list[float], list[float]]:
    """Return ``(xs, ys)`` for one recorded attribute, skipping unusable values.

    Unavailable and non-finite values are dropped rather than imputed, and they
    are *not* drawn: Matplotlib ignores a ``nan`` when it renders a line but
    still uses it when auto-scaling, so an unfiltered diverged run produces axis
    limits that hide the steps which did record finite values.

    Both the static overview figure and the live dashboard go through this
    function, so the two views cannot disagree about which points are real.
    """
    xs: list[float] = []
    ys: list[float] = []
    for record in records:
        value = getattr(record, attribute, None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        xs.append(float(record.step))
        ys.append(numeric)
    return xs, ys


def draw_recorded_points(
    ax: Any,
    records: Sequence[ParameterStepRecord],
    attribute: str,
    *,
    label: str | None = None,
    color: Any = None,
    linewidth: float = 1.2,
    **kwargs: Any,
) -> bool:
    """Draw one recorded attribute, with a visible mark at every recorded point.

    Returns ``True`` when at least one finite point was drawn and ``False`` when
    there was nothing to draw, so the caller can say so rather than leaving an
    empty panel that looks like a plotted run.

    The marker is not decoration.  A single recorded step is a single point, and
    a bare ``plot`` of one point has no segment to draw - the panel would be
    empty and indistinguishable from "no run has happened yet".  The scatter is
    kept alongside the marker so the point is unmistakable at any zoom.

    This is the one drawing path for a recorded series; ``dashboard.LiveDashboard``
    calls it rather than reimplementing the filtering and the marks.
    """
    xs, ys = recorded_series(records, attribute)
    if not xs:
        return False
    ax.plot(
        xs,
        ys,
        linewidth=linewidth,
        label=label,
        color=color,
        marker=".",
        markersize=6,
        **kwargs,
    )
    ax.scatter(xs, ys, s=10, color=color)
    return True


def plot_recorded_series(
    ax: Any,
    records: Sequence[ParameterStepRecord],
    attribute: str,
    label: str,
    **kwargs: Any,
) -> bool:
    """Draw one recorded series in its own panel, with a title and a grid."""
    if not draw_recorded_points(ax, records, attribute, label=label, **kwargs):
        ax.set_title(f"{label}\n(no finite values recorded)")
        return False
    ax.set_title(label, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return True


def _maybe_symlog(ax: Any, values: Iterable[float]) -> None:
    finite = [abs(v) for v in values if math.isfinite(v) and v != 0.0]
    if not finite:
        return
    largest = max(finite)
    if largest > 0 and min(finite) / largest < 1e-6:
        try:
            ax.set_yscale("symlog", linthresh=max(1e-300, min(finite) * 0.5))
        except Exception:  # pragma: no cover - symlog rejects non-positive linthresh
            pass


# --------------------------------------------------------------------------- #
# the overview figure
# --------------------------------------------------------------------------- #


def build_overview_figure(
    recorder: Recorder,
    analyzer: Any = None,
    *,
    figsize: tuple[float, float] = (17.0, 20.0),
    dpi: int = 110,
) -> Any:
    """The multi-panel figure: every required live series, from recorded values."""
    plt = _require_pyplot()
    records = recorder.records
    # 13 series panels + landscape + stability + state transition = 16 axes.
    figure, axes = plt.subplots(4, 4, figsize=figsize, dpi=dpi)
    figure.suptitle(
        "Single float32 parameter: recorded trajectory "
        f"({recorder.config.optimizer}, {len(records)} real steps)\n"
        "every point is a value read from the run; nothing is interpolated",
        fontsize=12,
    )
    flat = axes.flatten()
    # The panels come from the module-level OVERVIEW_PANELS, so the constant and
    # the figure cannot describe different sets.
    for ax, (attribute, title) in zip(flat, OVERVIEW_PANELS):
        plot_recorded_series(ax, records, attribute, title)
        if attribute in ("actual_delta", "fp32_rounding_error_abs", "ulp_relative_movement"):
            _maybe_symlog(ax, (r.actual_delta for r in records))
        ax.set_xlabel("step", fontsize=8)

    # N. local loss landscape
    landscape_ax = flat[len(OVERVIEW_PANELS)]
    _draw_landscape(landscape_ax, recorder)

    # O. stability
    stability_ax = flat[len(OVERVIEW_PANELS) + 1]
    _draw_stability(stability_ax, recorder, analyzer)

    # Remaining panel: parameter state transition for the last step
    transition_ax = flat[len(OVERVIEW_PANELS) + 2]
    _draw_transition(transition_ax, records[-1] if records else None)

    with warnings.catch_warnings():
        # A dense 4x4 grid of annotated panels can defeat tight_layout; the
        # layout is cosmetic, so a warning about it is noise, not a finding.
        warnings.simplefilter("ignore")
        try:
            figure.tight_layout(rect=(0, 0, 1, 0.97))
        except Exception:  # pragma: no cover - never fail a run over layout
            pass
    return figure


def _draw_landscape(ax: Any, recorder: Recorder) -> None:
    if not recorder.landscapes:
        ax.set_title("N. local loss landscape\n(no probe was recorded)")
        ax.text(0.5, 0.5, na.ONE_DIMENSIONAL_DISCLAIMER, ha="center", va="center", wrap=True, fontsize=7)
        return
    last = recorder.landscapes[-1]
    if not last or not last.get("samples"):
        ax.set_title("N. local loss landscape\n(empty probe samples)")
        ax.text(0.5, 0.5, na.ONE_DIMENSIONAL_DISCLAIMER, ha="center", va="center", wrap=True, fontsize=7)
        return
    offsets = [pair[0] for pair in last["samples"]]
    losses = [pair[1] for pair in last["samples"]]
    ax.plot(offsets, losses, marker="o", markersize=3, linewidth=1.0)
    centre_index = min(range(len(offsets)), key=lambda i: abs(offsets[i]))
    ax.axvline(0.0, color="tab:green", linewidth=0.8, linestyle=":")
    ax.plot([offsets[centre_index]], [losses[centre_index]], marker="o", markersize=9, color="tab:green")
    ax.set_title(
        f"N. local loss landscape (step {last['step']}, {last['dtype']})\n"
        f"first deriv {last.get('central_first_difference')}, "
        f"second deriv {last.get('central_second_difference')}",
        fontsize=8,
    )
    ax.set_xlabel("offset from the live parameter", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.text(
        0.01, -0.22, na.ONE_DIMENSIONAL_DISCLAIMER, transform=ax.transAxes, fontsize=6, wrap=True
    )


def _draw_stability(ax: Any, recorder: Recorder, analyzer: Any) -> None:
    if analyzer is None or not analyzer.metrics:
        ax.set_title("O. stability indicators\n(no analysis recorded)")
        return
    steps = [metric.step for metric in analyzer.metrics]
    ax.plot(steps, [m.rolling_mean_absolute_update for m in analyzer.metrics], label="mean |update| (window)")
    ax.plot(steps, [m.rolling_mean_absolute_gradient for m in analyzer.metrics], label="mean |gradient| (window)")
    ax.set_title("O. stability indicators (rolling window)", fontsize=9)
    ax.set_xlabel("step", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7)
    report = analyzer.report()
    ax.text(
        0.01,
        0.02,
        f"observed regime: {report.get('regime')}\n{STABILITY_DISCLAIMER.split(':')[0]}",
        transform=ax.transAxes,
        fontsize=6,
        va="bottom",
        wrap=True,
    )


def _draw_transition(ax: Any, record: ParameterStepRecord | None) -> None:
    """The previous -> gradient -> optimizer -> rounding -> current chain."""
    ax.axis("off")
    if record is None:
        ax.text(0.5, 0.5, "no step recorded", ha="center", va="center")
        return
    lines = [
        f"step {record.step}",
        "",
        f"previous state   {record.parameter_before!r}  {record.parameter_before_hex}",
        f"gradient         {record.gradient!r}",
        f"reconstructed update {record.optimizer_total_update!r}",
        f"  adaptive (recon)  {record.optimizer_adaptive_update!r}",
        f"  decay (recon)     {record.optimizer_weight_decay_update!r}",
        f"fp32 candidate   {record.fp32_candidate!r}",
        f"rounding resid.  {record.fp32_rounding_error_abs!r} "
        f"({record.fp32_rounding_ulp!r} ULP)",
        f"current state    {record.parameter_after!r}  {record.parameter_hex}",
        f"actual delta     {record.actual_delta!r} ({record.ulp_distance_count:+d} ULP)",
        "",
        f"reconstruction   {record.reconstruction_status} "
        f"(variant {record.reconstruction_variant})",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=7)
    ax.set_title("parameter state transition (last recorded step)", fontsize=9)


# --------------------------------------------------------------------------- #
# bit visualisation
# --------------------------------------------------------------------------- #


def draw_bit_pattern(ax: Any, record: ParameterStepRecord) -> None:
    """Draw the 32 stored bits of one step, marking the bits that changed.

    The bits highlighted are those this step changed *relative to its own
    previous value* - the transition recorded in
    ``ParameterStepRecord.changed_bit_positions``, which is the XOR of the
    before and after words.  No separate previous step is drawn or consulted: a
    two-row before/after diagram would compare steps rather than show the
    transition, and the caller that wants step-to-step comparison should read
    ``changed_bit_positions`` for each step instead.
    """
    from matplotlib.patches import Rectangle  # noqa: PLC0415 - optional dependency

    ax.axis("off")
    changed_positions = set(record.changed_bit_positions)
    for index in range(32):
        bit = (record.parameter_after_uint32 >> (31 - index)) & 1
        is_changed = index in changed_positions
        colour = "tab:orange" if is_changed else ("0.85" if bit == 0 else "0.4")
        ax.add_patch(
            Rectangle(
                (index, 0), 0.9, 0.8, facecolor=colour, edgecolor="black", linewidth=0.3
            )
        )
        ax.text(
            index + 0.45, 0.4, str(bit), ha="center", va="center", fontsize=7,
            color="white" if bit else "black",
        )
    ax.set_xlim(-0.2, 32.2)
    ax.set_ylim(-0.6, 1.2)
    ax.set_title(
        f"stored float32 bits after step {record.step}: {record.parameter_hex}\n"
        f"sign={record.parameter_sign_bit} exponent={record.parameter_exponent_bits} "
        f"fraction={record.parameter_fraction_bits} (orange = changed this step)",
        fontsize=8,
    )


def build_bit_figure(recorder: Recorder, *, dpi: int = 110) -> Any:
    """Bit-level history: the stored word and the changed bits, step by step."""
    plt = _require_pyplot()
    records = recorder.records
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), dpi=dpi, gridspec_kw={"height_ratios": [1, 2]})
    if records:
        draw_bit_pattern(axes[0], records[-1])
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "no step recorded", ha="center")
    ax = axes[1]
    steps = _steps(records)
    if steps:
        ax.bar(steps, [r.changed_bit_count for r in records], width=0.8, color="tab:orange")
        ax.set_ylabel("changed bits")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25, linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "no step recorded", ha="center")
    ax.set_title("M. bit-change history: how many of the 32 stored bits changed per step", fontsize=10)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# causal graph
# --------------------------------------------------------------------------- #


def draw_causal_chain(ax: Any, active_phase: str | None = None) -> None:
    """The causal diagram, with the active phase highlighted."""
    ax.axis("off")
    count = len(CAUSAL_CHAIN)
    for index, label in enumerate(CAUSAL_CHAIN):
        y = 1.0 - (index + 0.5) / count
        is_active = active_phase is not None and active_phase.upper() in label
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold" if is_active else "normal",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "tab:orange" if is_active else "0.9",
                "alpha": 0.9 if is_active else 0.6,
            },
        )
        if index < count - 1:
            ax.annotate(
                "",
                xy=(0.5, 1.0 - (index + 1.5) / count),
                xytext=(0.5, 1.0 - (index + 0.9) / count),
                arrowprops={"arrowstyle": "->", "color": "0.4"},
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"causal chain (active phase: {active_phase or 'none'})", fontsize=9)


# --------------------------------------------------------------------------- #
# saving
# --------------------------------------------------------------------------- #


def save_figure(figure: Any, path: str, *, dpi: int = 130) -> str:
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def render_figures(
    recorder: Recorder,
    directory: str,
    analyzer: Any = None,
) -> dict[str, str]:
    """Write the overview, bit history, and a landscape figure."""
    os.makedirs(directory, exist_ok=True)
    plt = _require_pyplot()
    written: dict[str, str] = {}
    overview = build_overview_figure(recorder, analyzer)
    written["overview"] = save_figure(overview, os.path.join(directory, "overview.png"))
    plt.close(overview)
    bits = build_bit_figure(recorder)
    written["bit_history"] = save_figure(bits, os.path.join(directory, "bit_history.png"))
    plt.close(bits)
    return written


# --------------------------------------------------------------------------- #
# textual state card
# --------------------------------------------------------------------------- #


def render_state_card(record: ParameterStepRecord, config: ExperimentConfig) -> str:
    """The exact textual inspector for one step (no rounded display values)."""
    rows: list[tuple[str, Any]] = [
        ("Step", record.step),
        ("Epoch", record.epoch),
        ("Batch", record.batch_index),
        ("Parameter", repr(record.parameter_after)),
        ("dtype", "float32"),
        ("shape", "()"),
        ("Before", f"{record.parameter_before!r}  ({record.parameter_before_hex})"),
        ("After", f"{record.parameter_after!r}  ({record.parameter_hex})"),
        ("Actual dw (observed)", f"{record.actual_delta!r}"),
        ("Attempted update (fp64 candidate)", repr(record.optimizer_attempted_update)),
        ("Reconstructed update (fp32, derived)", repr(record.optimizer_total_update)),
        ("Stored FP32 changed", record.stored_fp32_unchanged is False),
        ("Update below FP32 resolution", record.update_attempted_but_storage_frozen),
        ("Gradient", repr(record.gradient)),
        ("Gradient dtype/shape", f"{record.gradient_dtype} {record.gradient_shape}"),
        ("Gradient sign/magnitude", f"{record.gradient_sign:+d} / {record.gradient_magnitude!r}"),
        ("Loss", repr(record.loss)),
        ("Prediction", repr(record.prediction)),
        ("Input", repr(record.input_value)),
        ("Target", repr(record.target_value)),
        ("Learning Rate", repr(record.learning_rate)),
        ("Optimizer", record.optimizer_name),
        ("Optimizer step", f"{record.optimizer_step_before} -> {record.optimizer_step_after}"),
        ("m", repr(record.moment1)),
        ("v", repr(record.moment2)),
        ("Bias-corrected m", repr(record.bias_corrected_moment1)),
        ("Bias-corrected v", repr(record.bias_corrected_moment2)),
        ("Weight Decay (reconstructed)", repr(record.optimizer_weight_decay_update)),
        (
            "Optimizer Update (reconstructed)",
            f"total {record.optimizer_total_update!r} = "
            f"adaptive {record.optimizer_adaptive_update!r} + "
            f"decay {record.optimizer_weight_decay_update!r}",
        ),
        (
            "Reconstruction",
            f"{record.reconstruction_status} "
            f"(variant {record.reconstruction_variant}, "
            f"{record.reconstruction_ulp_error} ULP)",
        ),
        ("FP32 Candidate", repr(record.fp32_candidate)),
        ("FP32 Stored Value", f"{record.parameter_after!r} ({record.parameter_hex})"),
        (
            "Rounding Residual",
            f"{record.fp32_rounding_error_abs!r} "
            f"({record.fp32_rounding_ulp!r} ULP)",
        ),
        ("ULP at w_before", repr(record.ulp_size)),
        ("Movement (ULPs of w_before)", f"{record.ulp_distance_same_binade:+d}"),
        ("Displacement / ULP(w_before)", repr(record.ulp_relative_movement)),
        ("Representable f32 values between", f"{record.ulp_distance_count}"),
        ("Hex", record.parameter_hex),
        ("Raw uint32", record.parameter_after_uint32),
        ("Bits", record.parameter_bits),
        ("Sign", record.parameter_sign_bit),
        ("Exponent", record.parameter_exponent_bits),
        ("Fraction", record.parameter_fraction_bits),
        ("Changed Bits", f"{record.changed_bit_count} at {list(record.changed_bit_positions)}"),
        ("FP64 reference (not truth)", repr(record.fp64_track_value)),
        ("Gradient validation", record.gradient_validation_status),
        ("Finite", "YES" if record.finite else "NO"),
        ("Events", ", ".join(record.events) or "(none)"),
    ]
    width = max(len(name) for name, _ in rows)
    header = "SINGLE PARAMETER FORENSIC STATE\n" + "=" * 62
    body = "\n".join(f"{name.ljust(width)} : {value}" for name, value in rows)
    footer = (
        f"\nParameter / gradient / loss / optimizer state are four different "
        f"tensors.\nForward and backward do not change the parameter; only "
        f"optimizer.step() does."
    )
    return f"{header}\n{body}{footer}"


__all__ = [
    "CAUSAL_CHAIN",
    "STEP_PHASES",
    "OVERVIEW_PANELS",
    "configure_backend",
    "recorded_series",
    "draw_recorded_points",
    "plot_recorded_series",
    "build_overview_figure",
    "build_bit_figure",
    "draw_bit_pattern",
    "draw_causal_chain",
    "render_figures",
    "save_figure",
    "render_state_card",
]
