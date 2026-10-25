"""Stability and regime analysis.

The important part of this module is what it refuses to say.  A run whose
updates have become tiny can be in several genuinely different situations, and
they mean different things:
* the gradient is genuinely small (a flat region of the loss);
* the update is being cancelled by the optimizer (sign oscillation, or a
  near-stationary second moment);
* the update is smaller than one ULP, so float32 storage cannot represent it;
* the run is at a stationary point of *this one parameter* while the rest of the
  model is still moving;
* or the loss is non-finite and nothing can be concluded at all.

So the analyser reports an **observed regime** with the criteria that produced
it, plus the separate measurements that distinguish the cases above.  It never
reports convergence, because nothing local to a single coordinate can establish
that.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .config import StabilityCriteria
from .recorder import ParameterStepRecord

#: Regime labels the analyser may report.
REGIMES: tuple[str, ...] = (
    "still moving",
    "oscillating",
    "approximately stable",
    "diverging",
    "non-finite",
    "not enough data",
)

#: Always printed with the report, because the distinction matters more than
#: any single number in it.
STABILITY_DISCLAIMER: str = (
    "A small update, a small gradient, float32 storage stagnation, an "
    "oscillating update and a numerical plateau are five different "
    "observations. None of them establishes that the optimizer has converged, "
    "and none of them says anything about a global optimum: this instrument "
    "observes one coordinate of one parameter out of a high-dimensional "
    "optimization state."
)


@dataclass(frozen=True)
class StepStability:
    """Per-step stability measurements, derived but stored separately."""

    step: int
    parameter: float
    loss: float
    gradient: float
    update: float
    absolute_update: float
    absolute_gradient: float
    ulp_movement: float
    stored_fp32_unchanged: bool
    update_attempted_but_storage_frozen: bool
    finite: bool
    rolling_mean_absolute_update: float
    rolling_std_absolute_update: float
    rolling_mean_absolute_gradient: float
    running_gradient_sign_changes: int
    running_update_sign_changes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_row(self) -> dict[str, Any]:
        return {key: (repr(v) if isinstance(v, float) else v) for key, v in asdict(self).items()}


class ConvergenceAnalyzer:
    """Streaming stability analysis over the recorded trajectory."""

    def __init__(self, criteria: StabilityCriteria) -> None:
        self.criteria = criteria
        self._updates: list[float] = []
        self._gradients: list[float] = []
        self._losses: list[float] = []
        self._parameters: list[float] = []
        self._nonfinite_steps: list[int] = []
        self.metrics: list[StepStability] = []

    # -- streaming ----------------------------------------------------------
    def observe(self, record: ParameterStepRecord) -> StepStability:
        """Fold one recorded step into the running analysis."""
        self._updates.append(record.actual_delta)
        self._gradients.append(record.gradient)
        self._losses.append(record.loss)
        self._parameters.append(record.parameter_after)
        if not record.finite:
            self._nonfinite_steps.append(record.step)

        window_updates = self._updates[-self.criteria.window :]
        window_gradients = self._gradients[-self.criteria.window :]
        metrics = StepStability(
            step=record.step,
            parameter=record.parameter_after,
            loss=record.loss,
            gradient=record.gradient,
            update=record.actual_delta,
            absolute_update=abs(record.actual_delta),
            absolute_gradient=abs(record.gradient),
            ulp_movement=abs(record.ulp_distance_same_binade),
            stored_fp32_unchanged=record.stored_fp32_unchanged,
            update_attempted_but_storage_frozen=record.update_attempted_but_storage_frozen,
            finite=record.finite,
            rolling_mean_absolute_update=_mean([abs(u) for u in window_updates]),
            rolling_std_absolute_update=_std([abs(u) for u in window_updates]),
            rolling_mean_absolute_gradient=_mean([abs(g) for g in window_gradients]),
            running_gradient_sign_changes=_sign_changes(self._gradients),
            running_update_sign_changes=_sign_changes(self._updates),
        )
        self.metrics.append(metrics)
        return metrics

    # -- whole-run verdict --------------------------------------------------
    def report(self, records: Sequence[ParameterStepRecord] | None = None) -> dict[str, Any]:
        """Classify the observed regime, with the evidence that produced it."""
        if not self._updates:
            return {
                "regime": "not enough data",
                "rationale": ["no steps were recorded"],
                "disclaimer": STABILITY_DISCLAIMER,
            }

        size = self.criteria.window
        updates = self._updates[-size:]
        gradients = self._gradients[-size:]
        losses = self._losses[-size:]
        parameters = self._parameters[-size:]

        mean_abs_update = _mean([abs(u) for u in updates])
        std_abs_update = _std([abs(u) for u in updates])
        mean_abs_gradient = _mean([abs(g) for g in gradients])
        update_sign_changes = _sign_changes(updates)
        gradient_sign_changes = _sign_changes(gradients)
        unchanged_fraction = (
            sum(1 for m in self.metrics[-size:] if m.stored_fp32_unchanged) / len(self.metrics[-size:])
            if self.metrics
            else 0.0
        )
        total_ulp_movement = sum(abs(m.ulp_movement) for m in self.metrics[-size:])
        loss_growth = _growth_ratio(losses)

        regime, rationale = self._classify(
            mean_abs_update=mean_abs_update,
            mean_abs_gradient=mean_abs_gradient,
            update_sign_changes=update_sign_changes,
            unchanged_fraction=unchanged_fraction,
            loss_growth=loss_growth,
        )

        return {
            "regime": regime,
            "rationale": rationale,
            "final_window": len(updates),
            "criteria": asdict(self.criteria),
            "mean_update_magnitude": mean_abs_update,
            "std_update_magnitude": std_abs_update,
            "mean_gradient_magnitude": mean_abs_gradient,
            "max_absolute_gradient_in_window": max((abs(g) for g in gradients), default=0.0),
            "max_absolute_update_in_window": max((abs(u) for u in updates), default=0.0),
            "gradient_sign_changes_in_window": gradient_sign_changes,
            "update_sign_changes_in_window": update_sign_changes,
            "unchanged_fraction_in_window": unchanged_fraction,
            "total_ulp_movement_in_window": total_ulp_movement,
            "loss_growth_ratio_in_window": loss_growth,
            "parameter_std_in_window": _std(parameters),
            "fp32_storage_stagnant": unchanged_fraction >= self.criteria.stagnant_fraction,
            "nonfinite_steps": list(self._nonfinite_steps),
            "disclaimer": STABILITY_DISCLAIMER,
        }

    def _absolute_loss_growth(self) -> float | None:
        """Loss at the end of the window minus the loss at its start."""
        window = self._losses[-self.criteria.window :]
        finite = [v for v in window if math.isfinite(v)]
        if len(finite) < 2:
            return None
        return finite[-1] - finite[0]

    def _classify(
        self,
        *,
        mean_abs_update: float,
        mean_abs_gradient: float,
        update_sign_changes: int,
        unchanged_fraction: float,
        loss_growth: float | None,
    ) -> tuple[str, list[str]]:
        criteria = self.criteria
        rationale: list[str] = []

        if self._nonfinite_steps:
            return "non-finite", [
                f"non-finite state observed at steps {self._nonfinite_steps[:10]}",
                "no stability conclusion is possible for this run",
            ]

        # A ratio alone is not evidence: a loss sitting near zero can grow by a
        # large factor from float32 noise alone. Divergence therefore requires a
        # material absolute increase as well, and both thresholds are explicit.
        absolute_growth = self._absolute_loss_growth()
        ratio_rose = loss_growth is not None and loss_growth > criteria.divergence_loss_growth
        if ratio_rose and absolute_growth is not None and absolute_growth > criteria.divergence_absolute:
            rationale.append(
                f"loss grew by a factor of {loss_growth:.3g} over the final window "
                f"and by {absolute_growth:.3g} in absolute terms, above the "
                f"configured thresholds {criteria.divergence_loss_growth} and "
                f"{criteria.divergence_absolute}"
            )
            if mean_abs_gradient > criteria.gradient_eps:
                rationale.append(
                    f"mean |gradient| over the window is {mean_abs_gradient:.3g}, "
                    "which is not small"
                )
            return "diverging", rationale

        if ratio_rose:
            rationale.append(
                f"the loss ratio rose to {loss_growth:.3g} but the absolute increase "
                f"is only {absolute_growth!r}, below the divergence threshold "
                f"{criteria.divergence_absolute}: a near-zero loss changes by a large "
                f"factor without anything diverging"
            )

        rationale.append(
            f"mean |update| over the final window = {mean_abs_update:.3g} "
            f"(threshold {criteria.update_eps:.3g})"
        )
        rationale.append(
            f"mean |gradient| over the final window = {mean_abs_gradient:.3g} "
            f"(threshold {criteria.gradient_eps:.3g})"
        )
        rationale.append(
            f"update sign changes in window = {update_sign_changes} "
            f"(oscillation threshold {criteria.oscillation_min_sign_changes})"
        )
        rationale.append(
            f"fraction of window steps whose stored float32 bits did not change = "
            f"{unchanged_fraction:.3g} (stagnation threshold {criteria.stagnant_fraction})"
        )

        if update_sign_changes >= criteria.oscillation_min_sign_changes:
            rationale.append(
                "the update keeps reversing direction, which is oscillation rather "
                "than steady progress"
            )
            return "oscillating", rationale

        small_updates = mean_abs_update < criteria.update_eps
        small_gradients = mean_abs_gradient < criteria.gradient_eps
        if small_updates and small_gradients:
            rationale.append(
                "both the mean update and the mean gradient are below their "
                "thresholds in this window"
            )
            if unchanged_fraction >= criteria.stagnant_fraction:
                rationale.append(
                    "note: much of that is float32 storage stagnation - the "
                    "optimizer attempted updates the stored value could not "
                    "represent, which is not the same as a vanishing gradient"
                )
            return "approximately stable", rationale

        if unchanged_fraction >= criteria.stagnant_fraction and not small_updates:
            rationale.append(
                "the candidate updates are non-trivial but the stored float32 "
                "value did not move: this is a representation-limit effect, not "
                "an absence of updates"
            )
            return "still moving", rationale

        return "still moving", rationale


# --------------------------------------------------------------------------- #
# small statistics helpers
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _std(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    mean = sum(finite) / len(finite)
    variance = max(0.0, sum((v - mean) ** 2 for v in finite) / (len(finite) - 1))
    return math.sqrt(variance)


def _sign_changes(values: Sequence[float]) -> int:
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in values if v != 0.0 and math.isfinite(v)]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _growth_ratio(values: Sequence[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v) and v >= 0.0]
    if len(finite) < 2:
        return None
    if finite[0] == 0.0:
        return float("inf") if finite[-1] > 0.0 else 1.0
    return finite[-1] / finite[0]


__all__ = [
    "REGIMES",
    "STABILITY_DISCLAIMER",
    "StepStability",
    "ConvergenceAnalyzer",
]
