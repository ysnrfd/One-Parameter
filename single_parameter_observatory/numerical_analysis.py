"""Numerical analysis that is deliberately kept *outside* the training path.

Three things live here:
* the analysis of how a mathematical candidate update becomes a stored float32
  value (rounding residual, in ULPs);
* an independent float64 reference track and reference candidates;
* the one-dimensional loss-landscape probe.

Nothing in this module writes to the tracked parameter.  The landscape probe
receives a *pure* evaluation callable that builds a throwaway tensor, so the
real parameter cannot be perturbed even in principle; the engine additionally
hashes the parameter bits before and after each probe and records whether
restoration verified, rather than assuming it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any
from collections.abc import Callable, Sequence

import torch

from . import fp32_inspector as f32
from .errors import NumericalConsistencyError

#: A callable that evaluates the real model at an arbitrary scalar parameter
#: value and returns the loss as a Python float.  Implementations must not touch
#: the live parameter.
LossEvaluator = Callable[[float], float]

#: A callable that returns an analytic derivative of the loss with respect to the
#: tracked parameter, evaluated at a given scalar value.
DerivativeEvaluator = Callable[[float], float]

#: Reminder text attached to every landscape artefact.
ONE_DIMENSIONAL_DISCLAIMER = (
    "This is a one-dimensional diagnostic slice for the displayed batch "
    "objective. This experiment has exactly one trainable scalar; the slice "
    "does not describe other data batches or a high-dimensional network."
)

#: Reminder text attached to every float64 artefact.
FP64_REFERENCE_DISCLAIMER = (
    "Reference numerical experiment, not ground truth: float64 arithmetic is a "
    "different computation with different rounding, not a more correct one."
)

# --------------------------------------------------------------------------- #
# error helpers
# --------------------------------------------------------------------------- #


def absolute_error(measured: float, reference: float) -> float:
    return abs(float(measured) - float(reference))


def relative_error(measured: float, reference: float) -> float:
    """Relative error, with an explicit convention for a zero reference."""
    denom = abs(float(reference))
    diff = abs(float(measured) - float(reference))
    if denom == 0.0:
        if diff == 0.0:
            return 0.0
        return float("inf")
    return diff / denom


def sign_of(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# rounding analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RoundingAnalysis:
    """How a mathematical candidate became the value actually stored in float32."""

    candidate_fp64: float
    candidate_as_float32: float
    stored_float32: float
    stored_uint32: int
    absolute_residual: float
    relative_residual: float
    residual_sign: int
    residual_in_ulp: float
    ulp_at_stored: float
    candidate_rounds_to_stored: bool
    #: True only when the candidate is *exactly* the midpoint between the stored
    #: value and its neighbour on the candidate's side, decided in exact rational
    #: arithmetic. Not an approximate "about half a ULP" test.
    exact_half_ulp_tie: bool
    interpretation: str
    disclaimer: str = FP64_REFERENCE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self, indent: str = "  ") -> str:
        return "\n".join(
            indent + line
            for line in (
                f"mathematical candidate (fp64) : {self.candidate_fp64!r}",
                f"candidate rounded to fp32    : {self.candidate_as_float32!r}",
                f"actual stored fp32           : {self.stored_float32!r} "
                f"(0x{self.stored_uint32:08X})",
                f"rounding residual            : {self.absolute_residual!r}",
                f"relative rounding residual   : {self.relative_residual!r}",
                f"residual sign                : {self.residual_sign:+d}",
                f"residual in ULPs             : {self.residual_in_ulp!r}",
                f"exact half-ULP tie (exact)  : {self.exact_half_ulp_tie}",
                f"interpretation               : {self.interpretation}",
            )
        )


def analyse_rounding(candidate_fp64: float, stored_tensor: torch.Tensor) -> RoundingAnalysis:
    """Compare a float64 candidate update with the float32 value actually stored.

    ``stored_tensor`` is read, never written.  ``candidate_as_float32`` is
    produced by letting PyTorch round the candidate to float32, so the comparison
    distinguishes "the optimizer's float32 arithmetic landed here" from "the
    candidate rounds to a different float32 than what was stored".
    """
    stored = f32.inspect_float32(stored_tensor)
    candidate = float(candidate_fp64)
    candidate_f32 = float(f32.float32_tensor(candidate).item())
    residual = stored.value - candidate
    ulp = f32.ulp_size(stored.value)
    residual_ulp = (float("nan") if not math.isfinite(residual) else (residual / ulp) if ulp > 0.0 else (0.0 if residual == 0.0 else float("inf")))
    matches = raw_word_from_value(candidate_f32) == stored.raw_uint32
    tie = _is_exact_half_ulp_tie(candidate, stored, candidate_f32)
    if not (math.isfinite(candidate) and stored.is_finite):
        interpretation = "a non-finite candidate or stored value has no finite rounding residual"
    elif residual == 0.0:
        interpretation = "the candidate is exactly representable in float32"
    elif matches:
        interpretation = (
            "the float32 rounding of the candidate is the stored value; the "
            "residual is pure rounding error"
        )
    else:
        interpretation = (
            "the float32 rounding of the candidate differs from the stored "
            "value, so the real update path performed additional float32 "
            "arithmetic before storing"
        )
    return RoundingAnalysis(
        candidate_fp64=candidate,
        candidate_as_float32=candidate_f32,
        stored_float32=stored.value,
        stored_uint32=stored.raw_uint32,
        absolute_residual=residual,
        relative_residual=relative_error(stored.value, candidate),
        residual_sign=sign_of(residual),
        residual_in_ulp=residual_ulp,
        ulp_at_stored=ulp,
        candidate_rounds_to_stored=matches,
        exact_half_ulp_tie=tie,
        interpretation=interpretation,
    )


def raw_word_from_value(value: float) -> int:
    return f32.raw_word_from_python_float(float(value))


def _is_exact_half_ulp_tie(
    candidate: float, stored: f32.Float32Bits, candidate_f32: float
) -> bool:
    """Is ``candidate`` *exactly* halfway between two adjacent float32 values?

    A round-to-nearest-even tie means the candidate sits exactly on the midpoint
    between the rounded value and its neighbour on the candidate's side. The
    midpoint of adjacent finite binary32 values is representable in binary64;
    exact rational comparison keeps this definition explicit and avoids an
    approximate tolerance test near a midpoint.
    """
    if not (math.isfinite(candidate) and math.isfinite(candidate_f32)) or candidate == candidate_f32:
        return False  # exactly representable, or not a rounding decision at all
    if candidate > candidate_f32:
        neighbour = f32.ulp_neighbourhood(candidate_f32).next_representable
    else:
        neighbour = f32.ulp_neighbourhood(candidate_f32).previous_representable
    if not math.isfinite(neighbour):
        return False
    neighbour_raw = raw_word_from_value(neighbour)
    candidate_raw = raw_word_from_value(candidate_f32)
    exact_midpoint = (
        f32._exact_fraction(candidate_raw) + f32._exact_fraction(neighbour_raw)
    ) / 2
    return Fraction(candidate) == exact_midpoint


# --------------------------------------------------------------------------- #
# movement analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MovementAnalysis:
    """The authoritative delta: stored-after minus stored-before."""

    before: float
    after: float
    delta: float
    absolute_delta: float
    relative_delta: float
    direction: str
    before_uint32: int
    after_uint32: int
    bits_identical: bool
    ulp_size: float
    ulp_movement: float
    #: Representable float32 values strictly between before and after.  Correct
    #: but dominated by the subnormal range when a step crosses zero.
    ulp_distance: int
    #: Signed movement in ULPs of the *starting* value.  This is the per-step
    #: movement metric; it does not blow up at binade or sign boundaries.
    ulp_distance_same_binade: int
    attempted_but_not_representable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self, indent: str = "  ") -> str:
        return "\n".join(
            indent + line
            for line in (
                f"w before                 : {self.before!r} (0x{self.before_uint32:08X})",
                f"w after                  : {self.after!r} (0x{self.after_uint32:08X})",
                f"actual delta (after-before): {self.delta!r}",
                f"absolute delta           : {self.absolute_delta!r}",
                f"relative delta           : {self.relative_delta!r}",
                f"direction                : {self.direction}",
                f"stored bits identical    : {self.bits_identical}",
                f"ULP at w_before          : {self.ulp_size!r}",
                f"displacement / ULP(before): {self.ulp_movement!r}",
                f"ULPs of starting value   : {self.ulp_distance_same_binade:+d}",
                f"representable values between: {self.ulp_distance}",
                f"attempted but below ULP  : {self.attempted_but_not_representable}",
            )
        )


def analyse_movement(
    before_tensor: torch.Tensor,
    after_tensor: torch.Tensor,
    attempted_delta: float | None = None,
) -> MovementAnalysis:
    """Delta of the *stored* values, plus whether an update was attempted.

    ``attempted_delta`` is the candidate displacement reported by the optimizer
    reconstruction.  When it is non-zero while the stored bits are identical, the
    step is recorded as "update attempted, FP32 storage did not move", which is a
    fundamentally different event from "no update was attempted".
    """
    before = f32.inspect_float32(before_tensor)
    after = f32.inspect_float32(after_tensor)
    delta = after.value - before.value
    ulp = f32.ulp_size(before.value)
    identical = before.raw_uint32 == after.raw_uint32
    return MovementAnalysis(
        before=before.value,
        after=after.value,
        delta=delta,
        absolute_delta=abs(delta),
        relative_delta=relative_error(after.value, before.value),
        direction=("non-finite" if not math.isfinite(delta) else "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged"),
        before_uint32=before.raw_uint32,
        after_uint32=after.raw_uint32,
        bits_identical=identical,
        ulp_size=ulp,
        ulp_movement=(float("nan") if not (before.is_finite and after.is_finite) else (abs(delta) / ulp) if ulp > 0.0 else 0.0),
        # NaN has no ordering. Keep the integer transport field neutral and
        # expose the undefined movement via direction/finite and NaN magnitude.
        ulp_distance=(f32.ulp_distance(before.value, after.value) if not (math.isnan(before.value) or math.isnan(after.value)) else 0),
        ulp_distance_same_binade=f32.ulp_distance_same_binade(
            before.value, after.value
        ),
        attempted_but_not_representable=bool(
            identical and attempted_delta is not None and math.isfinite(attempted_delta) and attempted_delta != 0.0
        ),
    )


# --------------------------------------------------------------------------- #
# finite differences
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FiniteDifferenceEstimate:
    """One central-difference gradient estimate and its observed error."""

    half_width: float
    estimate: float
    absolute_error: float
    relative_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def central_difference(evaluate: LossEvaluator, w: float, half_width: float) -> float:
    """(L(w+h) - L(w-h)) / 2h, evaluated by actually running the model twice."""
    if not math.isfinite(half_width) or half_width == 0.0:
        raise NumericalConsistencyError("finite-difference half width must be non-zero and finite")
    plus = float(evaluate(float(w) + half_width))
    minus = float(evaluate(float(w) - half_width))
    return (plus - minus) / (2.0 * half_width)


def finite_difference_sweep(
    evaluate: LossEvaluator,
    w: float,
    reference_derivative: float,
    base_half_width: float,
    decades: Sequence[float],
) -> tuple[FiniteDifferenceEstimate, ...]:
    """Sweep the perturbation size and report the best observed estimate.

    The sweep exists because a single finite-difference step is never right on
    its own: too large and the truncation error of the Taylor expansion
    dominates, too small and catastrophic cancellation in ``L(w+h) - L(w-h)``
    dominates.  Reporting the whole sweep makes that trade-off visible instead of
    hiding it behind one number.
    """
    estimates: list[FiniteDifferenceEstimate] = []
    for exponent in decades:
        half_width = base_half_width * (10.0 ** float(exponent))
        estimate = central_difference(evaluate, w, half_width)
        estimates.append(
            FiniteDifferenceEstimate(
                half_width=half_width,
                estimate=estimate,
                absolute_error=absolute_error(estimate, reference_derivative),
                relative_error=relative_error(estimate, reference_derivative),
            )
        )
    return tuple(estimates)


def finite_difference_limitation_note() -> str:
    return (
        "Finite differences are themselves numerically inaccurate: the central "
        "difference has O(h^2) truncation error and its roundoff term grows like "
        "eps*|L|/h, so no single perturbation size is optimal. The sweep above "
        "shows the trade-off directly."
    )


# --------------------------------------------------------------------------- #
# one-dimensional landscape probe
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LandscapeSlice:
    """A one-dimensional diagnostic slice of the loss around the live parameter."""

    step: int
    center: float
    center_uint32: int
    radius: float
    points: int
    dtype: str
    samples: tuple[tuple[float, float], ...]
    loss_at_center: float
    loss_at_center_remeasured: float
    central_first_difference: float
    central_second_difference: float
    minimum_loss: float
    argmin_offset: float
    restoration_verified: bool
    disclaimer: str = ONE_DIMENSIONAL_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["samples"] = [list(pair) for pair in self.samples]
        return data

    def render(self, indent: str = "  ") -> str:
        lines = [
            f"slice centre (w)      : {self.center!r} (0x{self.center_uint32:08X})",
            f"slice radius          : {self.radius!r} over {self.points} points",
            f"evaluation dtype      : {self.dtype}",
            f"loss at centre        : {self.loss_at_center!r}",
            f"loss re-measured      : {self.loss_at_center_remeasured!r}",
            f"local first derivative: {self.central_first_difference!r}",
            f"local second deriv.   : {self.central_second_difference!r}",
            f"minimum loss on slice : {self.minimum_loss!r} at offset {self.argmin_offset!r}",
            f"parameter restored    : {self.restoration_verified}",
            f"disclaimer            : {self.disclaimer}",
        ]
        return "\n".join(indent + line for line in lines)


def probe_landscape(
    evaluate: LossEvaluator,
    center: float,
    radius: float,
    points: int,
    *,
    step: int,
    dtype: str = "float64",
    center_uint32: int = 0,
    evaluate_type: str = "float64",
    restoration_verified: bool = True,
) -> LandscapeSlice:
    """Sample ``L(w + offset)`` on a symmetric grid around ``center``.

    ``evaluate`` must not touch the live parameter.  ``evaluate_type`` records
    which arithmetic the probe itself used, because probing in float64 and
    probing in float32 answer slightly different questions.
    """
    if points < 5 or points % 2 == 0:
        raise NumericalConsistencyError("landscape probe needs an odd point count >= 5")
    span = 2.0 * float(radius)
    step_size = span / (points - 1)
    samples: list[tuple[float, float]] = []
    for index in range(points):
        offset = -float(radius) + index * step_size
        samples.append((offset, float(evaluate(float(center) + offset))))
    # Snap the centre sample to exactly the live value (grid arithmetic can
    # otherwise introduce a last-bit offset) and re-measure it independently.
    centre_index = points // 2
    loss_at_center = samples[centre_index][1]
    loss_remeasured = float(evaluate(float(center)))
    samples[centre_index] = (0.0, loss_remeasured)
    first = (samples[centre_index + 1][1] - samples[centre_index - 1][1]) / (
        2.0 * step_size
    )
    second = (
        samples[centre_index + 1][1]
        - 2.0 * samples[centre_index][1]
        + samples[centre_index - 1][1]
    ) / (step_size**2)
    best = min(samples, key=lambda pair: pair[1])
    return LandscapeSlice(
        step=step,
        center=float(center),
        center_uint32=center_uint32,
        radius=float(radius),
        points=points,
        dtype=evaluate_type,
        samples=tuple(samples),
        loss_at_center=resolve_centre_loss(loss_at_center, loss_remeasured),
        loss_at_center_remeasured=loss_remeasured,
        central_first_difference=first,
        central_second_difference=second,
        minimum_loss=best[1],
        argmin_offset=best[0],
        restoration_verified=restoration_verified,
    )


def resolve_centre_loss(centre_loss: float, remeasured_loss: float) -> float:
    """Return the centre loss, flagging disagreement loudly via a raised error."""
    if not (math.isnan(centre_loss) and math.isnan(remeasured_loss)) and not (
        math.isclose(centre_loss, remeasured_loss, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise NumericalConsistencyError(
            "landscape centre re-measurement disagreed with the grid sample: "
            f"{centre_loss!r} vs {remeasured_loss!r}"
        )
    return remeasured_loss


# --------------------------------------------------------------------------- #
# float64 reference track
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FP64ReferenceComparison:
    """FP32 trajectory against the independent FP64 reference trajectory."""

    step: int
    fp32_value: float
    fp64_reference_value: float
    absolute_difference: float
    relative_difference: float
    difference_in_ulp_of_fp32: float
    disclaimer: str = FP64_REFERENCE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_fp64_reference(
    step: int, fp32_value: float, fp64_value: float
) -> FP64ReferenceComparison:
    fp32_repr = float(f32.float32_tensor(fp32_value).item())
    diff = fp32_repr - float(fp64_value)
    ulp = f32.ulp_size(fp32_repr)
    return FP64ReferenceComparison(
        step=step,
        fp32_value=fp32_repr,
        fp64_reference_value=float(fp64_value),
        absolute_difference=abs(diff),
        relative_difference=relative_error(fp32_repr, fp64_value),
        difference_in_ulp_of_fp32=(abs(diff) / ulp) if ulp > 0.0 else 0.0,
    )


def fp64_candidate_from_state(
    *,
    weight_before: float,
    gradient: float,
    moment1_before: float,
    moment2_before: float,
    step_index: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    optimizer: str,
    momentum_before: float | None = None,
    momentum: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False,
    coupled_decay: bool = False,
    maximize: bool = False,
    amsgrad: bool = False,
    max_moment2_before: float | None = None,
) -> float:
    """A textbook float64 reconstruction of the update, from stored FP32 inputs.

    This is a *reference* computation, not a claim about what PyTorch did: it
    uses the textbook recurrences in double precision, which differ from the
    installed float32 implementation by construction (different rounding, and in
    some versions a different op order for the moment update).  Its purpose is
    to quantify the size of that difference.
    """
    if optimizer in ("adam", "adamw"):
        gradient_effective = -gradient if maximize else gradient
        if optimizer == "adam" and coupled_decay and weight_decay != 0.0:
            gradient_effective += weight_decay * weight_before
        moment1 = beta1 * moment1_before + (1.0 - beta1) * gradient_effective
        moment2 = beta2 * moment2_before + (1.0 - beta2) * gradient_effective**2
        bias_correction1 = 1.0 - beta1**step_index
        bias_correction2 = 1.0 - beta2**step_index
        if bias_correction1 == 0.0 or bias_correction2 == 0.0:
            raise NumericalConsistencyError("bias correction underflowed to zero")
        step_size = lr / bias_correction1
        if amsgrad and max_moment2_before is not None:
            moment2 = max(max_moment2_before, moment2) if not math.isnan(moment2) else moment2
        denom = math.sqrt(moment2) / math.sqrt(bias_correction2) + eps
        ratio = moment1 / denom if denom != 0.0 else (float("nan") if moment1 == 0.0 or math.isnan(moment1) else math.copysign(float("inf"), moment1))
        adaptive = -step_size * ratio
        decay = 0.0
        if optimizer == "adamw" and weight_decay != 0.0:
            decay = -lr * weight_decay * weight_before
        return weight_before + adaptive + decay
    if optimizer == "sgd":
        grad_eff = -gradient if maximize else gradient
        if coupled_decay and weight_decay != 0.0:
            grad_eff += weight_decay * weight_before
        if momentum == 0.0 or momentum_before is None:
            # No momentum buffer yet: SGD seeds the buffer with the effective gradient.
            buffer_source = grad_eff
        else:
            buffer_source = momentum * momentum_before + (1.0 - dampening) * grad_eff
        direction = grad_eff + momentum * buffer_source if nesterov else buffer_source
        decay = 0.0
        if not coupled_decay and weight_decay != 0.0:
            decay = -lr * weight_decay * weight_before
        return weight_before - lr * direction + decay
    raise NumericalConsistencyError(f"no float64 reference for optimizer {optimizer!r}")


__all__ = [
    "LossEvaluator",
    "DerivativeEvaluator",
    "ONE_DIMENSIONAL_DISCLAIMER",
    "FP64_REFERENCE_DISCLAIMER",
    "RoundingAnalysis",
    "MovementAnalysis",
    "FiniteDifferenceEstimate",
    "LandscapeSlice",
    "FP64ReferenceComparison",
    "absolute_error",
    "relative_error",
    "sign_of",
    "analyse_rounding",
    "analyse_movement",
    "central_difference",
    "finite_difference_sweep",
    "finite_difference_limitation_note",
    "probe_landscape",
    "compare_fp64_reference",
    "fp64_candidate_from_state",
]
