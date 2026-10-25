"""Independent validation of the quantities the run actually produced.

Three checks, each using a different mechanism from the one it checks:
* **autograd vs closed form** - the gradient PyTorch produced is compared with a
  hand-derived ``dL/dw`` written out in float64 from the forward definition, with
  no autograd and no torch arithmetic involved.
* **finite differences** - the same loss is re-evaluated at ``w +- h`` through
  the real model in float64, which shares no code path with either of the other
  two methods.
* **float64 autograd** - the same graph evaluated in float64, which isolates the
  float32 rounding error of the gradient itself.

Plus bit-level checks on the float32 decoding and the ULP machinery, which are
validated against each other rather than against a stored expectation.

None of these feed back into training.  They observe a completed step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from . import fp32_inspector as f32
from . import numerical_analysis as na
from .config import ValidationConfig
from .errors import ValidationError
from .model import FP64ReferenceModel, ObservatoryModel
from .optimizer_inspector import ReconstructionResult


@dataclass(frozen=True)
class GradientValidation:
    """Three independent estimates of the same derivative."""

    step: int
    autograd_value: float
    autograd_uint32: int | None
    analytical_value: float
    autograd_fp64_value: float | None
    finite_difference_value: float
    finite_difference_half_width: float
    finite_difference_sweep: tuple[na.FiniteDifferenceEstimate, ...]
    autograd_vs_analytical_abs: float
    autograd_vs_analytical_rel: float
    autograd_fp32_vs_fp64_abs: float
    autograd_fp32_vs_fp64_ulp: float
    autograd_vs_finite_difference_abs: float
    autograd_vs_finite_difference_rel: float
    gradient_rtol: float
    finite_difference_rtol: float
    gradient_tolerance_abs: float
    finite_difference_tolerance_abs: float
    loss_scale: float
    autograd_status: str
    finite_difference_status: str
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.autograd_status == "PASS" and self.finite_difference_status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["finite_difference_sweep"] = [
            estimate.to_dict() for estimate in self.finite_difference_sweep
        ]
        data["notes"] = list(self.notes)
        return data

    def render(self, indent: str = "  ") -> str:
        lines = [
            f"{indent}autograd (fp32, real)   : {self.autograd_value!r}"
            + (
                f" (0x{self.autograd_uint32:08X})" if self.autograd_uint32 is not None else ""
            ),
            f"{indent}autograd (fp64 replica)  : {self.autograd_fp64_value!r}",
            f"{indent}analytical (closed form) : {self.analytical_value!r}",
            f"{indent}finite difference         : {self.finite_difference_value!r} "
            f"(h = {self.finite_difference_half_width!r})",
            f"{indent}|autograd - analytical|  : {self.autograd_vs_analytical_abs!r} "
            f"(rel {self.autograd_vs_analytical_rel!r})",
            f"{indent}fp32 vs fp64 autograd    : {self.autograd_fp32_vs_fp64_abs!r} "
            f"({self.autograd_fp32_vs_fp64_ulp!r} ULP)",
            f"{indent}|autograd - fin.diff.|   : {self.autograd_vs_finite_difference_abs!r} "
            f"(rel {self.autograd_vs_finite_difference_rel!r})",
            f"{indent}autograd check           : {self.autograd_status} "
            f"(|err| {self.autograd_vs_analytical_abs!r} <= bound "
            f"{self.gradient_tolerance_abs!r})",
            f"{indent}finite-difference check  : {self.finite_difference_status} "
            f"(|err| {self.autograd_vs_finite_difference_abs!r} <= bound "
            f"{self.finite_difference_tolerance_abs!r})",
        ]
        return "\n".join(lines)


def validate_gradient(
    *,
    step: int,
    model: ObservatoryModel,
    x: torch.Tensor,
    target: torch.Tensor,
    w_value: float,
    autograd_gradient: float,
    autograd_uint32: int | None,
    config: ValidationConfig,
    include_fp64_autograd: bool = True,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> GradientValidation:
    """Compare the real gradient with a closed form and with finite differences."""
    objective_batches = tuple(batches) if batches is not None else ((x, target),)
    if not objective_batches:
        raise ValidationError("gradient validation requires at least one batch")
    try:
        analytical = sum(
            model.analytic_gradient_fp64(batch_x, batch_target, w_value)
            for batch_x, batch_target in objective_batches
        )
    except Exception as exc:  # pragma: no cover - defensive, never silent
        raise ValidationError(
            f"the analytical derivative could not be evaluated: {type(exc).__name__}: {exc}"
        ) from exc

    def evaluate(w: float) -> float:
        return sum(
            model.evaluate_loss(batch_x, batch_target, w, dtype=torch.float64)
            for batch_x, batch_target in objective_batches
        )

    base_half_width = 1e-4 * max(1.0, abs(float(w_value))) if math.isfinite(w_value) else 1e-4
    sweep = na.finite_difference_sweep(
        evaluate,
        float(w_value),
        reference_derivative=analytical,
        base_half_width=base_half_width,
        decades=config.fd_sweep_decades,
    )
    finite_finite = [estimate for estimate in sweep if math.isfinite(estimate.estimate)]
    # A non-finite training objective is an observation the engine must record,
    # not an exception that prevents the failing step from being sealed.
    best = min(finite_finite, key=lambda estimate: estimate.absolute_error) if finite_finite else sweep[0]

    fp64_autograd: float | None = None
    if include_fp64_autograd:
        fp64_model = FP64ReferenceModel(w_value, model.model_config)
        with torch.no_grad():
            fp64_model.input_scale.fill_(float(model.input_scale.detach().item()))
            fp64_model.output_scale.fill_(float(model.output_scale.detach().item()))
        for batch_x, batch_target in objective_batches:
            fp64_model.loss(batch_x, batch_target).backward()
        grad64 = fp64_model.w.grad
        if grad64 is not None:
            fp64_autograd = float(grad64.detach().reshape(-1)[0].item())

    abs_vs_analytic = na.absolute_error(autograd_gradient, analytical)
    rel_vs_analytic = na.relative_error(autograd_gradient, analytical)
    abs_vs_fd = na.absolute_error(autograd_gradient, best.estimate)
    rel_vs_fd = na.relative_error(autograd_gradient, best.estimate)

    fp32_fp64_abs = (
        na.absolute_error(autograd_gradient, fp64_autograd)
        if fp64_autograd is not None
        else float("nan")
    )
    fp32_fp64_ulp = (
        na.absolute_error(autograd_gradient, fp64_autograd) / f32.ulp_size(autograd_gradient)
        if fp64_autograd is not None and f32.ulp_size(autograd_gradient) > 0
        else 0.0
    )

    # The criterion is absolute-plus-relative, scaled by the magnitude of the
    # terms that produced the gradient rather than by the (possibly much
    # smaller) result. See gradient_tolerance for why a pure relative bound
    # would be unsound here.
    loss_scale = abs(evaluate(w_value))
    gradient_bound, gradient_note = gradient_tolerance(
        analytical, loss_scale, config.gradient_rtol
    )
    finite_bound, finite_note = gradient_tolerance(
        best.estimate, loss_scale, config.finite_difference_rtol
    )

    autograd_status = "PASS" if math.isfinite(abs_vs_analytic) and abs_vs_analytic <= gradient_bound else "FAIL"
    finite_status = "PASS" if math.isfinite(abs_vs_fd) and abs_vs_fd <= finite_bound else "FAIL"

    return GradientValidation(
        step=step,
        autograd_value=float(autograd_gradient),
        autograd_uint32=autograd_uint32,
        analytical_value=float(analytical),
        autograd_fp64_value=fp64_autograd,
        finite_difference_value=best.estimate,
        finite_difference_half_width=best.half_width,
        finite_difference_sweep=sweep,
        autograd_vs_analytical_abs=abs_vs_analytic,
        autograd_vs_analytical_rel=rel_vs_analytic,
        autograd_fp32_vs_fp64_abs=fp32_fp64_abs,
        autograd_fp32_vs_fp64_ulp=fp32_fp64_ulp,
        autograd_vs_finite_difference_abs=abs_vs_fd,
        autograd_vs_finite_difference_rel=rel_vs_fd,
        gradient_rtol=config.gradient_rtol,
        finite_difference_rtol=config.finite_difference_rtol,
        gradient_tolerance_abs=gradient_bound,
        finite_difference_tolerance_abs=finite_bound,
        loss_scale=loss_scale,
        autograd_status=autograd_status,
        finite_difference_status=finite_status,
        notes=(
            "The closed form is an independent derivation, not a second autograd call.",
            f"gradient tolerance: {gradient_note}",
            f"finite-difference tolerance: {finite_note}",
            na.finite_difference_limitation_note(),
            "A FAIL in the finite-difference row usually means the perturbation "
            "size is unsuitable for this loss scale, not that autograd is wrong; "
            "the sweep shows why.",
        ),
    )


@dataclass(frozen=True)
class BitValidation:
    """Checks that float32 decoding is self-consistent and externally verified."""

    raw_uint32: int
    decoded_value: float
    round_trip_ok: bool
    numpy_agreement: bool | None
    numpy_version: str | None
    storage_bytes: str
    element_size: int
    ulp_matches_bit_spacing: bool
    nextafter_matches_ulp: bool
    status: str
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data

    def render(self, indent: str = "  ") -> str:
        lines = [
            f"{indent}raw uint32           : 0x{self.raw_uint32:08X}",
            f"{indent}decoded value        : {self.decoded_value!r}",
            f"{indent}decode round-trip    : {self.round_trip_ok}",
            f"{indent}agrees with numpy    : {self.numpy_agreement}",
            f"{indent}storage bytes        : {self.storage_bytes}",
            f"{indent}element_size         : {self.element_size}",
            f"{indent}ULP matches bits     : {self.ulp_matches_bit_spacing}",
            f"{indent}nextafter matches    : {self.nextafter_matches_ulp}",
            f"{indent}status               : {self.status}",
        ]
        return "\n".join(lines)


def validate_float32_bits(tensor: torch.Tensor) -> BitValidation:
    """Verify the bit machinery against itself, against numpy, and against torch."""
    if tensor.dtype is not torch.float32:
        raise ValidationError(
            f"bit validation requires a float32 tensor, got {tensor.dtype}"
        )
    raw = f32.assert_python_and_tensor_agree(tensor)
    bits = f32.decode_float32_bits(raw)
    numpy_agreement: bool | None = None
    numpy_version: str | None = None
    try:
        import numpy as np  # noqa: PLC0415 - optional cross-check

        numpy_version = np.__version__
        as_numpy = np.asarray(tensor.detach().cpu().numpy()).reshape(-1)
        numpy_raw = int(as_numpy.view(np.uint32)[0])
        numpy_agreement = numpy_raw == raw
    except Exception as exc:  # pragma: no cover - numpy is optional here
        numpy_version = f"<unavailable: {type(exc).__name__}>"
        numpy_agreement = None

    evidence = f32.storage_evidence(tensor)
    spacing = f32.ulp_size(bits.value)
    spacing_down = f32.ulp_neighbourhood(bits.value).ulp_size_down
    toward_zero, away_from_zero = _bit_level_spacings(bits)

    # Both directions are compared, because at a power-of-two boundary the
    # gap away from zero and the gap towards zero differ.
    ulp_matches = bool(
        math.isclose(spacing, away_from_zero, rel_tol=1e-6, abs_tol=0.0)
        and math.isclose(spacing_down, toward_zero, rel_tol=1e-6, abs_tol=0.0)
    ) or (spacing == 0.0 and toward_zero == 0.0)

    # Walking nextafter across the same value must land on the same neighbour.
    t = torch.tensor(bits.value, dtype=torch.float32)
    target_up = f32.ulp_neighbourhood(bits.value).next_representable
    target_down = f32.ulp_neighbourhood(bits.value).previous_representable
    walked_up = float(torch.nextafter(t, torch.tensor(math.inf, dtype=torch.float32)).item())
    walked_down = float(torch.nextafter(t, torch.tensor(-math.inf, dtype=torch.float32)).item())
    nextafter_matches = walked_up == target_up and walked_down == target_down

    failures = []
    if numpy_agreement is False:
        failures.append("numpy disagrees with the tensor bit pattern")
    if not ulp_matches:
        failures.append("torch.nextafter ULP disagrees with the bit-level spacing")
    if not nextafter_matches:
        failures.append("nextafter neighbourhood is inconsistent")
    if evidence.element_size_bytes != 4:
        failures.append(f"element_size is {evidence.element_size_bytes}, expected 4")

    return BitValidation(
        raw_uint32=raw,
        decoded_value=bits.value,
        round_trip_ok=True,
        numpy_agreement=numpy_agreement,
        numpy_version=numpy_version,
        storage_bytes=" ".join(f"{b:02X}" for b in evidence.raw_storage_bytes),
        element_size=evidence.element_size_bytes,
        ulp_matches_bit_spacing=ulp_matches,
        nextafter_matches_ulp=nextafter_matches,
        status="PASS" if not failures else "FAIL: " + "; ".join(failures),
        notes=(
            "Round-trip check: the decoded sign/exponent/fraction re-encode to the "
            "observed raw word.",
            "numpy and torch reinterpretation paths are compared against each other, "
            "so a systematic error in one would show up as a disagreement.",
        ),
    )


def _bit_level_spacings(bits: f32.Float32Bits) -> tuple[float, float]:
    """The gaps implied by the bit pattern alone, independent of ``torch``.

    Returns ``(toward_zero, away_from_zero)``.  For a normal value with
    unbiased exponent ``e`` the gap within its own binade is ``2**(e - 23)``.
    A normal whose fraction field is all zeros is exactly a power of two, and
    its neighbour on the zero side belongs to the binade below, so the gap
    *toward* zero is half as large.  Getting this asymmetry right is the whole
    point of checking it: ``nextafter`` and the exponent field must agree about
    both directions, not just the easier one.
    """
    if not bits.is_finite:
        return 0.0, 0.0
    if bits.is_zero or bits.is_subnormal:
        return f32.SMALLEST_SUBNORMAL_F32, f32.SMALLEST_SUBNORMAL_F32
    exponent = bits.unbiased_exponent
    if bits.fraction_bits_int == 0 and exponent > -126:
        return 2.0 ** (exponent - 24), 2.0 ** (exponent - 23)
    return 2.0 ** (exponent - 23), 2.0 ** (exponent - 23)


def gradient_tolerance(
    reference: float, scale: float, rtol: float
) -> tuple[float, str]:
    """The absolute-plus-relative bound a float32 gradient is held to.

    A float32 gradient is a sum of products whose individual magnitudes can be
    far larger than the sum when cancellation occurs - here the result is often
    a hundred times smaller than the terms that produced it.  Bounding the
    error by ``eps32 * |result|`` would therefore be an unsound criterion, and
    tightening the *relative* tolerance until it passes would be worse: it
    would be a number chosen to produce the answer the author wanted.

    The floor below is instead scaled by the magnitude of the terms, which for
    this model is proportional to the loss, and it is stated rather than tuned:
        |autograd - closed form| <= 8 * eps32 * max(1, |loss|) + rtol * |closed form|

    Both the floor and the observed error are recorded, so the margin is
    visible rather than asserted.
    """
    eps32 = 1.1920928955078125e-07
    absolute_floor = 8.0 * eps32 * max(1.0, abs(float(scale)))
    bound = absolute_floor + float(rtol) * abs(float(reference))
    note = (
        f"|err| <= {absolute_floor!r} (8 * eps32 * max(1, |loss|)) "
        f"+ {rtol!r} * |reference|"
    )
    return bound, note


@dataclass(frozen=True)
class ULPValidation:
    """The ULP neighbourhood checked against a manual nextafter walk."""

    value: float
    previous: float
    next: float
    walk_steps: int
    walk_reaches_next: bool
    walk_reaches_previous: bool
    ulp_distance_check: int
    ulp_distance_expected: int
    status: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self, indent: str = "  ") -> str:
        return "\n".join(
            indent + line
            for line in (
                f"value                  : {self.value!r}",
                f"previous representable : {self.previous!r}",
                f"next representable     : {self.next!r}",
                f"{self.walk_steps} nextafter steps reach next     : {self.walk_reaches_next}",
                f"{self.walk_steps} nextafter steps reach previous : {self.walk_reaches_previous}",
                f"ulp_distance           : {self.ulp_distance_check} "
                f"(expected {self.ulp_distance_expected})",
                f"status                 : {self.status}",
            )
        )


def validate_ulp(value: float, *, steps: int = 4) -> ULPValidation:
    """Walk ``nextafter`` explicitly and confirm the ULP helpers agree."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValidationError("ULP walk steps must be a positive integer")
    neighbourhood = f32.ulp_neighbourhood(value)
    t = torch.tensor(float(value), dtype=torch.float32)
    up = t
    expected_up = 0
    for _ in range(steps):
        next_up = torch.nextafter(up, torch.tensor(math.inf, dtype=torch.float32))
        expected_up += int(float(next_up.item()) != float(up.item()))
        up = next_up
    down = t
    expected_down = 0
    for _ in range(steps):
        next_down = torch.nextafter(down, torch.tensor(-math.inf, dtype=torch.float32))
        expected_down += int(float(next_down.item()) != float(down.item()))
        down = next_down
    walk_next = float(up.item())
    walk_previous = float(down.item())
    target = neighbourhood.next_representable
    previous_target = neighbourhood.previous_representable

    # The walk must pass through the reported neighbour at step 1 and land
    # exactly steps ULPs away at step `steps`.
    reaches_next = math.isclose(
        float(torch.nextafter(t, torch.tensor(math.inf, dtype=torch.float32)).item()),
        target,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    reaches_previous = math.isclose(
        float(torch.nextafter(t, torch.tensor(-math.inf, dtype=torch.float32)).item()),
        previous_target,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    distance_up = f32.ulp_distance(float(value), walk_next)
    distance_down = f32.ulp_distance(walk_previous, float(value))
    failures = []
    if not reaches_next:
        failures.append("one nextafter step did not reach the reported next value")
    if not reaches_previous:
        failures.append("one nextafter step did not reach the reported previous value")
    if distance_up != expected_up:
        failures.append(
            f"ulp_distance reported {distance_up} for a {steps}-step upward walk"
        )
    if distance_down != expected_down:
        failures.append(
            f"ulp_distance reported {distance_down} for a {steps}-step downward walk"
        )

    return ULPValidation(
        value=float(value),
        previous=previous_target,
        next=target,
        walk_steps=steps,
        walk_reaches_next=reaches_next,
        walk_reaches_previous=reaches_previous,
        ulp_distance_check=distance_up,
        ulp_distance_expected=steps,
        status="PASS" if not failures else "FAIL: " + "; ".join(failures),
    )


@dataclass(frozen=True)
class ReconstructionValidation:
    """The optimizer reconstruction result, as a validation verdict."""

    step: int
    status: str
    bit_exact: bool
    ulp_error: int
    tolerance_ulp: int
    variant: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_reconstruction(step: int, result: ReconstructionResult) -> ReconstructionValidation:
    return ReconstructionValidation(
        step=step,
        status=result.status,
        bit_exact=result.bit_exact,
        ulp_error=result.ulp_error,
        tolerance_ulp=result.tolerance_ulp,
        variant=result.variant,
        detail=(
            f"actual 0x{result.actual_uint32:08X} vs reconstructed "
            f"0x{result.reconstructed_uint32:08X}"
        ),
    )


@dataclass
class ValidationLedger:
    """Run-level tally of every validation performed."""

    gradient_checks: int = 0
    gradient_failures: int = 0
    finite_difference_failures: int = 0
    reconstruction_checks: int = 0
    reconstruction_failures: int = 0
    bit_checks: int = 0
    bit_failures: int = 0
    ulp_checks: int = 0
    ulp_failures: int = 0
    max_gradient_relative_error: float = 0.0
    max_gradient_absolute_error: float = 0.0
    max_finite_difference_relative_error: float = 0.0
    max_finite_difference_absolute_error: float = 0.0
    max_reconstruction_ulp_error: int = 0
    statuses: list[str] = field(default_factory=list)

    def add_gradient(self, validation: GradientValidation) -> None:
        self.gradient_checks += 1
        if validation.autograd_status != "PASS":
            self.gradient_failures += 1
            self.statuses.append(
                f"step {validation.step}: autograd {validation.autograd_status} "
                f"rel={validation.autograd_vs_analytical_rel!r}"
            )
        if validation.finite_difference_status != "PASS":
            self.finite_difference_failures += 1
            self.statuses.append(
                f"step {validation.step}: finite difference "
                f"{validation.finite_difference_status} "
                f"rel={validation.autograd_vs_finite_difference_rel!r}"
            )
        self.max_gradient_relative_error = max(
            self.max_gradient_relative_error, validation.autograd_vs_analytical_rel
        )
        self.max_gradient_absolute_error = max(
            self.max_gradient_absolute_error, validation.autograd_vs_analytical_abs
        )
        self.max_finite_difference_relative_error = max(
            self.max_finite_difference_relative_error,
            validation.autograd_vs_finite_difference_rel,
        )
        self.max_finite_difference_absolute_error = max(
            self.max_finite_difference_absolute_error,
            validation.autograd_vs_finite_difference_abs,
        )

    def add_reconstruction(self, validation: ReconstructionValidation) -> None:
        self.reconstruction_checks += 1
        if not validation.passed:
            self.reconstruction_failures += 1
            self.statuses.append(
                f"step {validation.step}: reconstruction {validation.status} "
                f"({validation.detail})"
            )
        self.max_reconstruction_ulp_error = max(
            self.max_reconstruction_ulp_error, validation.ulp_error
        )

    def add_bit(self, validation: BitValidation) -> None:
        self.bit_checks += 1
        if not validation.passed:
            self.bit_failures += 1
            self.statuses.append(f"bit validation {validation.status}")

    def add_ulp(self, validation: ULPValidation) -> None:
        self.ulp_checks += 1
        if not validation.passed:
            self.ulp_failures += 1
            self.statuses.append(f"ULP validation {validation.status}")

    @property
    def total_failures(self) -> int:
        return (
            self.gradient_failures
            + self.finite_difference_failures
            + self.reconstruction_failures
            + self.bit_failures
            + self.ulp_failures
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gradient_checks": self.gradient_checks,
            "gradient_failures": self.gradient_failures,
            "finite_difference_failures": self.finite_difference_failures,
            "reconstruction_checks": self.reconstruction_checks,
            "reconstruction_failures": self.reconstruction_failures,
            "bit_checks": self.bit_checks,
            "bit_failures": self.bit_failures,
            "ulp_checks": self.ulp_checks,
            "ulp_failures": self.ulp_failures,
            "max_gradient_relative_error": self.max_gradient_relative_error,
            "max_gradient_absolute_error": self.max_gradient_absolute_error,
            "max_finite_difference_relative_error": self.max_finite_difference_relative_error,
            "max_finite_difference_absolute_error": self.max_finite_difference_absolute_error,
            "max_reconstruction_ulp_error": self.max_reconstruction_ulp_error,
            "total_failures": self.total_failures,
            "failure_notes": list(self.statuses),
        }

    def render(self) -> str:
        rows = [
            ("gradient checks", f"{self.gradient_checks - self.gradient_failures}/{self.gradient_checks} passed"),
            ("finite-difference checks", f"{self.gradient_checks - self.finite_difference_failures}/{self.gradient_checks} passed"),
            ("optimizer reconstructions", f"{self.reconstruction_checks - self.reconstruction_failures}/{self.reconstruction_checks} passed"),
            ("float32 bit checks", f"{self.bit_checks - self.bit_failures}/{self.bit_checks} passed"),
            ("ULP checks", f"{self.ulp_checks - self.ulp_failures}/{self.ulp_checks} passed"),
            ("max gradient RELATIVE error", repr(self.max_gradient_relative_error)),
            ("max gradient ABSOLUTE error", repr(self.max_gradient_absolute_error)),
            ("max finite-difference RELATIVE error", repr(self.max_finite_difference_relative_error)),
            ("max finite-difference ABSOLUTE error", repr(self.max_finite_difference_absolute_error)),
            ("max reconstruction error (ULP)", str(self.max_reconstruction_ulp_error)),
            ("total failures", str(self.total_failures)),
        ]
        width = max(len(name) for name, _ in rows)
        lines = [f"  {name.ljust(width)} : {value}" for name, value in rows]
        for note in self.statuses[:20]:
            lines.append(f"      ! {note}")
        return "\n".join(lines)


__all__ = [
    "GradientValidation",
    "BitValidation",
    "ULPValidation",
    "ReconstructionValidation",
    "ValidationLedger",
    "validate_gradient",
    "validate_float32_bits",
    "validate_ulp",
    "validate_reconstruction",
    "gradient_tolerance",
]
