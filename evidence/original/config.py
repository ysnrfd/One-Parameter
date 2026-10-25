"""Experiment configuration and the real-training presets.

Every knob that can change the numerical trajectory of the tracked parameter is
declared here.  ``ExperimentConfig`` is frozen: a run is defined by the value it
was created with, and the recorder stores that exact object in every export, so a
trajectory can always be traced back to the configuration that produced it.

Presets differ from one another only through these fields.  A preset is a real
training configuration; no preset supplies a trajectory, and every preset is
executed by exactly the same code path as a hand-written configuration.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .errors import ConfigError

#: Optimizers this project can build, reconstruct and inspect.
SUPPORTED_OPTIMIZERS: tuple[str, ...] = ("sgd", "adam", "adamw")

#: Presets that deliberately drive the parameter into an FP32 dead zone.
#: These exist because "the update was attempted but the stored float32 value
#: did not move" is a distinct phenomenon from "no update was attempted".
RESOLUTION_SENSITIVE_OPTIMIZER: str = "adamw"


@dataclass(frozen=True)
class DatasetConfig:
    """Deterministic, real input/target stream consumed by both tracks.

    The targets are constructed so that the loss has a *known* global optimum in
    one dimension: the model evaluates ``o * tanh(s * x * w)`` and the targets
    are produced from a known ``w_true``.  At ``w == w_true`` the loss is exactly
    zero, which makes the approach to the optimum directly observable.  This is
    a property of the experiment design, not a substitute for real training:
    the data is still fed to a real forward pass and the gradient still comes
    from autograd.
    """

    samples: int = 512
    batch_size: int = 1
    input_min: float = -2.0
    input_max: float = 2.0
    target_true_parameter: float = 1.5
    noise_std: float = 0.0
    shuffle_batches: bool = False
    seed: int = 20260926
    #: 'linear' walks the batch index space in order; 'alternating' flips the
    #: sign of every second input (keeps |x| constant, flips the gradient sign).
    walk: str = "linear"

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ConfigError("DatasetConfig.samples must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("DatasetConfig.batch_size must be >= 1")
        if self.batch_size > self.samples:
            raise ConfigError("DatasetConfig.batch_size cannot exceed samples")
        if not math.isfinite(self.input_min) or not math.isfinite(self.input_max):
            raise ConfigError("DatasetConfig input range bounds must be finite")
        if self.input_max < self.input_min:
            raise ConfigError("DatasetConfig.input_max must be >= input_min")
        if not math.isfinite(self.target_true_parameter):
            raise ConfigError("DatasetConfig.target_true_parameter must be finite")
        if self.walk not in ("linear", "alternating"):
            raise ConfigError(f"unsupported DatasetConfig.walk: {self.walk!r}")
        if not math.isfinite(self.noise_std) or self.noise_std < 0.0:
            raise ConfigError("DatasetConfig.noise_std must be finite and >= 0")


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the tiny real PyTorch model that owns the tracked parameter.

    ``input_scale`` and ``output_scale`` are registered as *buffers*, not
    parameters: the module therefore contains exactly one trainable tensor in
    total, so "exactly one parameter" holds for the whole model and not merely
    for the tracked handle.
    """

    input_scale: float = 1.0
    output_scale: float = 1.0
    use_tanh: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.input_scale) or self.input_scale == 0.0:
            raise ConfigError("ModelConfig.input_scale must be finite and non-zero")
        if not math.isfinite(self.output_scale):
            raise ConfigError("ModelConfig.output_scale must be finite")


@dataclass(frozen=True)
class LandscapeConfig:
    """One-dimensional loss-landscape probe (a diagnostic slice, not a landscape)."""

    enabled: bool = True
    every: int = 25
    radius: float = 0.25
    points: int = 17
    dtype: str = "float64"  # 'float64' or 'float32'

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ConfigError("LandscapeConfig.every must be >= 1")
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ConfigError("LandscapeConfig.radius must be finite and > 0")
        if self.points < 5 or self.points % 2 == 0:
            raise ConfigError("LandscapeConfig.points must be an odd integer >= 5")
        if self.dtype not in ("float64", "float32"):
            raise ConfigError("LandscapeConfig.dtype must be 'float64' or 'float32'")


@dataclass(frozen=True)
class ValidationConfig:
    """Independent cross-checks of the real gradient and the real update."""

    enabled: bool = True
    every: int = 1
    #: Relative tolerance for autograd vs the closed-form derivative.
    gradient_rtol: float = 1e-6
    #: Relative tolerance for the central-difference estimate.  Finite
    #: differences are themselves inaccurate, so this is deliberately looser.
    finite_difference_rtol: float = 5e-5
    #: Half-widths (in decades) for the finite-difference step-size sweep.
    fd_sweep_decades: tuple[float, ...] = (-10.0, -8.0, -6.0, -4.0, -2.0)
    #: Tolerance in ULPs for the bit-level optimizer reconstruction.
    reconstruction_ulp_tolerance: int = 0
    #: Try every documented update-order variant and keep the bit-exact one.
    search_update_variants: bool = True

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ConfigError("ValidationConfig.every must be >= 1")
        if not math.isfinite(self.gradient_rtol) or not math.isfinite(self.finite_difference_rtol):
            raise ConfigError("validation tolerances must be finite and > 0")
        if self.gradient_rtol <= 0.0 or self.finite_difference_rtol <= 0.0:
            raise ConfigError("validation tolerances must be > 0")
        if self.reconstruction_ulp_tolerance < 0:
            raise ConfigError("ValidationConfig.reconstruction_ulp_tolerance must be >= 0")


@dataclass(frozen=True)
class StabilityCriteria:
    """Explicit, inspectable thresholds for the regime classifier.

    The classifier reports an *observed regime* only.  It never reports
    convergence, because a small update magnitude, a small gradient magnitude
    and FP32 storage stagnation are three different observations and none of
    them implies anything about a global optimum.
    """

    window: int = 50
    update_eps: float = 1e-12
    gradient_eps: float = 1e-12
    stationary_rel: float = 1e-9
    oscillation_min_sign_changes: int = 3
    divergence_loss_growth: float = 1.05
    #: Divergence also requires the loss to grow by at least this much in
    #: absolute terms. Without it, a loss sitting near zero can change by a
    #: large *factor* purely from float32 noise and be mislabelled as diverging.
    divergence_absolute: float = 1e-3
    #: Fraction of window steps whose stored FP32 bits did not change at all.
    stagnant_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ConfigError("StabilityCriteria.window must be >= 2")
        if not math.isfinite(self.update_eps) or self.update_eps <= 0.0:
            raise ConfigError("StabilityCriteria.update_eps must be finite and > 0")
        if not math.isfinite(self.gradient_eps) or self.gradient_eps <= 0.0:
            raise ConfigError("StabilityCriteria.gradient_eps must be finite and > 0")
        if not math.isfinite(self.stationary_rel) or self.stationary_rel <= 0.0:
            raise ConfigError("StabilityCriteria.stationary_rel must be finite and > 0")
        if self.oscillation_min_sign_changes < 1:
            raise ConfigError("StabilityCriteria.oscillation_min_sign_changes must be >= 1")
        if not math.isfinite(self.divergence_loss_growth) or self.divergence_loss_growth <= 1.0:
            raise ConfigError("StabilityCriteria.divergence_loss_growth must be finite and > 1.0")
        if not math.isfinite(self.divergence_absolute) or self.divergence_absolute <= 0.0:
            raise ConfigError("StabilityCriteria.divergence_absolute must be finite and > 0")
        if not 0.0 <= self.stagnant_fraction <= 1.0:
            raise ConfigError("StabilityCriteria.stagnant_fraction must satisfy 0 <= f <= 1")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, self-contained description of one real training run."""

    # --- identity / reproducibility -------------------------------------
    name: str = "default"
    seed: int = 20260926

    # --- the tracked parameter -------------------------------------------
    initial_value: float = -2.0

    # --- optimizer --------------------------------------------------------
    optimizer: str = RESOLUTION_SENSITIVE_OPTIMIZER
    lr: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    amsgrad: bool = False
    maximize: bool = False

    # --- learning-rate schedule ------------------------------------------
    lr_schedule: str = "constant"  # 'constant' | 'cosine' | 'exponential'
    lr_final_factor: float = 0.1  # cosine: lr_min/lr_max; exponential: gamma

    # --- training loop ----------------------------------------------------
    steps: int = 200
    zero_grad_set_to_none: bool = True
    #: Number of forward/backward micro-batches whose gradients are summed
    #: before a single optimizer step (real gradient accumulation).
    accumulation_micro_batches: int = 1
    stop_on_nonfinite: bool = True

    # --- components -------------------------------------------------------
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    landscape: LandscapeConfig = field(default_factory=LandscapeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    stability: StabilityCriteria = field(default_factory=StabilityCriteria)

    # --- instrumentation --------------------------------------------------
    run_fp64_reference_track: bool = True
    record_full_state_snapshots: bool = True
    #: Default step rate for the live view.  High enough that the recorded
    #: series grows visibly; the view's own frame rate is set separately.
    live_speed_steps_per_second: float = 30.0

    def __post_init__(self) -> None:
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ConfigError(
                f"unsupported optimizer {self.optimizer!r}; "
                f"expected one of {SUPPORTED_OPTIMIZERS}"
            )
        if self.steps < 1:
            raise ConfigError("ExperimentConfig.steps must be >= 1")
        if not math.isfinite(self.initial_value):
            raise ConfigError("ExperimentConfig.initial_value must be finite")
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ConfigError("ExperimentConfig.lr must be finite and > 0")
        if len(self.betas) != 2:
            raise ConfigError("ExperimentConfig.betas must have exactly two entries")
        for name, beta in zip(("beta1", "beta2"), self.betas):
            if not 0.0 <= beta < 1.0:
                raise ConfigError(f"{name} must satisfy 0 <= beta < 1, got {beta!r}")
        if not math.isfinite(self.eps) or self.eps < 0.0:
            raise ConfigError("ExperimentConfig.eps must be finite and >= 0")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ConfigError("ExperimentConfig.weight_decay must be finite and >= 0")
        if not 0.0 <= self.momentum < 1.0:
            raise ConfigError("ExperimentConfig.momentum must satisfy 0 <= m < 1")
        if not 0.0 <= self.dampening < 1.0:
            raise ConfigError("ExperimentConfig.dampening must satisfy 0 <= d < 1")
        if self.nesterov and (self.momentum <= 0.0 or self.dampening != 0.0):
            raise ConfigError("nesterov requires momentum > 0 and dampening == 0")
        if self.accumulation_micro_batches < 1:
            raise ConfigError("accumulation_micro_batches must be >= 1")
        if self.lr_schedule not in ("constant", "cosine", "exponential"):
            raise ConfigError(f"unsupported lr_schedule: {self.lr_schedule!r}")
        if self.lr_schedule == "exponential" and not 0.0 < self.lr_final_factor <= 1.0:
            raise ConfigError("exponential schedule requires 0 < factor <= 1")
        if not math.isfinite(self.live_speed_steps_per_second) or self.live_speed_steps_per_second <= 0.0:
            raise ConfigError("live_speed_steps_per_second must be > 0")

    # -- derived helpers ---------------------------------------------------
    def optimizer_kwargs(self) -> dict[str, Any]:
        """Optimizer keyword arguments for this configuration (unsupported keys
        are filtered later against the installed constructor signature)."""
        common: dict[str, Any] = {"lr": self.lr, "weight_decay": self.weight_decay}
        if self.optimizer == "sgd":
            common.update(
                momentum=self.momentum,
                dampening=self.dampening,
                nesterov=self.nesterov,
            )
        else:
            common.update(betas=self.betas, eps=self.eps, amsgrad=self.amsgrad)
        if self.maximize:
            common["maximize"] = True
        return common

    def replace(self, **changes: Any) -> "ExperimentConfig":
        """Return a new validated configuration with ``changes`` applied."""
        return dataclasses.replace(self, **changes)

    def replace_dataset(self, **changes: Any) -> "ExperimentConfig":
        return self.replace(dataset=dataclasses.replace(self.dataset, **changes))

    def replace_model(self, **changes: Any) -> "ExperimentConfig":
        return self.replace(model=dataclasses.replace(self.model, **changes))

    def replace_landscape(self, **changes: Any) -> "ExperimentConfig":
        return self.replace(landscape=dataclasses.replace(self.landscape, **changes))

    def replace_validation(self, **changes: Any) -> "ExperimentConfig":
        return self.replace(validation=dataclasses.replace(self.validation, **changes))

    def replace_stability(self, **changes: Any) -> "ExperimentConfig":
        return self.replace(stability=dataclasses.replace(self.stability, **changes))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _preset(name: str, description: str, **changes: Any) -> tuple[str, str, ExperimentConfig]:
    base = ExperimentConfig(name=name, **changes)
    return name, description, base


def _build_preset_specs() -> list[tuple[str, str, ExperimentConfig]]:
    return [
        _preset(
            "adamw",
            "Default. Decoupled weight decay, real autograd, moderate steps.",
            optimizer="adamw",
            lr=0.1,
            weight_decay=0.01,
            steps=200,
        ),
        _preset(
            "adamw_decay_only",
            "Weight decay as the only driving force (zero-initial gradient is "
            "impossible to force, so this preset uses a far-from-optimum start "
            "and isolates the decoupled decay term in the update breakdown).",
            optimizer="adamw",
            lr=0.1,
            weight_decay=0.5,
            initial_value=8.0,
            steps=200,
        ),
        _preset(
            "adam",
            "L2 weight decay coupled into the gradient by the optimizer itself.",
            optimizer="adam",
            lr=0.1,
            weight_decay=0.01,
            steps=200,
        ),
        _preset(
            "sgd",
            "Plain SGD, no momentum, no weight decay: the simplest real update.",
            optimizer="sgd",
            lr=0.05,
            weight_decay=0.0,
            steps=200,
        ),
        _preset(
            "sgd_momentum",
            "SGD with momentum and Nesterov acceleration.",
            optimizer="sgd",
            lr=0.02,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0,
            steps=200,
        ),
        _preset(
            "small_gradient",
            "Small-gradient regime: |x| is tiny, so dL/dw is tiny as well.",
            optimizer="adamw",
            lr=0.1,
            steps=200,
        ),
        _preset(
            "large_gradient",
            "Large-gradient regime: |x| and the initial distance are both large.",
            optimizer="adamw",
            lr=0.05,
            initial_value=-12.0,
            steps=200,
        ),
        _preset(
            "near_stationary",
            "Near-stationary regime: the parameter starts at the optimum, so the "
            "gradient is ~0 from the first step onward.",
            optimizer="adamw",
            lr=0.1,
            initial_value=1.5,
            steps=200,
        ),
        _preset(
            "oscillatory",
            "Oscillatory regime: a large learning rate on a sign-alternating "
            "input stream makes the update sign change repeatedly.",
            optimizer="adamw",
            lr=1.2,
            initial_value=-3.0,
            steps=200,
        ),
        _preset(
            "fp32_resolution",
            "FP32-resolution-sensitive regime: the learning rate is small "
            "enough that the optimizer's candidate update falls below one ULP "
            "of the stored float32 value, so updates are attempted but the "
            "stored parameter does not move. The start is offset from the known "
            "optimum so the gradient is genuinely non-zero - starting exactly at "
            "it would make the candidate update zero and demonstrate nothing.",
            optimizer=RESOLUTION_SENSITIVE_OPTIMIZER,
            lr=1e-9,
            weight_decay=0.0,
            initial_value=1.5001,
            steps=60,
        ),
        _preset(
            "diverging",
            "Divergence regime: the learning rate is far outside the stable "
            "range, which drives the real loss and the real parameter to "
            "non-finite values so the non-finite detector is exercised. Uses the "
            "linear (non-tanh) model, because a saturated tanh kills the gradient "
            "and the run would stall at a large finite value instead of diverging.",
            optimizer="sgd",
            lr=1e9,
            weight_decay=0.0,
            initial_value=-5.0,
            steps=40,
        ),
    ]


def build_presets() -> dict[str, ExperimentConfig]:
    """Real training presets.  Every one is executed by the normal engine."""
    specs = _build_preset_specs()
    # Presets that need a different dataset get it applied here, in one place,
    # so the table above stays readable.
    overrides: dict[str, Callable[[ExperimentConfig], ExperimentConfig]] = {
        "small_gradient": lambda c: c.replace_dataset(
            input_min=-1e-3, input_max=1e-3
        ),
        "large_gradient": lambda c: c.replace_dataset(
            input_min=-6.0, input_max=6.0
        ),
        "oscillatory": lambda c: c.replace_dataset(walk="alternating"),
        "diverging": lambda c: c.replace_dataset(
            input_min=-8.0, input_max=8.0
        ).replace_model(use_tanh=False),
    }
    presets: dict[str, ExperimentConfig] = {}
    for name, _description, cfg in specs:
        apply = overrides.get(name)
        presets[name] = apply(cfg) if apply else cfg
    return presets


#: Human-readable one-line descriptions, kept next to the presets themselves.
PRESET_DESCRIPTIONS: Mapping[str, str] = {
    name: description
    for name, description, _cfg in _build_preset_specs()
}

PRESETS: dict[str, ExperimentConfig] = build_presets()

__all__ = [
    "SUPPORTED_OPTIMIZERS",
    "DatasetConfig",
    "ModelConfig",
    "LandscapeConfig",
    "ValidationConfig",
    "StabilityCriteria",
    "ExperimentConfig",
    "PRESETS",
    "PRESET_DESCRIPTIONS",
    "build_presets",
]
