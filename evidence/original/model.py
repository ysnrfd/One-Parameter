"""The experimental substrate: a real PyTorch model with exactly one trainable scalar.

The model is deliberately tiny so the whole causal chain can be exposed, but
nothing about it is a placeholder:
* ``w`` is a genuine ``nn.Parameter`` with ``requires_grad=True``;
* it is multiplied into a real autograd graph and differentiated by
  ``loss.backward()``;
* the only other tensors in the module are *buffers* (``input_scale``,
  ``output_scale``), so ``model.parameters()`` yields exactly one tensor and
  "exactly one trainable parameter" holds for the entire module;
* the same forward arithmetic is reachable both through the graph (``forward``)
  and through a pure function (``evaluate_loss``) that builds a throwaway tensor.
  The second path exists for diagnostics only and can never write to ``w``.

This is a controlled instrument for observing one scalar, not an LLM.  The README
states that distinction explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from . import fp32_inspector as f32
from .config import DatasetConfig, ModelConfig
from .errors import ConfigError, TrackingInvalidError

#: Names of the recorded forward stages, in execution order.
FORWARD_STAGE_NAMES: tuple[str, ...] = (
    "parameter",
    "input",
    "z",
    "activation",
    "prediction",
    "target",
    "loss",
)


@dataclass(frozen=True)
class TensorSnapshot:
    """One recorded tensor: what it was, and exactly what it contained."""

    name: str
    value: float
    mean: float
    dtype: str
    shape: tuple[int, ...]
    device: str
    requires_grad: bool
    grad_fn: str | None
    is_leaf: bool
    finite: bool
    numel: int
    value_is_first_element: bool
    raw_uint32: int | None = None
    bits: str | None = None
    hex: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "mean": self.mean,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "device": self.device,
            "requires_grad": self.requires_grad,
            "grad_fn": self.grad_fn,
            "is_leaf": self.is_leaf,
            "finite": self.finite,
            "numel": self.numel,
            "value_is_first_element": self.value_is_first_element,
            "raw_uint32": self.raw_uint32,
            "bits": self.bits,
            "hex": self.hex,
        }

    def render(self, indent: str = "  ") -> str:
        value_note = " (first element)" if self.value_is_first_element else ""
        return "\n".join(
            indent + line
            for line in (
                f"{self.name:<12} value={self.value!r}{value_note} mean={self.mean!r} "
                f"dtype={self.dtype} shape={self.shape} device={self.device}",
                f"{'':<12} requires_grad={self.requires_grad} "
                f"is_leaf={self.is_leaf} grad_fn={self.grad_fn} finite={self.finite}",
            )
        )


def describe_tensor(name: str, tensor: torch.Tensor, *, with_bits: bool = False) -> TensorSnapshot:
    """Snapshot a tensor without detaching it from its graph.

    The snapshot stores Python scalars and strings only, so the graph itself is
    never retained: keeping a reference to every recorded graph would leak the
    whole autograd history.  For a multi-element tensor, ``value`` is documented
    as the first element and ``mean`` carries the aggregate, rather than either
    one silently standing in for the other.
    """
    detached = tensor.detach()
    if detached.numel() < 1:
        raise ConfigError(f"describe_tensor received an empty tensor for {name!r}")
    raw = bits = hex_repr = None
    if with_bits and detached.dtype is torch.float32 and detached.numel() == 1:
        raw = f32.raw_word_from_tensor(detached)
        bits = f"{raw:032b}"
        hex_repr = f"0x{raw:08X}"
    grad_fn = tensor.grad_fn
    return TensorSnapshot(
        name=name,
        value=float(detached.reshape(-1)[0].item()),
        mean=float(detached.double().mean().item()),
        dtype=str(tensor.dtype),
        shape=tuple(int(s) for s in tensor.shape),
        device=str(tensor.device),
        requires_grad=bool(tensor.requires_grad),
        grad_fn=None if grad_fn is None else type(grad_fn).__name__,
        is_leaf=bool(tensor.is_leaf),
        finite=bool(torch.isfinite(detached).all().item()),
        numel=int(tensor.numel()),
        value_is_first_element=bool(detached.numel() != 1),
        raw_uint32=raw,
        bits=bits,
        hex=hex_repr,
    )


@dataclass(frozen=True)
class ForwardPass:
    """Everything one real forward pass produced, all of it observed."""

    prediction: float
    loss: float
    loss_grad_fn: str | None
    prediction_grad_fn: str | None
    stages: tuple[TensorSnapshot, ...]

    def stage(self, name: str) -> TensorSnapshot:
        for snapshot in self.stages:
            if snapshot.name == name:
                return snapshot
        raise KeyError(f"no forward stage named {name!r}")

    def stage_values(self) -> dict[str, float]:
        return {snapshot.name: snapshot.value for snapshot in self.stages}

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "loss": self.loss,
            "loss_grad_fn": self.loss_grad_fn,
            "prediction_grad_fn": self.prediction_grad_fn,
            "stages": [snapshot.to_dict() for snapshot in self.stages],
        }

    def render(self, indent: str = "  ") -> str:
        return "\n".join(snapshot.render(indent) for snapshot in self.stages)


def _forward_math(
    x: torch.Tensor,
    w: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    use_tanh: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The forward arithmetic, in one place, shared by the graph and diagnostics.

    ``w`` is a *scalar* tensor and broadcasts against the batch dimension, so
    ``dL/dw`` exists and the graph has a single trainable leaf.
    """
    z = x * w
    scaled = z * input_scale
    activation = torch.tanh(scaled) if use_tanh else scaled
    prediction = activation * output_scale
    return z, activation, prediction


class ObservatoryModel(nn.Module):
    """One real float32 trainable scalar inside a real computational graph."""

    def __init__(
        self,
        initial_value: float,
        model_config: ModelConfig | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.model_config = model_config or ModelConfig()
        if dtype is not torch.float32:
            # The float64 track exists as a *reference* instrument; the primary
            # tracked parameter is always float32 and this guard makes that
            # structural rather than conventional.
            raise ConfigError(
                "ObservatoryModel is the primary float32 instrument; the float64 "
                "reference track is built by numerical_analysis.evaluate_loss"
            )
        self.w = nn.Parameter(torch.tensor(float(initial_value), dtype=torch.float32))
        self.register_buffer(
            "input_scale", torch.tensor(self.model_config.input_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "output_scale", torch.tensor(self.model_config.output_scale, dtype=torch.float32)
        )

    # -- graph path ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _z, _a, prediction = _forward_math(
            x, self.w, self.input_scale, self.output_scale, self.model_config.use_tanh
        )
        return prediction

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(x)
        return ((prediction - target) ** 2).mean()

    def forward_with_trace(self, x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, ForwardPass]:
        """Run a real forward pass and record every intermediate it produced."""
        z, activation, prediction = _forward_math(
            x, self.w, self.input_scale, self.output_scale, self.model_config.use_tanh
        )
        loss = ((prediction - target) ** 2).mean()
        stages = (
            # Recorded inside the trace as well as outside it: the forward pass
            # must not move the parameter, and the trace is where that is shown.
            describe_tensor("parameter", self.w, with_bits=True),
            describe_tensor("input", x),
            describe_tensor("z", z),
            describe_tensor("activation", activation),
            describe_tensor("prediction", prediction),
            describe_tensor("target", target),
            describe_tensor("loss", loss),
        )
        record = ForwardPass(
            prediction=float(prediction.detach().reshape(-1)[0].item()),
            loss=float(loss.detach().item()),
            loss_grad_fn=None if loss.grad_fn is None else type(loss.grad_fn).__name__,
            prediction_grad_fn=(
                None if prediction.grad_fn is None else type(prediction.grad_fn).__name__
            ),
            stages=stages,
        )
        return prediction, loss, record

    # -- diagnostic path (never touches self.w) ----------------------------
    def evaluate_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        w_value: float,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> float:
        """Loss of the *same* model at an arbitrary ``w_value``.

        The candidate weight is a fresh detached tensor, so this function
        structurally cannot perturb the live parameter or the optimizer state.
        """
        with torch.no_grad():
            candidate = torch.tensor(float(w_value), dtype=dtype)
            _z, _a, prediction = _forward_math(
                x.to(dtype),
                candidate,
                self.input_scale.to(dtype),
                self.output_scale.to(dtype),
                self.model_config.use_tanh,
            )
            loss = ((prediction - target.to(dtype)) ** 2).mean()
        return float(loss.item())

    def analytic_gradient_fp64(
        self, x: torch.Tensor, target: torch.Tensor, w_value: float
    ) -> float:
        """Closed-form dL/dw in pure Python float64.

        Written out by hand from the forward definition, using no autograd and no
        torch arithmetic, precisely so it can disagree with autograd:
            dL/dw = mean_b [ 2 (p_b - t_b) * o * s (1 - a_b^2) * x_b ]
        with ``a = tanh(s * x * w)`` and ``p = o * a``.
        """
        inputs = [float(v) for v in x.detach().reshape(-1).tolist()]
        targets = [float(v) for v in target.detach().reshape(-1).tolist()]
        if len(inputs) != len(targets) or not inputs:
            raise ConfigError("analytic gradient needs equally sized, non-empty x and target")
        s = self.model_config.input_scale
        o = self.model_config.output_scale
        w = float(w_value)
        total = 0.0
        for x_b, t_b in zip(inputs, targets):
            z = x_b * w
            scaled = z * s
            if self.model_config.use_tanh:
                a = math.tanh(scaled)
                d_activation = s * (1.0 - a * a)
            else:
                a = scaled
                d_activation = s
            p = a * o
            d_loss_d_p = 2.0 * (p - t_b)
            total += d_loss_d_p * o * d_activation * x_b
        return total / len(inputs)

    def loss_fp64_python(self, x: torch.Tensor, target: torch.Tensor, w_value: float) -> float:
        """Loss in pure Python float64, independent of torch arithmetic."""
        inputs = [float(v) for v in x.detach().reshape(-1).tolist()]
        targets = [float(v) for v in target.detach().reshape(-1).tolist()]
        if len(inputs) != len(targets) or not inputs:
            raise ConfigError("loss_fp64_python needs equally sized, non-empty x and target")
        s = self.model_config.input_scale
        o = self.model_config.output_scale
        w = float(w_value)
        total = 0.0
        for x_b, t_b in zip(inputs, targets):
            a = math.tanh(x_b * w * s) if self.model_config.use_tanh else x_b * w * s
            p = a * o
            total += (p - t_b) ** 2
        return total / len(inputs)

    # -- invariants ---------------------------------------------------------
    def assert_single_float32_parameter(self) -> torch.Tensor:
        """Verify the module holds exactly one trainable float32 scalar.

        Raises :class:`TrackingInvalidError` naming the exact violated invariant.
        """
        parameters = list(self.named_parameters())
        if len(parameters) != 1:
            raise TrackingInvalidError(
                "the model must contain exactly one parameter",
                {"found": [name for name, _ in parameters]},
            )
        name, parameter = parameters[0]
        problems: dict[str, Any] = {"name": name}
        if not isinstance(parameter, nn.Parameter):
            problems["is_parameter"] = type(parameter).__name__
        if tuple(parameter.shape) != ():
            problems["shape"] = tuple(parameter.shape)
        if int(parameter.numel()) != 1:
            problems["numel"] = int(parameter.numel())
        if parameter.dtype is not torch.float32:
            problems["dtype"] = str(parameter.dtype)
        if int(parameter.element_size()) != 4:
            problems["element_size"] = int(parameter.element_size())
        if parameter.requires_grad is not True:
            problems["requires_grad"] = parameter.requires_grad
        if parameter.is_leaf is not True:
            problems["is_leaf"] = bool(parameter.is_leaf)
        if len(problems) > 1:
            # State the expectation, not just the observation: a message that
            # only reports what was found leaves the reader guessing what the
            # instrument actually requires.
            raise TrackingInvalidError(
                "the tracked parameter violates a required invariant "
                "(required: one scalar, shape (), dtype float32, element_size 4, "
                "requires_grad True, is_leaf True, and the only parameter in the model)",
                problems,
            )
        return parameter

    def describe(self) -> Mapping[str, Any]:
        """The header block: what exactly is being tracked."""
        parameter = self.w
        return {
            "name": "w",
            "class": type(parameter).__name__,
            "shape": tuple(int(s) for s in parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "device": str(parameter.device),
            "element_size": int(parameter.element_size()),
            "is_leaf": bool(parameter.is_leaf),
            "buffers": [name for name, _ in self.named_buffers()],
            "trainable_parameter_count": len(list(self.parameters())),
            "raw_storage": f32.storage_evidence(parameter).to_dict(),
        }


# --------------------------------------------------------------------------- #
# float64 reference track
# --------------------------------------------------------------------------- #


class FP64ReferenceModel(nn.Module):
    """A real, separate float64 model used only as a numerical reference.

    This is not a shadow copy of the primary instrument and it is not "the truth".
    It is a second real run of the same forward shape in a different precision,
    which makes the *difference* between the two trajectories a measurable
    quantity (accumulated float32 rounding) instead of an assumption.  It never
    shares a tensor, an optimizer or a parameter with the float32 model.
    """

    def __init__(
        self,
        initial_value: float,
        model_config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.w = nn.Parameter(torch.tensor(float(initial_value), dtype=torch.float64))
        self.register_buffer(
            "input_scale", torch.tensor(self.model_config.input_scale, dtype=torch.float64)
        )
        self.register_buffer(
            "output_scale", torch.tensor(self.model_config.output_scale, dtype=torch.float64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _z, _a, prediction = _forward_math(
            x.to(torch.float64),
            self.w,
            self.input_scale,
            self.output_scale,
            self.model_config.use_tanh,
        )
        return prediction

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((self.forward(x) - target.to(torch.float64)) ** 2).mean()


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dataset:
    """A deterministic real input/target stream shared by both tracks."""

    inputs: torch.Tensor
    targets: torch.Tensor
    steps_per_epoch: int
    walk: str
    shuffle_permutation: tuple[int, ...] | None
    target_true_parameter: float
    noise_std: float

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """Return ``(x, target, batch_index, epoch, global_batch)`` for ``step``.

        A pure function of ``step``, so the float32 track and the float64
        reference track see byte-identical batches and any difference between
        their trajectories is attributable to arithmetic alone.
        """
        count = int(self.inputs.shape[0])
        size = int(self.inputs.shape[1])
        global_batch = int(step)
        if self.walk == "alternating" and global_batch % 2 == 1:
            index = (global_batch - 1) % self.steps_per_epoch
        else:
            index = global_batch % self.steps_per_epoch
        if self.shuffle_permutation is not None:
            start = (index * size) % count
            rows = [
                self.shuffle_permutation[(start + offset) % count] for offset in range(size)
            ]
        else:
            start = (index * size) % count
            rows = [(start + offset) % count for offset in range(size)]
        index_tensor = torch.tensor(rows, dtype=torch.long)
        x = self.inputs.index_select(0, index_tensor).clone()
        target = self.targets.index_select(0, index_tensor).clone()
        epoch = global_batch // self.steps_per_epoch
        batch_index = global_batch % self.steps_per_epoch
        return x, target, batch_index, epoch, global_batch

    def describe(self) -> Mapping[str, Any]:
        return {
            "samples": int(self.inputs.shape[0]),
            "batch_size": int(self.inputs.shape[1]),
            "steps_per_epoch": self.steps_per_epoch,
            "walk": self.walk,
            "shuffled": self.shuffle_permutation is not None,
            "target_true_parameter": self.target_true_parameter,
            "noise_std": self.noise_std,
            "input_min": float(self.inputs.min().item()),
            "input_max": float(self.inputs.max().item()),
        }


def build_dataset(config: DatasetConfig, model_config: ModelConfig) -> Dataset:
    """Build the real stream once, on CPU, from an explicit seed.

    Targets are generated from a known ``w_true`` through the model's own
    forward shape, so the one-dimensional problem has an exactly known optimum.
    That is an experimental design choice; the forward pass, the loss, the
    gradient and the update are all still produced by PyTorch at run time.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    count = int(config.samples)
    if config.walk == "alternating":
        # Constant magnitude, alternating sign: identical gradient magnitude,
        # alternating gradient sign.  This is what makes the oscillatory
        # preset actually oscillate instead of merely wandering.
        magnitude = 0.5 * (abs(config.input_min) + abs(config.input_max))
        signs = torch.where(
            torch.arange(count) % 2 == 0,
            torch.full((count,), 1.0),
            torch.full((count,), -1.0),
        )
        inputs = (signs * magnitude).reshape(count, 1).to(torch.float32)
    else:
        inputs = torch.linspace(
            float(config.input_min), float(config.input_max), count, dtype=torch.float32
        ).reshape(count, 1)
    w_true = float(config.target_true_parameter)
    scale = float(model_config.input_scale)
    out = float(model_config.output_scale)
    if model_config.use_tanh:
        targets = out * torch.tanh(inputs * w_true * scale)
    else:
        targets = out * (inputs * w_true * scale)
    targets = targets.to(torch.float32)
    if config.noise_std > 0.0:
        targets = targets + config.noise_std * torch.randn(
            count, 1, generator=generator, dtype=torch.float32
        )
    permutation: tuple[int, ...] | None = None
    if config.shuffle_batches:
        permutation = tuple(
            int(v)
            for v in torch.randperm(count, generator=generator).tolist()
        )
    return Dataset(
        inputs=inputs.contiguous(),
        targets=targets.contiguous(),
        steps_per_epoch=max(1, count // int(config.batch_size)),
        walk=config.walk,
        shuffle_permutation=permutation,
        target_true_parameter=w_true,
        noise_std=float(config.noise_std),
    )


__all__ = [
    "FORWARD_STAGE_NAMES",
    "TensorSnapshot",
    "ForwardPass",
    "ObservatoryModel",
    "FP64ReferenceModel",
    "Dataset",
    "build_dataset",
    "describe_tensor",
]
