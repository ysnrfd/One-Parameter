"""Ownership and invariant checking for the one tracked parameter.

The tracker is deliberately paranoid.  It re-verifies the tracked object at the
start of every step rather than trusting that nothing changed, because the whole
value of the instrument depends on those invariants still holding: a tracked
object that had silently stopped being a trainable float32 scalar would make
every downstream number meaningless without anything looking wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any
from collections.abc import Mapping

import torch
from torch import nn

from . import fp32_inspector as f32
from .errors import TrackingInvalidError
from .model import ObservatoryModel

#: The invariants that must hold for the tracked object, and the reason each one
#: matters.  Stored as data so the report can print them instead of the code
#: merely asserting them.
REQUIRED_INVARIANTS: tuple[tuple[str, str], ...] = (
    ("isinstance(parameter, torch.nn.Parameter)", "the tracked object is a Parameter"),
    ("parameter.numel() == 1", "exactly one scalar, not a vector or a view of one"),
    ("parameter.shape == torch.Size([])", "a true scalar, not shape (1,)"),
    ("parameter.dtype == torch.float32", "the subject of the investigation is binary32"),
    ("parameter.requires_grad is True", "autograd must be able to reach it"),
    ("parameter.element_size() == 4", "four bytes of storage, proven not asserted"),
    ("parameter.is_leaf is True", "the optimizer must write to it directly"),
    ("model has exactly one parameter", "no second trainable tensor to confuse the analysis"),
)

#: Explicit statements about what the tracked object is not.
TENSOR_VS_PARAMETER_NOTES: tuple[str, ...] = (
    "A tensor is not a parameter. torch.Tensor is the base class; "
    "torch.nn.Parameter is a subclass that additionally requires_grad=True by "
    "default and is registered in a module's parameter list.",
    "The tracked object is exactly one trainable scalar element. The gradient, "
    "the activations, the prediction, the loss and the optimizer state are five "
    "different tensors, and none of them is the parameter.",
    "parameter.grad is a separate tensor that autograd allocates and writes; it "
    "is not a view of the parameter and does not alias it.",
    "Optimizer state (exp_avg, exp_avg_sq, momentum_buffer) is memory about the "
    "parameter's update history. It is not a second copy of the parameter and "
    "does not move when the parameter moves.",
)


@dataclass(frozen=True)
class ParameterSnapshot:
    """A read-only observation of the parameter at one labelled instant."""

    label: str
    step: int
    value: float
    uint32: int
    hex: str
    bits: str
    exponent_bits: int
    fraction_bits_int: int
    sign_bit: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self, indent: str = "  ") -> str:
        return (
            f"{indent}{self.label:<28} {self.value!r}  {self.hex}  "
            f"exp={self.exponent_bits} frac={self.fraction_bits_int} sign={self.sign_bit}"
        )


@dataclass(frozen=True)
class TrackedParameterInfo:
    """The full identity card of the tracked parameter."""

    name: str
    class_name: str
    shape: tuple[int, ...]
    numel: int
    dtype: str
    requires_grad: bool
    device: str
    element_size: int
    is_leaf: bool
    raw_storage_bytes: str
    storage_nbytes: int | None
    storage_nbytes_method: str
    total_tensor_bytes: int
    value: float
    hex: str
    bits: str
    exponent_bits: int
    fraction_bits_int: int
    sign_bit: int
    trainable_parameter_count: int
    invariants: tuple[str, ...]
    invariants_satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        data["invariants"] = list(self.invariants)
        return data

    def render(self) -> str:
        rows = [
            ("name", self.name),
            ("class", self.class_name),
            ("shape", str(self.shape)),
            ("numel", str(self.numel)),
            ("dtype", self.dtype),
            ("requires_grad", str(self.requires_grad)),
            ("device", self.device),
            ("element_size", f"{self.element_size} bytes"),
            ("is_leaf", str(self.is_leaf)),
            ("raw storage bytes", self.raw_storage_bytes),
            ("storage nbytes", f"{self.storage_nbytes} ({self.storage_nbytes_method})"),
            ("tensor bytes", str(self.total_tensor_bytes)),
            ("initial value", repr(self.value)),
            ("hex", self.hex),
            ("bits", self.bits),
            ("exponent (biased)", str(self.exponent_bits)),
            ("fraction (23 bits)", str(self.fraction_bits_int)),
            ("sign bit", str(self.sign_bit)),
            ("trainable parameters in model", str(self.trainable_parameter_count)),
        ]
        width = max(len(name) for name, _ in rows)
        lines = [f"{name.ljust(width)} : {value}" for name, value in rows]
        lines.append("invariants checked      :")
        lines.extend(f"    [ok] {text}" for text in self.invariants)
        return "\n".join(lines)


class TrackedParameter:
    """Holds the one tracked parameter and enforces its invariants."""

    def __init__(self, model: ObservatoryModel) -> None:
        self._model = model
        # assert_single_float32_parameter raises if the model is not a valid
        # single-scalar float32 instrument, so this cannot capture a bad object.
        self._parameter = model.assert_single_float32_parameter()
        self._name = "w"

    @property
    def parameter(self) -> nn.Parameter:
        return self._parameter

    @property
    def model(self) -> ObservatoryModel:
        return self._model

    # -- invariants ---------------------------------------------------------
    def verify(self, *, step: int | None = None) -> None:
        """Re-check every invariant; raise :class:`TrackingInvalidError` on failure."""
        where = "before the first step" if step is None else f"at step {step}"
        parameter = self._parameter
        problems: dict[str, Any] = {"where": where}
        if not isinstance(parameter, nn.Parameter):
            problems["type"] = type(parameter).__name__
        if int(parameter.numel()) != 1:
            problems["numel"] = int(parameter.numel())
        if tuple(parameter.shape) != ():
            problems["shape"] = tuple(parameter.shape)
        if parameter.dtype is not torch.float32:
            problems["dtype"] = str(parameter.dtype)
        if int(parameter.element_size()) != 4:
            problems["element_size"] = int(parameter.element_size())
        if parameter.requires_grad is not True:
            problems["requires_grad"] = bool(parameter.requires_grad)
        if parameter.is_leaf is not True:
            problems["is_leaf"] = bool(parameter.is_leaf)
        registered = list(self._model.named_parameters())
        if len(registered) != 1:
            problems["model_parameters"] = [name for name, _ in registered]
        elif registered[0][1] is not parameter or getattr(self._model, "w", None) is not parameter:
            problems["parameter_identity"] = "the model no longer owns the original tracked parameter"
        if len(problems) > 1:
            raise TrackingInvalidError(
                "the tracked object is no longer a single trainable float32 scalar",
                problems,
            )

    def info(self) -> TrackedParameterInfo:
        """Build the identity card, verifying invariants as a side effect."""
        self.verify()
        parameter = self._parameter
        evidence = f32.storage_evidence(parameter)
        bits = f32.inspect_float32(parameter)
        # The identity card reports the bare dtype name ("float32"), which is
        # what the reader needs; str(torch.float32) would say "torch.float32".
        dtype_name = evidence.dtype.replace("torch.", "")
        return TrackedParameterInfo(
            name=self._name,
            class_name=type(parameter).__name__,
            shape=evidence.shape,
            numel=evidence.numel,
            dtype=dtype_name,
            requires_grad=bool(parameter.requires_grad),
            device=evidence.device,
            element_size=evidence.element_size_bytes,
            is_leaf=bool(parameter.is_leaf),
            raw_storage_bytes=" ".join(f"{b:02X}" for b in evidence.raw_storage_bytes),
            storage_nbytes=evidence.storage_nbytes,
            storage_nbytes_method=evidence.storage_nbytes_method,
            total_tensor_bytes=evidence.total_bytes,
            value=bits.value,
            hex=bits.hex,
            bits=bits.binary,
            exponent_bits=bits.exponent_bits,
            fraction_bits_int=bits.fraction_bits_int,
            sign_bit=bits.sign_bit,
            trainable_parameter_count=len(list(self._model.parameters())),
            invariants=tuple(text for _expr, text in REQUIRED_INVARIANTS),
            invariants_satisfied=True,
        )

    # -- observations -------------------------------------------------------
    def snapshot(self, label: str, step: int) -> ParameterSnapshot:
        """Read the parameter now and return an immutable observation of it."""
        bits = f32.inspect_float32(self._parameter)
        return ParameterSnapshot(
            label=label,
            step=step,
            value=bits.value,
            uint32=bits.raw_uint32,
            hex=bits.hex,
            bits=bits.binary,
            exponent_bits=bits.exponent_bits,
            fraction_bits_int=bits.fraction_bits_int,
            sign_bit=bits.sign_bit,
        )

    def value(self) -> float:
        return float(self._parameter.detach().reshape(-1)[0].item())

    def raw_uint32(self) -> int:
        return f32.raw_word_from_tensor(self._parameter)

    def clone(self) -> torch.Tensor:
        """A detached, bit-exact copy, safe to hand to a reconstruction."""
        return self._parameter.detach().clone()

    def content_hash(self) -> str:
        """A short hash of the stored bytes, used to prove restoration."""
        digest = hashlib.sha256()
        digest.update(str(self._parameter.dtype).encode("utf-8"))
        digest.update(str(tuple(self._parameter.shape)).encode("utf-8"))
        digest.update(f32.storage_evidence(self._parameter).raw_storage_bytes)
        return digest.hexdigest()[:16]

    def gradient_value(self) -> float | None:
        """``parameter.grad`` as a float, or ``None`` when autograd has not written one."""
        grad = self._parameter.grad
        if grad is None:
            return None
        return float(grad.detach().reshape(-1)[0].item())

    def describe_gradients(self) -> Mapping[str, Any]:
        grad = self._parameter.grad
        if grad is None:
            return {"present": False, "note": "parameter.grad is None"}
        return {
            "present": True,
            "dtype": str(grad.dtype),
            "shape": tuple(int(s) for s in grad.shape),
            "device": str(grad.device),
            "is_leaf": bool(grad.is_leaf),
            "layout": str(grad.layout),
            "value": float(grad.detach().reshape(-1)[0].item()),
            "finite": bool(torch.isfinite(grad).all().item()),
            "raw_uint32": (
                f32.raw_word_from_tensor(grad) if grad.dtype is torch.float32 else None
            ),
        }


__all__ = [
    "REQUIRED_INVARIANTS",
    "TENSOR_VS_PARAMETER_NOTES",
    "ParameterSnapshot",
    "TrackedParameterInfo",
    "TrackedParameter",
]
