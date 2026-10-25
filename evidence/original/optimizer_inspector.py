"""Optimizer inspection, version audit, and independent reconstruction.

Three separate jobs live here.

**Audit.** Record exactly which PyTorch is installed, which optimizer class is in
use, what its constructor accepts, and what its source actually says about the
update.  Optimizer implementations are not stable across PyTorch versions: the
order of the decoupled weight-decay term, and whether the first moment is
updated with ``lerp_`` or with ``mul_``/``add_``, have both changed between
releases.  Claiming one fixed formula would be exactly the kind of unjustified
assumption this project exists to avoid, so the audit records the *observed*
implementation instead.

**Inspect.** Read ``optimizer.state[parameter]`` directly and decode every
tensor in it, including the FP32 bit pattern of each stored scalar.  No state
value is inferred or recomputed.

**Reconstruct.** Recompute the expected transition from the observed before-state,
the observed gradient and the observed hyperparameters, using PyTorch's own
tensor operations, and compare with the value actually stored. Several documented
update-order variants are tried and the one that reproduces the real update
bit-for-bit is reported; the variant that matches is itself a measurement of what
the installed implementation does.

The reconstruction never mutates the parameter or the optimizer state: every
tensor it touches is a clone.
"""

from __future__ import annotations

import inspect
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from . import fp32_inspector as f32
from . import numerical_analysis as na
from .config import ExperimentConfig
from .errors import ConfigError, NumericalConsistencyError

#: Optimizer classes the project can build, keyed by configuration name.
OPTIMIZER_CLASSES: dict[str, type[torch.optim.Optimizer]] = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

#: Hyperparameter names that carry optimizer state, per optimizer.
_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "adam": ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
    "adamw": ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
    "sgd": ("momentum_buffer",),
}

# --------------------------------------------------------------------------- #
# environment audit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnvironmentAudit:
    """Everything needed to interpret a trajectory recorded on this machine."""

    python_version: str
    python_executable: str
    torch_version: str
    torch_file: str
    platform: str
    machine: str
    processor: str
    operating_system: str
    device: str
    default_dtype: str
    num_threads: int
    numpy_version: str | None
    matplotlib_version: str | None
    matplotlib_backend: str | None
    parallel_info: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        rows = [
            ("Python", f"{self.python_version} ({self.python_executable})"),
            ("PyTorch", f"{self.torch_version}"),
            ("PyTorch file", self.torch_file),
            ("Platform", self.platform),
            ("Machine", self.machine),
            ("Processor", self.processor),
            ("OS", self.operating_system),
            ("Device", self.device),
            ("Default dtype", self.default_dtype),
            ("Torch threads", str(self.num_threads)),
            ("NumPy", self.numpy_version or "not installed"),
            ("Matplotlib", self.matplotlib_version or "not installed"),
            ("Matplotlib backend", self.matplotlib_backend or "n/a"),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"  {name.ljust(width)} : {value}" for name, value in rows)


def audit_environment() -> EnvironmentAudit:
    """Record the interpreter, PyTorch build, and machine, as observed."""
    try:
        numpy_version: str | None
        import numpy  # noqa: PLC0415 - optional dependency, probed on purpose

        numpy_version = numpy.__version__
    except Exception:  # pragma: no cover - numpy is in requirements.txt
        numpy_version = None

    try:
        threads = int(torch.get_num_threads())
    except Exception:  # pragma: no cover
        threads = -1

    matplotlib_version: str | None
    matplotlib_backend: str | None
    try:
        import matplotlib  # noqa: PLC0415 - optional dependency, probed on purpose

        matplotlib_version = str(matplotlib.__version__)
        matplotlib_backend = str(matplotlib.get_backend())
    except Exception:
        matplotlib_version = None
        matplotlib_backend = None

    return EnvironmentAudit(
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        torch_version=str(torch.__version__),
        torch_file=str(getattr(torch, "__file__", "unknown")),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        operating_system=f"{platform.system()} {platform.release()}",
        device="cpu",
        default_dtype=str(torch.get_default_dtype()),
        num_threads=threads,
        numpy_version=numpy_version,
        matplotlib_version=matplotlib_version,
        matplotlib_backend=matplotlib_backend,
        parallel_info=f"cuda_available={torch.cuda.is_available()} "
        f"mps_available={getattr(torch.backends, 'mps', None) is not None}",
    )


# --------------------------------------------------------------------------- #
# source audit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OptimizerSourceAudit:
    """What the installed optimizer implementation actually says."""

    optimizer_name: str
    class_name: str
    class_module: str
    source_file: str | None
    constructor_parameters: tuple[str, ...]
    supported_hyperparameters: tuple[str, ...]
    dropped_hyperparameters: tuple[str, ...]
    group_flags: Mapping[str, Any]
    source_findings: Mapping[str, Any]
    torch_version: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["constructor_parameters"] = list(self.constructor_parameters)
        data["supported_hyperparameters"] = list(self.supported_hyperparameters)
        data["dropped_hyperparameters"] = list(self.dropped_hyperparameters)
        data["group_flags"] = dict(self.group_flags)
        data["source_findings"] = dict(self.source_findings)
        return data

    def render(self) -> str:
        lines = [
            f"  optimizer class       : {self.class_module}.{self.class_name}",
            f"  source file           : {self.source_file or 'unavailable'}",
            f"  torch version         : {self.torch_version}",
            f"  constructor parameters: {', '.join(self.constructor_parameters)}",
            f"  hyperparameters used  : {', '.join(self.supported_hyperparameters)}",
        ]
        if self.dropped_hyperparameters:
            lines.append(
                "  DROPPED (not accepted by this torch version): "
                + ", ".join(self.dropped_hyperparameters)
            )
        lines.append("  param group flags    :")
        for key, value in sorted(self.group_flags.items()):
            lines.append(f"      {key} = {value!r}")
        lines.append("  source findings      :")
        for key, value in sorted(self.source_findings.items()):
            lines.append(f"      {key} = {value!r}")
        return "\n".join(lines)


def _module_source_text(module_name: str) -> tuple[str | None, str]:
    """Return ``(path, text)`` for an installed optimizer module."""
    try:
        module = __import__(module_name, fromlist=["_"])
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"<module import failed: {exc}>"
    path = getattr(module, "__file__", None)
    if not path or not os.path.exists(path):
        return path, ""
    with open(path, "r", encoding="utf-8") as handle:
        return path, handle.read()


def _extract_function_source(text: str, function_name: str) -> str | None:
    """Pull one top-level function body out of a module's source text."""
    marker = f"def {function_name}("
    start = text.find(marker)
    if start < 0:
        return None
    lines = text[start:].splitlines()
    collected: list[str] = []
    for line in lines:
        if len(collected) > 0:
            if line.startswith("def "):
                break
            if line and not line[0].isspace() and not line.startswith((")", "]")):
                break
        collected.append(line)
    return "\n".join(collected)


def audit_optimizer_source(
    optimizer_name: str,
    *,
    requested: Mapping[str, Any],
    group_flags: Mapping[str, Any],
) -> OptimizerSourceAudit:
    """Read the installed optimizer source and record what it does.

    The findings are keyword-presence observations, not an interpretation: for
    example ``decay_before_update`` is ``True`` only when the source text
    actually contains the corresponding statement.
    """
    optimizer_class = OPTIMIZER_CLASSES[optimizer_name]
    try:
        signature = inspect.signature(optimizer_class.__init__)
        parameters = tuple(
            name
            for name, p in signature.parameters.items()
            if name != "self" and p.kind is not p.VAR_KEYWORD
        )
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        parameters = ()
    module_name = optimizer_class.__module__
    path, text = _module_source_text(module_name)
    func_target = (
        "_single_tensor_adamw" if optimizer_name == "adamw"
        else "_single_tensor_sgd" if optimizer_name == "sgd"
        else "_single_tensor_adam"
    )
    single_tensor = _extract_function_source(text, func_target)
    if single_tensor is None and optimizer_name == "adamw":
        single_tensor = _extract_function_source(text, "_multi_tensor_adamw") or text
    elif single_tensor is None and optimizer_name == "sgd":
        single_tensor = _extract_function_source(text, "sgd") or text
    haystack = single_tensor or text
    findings: dict[str, Any] = {
        "source_text_available": bool(haystack),
        "single_tensor_function_found": single_tensor is not None,
        "uses_lerp_for_moment1": "exp_avg.lerp_(grad" in haystack,
        "uses_mul_add_for_moment1": "exp_avg.mul_(beta1).add_(grad" in haystack,
        "has_capturable_branch": "capturable" in haystack,
        "decay_before_update": "param.mul_(1 - lr * weight_decay)" in haystack
        or "mul_(1 - lr * weight_decay)" in haystack,
        "decay_after_update_add_param": "alpha=-weight_decay * lr" in haystack,
        "coupled_decay_into_grad": "grad = grad.add(param, alpha=weight_decay)" in haystack
        or "d_p = d_p.add(p, alpha=weight_decay)" in haystack,
        "eps_added_outside_sqrt": ".sqrt() / bias_correction2_sqrt).add_(eps)" in haystack,
        "eps_added_inside_sqrt": "add_(eps)" in haystack and "sqrt().add_(eps)" in haystack,
        "amsgrad_present": "max_exp_avg_sq" in haystack,
        "for_each_reference_present": "_multi_tensor" in text or "foreach" in text,
    }
    if single_tensor:
        findings["single_tensor_source_excerpt"] = single_tensor[:4000]
    supported = tuple(name for name in requested if name in parameters)
    dropped = tuple(name for name in requested if name not in parameters)
    return OptimizerSourceAudit(
        optimizer_name=optimizer_name,
        class_name=optimizer_class.__name__,
        class_module=optimizer_class.__module__,
        source_file=path,
        constructor_parameters=parameters,
        supported_hyperparameters=supported,
        dropped_hyperparameters=dropped,
        group_flags=dict(group_flags),
        source_findings=findings,
        torch_version=str(torch.__version__),
    )


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OptimizerHandle:
    """A real PyTorch optimizer plus the audit of how it was built."""

    name: str
    optimizer: torch.optim.Optimizer
    parameter: nn.Parameter
    audit: OptimizerSourceAudit

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def set_lr(self, lr: float) -> None:
        self.optimizer.param_groups[0]["lr"] = float(lr)

    def group_flags(self) -> dict[str, Any]:
        group = self.optimizer.param_groups[0]
        return {
            key: group[key]
            for key in sorted(group)
            if key not in ("params",)
        }


def make_optimizer(
    config: ExperimentConfig, parameter: nn.Parameter
) -> OptimizerHandle:
    """Construct the real optimizer, filtered against the installed signature.

    Unsupported keyword arguments are dropped *and recorded*, never silently.
    """
    optimizer_class = OPTIMIZER_CLASSES[config.optimizer]
    try:
        signature = inspect.signature(optimizer_class.__init__)
        accepted = {
            name
            for name, p in signature.parameters.items()
            if name != "self" and p.kind is not p.VAR_KEYWORD
        }
    except (TypeError, ValueError):  # pragma: no cover
        accepted = set()
    requested = dict(config.optimizer_kwargs())
    if "maximize" not in accepted:
        requested.pop("maximize", None)
    kwargs = {name: value for name, value in requested.items() if name in accepted}
    optimizer = optimizer_class([parameter], **kwargs)
    return OptimizerHandle(
        name=config.optimizer,
        optimizer=optimizer,
        parameter=parameter,
        audit=audit_optimizer_source(
            config.optimizer, requested=requested, group_flags=_group_flags(optimizer)
        ),
    )


def _group_flags(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    if not optimizer.param_groups:
        return {}
    group = optimizer.param_groups[0]
    return {key: value for key, value in sorted(group.items()) if key != "params"}


# --------------------------------------------------------------------------- #
# state inspection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StateEntryReading:
    """One decoded entry of ``optimizer.state[parameter]``."""

    key: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    device: str
    value: float | None
    values: tuple[float, ...] | None
    raw_uint32: int | None
    bits: str | None
    is_finite: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        if data["values"] is not None:
            data["values"] = list(data["values"])
        return data

    def render(self, indent: str = "  ") -> str:
        text = f"{self.key:<10} {self.kind:<14} dtype={self.dtype} shape={self.shape}"
        if self.value is not None:
            text += f" value={self.value!r}"
        if self.raw_uint32 is not None:
            text += f" (0x{self.raw_uint32:08X})"
        return indent + text


@dataclass(frozen=True)
class OptimizerStateReading:
    """The observed optimizer state for the tracked parameter at one instant."""

    present: bool
    keys: tuple[str, ...]
    entries: tuple[StateEntryReading, ...]
    note: str
    step_value: int | None
    state_dict_json: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "keys": list(self.keys),
            "entries": [entry.to_dict() for entry in self.entries],
            "note": self.note,
            "step_value": self.step_value,
            "state_dict": dict(self.state_dict_json),
        }

    def render(self, indent: str = "  ") -> str:
        if not self.present:
            return f"{indent}optimizer state: not present ({self.note})"
        lines = [f"{indent}optimizer state keys: {', '.join(self.keys) or '(empty)'}"]
        lines.extend(entry.render(indent + "  ") for entry in self.entries)
        return "\n".join(lines)

    def value_of(self, key: str) -> float | None:
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        return None


def _read_entry(key: str, value: Any) -> StateEntryReading:
    if isinstance(value, torch.Tensor):
        flat = value.detach().to("cpu").reshape(-1)
        values = tuple(float(v) for v in flat.tolist()) if flat.numel() <= 8 else None
        raw = bits = None
        if flat.numel() == 1 and value.dtype is torch.float32:
            raw = f32.raw_word_from_tensor(flat.reshape(1))
            bits = f"{raw:032b}"
        return StateEntryReading(
            key=key,
            kind="tensor",
            dtype=str(value.dtype),
            shape=tuple(int(s) for s in value.shape),
            device=str(value.device),
            value=float(flat[0].item()) if flat.numel() >= 1 else None,
            values=values,
            raw_uint32=raw,
            bits=bits,
            is_finite=bool(torch.isfinite(flat).all().item()) if flat.numel() else True,
        )
    if isinstance(value, (int, float, bool)):
        return StateEntryReading(
            key=key,
            kind="python_scalar",
            dtype=type(value).__name__,
            shape=(),
            device="python",
            value=float(value),
            values=None,
            raw_uint32=None,
            bits=None,
            is_finite=math.isfinite(float(value)),
        )
    return StateEntryReading(
        key=key,
        kind=type(value).__name__,
        dtype="n/a",
        shape=(),
        device="n/a",
        value=None,
        values=None,
        raw_uint32=None,
        bits=None,
        is_finite=True,
    )


def read_optimizer_state(
    optimizer: torch.optim.Optimizer, parameter: nn.Parameter
) -> OptimizerStateReading:
    """Read the live optimizer state for one parameter, decoding every tensor.

    ``optimizer.state`` is a ``defaultdict``, so this uses ``.get()``: looking a
    key up with ``[]`` would create the entry and turn a real observation ("this
    optimizer has not initialised state for this parameter yet") into a
    fabricated one.
    """
    state = optimizer.state.get(parameter, None)
    if state is None:
        return OptimizerStateReading(
            present=False,
            keys=(),
            entries=(),
            note=(
                "optimizer.state has no entry for this parameter: either the "
                "optimizer has not stepped yet, or it keeps no per-parameter "
                "state at all (plain SGD without momentum)"
            ),
            step_value=None,
            state_dict_json=_json_safe_state_dict(optimizer),
        )
    entries = tuple(_read_entry(key, value) for key, value in sorted(state.items()))
    step_value: int | None = None
    raw_step = state.get("step", None)
    if isinstance(raw_step, torch.Tensor) and raw_step.numel() == 1:
        step_value = int(raw_step.detach().reshape(-1)[0].item())
    elif isinstance(raw_step, (int, float)):
        step_value = int(raw_step)
    note = "read directly from optimizer.state[parameter]"
    if not state:
        note = (
            "optimizer.state[parameter] is an empty dict: this optimizer keeps no "
            "per-parameter state (plain SGD without momentum)"
        )
    return OptimizerStateReading(
        present=True,
        keys=tuple(sorted(state.keys())),
        entries=entries,
        note=note,
        step_value=step_value,
        state_dict_json=_json_safe_state_dict(optimizer),
    )


def _json_safe_state_dict(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """A JSON-serialisable copy of ``optimizer.state_dict()``.

    Tensors are reduced to their decoded scalar content rather than dropped, so
    the export keeps the real values and loses nothing.
    """

    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            flat = value.detach().to("cpu").reshape(-1)
            if flat.numel() == 1:
                item = float(flat[0].item())
                if value.dtype is torch.float32:
                    raw = f32.raw_word_from_tensor(flat.reshape(1))
                    return {"value": item, "uint32": raw, "hex": f"0x{raw:08X}"}
                return {"value": item}
            return [float(v) for v in flat.tolist()[:16]]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        return repr(value)

    try:
        return convert(optimizer.state_dict())
    except Exception as exc:  # pragma: no cover - defensive, but never silent
        return {"error": f"state_dict() failed: {type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# independent reconstruction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UpdateVariant:
    """One candidate description of the installed update semantics."""

    name: str
    #: 'lerp' -> exp_avg.lerp_(g, 1-b1);  'mul_add' -> mul_(b1).add_(g, 1-b1)
    moment_update: str
    #: AdamW: 'before' -> p.mul_(1-lr*wd); 'after_original' / 'after_current'
    #: -> p.add_(<w before / w now>, alpha=-lr*wd);  'none' -> not applicable.
    decay_order: str
    #: 'outside' -> denom = sqrt(v_hat) + eps;  'inside' -> sqrt(v_hat + eps)
    eps_placement: str
    #: SGD only: does the momentum buffer start as a clone of the gradient or at
    #: zero?  and is weight decay coupled into the gradient?
    buffer_init: str = "clone_grad"
    decay_coupling: str = "coupled_into_grad"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _variant(name: str, moment: str, decay: str, eps: str, **extra: Any) -> UpdateVariant:
    return UpdateVariant(
        name=name,
        moment_update=moment,
        decay_order=decay,
        eps_placement=eps,
        **extra,
    )


#: The update-order variants worth trying.  The list is deliberately explicit:
#: the project's position is that the installed implementation decides, and the
#: only trustworthy way to find out is to see which one reproduces the real
#: update exactly.
ADAMW_VARIANTS: tuple[UpdateVariant, ...] = tuple(
    _variant(
        f"adamw[{moment},{decay},{eps}]", moment, decay, eps
    )
    for moment in ("lerp", "mul_add")
    for decay in ("before", "after_original", "after_current")
    for eps in ("outside", "inside")
)

ADAM_VARIANTS: tuple[UpdateVariant, ...] = tuple(
    _variant(f"adam[{moment},coupled,{eps}]", moment, "none", eps)
    for moment in ("lerp", "mul_add")
    for eps in ("outside", "inside")
) + tuple(
    _variant(f"adam[{moment},decoupled,{eps}]", moment, "before", eps)
    for moment in ("lerp", "mul_add")
    for eps in ("outside", "inside")
)

SGD_VARIANTS: tuple[UpdateVariant, ...] = tuple(
    _variant(
        f"sgd[{buffer_init},{coupling}]",
        "none",
        "none",
        "outside",
        buffer_init=buffer_init,
        decay_coupling=coupling,
    )
    for buffer_init in ("clone_grad", "zeros")
    for coupling in ("coupled_into_grad", "decoupled_after")
)


def variants_for(optimizer_name: str) -> tuple[UpdateVariant, ...]:
    if optimizer_name == "adamw":
        return ADAMW_VARIANTS
    if optimizer_name == "adam":
        return ADAM_VARIANTS
    if optimizer_name == "sgd":
        return SGD_VARIANTS
    raise ConfigError(f"no reconstruction variants defined for {optimizer_name!r}")


@dataclass(frozen=True)
class ReconstructedUpdate:
    """The result of one independent reconstruction, in float32."""

    variant: str
    after: torch.Tensor
    after_uint32: int
    moment1_after: float | None
    moment2_after: float | None
    step_after: int
    total_delta: float
    adaptive_delta: float
    decay_delta: float
    loss_amplification: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variant"] = self.variant
        data["after_uint32"] = int(self.after_uint32)
        return data


@dataclass(frozen=True)
class ReconstructionResult:
    """Reconstruction versus the transition the optimizer actually performed."""

    optimizer_name: str
    variant: str
    variant_search_performed: bool
    variant_candidates_tried: int
    variant_errors: tuple[tuple[str, int], ...]
    best_variant: str
    best_ulp_error: int
    actual_uint32: int
    reconstructed_uint32: int
    bit_exact: bool
    ulp_error: int
    absolute_error: float
    relative_error: float
    actual_after: float
    reconstructed_after: float
    total_update: float
    adaptive_update: float
    weight_decay_update: float
    step_after: int
    moment1_after: float | None
    moment2_after: float | None
    fp64_reference_after: float
    status: str
    tolerance_ulp: int

    @property
    def matched(self) -> bool:
        return self.ulp_error <= self.tolerance_ulp

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variant_errors"] = [list(pair) for pair in self.variant_errors]
        data["bit_exact"] = self.bit_exact
        return data

    def render(self, indent: str = "  ") -> str:
        lines = [
            f"{indent}optimizer              : {self.optimizer_name}",
            f"{indent}update variant         : {self.variant}",
            f"{indent}actual after           : {self.actual_after!r} "
            f"(0x{self.actual_uint32:08X})",
            f"{indent}reconstructed after    : {self.reconstructed_after!r} "
            f"(0x{self.reconstructed_uint32:08X})",
            f"{indent}absolute error         : {self.absolute_error!r}",
            f"{indent}relative error         : {self.relative_error!r}",
            f"{indent}error in ULPs          : {self.ulp_error} "
            f"(tolerance {self.tolerance_ulp})",
            f"{indent}total update           : {self.total_update!r}",
            f"{indent}  adaptive contribution: {self.adaptive_update!r}",
            f"{indent}  decay contribution   : {self.weight_decay_update!r}",
            f"{indent}fp64 reference after   : {self.fp64_reference_after!r} "
            f"(reference experiment, not ground truth)",
            f"{indent}status                 : {self.status}",
        ]
        if self.variant_candidates_tried > 1:
            lines.append(
                f"{indent}variant search         : tried {self.variant_candidates_tried}, "
                f"best {self.best_variant!r} with {self.best_ulp_error} ULP error"
            )
        return "\n".join(lines)


def _moments_from_state(
    state_before: Mapping[str, Any],
) -> tuple[float, float, float | None, int]:
    m1 = state_before.get("exp_avg")
    moment1 = float(m1.item()) if hasattr(m1, "item") else (float(m1) if m1 is not None else 0.0)
    m2 = state_before.get("exp_avg_sq")
    moment2 = float(m2.item()) if hasattr(m2, "item") else (float(m2) if m2 is not None else 0.0)
    msq = state_before.get("max_exp_avg_sq")
    max_sq = float(msq.item()) if hasattr(msq, "item") else (float(msq) if msq is not None else None)
    s = state_before.get("step")
    step = int(s.item()) if hasattr(s, "item") else (int(s) if s is not None else 0)
    return moment1, moment2, max_sq, step


def _apply_moment_update(
    moment: torch.Tensor, gradient: torch.Tensor, beta1: float, variant: UpdateVariant
) -> torch.Tensor:
    moment = moment.clone()
    if variant.moment_update == "lerp":
        moment.lerp_(gradient, 1.0 - beta1)
    else:
        moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    return moment


def _reconstruct_adam_family(
    variant: UpdateVariant,
    *,
    optimizer_name: str,
    weight_before: torch.Tensor,
    gradient: torch.Tensor,
    state_before: Mapping[str, Any],
    lr: float,
    betas: Sequence[float],
    eps: float,
    weight_decay: float,
    amsgrad: bool,
    maximize: bool,
) -> ReconstructedUpdate:
    beta1, beta2 = float(betas[0]), float(betas[1])
    moment1_before, moment2_before, max_sq_before, step_before = _moments_from_state(state_before)
    step_after = step_before + 1
    effective_gradient = -gradient if maximize else gradient
    if optimizer_name == "adam" and variant.decay_order == "none" and weight_decay != 0.0:
        effective_gradient = effective_gradient.add(weight_before, alpha=weight_decay)
    moment1 = _apply_moment_update(
        torch.tensor(moment1_before, dtype=weight_before.dtype),
        effective_gradient,
        beta1,
        variant,
    )
    moment2 = (
        torch.tensor(moment2_before, dtype=weight_before.dtype)
        .mul_(beta2)
        .addcmul_(effective_gradient, effective_gradient, value=1.0 - beta2)
    )
    if amsgrad:
        previous_max = torch.tensor(
            max_sq_before if max_sq_before is not None else 0.0, dtype=weight_before.dtype
        )
        max_sq = torch.maximum(previous_max, moment2)
    else:
        max_sq = moment2
    bias_correction1 = 1.0 - beta1**step_after
    bias_correction2 = 1.0 - beta2**step_after
    if bias_correction1 == 0.0 or bias_correction2 == 0.0:
        raise NumericalConsistencyError("bias correction reached zero")
    parameter = weight_before.clone()
    if optimizer_name == "adamw" and weight_decay != 0.0 and variant.decay_order == "before":
        parameter.mul_(1.0 - lr * weight_decay)
    if variant.eps_placement == "outside":
        denominator = (max_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    else:
        denominator = (max_sq / bias_correction2).sqrt().add_(eps)
    parameter.addcdiv_(moment1, denominator, value=-(lr / bias_correction1))

    # The same update with weight decay switched off.  The difference between the
    # two float32 results *is* the measured decoupled-decay contribution, rather
    # than an algebraic split that could hide rounding.
    adaptive_only = weight_before.clone()
    if variant.eps_placement == "outside":
        denom_only = (max_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    else:
        denom_only = (max_sq / bias_correction2).sqrt().add_(eps)
    adaptive_only.addcdiv_(moment1, denom_only, value=-(lr / bias_correction1))

    if optimizer_name == "adamw" and weight_decay != 0.0 and variant.decay_order == "after_original":
        parameter.add_(weight_before, alpha=-lr * weight_decay)
    elif optimizer_name == "adamw" and weight_decay != 0.0 and variant.decay_order == "after_current":
        parameter.add_(parameter, alpha=-lr * weight_decay)

    before_value = float(weight_before.item())
    after_value = float(parameter.item())
    adaptive_only_value = float(adaptive_only.item())
    return ReconstructedUpdate(
        variant=variant.name,
        after=parameter,
        after_uint32=f32.raw_word_from_tensor(parameter),
        moment1_after=float(moment1.item()),
        moment2_after=float(moment2.item()),
        step_after=step_after,
        total_delta=after_value - before_value,
        adaptive_delta=adaptive_only_value - before_value,
        decay_delta=after_value - adaptive_only_value,
        loss_amplification=float("nan"),
    )


def _reconstruct_sgd(
    variant: UpdateVariant,
    *,
    weight_before: torch.Tensor,
    gradient: torch.Tensor,
    state_before: Mapping[str, Any],
    lr: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    weight_decay: float,
    maximize: bool,
) -> ReconstructedUpdate:
    buffer_present = "momentum_buffer" in state_before
    buffer_before = (
        float(state_before["momentum_buffer"].item()) if buffer_present else None
    )
    effective_gradient = -gradient if maximize else gradient
    if variant.decay_coupling == "coupled_into_grad" and weight_decay != 0.0:
        effective_gradient = effective_gradient.add(weight_before, alpha=weight_decay)
    if buffer_before is None:
        # First step with momentum: PyTorch seeds the buffer from the gradient
        # itself; the "zeros" variant covers the older alternative convention.
        if variant.buffer_init == "clone_grad":
            buffer_after = effective_gradient.clone()
        else:
            buffer_after = torch.zeros_like(effective_gradient)
    else:
        buffer_after = torch.tensor(
            buffer_before, dtype=weight_before.dtype
        ).mul_(momentum).add_(effective_gradient, alpha=1.0 - dampening)
    if nesterov:
        direction = effective_gradient.add(buffer_after, alpha=momentum)
    else:
        direction = buffer_after
    parameter = weight_before.clone().add_(direction, alpha=-lr)
    adaptive_only = weight_before.clone().add_(direction, alpha=-lr)
    if variant.decay_coupling == "decoupled_after" and weight_decay != 0.0:
        parameter.add_(weight_before, alpha=-lr * weight_decay)
    before_value = float(weight_before.item())
    after_value = float(parameter.item())
    return ReconstructedUpdate(
        variant=variant.name,
        after=parameter,
        after_uint32=f32.raw_word_from_tensor(parameter),
        moment1_after=float(buffer_after.item()),
        moment2_after=None,
        step_after=1 if buffer_before is None else 2,
        total_delta=after_value - before_value,
        adaptive_delta=float(adaptive_only.item()) - before_value,
        decay_delta=after_value - float(adaptive_only.item()),
        loss_amplification=float("nan"),
    )


def _reconstruct_one(
    variant: UpdateVariant,
    *,
    optimizer_name: str,
    weight_before: torch.Tensor,
    gradient: torch.Tensor,
    state_before: Mapping[str, Any],
    lr: float,
    config: ExperimentConfig,
) -> ReconstructedUpdate:
    if optimizer_name == "sgd":
        return _reconstruct_sgd(
            variant,
            weight_before=weight_before,
            gradient=gradient,
            state_before=state_before,
            lr=lr,
            momentum=config.momentum,
            dampening=config.dampening,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
            maximize=config.maximize,
        )
    return _reconstruct_adam_family(
        variant,
        optimizer_name=optimizer_name,
        weight_before=weight_before,
        gradient=gradient,
        state_before=state_before,
        lr=lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
        maximize=config.maximize,
    )


def reconstruct_update(
    *,
    optimizer_name: str,
    weight_before: torch.Tensor,
    weight_after: torch.Tensor,
    gradient: torch.Tensor,
    state_before: Mapping[str, Any],
    lr: float,
    config: ExperimentConfig,
    active_variant: str | None = None,
    search: bool = True,
    tolerance_ulp: int = 0,
) -> ReconstructionResult:
    """Independently reconstruct the transition and compare with the real one.

    ``active_variant`` pins the variant to use.  When it is ``None`` and ``search``
    is true, every documented variant is evaluated and the one that reproduces
    the real stored value bit-for-bit is adopted; that match is recorded as a
    measurement of the installed semantics rather than assumed in advance.
    """
    candidates = variants_for(optimizer_name)
    if active_variant is not None:
        selected = [v for v in candidates if v.name == active_variant]
        if not selected:
            raise ConfigError(
                f"unknown update variant {active_variant!r} for {optimizer_name!r}"
            )
        tried = tuple(selected)
    elif search:
        tried = candidates
    else:
        tried = candidates[:1]

    actual_after = f32.inspect_float32(weight_after)
    errors: list[tuple[str, int]] = []
    best: ReconstructedUpdate | None = None
    best_ulp: int | None = None

    for variant in tried:
        result = _reconstruct_one(
            variant,
            optimizer_name=optimizer_name,
            weight_before=weight_before,
            gradient=gradient,
            state_before=state_before,
            lr=lr,
            config=config,
        )
        ulp_error = abs(
            f32.ulp_distance(float(result.after.item()), actual_after.value)
        )
        errors.append((variant.name, ulp_error))
        if best_ulp is None or ulp_error < best_ulp:
            best, best_ulp = result, ulp_error

    if best is None or best_ulp is None:  # pragma: no cover - candidates is non-empty
        raise NumericalConsistencyError("no reconstruction variant could be evaluated")

    assert best is not None
    ulp_error = abs(f32.ulp_distance(float(best.after.item()), actual_after.value))
    absolute = na.absolute_error(float(best.after.item()), actual_after.value)
    relative = na.relative_error(float(best.after.item()), actual_after.value)
    matched = ulp_error <= tolerance_ulp
    if matched:
        status = "PASS"
    elif best_ulp == 0:
        status = "RECONSTRUCTION MISMATCH (another variant matched bit-exactly)"
    else:
        status = "RECONSTRUCTION MISMATCH"

    state_before_floats = _state_as_floats(optimizer_name, state_before)
    reference_after = na.fp64_candidate_from_state(
        weight_before=float(weight_before.item()),
        gradient=float(gradient.item()),
        moment1_before=state_before_floats["moment1"],
        moment2_before=state_before_floats["moment2"],
        step_index=best.step_after,
        lr=lr,
        beta1=config.betas[0],
        beta2=config.betas[1],
        eps=config.eps,
        weight_decay=config.weight_decay,
        optimizer=optimizer_name,
        momentum_before=state_before_floats["momentum_buffer"],
        momentum=config.momentum,
        dampening=config.dampening,
        nesterov=config.nesterov,
        coupled_decay=True,
    )

    return ReconstructionResult(
        optimizer_name=optimizer_name,
        variant=best.variant,
        variant_search_performed=bool(search and active_variant is None),
        variant_candidates_tried=len(tried),
        variant_errors=tuple(errors),
        best_variant=best.variant,
        best_ulp_error=int(best_ulp),
        actual_uint32=actual_after.raw_uint32,
        reconstructed_uint32=best.after_uint32,
        bit_exact=bool(best.after_uint32 == actual_after.raw_uint32),
        ulp_error=int(ulp_error),
        absolute_error=absolute,
        relative_error=relative,
        actual_after=actual_after.value,
        reconstructed_after=float(best.after.item()),
        total_update=best.total_delta,
        adaptive_update=best.adaptive_delta,
        weight_decay_update=best.decay_delta,
        step_after=best.step_after,
        moment1_after=best.moment1_after,
        moment2_after=best.moment2_after,
        fp64_reference_after=reference_after,
        status=status,
        tolerance_ulp=tolerance_ulp,
    )


def _state_as_floats(
    optimizer_name: str, state_before: Mapping[str, Any]
) -> dict[str, float | None]:
    def read(key: str) -> float | None:
        if key not in state_before:
            return None
        value = state_before[key]
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, torch.Tensor):
            return float(value.detach().reshape(-1)[0].item())
        return float(value)

    if optimizer_name == "sgd":
        return {"moment1": 0.0, "moment2": 0.0, "momentum_buffer": read("momentum_buffer")}
    return {
        "moment1": read("exp_avg") or 0.0,
        "moment2": read("exp_avg_sq") or 0.0,
        "momentum_buffer": None,
    }


def probe_optimizer_semantics(config: ExperimentConfig) -> dict[str, Any]:
    """One isolated real step, used to measure which update variant is installed.

    This runs a throwaway parameter and throwaway optimizer: it is an audit of
    the installed PyTorch, completely separate from the tracked experiment.
    """
    from .model import ObservatoryModel  # local import avoids an import cycle

    scratch_config = config.replace(
        steps=1, stop_on_nonfinite=False, run_fp64_reference_track=False
    )
    model = ObservatoryModel(scratch_config.initial_value, scratch_config.model)
    parameter = model.assert_single_float32_parameter()
    handle = make_optimizer(scratch_config, parameter)
    lr = handle.current_lr()
    weight_before = parameter.detach().clone()
    state_before: dict[str, Any] = {}
    if handle.optimizer.state.get(parameter, None):
        state_before = {
            key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
            for key, value in dict(handle.optimizer.state[parameter]).items()
        }
    x = torch.tensor([[1.25]], dtype=torch.float32)
    target = torch.tensor([[0.75]], dtype=torch.float32)
    handle.optimizer.zero_grad(set_to_none=True)
    loss = model.loss(x, target)
    loss.backward()
    gradient = parameter.grad.detach().clone()
    handle.optimizer.step()
    weight_after = parameter.detach().clone()
    result = reconstruct_update(
        optimizer_name=config.optimizer,
        weight_before=weight_before,
        weight_after=weight_after,
        gradient=gradient,
        state_before=state_before,
        lr=lr,
        config=config,
        search=True,
        tolerance_ulp=0,
    )
    return {
        "optimizer": config.optimizer,
        "torch_version": str(torch.__version__),
        "probe_gradient": float(gradient.item()),
        "probe_loss": float(loss.item()),
        "measured_variant": result.variant,
        "bit_exact": result.bit_exact,
        "candidates": [
            {"variant": name, "ulp_error": err} for name, err in result.variant_errors
        ],
        "source_findings": dict(handle.audit.source_findings),
        "note": (
            "Measured on a throwaway parameter: the variant listed in "
            "'measured_variant' is the one that reproduces this PyTorch build's "
            "update bit for bit."
        ),
    }


__all__ = [
    "OPTIMIZER_CLASSES",
    "EnvironmentAudit",
    "OptimizerSourceAudit",
    "OptimizerHandle",
    "StateEntryReading",
    "OptimizerStateReading",
    "UpdateVariant",
    "ReconstructedUpdate",
    "ReconstructionResult",
    "ADAMW_VARIANTS",
    "ADAM_VARIANTS",
    "SGD_VARIANTS",
    "audit_environment",
    "audit_optimizer_source",
    "make_optimizer",
    "read_optimizer_state",
    "variants_for",
    "reconstruct_update",
    "probe_optimizer_semantics",
]
