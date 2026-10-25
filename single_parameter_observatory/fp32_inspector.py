"""IEEE-754 binary32 forensics: bits, decoding, ULP neighbourhood, bit transitions.

Every routine in this module works from *observed* storage.  Bit patterns are
read out of the actual tensor memory (or re-derived from an actual Python float
with ``struct``, which is the same round-to-nearest-even rounding PyTorch uses
when it narrows a double to a float).  Nothing is estimated, and every decode is
verified by re-encoding: if a decoded field set does not reproduce the observed
raw word, :class:`~.errors.BitDecodeError` is raised instead of returning a
plausible-looking but wrong answer.
"""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any
from collections.abc import Mapping, Sequence

import torch

from .errors import BitDecodeError, NumericalConsistencyError

#: binary32 layout
_SIGN_SHIFT = 31
_EXPONENT_SHIFT = 23
_EXPONENT_MASK = 0xFF
_FRACTION_MASK = 0x7FFFFF
_UINT32_MASK = 0xFFFFFFFF
_SIGN_MASK = 0x80000000
_EXPONENT_FIELD_MASK = 0x7F800000

_F32 = torch.float32
_POS_INF = float("inf")
_NEG_INF = float("-inf")

#: Smallest positive normal and smallest positive subnormal of binary32.
SMALLEST_NORMAL_F32 = 1.1754943508222875e-38
SMALLEST_SUBNORMAL_F32 = 1.401298464324817e-45

#: Machine epsilon (the gap from 1.0 to the next representable float32).
FLOAT32_EPSILON = 1.1920928955078125e-07

# --------------------------------------------------------------------------- #
# raw word <-> value
# --------------------------------------------------------------------------- #


def raw_word_from_python_float(value: float) -> int:
    """Raw binary32 word of ``value`` after rounding to nearest float32.

    A Python ``float`` is a double.  ``struct.pack('<f', ...)`` narrows it with
    the same round-to-nearest-even rule that ``torch.tensor(x, torch.float32)``
    applies, so the two agree bit for bit; :func:`assert_python_and_tensor_agree`
    checks that on real tensors.
    """
    as_double = float(value)
    if math.isnan(as_double):
        return 0x7FC00000  # canonical quiet NaN, used only for NaN inputs
    try:
        return struct.unpack("<I", struct.pack("<f", as_double))[0]
    except OverflowError:  # pragma: no cover - struct raises for values > f32 max
        return 0x7F800000 if as_double > 0 else 0xFF800000


def python_float_from_raw_word(raw: int) -> float:
    """Exact float32 value of a raw binary32 word."""
    return struct.unpack("<f", struct.pack("<I", int(raw) & _UINT32_MASK))[0]


def raw_word_from_tensor(tensor: torch.Tensor) -> int:
    """Raw binary32 word read out of a real float32 tensor's memory.

    The primary path reinterprets the tensor's own bytes (``view(torch.int32)``);
    the ``struct`` path is a fallback for exotic devices/dtypes and is also used
    by the independent cross-check in :func:`assert_python_and_tensor_agree`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(tensor).__name__}")
    if tensor.numel() != 1:
        raise NumericalConsistencyError(
            "raw_word_from_tensor requires a single-element tensor, "
            f"got shape {tuple(tensor.shape)}"
        )
    flat = tensor.detach().to("cpu").reshape(-1)
    if flat.dtype is torch.float32:
        try:
            as_int = flat.contiguous().view(torch.int32)
            return int(as_int[0].item()) & _UINT32_MASK
        except RuntimeError:  # pragma: no cover - defensive
            pass
    return raw_word_from_python_float(float(flat[0].item()))


def float32_tensor(value: float) -> torch.Tensor:
    """A real float32 tensor holding ``value`` rounded to nearest float32.

    Used to produce the *stored* form of a mathematical candidate, so the
    rounding residual is measured against an actual float32, not against a
    hand-computed rounding.
    """
    return torch.tensor(float(value), dtype=_F32)


def assert_python_and_tensor_agree(tensor: torch.Tensor) -> int:
    """Return the raw word, asserting both extraction paths produce it.

    Raises :class:`NumericalConsistencyError` when the tensor-reinterpretation
    path and the ``struct`` path disagree, which would mean the project is
    reading something other than the stored value.
    """
    if tensor.numel() != 1:
        raise NumericalConsistencyError("expected a single-element tensor")
    from_tensor = raw_word_from_tensor(tensor)
    from_python = raw_word_from_python_float(float(tensor.detach().reshape(-1)[0].item()))
    if from_tensor != from_python:
        raise NumericalConsistencyError(
            "float32 extraction paths disagree: "
            f"tensor.view={from_tensor:#010x} struct={from_python:#010x}"
        )
    return from_tensor


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Float32Bits:
    """Complete forensic description of one stored binary32 value.

    Two decimal forms are reported and they are *not* the same quantity:

    ``decimal_exact``
        the exact rational expansion of the stored value, with no rounding
        applied anywhere (``float32(0.1)`` is
        ``0.100000001490116119384765625``);
    ``decimal_shortest_float32``
        the shortest decimal string that parses back to the same float32 bits
        (``0.1``).

    ``scientific`` is a rounded display form for reading magnitude at a glance
    and is not an exact value.

    The first two are produced by two distinct helpers that take a raw ``uint32``
    word and apply its sign bit themselves:
    :func:`_exact_decimal` returns the exact expansion as a ``str``, and
    :func:`_exact_fraction` returns the same value as a signed ``Fraction``.  Both
    are exact; :func:`_shortest_float32_repr` is the round-tripping form and is not
    exact.
    """

    raw_uint32: int
    value: float
    hex: str
    binary: str
    sign_bit: int
    exponent_bits: int
    fraction_bits_int: int
    fraction_bits_binary: str
    unbiased_exponent: int
    decimal_exact: str
    decimal_shortest_float32: str
    scientific: str
    is_zero: bool
    is_subnormal: bool
    is_infinite: bool
    is_nan: bool
    is_finite: bool
    effective_exponent: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self, indent: str = "  ") -> str:
        lines = [
            f"exact rational value   : {self.decimal_exact}",
            f"shortest round-tripping : {self.decimal_shortest_float32}",
            f"scientific (display)    : {self.scientific}",
            f"hex                     : {self.hex}",
            f"raw uint32              : 0x{self.raw_uint32:08X} ({self.raw_uint32})",
            f"binary32                : {self.binary}",
            f"sign bit                : {self.sign_bit}",
            f"exponent (biased)       : {self.exponent_bits}",
            f"exponent (unbiased)     : {self.unbiased_exponent}",
            f"fraction (23 bits)      : {self.fraction_bits_binary} ({self.fraction_bits_int})",
            f"classification          : {_classification(self)}",
        ]
        return "\n".join(indent + line for line in lines)


def _classification(bits: Float32Bits) -> str:
    if bits.is_nan:
        return "NaN"
    if bits.is_infinite:
        return "infinity"
    if bits.is_zero:
        return "zero"
    if bits.is_subnormal:
        return "subnormal"
    return "normal"


def decode_float32_bits(raw: int) -> Float32Bits:
    """Decode a raw binary32 word into sign / exponent / fraction fields."""
    word = int(raw) & _UINT32_MASK
    sign = (word >> _SIGN_SHIFT) & 0x1
    exponent = (word >> _EXPONENT_SHIFT) & _EXPONENT_MASK
    fraction = word & _FRACTION_MASK
    value = python_float_from_raw_word(word)
    is_nan = exponent == _EXPONENT_MASK and fraction != 0
    is_infinite = exponent == _EXPONENT_MASK and fraction == 0
    is_zero = exponent == 0 and fraction == 0
    is_subnormal = exponent == 0 and fraction != 0
    bits = Float32Bits(
        raw_uint32=word,
        value=value,
        hex=f"0x{word:08X}",
        binary=f"{word:032b}",
        sign_bit=sign,
        exponent_bits=exponent,
        fraction_bits_int=fraction,
        fraction_bits_binary=f"{fraction:023b}",
        unbiased_exponent=exponent - 127,
        decimal_exact=_exact_decimal(word),
        decimal_shortest_float32=_shortest_float32_repr(value),
        scientific=f"{value:.9e}" if math.isfinite(value) else str(value),
        is_zero=is_zero,
        is_subnormal=is_subnormal,
        is_infinite=is_infinite,
        is_nan=is_nan,
        is_finite=math.isfinite(value),
        effective_exponent=_effective_exponent(exponent, fraction),
    )
    verify_decode(bits)
    return bits


def _exact_decimal(word: int) -> str:
    """Exact decimal expansion of the stored float32; returns a ``str``.

    Takes a raw ``uint32`` word and applies the sign bit itself, carrying the sign
    as a leading ``-`` in the returned string.  Not a round-tripping form - see
    :func:`_shortest_float32_repr` and :func:`_exact_fraction`, which are distinct
    quantities with distinct return types.

    The expansion is exact and terminates.  Every finite binary32 is a rational
    ``mantissa * 2**e`` with integer ``mantissa`` and integer ``e``, and a negative
    ``e`` means division by a power of two - a finite decimal, because the
    denominator becomes a power of ten when numerator and denominator are scaled
    by a power of five.  Nothing here is shortened or rounded.

    For ``float32(0.1)`` this returns ``0.100000001490116119384765625``, where
    :func:`_shortest_float32_repr` returns ``0.1``.
    """
    raw = int(word) & _UINT32_MASK
    negative = bool(raw & _SIGN_MASK)
    exponent = (raw >> _EXPONENT_SHIFT) & _EXPONENT_MASK
    fraction = raw & _FRACTION_MASK
    if exponent == _EXPONENT_MASK:
        if fraction:
            return "nan"
        return "-inf" if negative else "inf"
    if exponent == 0 and fraction == 0:
        return "-0.0" if negative else "0.0"
    if exponent == 0:
        # Subnormal: no implicit leading 1, and the scale is 2**-149.
        mantissa, power = fraction, -149
    else:
        mantissa, power = (1 << 23) | fraction, exponent - 127 - 23
    if power >= 0:
        # Integral value. Rendered with a ".0" so that an exactly representable
        # value looks the same here as in the fractional branch.
        magnitude = f"{mantissa << power}.0"
    else:
        scale = -power
        # mantissa / 2**scale == (mantissa * 5**scale) / 10**scale
        digits = str(mantissa * 5**scale).rjust(scale + 1, "0")
        integer_part = digits[:-scale].lstrip("0") or "0"
        fractional_part = digits[-scale:].rstrip("0")
        magnitude = f"{integer_part}.{fractional_part}" if fractional_part else f"{integer_part}.0"
    return f"-{magnitude}" if negative else magnitude


def _exact_fraction(word: int) -> Fraction:
    """Exact rational *value* of a finite stored float32; returns a ``Fraction``.

    Takes a raw ``uint32`` word and applies the sign bit itself, so the returned
    Fraction is signed - a negative word yields a negative Fraction, and ``-0.0``
    yields zero.  The counterpart :func:`_exact_decimal` returns the same value as
    a decimal ``str`` with the sign in the string, and :func:`_shortest_float32_repr`
    returns a *different* quantity: the shortest decimal that round-trips.

    Used for the tie test in the rounding analysis, where a float comparison
    cannot distinguish an exact halfway case from a near miss.
    """
    raw = int(word) & _UINT32_MASK
    exponent = (raw >> _EXPONENT_SHIFT) & _EXPONENT_MASK
    fraction = raw & _FRACTION_MASK
    mantissa, power = (fraction, -149) if exponent == 0 else (
        (1 << 23) | fraction,
        exponent - 127 - 23,
    )
    if raw & _SIGN_MASK:
        mantissa = -mantissa
    value = Fraction(mantissa)
    return value * Fraction(2) ** power if power >= 0 else value / Fraction(2) ** (-power)


def _shortest_float32_repr(value: float) -> str:
    """Shortest decimal string that parses back to the same float32 bits.

    float32 needs at most 9 significant decimal digits, so the loop terminates
    quickly and needs no third-party shortest-repr library.
    """
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    for precision in range(1, 10):
        candidate = f"{value:.{precision}g}"
        try:
            if raw_word_from_python_float(float(candidate)) == raw_word_from_python_float(value):
                return candidate
        except (ValueError, OverflowError):
            continue
    return f"{value:.9g}"  # pragma: no cover - unreachable for binary32


def _effective_exponent(exponent: int, fraction: int) -> int:
    """The exponent the stored bits are actually scaled by.

    For a subnormal the exponent field is zero, but the value is still scaled as
    if the field were one, so the effective exponent is -126 - exactly as for
    the smallest normal.  Reporting -149 here would conflate "the exponent field"
    with "log2 of the smallest subnormal", which are different quantities.
    """
    if exponent == 0:
        return -126
    if exponent == _EXPONENT_MASK:
        return 128
    return exponent - 127


def verify_decode(bits: Float32Bits) -> None:
    """Re-encode the decoded fields and assert the raw word is reproduced."""
    word = encode_float32_bits(bits.sign_bit, bits.exponent_bits, bits.fraction_bits_int)
    if word != bits.raw_uint32:
        raise BitDecodeError(
            "float32 decode did not round-trip: "
            f"decoded {bits.sign_bit}/{bits.exponent_bits}/{bits.fraction_bits_int} "
            f"-> 0x{word:08X}, observed 0x{bits.raw_uint32:08X}"
        )
    if struct.pack("<f", bits.value) != struct.pack("<I", bits.raw_uint32):
        raise BitDecodeError("float32 value/raw-word pair is internally inconsistent")


def encode_float32_bits(sign: int, exponent: int, fraction: int) -> int:
    """Inverse of :func:`decode_float32_bits` (used by the bit-level tests)."""
    if sign not in (0, 1):
        raise BitDecodeError(f"sign must be 0 or 1, got {sign!r}")
    if not 0 <= exponent <= _EXPONENT_MASK:
        raise BitDecodeError(f"exponent out of range: {exponent!r}")
    if not 0 <= fraction <= _FRACTION_MASK:
        raise BitDecodeError(f"fraction out of range: {fraction!r}")
    return (sign << _SIGN_SHIFT) | (exponent << _EXPONENT_SHIFT) | fraction


def inspect_float32(value: Any, *, prefer_tensor_path: bool = False) -> Float32Bits:
    """Inspect a float32 value supplied as a tensor, a float, or a raw word.

    ``prefer_tensor_path`` re-reads a float through a real float32 tensor, which
    is how the tracked parameter's bits are always obtained.
    """
    if isinstance(value, torch.Tensor):
        return decode_float32_bits(raw_word_from_tensor(value.float()))
    if isinstance(value, bool):
        raise TypeError("refusing to inspect a bool as a float32")
    if isinstance(value, int):
        return decode_float32_bits(raw_word_from_python_float(float(value)))
    if prefer_tensor_path:
        return decode_float32_bits(raw_word_from_tensor(float32_tensor(float(value))))
    return decode_float32_bits(raw_word_from_python_float(float(value)))


# --------------------------------------------------------------------------- #
# ULP neighbourhood
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ULPNeighbourhood:
    """Representable float32 neighbours of one value."""

    value: float
    previous_representable: float
    next_representable: float
    distance_to_previous: float
    distance_to_next: float
    ulp_size: float
    ulp_size_down: float
    is_finite: bool
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self, indent: str = "  ") -> str:
        return "\n".join(
            indent + line
            for line in (
                f"previous representable : {self.previous_representable!r}",
                f"next representable     : {self.next_representable!r}",
                f"distance to previous   : {self.distance_to_previous!r}",
                f"distance to next       : {self.distance_to_next!r}",
                f"ULP (binade spacing)   : {self.ulp_size!r}",
                f"ULP (towards zero)     : {self.ulp_size_down!r}",
                f"method                 : {self.method}",
            )
        )


def ulp_neighbourhood(value: float) -> ULPNeighbourhood:
    """Neighbouring representable float32 values, via ``torch.nextafter``."""
    t = torch.tensor(float(value), dtype=_F32)
    up = torch.nextafter(t, torch.tensor(_POS_INF, dtype=_F32))
    down = torch.nextafter(t, torch.tensor(_NEG_INF, dtype=_F32))
    current = float(t.item())
    next_value = float(up.item())
    prev_value = float(down.item())
    distance_next = next_value - current
    distance_prev = current - prev_value
    # "ULP" is reported symmetrically as the binade spacing (away from zero),
    # which is distance_next for positive values and distance_prev for negative.
    # At a power-of-two boundary, the gap towards zero is half the gap away from zero.
    # Both are reported so no ambiguity is hidden.
    if current >= 0.0:
        binade_spacing = distance_next
        towards_zero_spacing = distance_prev
    else:
        binade_spacing = distance_prev
        towards_zero_spacing = distance_next
    # The next value beyond the largest finite float is infinity, but the
    # finite binade itself still has spacing 2**104.
    if math.isfinite(current) and math.isinf(binade_spacing):
        binade_spacing = 2.0**104

    return ULPNeighbourhood(
        value=current,
        previous_representable=prev_value,
        next_representable=next_value,
        distance_to_previous=distance_prev,
        distance_to_next=distance_next,
        ulp_size=binade_spacing,
        ulp_size_down=towards_zero_spacing,
        is_finite=math.isfinite(current),
        method="torch.nextafter",
    )


def ulp_size(value: float) -> float:
    return ulp_neighbourhood(value).ulp_size


def _order_key(raw: int) -> int:
    """Map a raw float32 word onto a monotonically increasing integer.

    Used to count representable intervals exactly.  Two float32 words ``a`` and
    ``b`` are exactly ``key(b) - key(a)`` representable values apart.
    """
    word = int(raw) & _UINT32_MASK
    magnitude = word & (_UINT32_MASK ^ _SIGN_MASK)  # drop the sign bit
    return -magnitude if (word >> _SIGN_SHIFT) & 0x1 else magnitude


def ulp_distance(before: float, after: float) -> int:
    """Signed number of representable float32 steps from ``before`` to ``after``."""
    raw_before = raw_word_from_python_float(before)
    raw_after = raw_word_from_python_float(after)
    for raw in (raw_before, raw_after):
        exponent = (raw >> _EXPONENT_SHIFT) & _EXPONENT_MASK
        if exponent == _EXPONENT_MASK and (raw & _FRACTION_MASK) != 0:
            raise NumericalConsistencyError("ULP distance is undefined for NaN")
    return _order_key(raw_after) - _order_key(raw_before)


def ulp_distance_same_binade(before: float, after: float) -> int:
    """Signed ULP count measured in ULPs of the *starting* value.

    :func:`ulp_distance` counts how many representable float32 values lie
    between two values, which is the right answer to "how many distinct float32
    values could the parameter have been" but a poor answer to "how far did it
    move": when a step crosses zero the count passes through the entire subnormal
    range and lands around 2.1 billion, which swamps every other step in an
    aggregate.

    This metric instead divides the displacement by the ULP at the starting
    value and rounds toward zero, so a binade boundary or a sign crossing does
    not blow it up.  It is an approximation of a per-step movement, and is
    reported next to :func:`ulp_distance` rather than instead of it.
    """
    start = float(before)
    end = float(after)
    if not (math.isfinite(start) and math.isfinite(end)):
        return 0
    spacing = ulp_size(start)
    if spacing <= 0.0:
        return 0 if start == end else int(math.copysign(float("inf"), end - start))
    delta = end - start
    steps = int(abs(delta) / spacing)  # truncates toward zero
    return steps if delta >= 0.0 else -steps


def ulp_movement(before: float, after: float) -> float:
    """|Δw| expressed in ULPs of the *starting* value (0.0 when w did not move)."""
    before_ulp = ulp_size(before)
    delta = abs(float(after) - float(before))
    if before_ulp <= 0.0:
        return 0.0 if delta == 0.0 else float("inf")
    return delta / before_ulp


# --------------------------------------------------------------------------- #
# bit transitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BitTransition:
    """Bit-level difference between two consecutive stored float32 words."""

    before_uint32: int
    after_uint32: int
    xor_mask: int
    changed_bit_count: int
    #: bit positions counted from the most significant bit (0 = sign, 31 = last
    #: fraction bit); these are positions in the actual stored words.
    changed_bit_positions: tuple[int, ...]
    sign_changed: bool
    exponent_changed: bool
    fraction_changed: bool
    ulp_distance: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["changed_bit_positions"] = list(self.changed_bit_positions)
        return data

    def render(self, indent: str = "  ") -> str:
        positions = ", ".join(str(p) for p in self.changed_bit_positions) or "(none)"
        return "\n".join(
            indent + line
            for line in (
                f"before                 : 0x{self.before_uint32:08X}",
                f"after                  : 0x{self.after_uint32:08X}",
                f"XOR mask               : 0x{self.xor_mask:08X} "
                f"({self.xor_mask:032b})",
                f"changed bit count      : {self.changed_bit_count}",
                f"changed bit positions  : {positions} (0 = sign ... 31 = fraction LSB)",
                f"sign changed           : {self.sign_changed}",
                f"exponent changed       : {self.exponent_changed}",
                f"fraction changed       : {self.fraction_changed}",
                f"distance in ULPs       : {self.ulp_distance}",
            )
        )


def bit_transition(before_uint32: int, after_uint32: int) -> BitTransition:
    """XOR two stored words and report which actual bits changed."""
    a = int(before_uint32) & _UINT32_MASK
    b = int(after_uint32) & _UINT32_MASK
    xor = a ^ b
    positions = tuple(i for i in range(32) if (xor >> (31 - i)) & 0x1)
    return BitTransition(
        before_uint32=a,
        after_uint32=b,
        xor_mask=xor,
        changed_bit_count=len(positions),
        changed_bit_positions=positions,
        sign_changed=bool(xor & _SIGN_MASK),
        exponent_changed=bool(xor & _EXPONENT_FIELD_MASK),
        fraction_changed=bool(xor & _FRACTION_MASK),
        ulp_distance=_order_key(b) - _order_key(a),
    )


# --------------------------------------------------------------------------- #
# storage evidence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StorageEvidence:
    """Direct evidence that a tensor is stored as binary32 in memory."""

    dtype: str
    element_size_bytes: int
    numel: int
    total_bytes: int
    raw_storage_bytes: bytes
    storage_nbytes: int | None
    storage_nbytes_method: str
    shape: tuple[int, ...]
    device: str
    is_floating_point: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw_storage_bytes"] = " ".join(f"{b:02X}" for b in self.raw_storage_bytes)
        return data

    def render(self, indent: str = "  ") -> str:
        lines = [
            f"dtype                  : {self.dtype}",
            f"element_size_bytes     : {self.element_size_bytes}",
            f"numel                  : {self.numel}",
            f"shape                  : {self.shape}",
            f"total_bytes            : {self.total_bytes}",
            "raw storage bytes      : "
            + " ".join(f"{b:02X}" for b in self.raw_storage_bytes),
            f"untyped_storage.nbytes : {self.storage_nbytes} "
            f"(via {self.storage_nbytes_method})",
            f"device                 : {self.device}",
        ]
        return "\n".join(indent + line for line in lines)


def storage_evidence(tensor: torch.Tensor) -> StorageEvidence:
    """Collect byte-level evidence of how a tensor is actually stored."""
    flat = tensor.detach().to("cpu").reshape(-1).contiguous()
    as_bytes = bytes(int(b) for b in flat.view(torch.uint8).tolist())
    storage_nbytes: int | None
    method: str
    try:
        storage_nbytes = int(tensor.untyped_storage().nbytes())
        method = "UntypedStorage.nbytes()"
    except (AttributeError, RuntimeError):  # pragma: no cover - very old torch
        storage_nbytes = int(tensor.untyped_storage().size()) * int(tensor.element_size())
        method = "UntypedStorage.size() * element_size()"
    return StorageEvidence(
        dtype=str(tensor.dtype),
        element_size_bytes=int(tensor.element_size()),
        numel=int(tensor.numel()),
        total_bytes=int(tensor.numel()) * int(tensor.element_size()),
        raw_storage_bytes=as_bytes,
        storage_nbytes=storage_nbytes,
        storage_nbytes_method=method,
        shape=tuple(int(s) for s in tensor.shape),
        device=str(tensor.device),
        is_floating_point=bool(tensor.is_floating_point()),
    )


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #


def summarise_bits_sequence(raw_words: Sequence[int]) -> Mapping[str, Any]:
    """Aggregate bit statistics over a recorded trajectory (recorded words only)."""
    if not raw_words:
        return {"steps": 0, "distinct_words": 0, "sign_flips": 0, "exponent_changes": 0}
    sign_flips = 0
    exponent_changes = 0
    previous = raw_words[0]
    for word in raw_words[1:]:
        if (previous ^ word) & _SIGN_MASK:
            sign_flips += 1
        if (previous ^ word) & _EXPONENT_FIELD_MASK:
            exponent_changes += 1
        previous = word
    return {
        "steps": len(raw_words),
        "distinct_words": len(set(int(w) for w in raw_words)),
        "sign_flips": sign_flips,
        "exponent_changes": exponent_changes,
    }


__all__ = [
    "FLOAT32_EPSILON",
    "SMALLEST_NORMAL_F32",
    "SMALLEST_SUBNORMAL_F32",
    "Float32Bits",
    "ULPNeighbourhood",
    "BitTransition",
    "StorageEvidence",
    "raw_word_from_python_float",
    "python_float_from_raw_word",
    "raw_word_from_tensor",
    "float32_tensor",
    "assert_python_and_tensor_agree",
    "decode_float32_bits",
    "encode_float32_bits",
    "verify_decode",
    "inspect_float32",
    "ulp_neighbourhood",
    "ulp_size",
    "ulp_distance",
    "ulp_distance_same_binade",
    "ulp_movement",
    "bit_transition",
    "storage_evidence",
    "summarise_bits_sequence",
]
