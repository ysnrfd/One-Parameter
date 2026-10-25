"""Error taxonomy for the observatory.

Every failure mode listed in the specification's "no silent failure" section has
a named exception here.  Nothing in this project catches an exception and
continues without recording the fact; a caught exception is either converted
into an explicit record (and flagged) or re-raised.

The module exists as a separate file (instead of living inside ``config.py``) so
that every other module can import the taxonomy without creating an import cycle:
config -> errors, and nothing in errors imports project code.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping


class ObservatoryError(RuntimeError):
    """Base class for every error raised by this project."""


class ConfigError(ObservatoryError, ValueError):
    """An experiment configuration is internally inconsistent or illegal."""


class TrackingInvalidError(ObservatoryError):
    """The tracked object is no longer a valid single float32 trainable scalar.

    The exact failing invariant is always available in :attr:`reason` so the
    failure can be reported verbatim rather than as a generic message.
    """

    def __init__(self, reason: str, details: Mapping[str, Any] | None = None) -> None:
        self.reason = reason
        self.details: dict[str, Any] = dict(details or {})
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.details.items()))
        super().__init__(f"{reason}{(' [' + rendered + ']') if rendered else ''}")


class BitDecodeError(ObservatoryError):
    """Decoded IEEE-754 fields do not re-encode to the observed raw bit pattern."""


class NumericalConsistencyError(ObservatoryError):
    """A measured quantity contradicts an independently computed one."""


class ValidationError(ObservatoryError):
    """A validation routine could not be executed at all (not merely failed)."""


class HistoryCorruptionError(ObservatoryError):
    """The immutable history hash chain does not verify."""


class NonFiniteStateError(ObservatoryError):
    """A tracked quantity became NaN or infinite during a real run."""


class ReconstructionMismatch(ObservatoryError):
    """An independent optimizer reconstruction did not match the real update."""


class DependencyMissingError(ObservatoryError):
    """An optional third-party dependency is required for the requested mode."""


__all__ = [
    "ObservatoryError",
    "ConfigError",
    "TrackingInvalidError",
    "BitDecodeError",
    "NumericalConsistencyError",
    "ValidationError",
    "HistoryCorruptionError",
    "NonFiniteStateError",
    "ReconstructionMismatch",
    "DependencyMissingError",
]
