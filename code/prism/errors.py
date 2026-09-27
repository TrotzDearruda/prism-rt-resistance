"""Exception hierarchy shared by every Prism subsystem.

Raised only at module boundaries that a caller can act on: configuration intake,
resource access, feasibility rejection and checkpoint integrity.
"""

from __future__ import annotations


class PrismError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(PrismError):
    """A configuration value is missing, of the wrong type, or contradictory."""


class CohortError(PrismError):
    """A cohort definition is internally inconsistent."""


class SupervisionError(CohortError):
    """Metabolite supervision is requested for a record that cannot carry it."""


class ResourceError(PrismError):
    """A public resource could not be read, or its digest does not match."""


class FeasibilityError(PrismError):
    """The pre-declared discrimination floor was breached and cannot be repaired."""


class CheckpointError(PrismError):
    """A checkpoint is absent, unreadable, or fails its recorded digest."""


class ProtocolError(PrismError):
    """An evaluation was requested that the pre-declared protocol does not permit."""
