"""Typed exceptions raised by DicomVolumeQC.

Every public failure mode has a dedicated exception so callers (and the CLI)
can distinguish geometry problems, QC failures, de-identification problems,
and configuration mistakes without string matching.
"""


class DvqcError(Exception):
    """Base class for all DicomVolumeQC errors."""


class GeometryError(DvqcError):
    """Raised when a volume's spatial geometry is unusable or inconsistent."""


class QCFailure(DvqcError):
    """Raised when a series fails a hard quality-control gate."""


class DeidError(DvqcError):
    """Raised when de-identification cannot be completed safely."""


class ConfigError(DvqcError):
    """Raised when a configuration file is missing, malformed, or invalid."""
