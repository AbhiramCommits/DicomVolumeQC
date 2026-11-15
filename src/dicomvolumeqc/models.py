"""Shared data models used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class Severity(StrEnum):
    """QC finding severity."""

    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Finding:
    """A structured, machine-readable QC or geometry finding."""

    code: str
    severity: Severity
    message: str
    observed: float | int | str | bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class SeriesRef:
    """A group of DICOM files believed to belong to one series."""

    series_uid: str
    directory: Path
    files: tuple[Path, ...] = ()
    unreadable_files: tuple[Path, ...] = ()
    patient_id: str = ""
    study_uid: str = ""
    mask_path: Path | None = None


@dataclass(frozen=True)
class SeriesMeta:
    """Geometric and intensity summary of a reconstructed series."""

    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    n_slices: int
    in_plane_spacing: tuple[float, float]
    slice_spacing_median: float
    slice_spacing_std: float
    orientation_cosines: tuple[float, ...]
    is_axial: bool
    volume_shape: tuple[int, int, int]
    intensity_min: float
    intensity_max: float
    file_count: int
    unreadable_count: int
    slice_positions: tuple[float, ...] = ()
    instance_numbers: tuple[int, ...] = ()
    duplicate_slices: int = 0
    inconsistent_geometry: bool = False
    constant_volume: bool = False
    mask_path: str | None = None
    mask_label_count: int | None = None
    mask_grid_matches: bool | None = None
    directory: str = ""
    series_description: str = ""
    extra_findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def slice_spacing_cv(self) -> float:
        if self.slice_spacing_median <= 0:
            return float("inf")
        return self.slice_spacing_std / self.slice_spacing_median


@dataclass
class Volume:
    """A reconstructed 3D volume with its physical grid."""

    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    meta: SeriesMeta

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.array.shape)  # type: ignore[return-value]

    @property
    def extent(self) -> tuple[float, float, float]:
        """Physical field of view in (x, y, z) order, matching ``spacing``."""
        return (
            float(self.spacing[0] * self.shape[2]),
            float(self.spacing[1] * self.shape[1]),
            float(self.spacing[2] * self.shape[0]),
        )
