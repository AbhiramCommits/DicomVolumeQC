"""QC rule registry: each rule maps a series to structured findings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dicomvolumeqc.config import PreprocessConfig
from dicomvolumeqc.geometry import (
    GeometryTolerances,
    check_axial,
    check_consistency,
    check_corrupt,
    check_orientation_cosines,
    check_slice_count,
    check_slice_gap,
    check_spacing,
)
from dicomvolumeqc.models import Finding, SeriesMeta, Severity, Volume

RuleFunc = Callable[[SeriesMeta, Volume | None], list[Finding]]


class Verdict(StrEnum):
    """Per-series QC verdict."""

    PASS = "PASS"
    WARN = "WARN"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class Rule:
    """A registered QC rule."""

    code: str
    description: str
    severity: Severity
    func: RuleFunc


class RuleRegistry:
    """Rule registry bound to one configuration (and its tolerances)."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self.tolerances = GeometryTolerances(
            min_slices=config.min_slices,
            spacing_cv_tolerance=config.spacing_cv_tolerance,
            gap_factor=config.gap_factor,
            axial_tolerance_deg=config.axial_tolerance_deg,
        )
        self.rules = self._build()

    def _build(self) -> dict[str, Rule]:
        tol = self.tolerances
        config = self.config

        def hu_range(meta: SeriesMeta, volume: Volume | None) -> list[Finding]:
            if volume is None or volume.array.size == 0:
                return []
            low = config.hu_min
            high = config.hu_max
            outside = np.count_nonzero((volume.array < low) | (volume.array > high))
            if outside == 0:
                return []
            observed = float(volume.array.min() if volume.array.min() < low else volume.array.max())
            return [
                Finding(
                    code="HU_OUT_OF_RANGE",
                    severity=Severity.FAIL,
                    message=(
                        f"{outside} voxels outside [{low:.0f}, {high:.0f}] HU; "
                        f"extreme value {observed:.1f}"
                    ),
                    observed=round(observed, 3),
                )
            ]

        def empty_or_constant(meta: SeriesMeta, volume: Volume | None) -> list[Finding]:
            if volume is not None and volume.array.size == 0:
                return [
                    Finding(
                        code="EMPTY_OR_CONSTANT_VOLUME",
                        severity=Severity.FAIL,
                        message="volume contains no voxels",
                        observed=0,
                    )
                ]
            if meta.constant_volume:
                return [
                    Finding(
                        code="EMPTY_OR_CONSTANT_VOLUME",
                        severity=Severity.FAIL,
                        message="volume has zero intensity range (constant)",
                        observed=True,
                    )
                ]
            return []

        def duplicates(meta: SeriesMeta, volume: Volume | None) -> list[Finding]:
            if meta.duplicate_slices <= 0:
                return []
            return [
                Finding(
                    code="DUPLICATE_SLICES",
                    severity=Severity.FAIL,
                    message=f"{meta.duplicate_slices} duplicate slice payload(s) detected",
                    observed=meta.duplicate_slices,
                )
            ]

        def mask_grid(meta: SeriesMeta, volume: Volume | None) -> list[Finding]:
            if meta.mask_grid_matches is not False:
                return []
            return [
                Finding(
                    code="MASK_GRID_MISMATCH",
                    severity=Severity.FAIL,
                    message="mask spacing/origin/shape does not match the image grid",
                    observed=False,
                )
            ]

        def mask_empty(meta: SeriesMeta, volume: Volume | None) -> list[Finding]:
            if meta.mask_label_count is None or meta.mask_label_count > 0:
                return []
            return [
                Finding(
                    code="MASK_EMPTY",
                    severity=Severity.FAIL,
                    message="mask contains no labeled voxels",
                    observed=0,
                )
            ]

        def mask_missing(meta: SeriesMeta, volume: Volume | None) -> list[Finding]:
            if meta.mask_path is not None:
                return []
            return [
                Finding(
                    code="MASK_MISSING",
                    severity=Severity.WARN,
                    message="no segmentation mask was found beside this series",
                    observed=False,
                )
            ]

        entries: list[tuple[str, str, Severity, RuleFunc]] = [
            (
                "TOO_FEW_SLICES",
                "Series has fewer than the configured minimum number of slices.",
                Severity.FAIL,
                lambda meta, volume: check_slice_count(meta, tol),
            ),
            (
                "NONUNIFORM_SPACING",
                "Slice spacing coefficient of variation exceeds tolerance.",
                Severity.FAIL,
                lambda meta, volume: check_spacing(meta, tol),
            ),
            (
                "SLICE_GAP",
                "A slice gap exceeds the configured multiple of median spacing.",
                Severity.FAIL,
                lambda meta, volume: check_slice_gap(meta, tol),
            ),
            (
                "DUPLICATE_SLICES",
                "Two or more slices have identical pixel payloads.",
                Severity.FAIL,
                duplicates,
            ),
            (
                "NON_AXIAL",
                "Slice normal is outside the axial tolerance.",
                Severity.FAIL,
                lambda meta, volume: check_axial(meta, tol),
            ),
            (
                "ORIENTATION_NOT_ORTHONORMAL",
                "Direction cosines are not unit and mutually orthogonal.",
                Severity.FAIL,
                lambda meta, volume: check_orientation_cosines(meta),
            ),
            (
                "CORRUPT_FILES",
                "One or more files in the series could not be decoded.",
                Severity.FAIL,
                lambda meta, volume: check_corrupt(meta),
            ),
            (
                "EMPTY_OR_CONSTANT_VOLUME",
                "Volume is empty or has no intensity variation.",
                Severity.FAIL,
                empty_or_constant,
            ),
            (
                "HU_OUT_OF_RANGE",
                "Voxel values fall outside the configured CT HU range.",
                Severity.FAIL,
                hu_range,
            ),
            (
                "INCONSISTENT_GEOMETRY",
                "Rows/Columns, PixelSpacing, or orientation vary across slices.",
                Severity.FAIL,
                lambda meta, volume: check_consistency(meta),
            ),
            (
                "MASK_GRID_MISMATCH",
                "Mask grid differs from the image grid.",
                Severity.FAIL,
                mask_grid,
            ),
            (
                "MASK_EMPTY",
                "Mask has zero labeled voxels.",
                Severity.FAIL,
                mask_empty,
            ),
            (
                "MASK_MISSING",
                "No segmentation mask is available for this series.",
                Severity.WARN,
                mask_missing,
            ),
        ]
        return {
            code: Rule(code, description, severity, func)
            for code, description, severity, func in entries
        }

    def catalog(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "code": rule.code,
                "description": rule.description,
                "severity": rule.severity.value,
            }
            for rule in sorted(self.rules.values(), key=lambda r: r.code)
        )

    def run(self, meta: SeriesMeta, volume: Volume | None) -> list[Finding]:
        findings: list[Finding] = list(meta.extra_findings)
        for code in sorted(self.rules):
            findings.extend(self.rules[code].func(meta, volume))
        unique: list[Finding] = []
        seen: set[tuple[str, str]] = set()
        for finding in findings:
            key = (finding.code, finding.message)
            if key not in seen:
                seen.add(key)
                unique.append(finding)
        return sorted(unique, key=lambda f: (f.code, f.message))
