"""Reusable geometry checks returning structured findings.

Both the ingest validator and the QC rule registry compose these functions,
so a tolerance change cannot make the two layers disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dicomvolumeqc.models import Finding, SeriesMeta, Severity


@dataclass(frozen=True)
class GeometryTolerances:
    """Tolerances applied to every geometry check."""

    min_slices: int = 3
    spacing_cv_tolerance: float = 0.05
    gap_factor: float = 1.5
    axial_tolerance_deg: float = 5.0


def check_slice_count(meta: SeriesMeta, tol: GeometryTolerances) -> list[Finding]:
    if meta.n_slices >= tol.min_slices:
        return []
    return [
        Finding(
            code="TOO_FEW_SLICES",
            severity=Severity.FAIL,
            message=f"series has {meta.n_slices} slices; at least {tol.min_slices} are required",
            observed=meta.n_slices,
        )
    ]


def check_orientation_cosines(meta: SeriesMeta) -> list[Finding]:
    """Check that the six direction cosines are unit and orthogonal."""
    cosines = np.asarray(meta.orientation_cosines, dtype=float)
    if cosines.shape != (6,):
        return [
            Finding(
                code="ORIENTATION_NOT_ORTHONORMAL",
                severity=Severity.FAIL,
                message="ImageOrientationPatient does not contain six cosines",
                observed=str(meta.orientation_cosines),
            )
        ]
    row = cosines[:3]
    col = cosines[3:]
    orthonormal = (
        abs(float(np.linalg.norm(row)) - 1.0) <= 1e-3
        and abs(float(np.linalg.norm(col)) - 1.0) <= 1e-3
        and abs(float(np.dot(row, col))) <= 1e-3
    )
    if orthonormal:
        return []
    return [
        Finding(
            code="ORIENTATION_NOT_ORTHONORMAL",
            severity=Severity.FAIL,
            message="ImageOrientationPatient cosines are not orthonormal",
            observed=round(float(np.dot(row, col)), 6),
        )
    ]


def check_axial(meta: SeriesMeta, tol: GeometryTolerances) -> list[Finding]:
    """Check that the slice normal is within tolerance of the axial axis."""
    cosines = np.asarray(meta.orientation_cosines, dtype=float)
    if cosines.shape != (6,):
        return []
    normal = np.cross(cosines[:3], cosines[3:])
    norm = float(np.linalg.norm(normal))
    if norm == 0:
        return []
    angle = float(np.degrees(np.arccos(np.clip(abs(normal[2] / norm), -1.0, 1.0))))
    if angle <= tol.axial_tolerance_deg:
        return []
    return [
        Finding(
            code="NON_AXIAL",
            severity=Severity.FAIL,
            message=(
                f"slice normal is {angle:.2f} deg from axial; "
                f"tolerance is {tol.axial_tolerance_deg:.2f} deg"
            ),
            observed=round(angle, 3),
        )
    ]


def check_orientation(meta: SeriesMeta, tol: GeometryTolerances) -> list[Finding]:
    """Combined orientation check for callers that want one call."""
    return check_orientation_cosines(meta) + check_axial(meta, tol)


def check_spacing(meta: SeriesMeta, tol: GeometryTolerances) -> list[Finding]:
    if np.isfinite(meta.slice_spacing_cv) and meta.slice_spacing_cv > tol.spacing_cv_tolerance:
        return [
            Finding(
                code="NONUNIFORM_SPACING",
                severity=Severity.FAIL,
                message=(
                    f"slice spacing CV {meta.slice_spacing_cv:.4f} exceeds "
                    f"tolerance {tol.spacing_cv_tolerance:.4f}"
                ),
                observed=round(meta.slice_spacing_cv, 6),
            )
        ]
    return []


def check_slice_gap(meta: SeriesMeta, tol: GeometryTolerances) -> list[Finding]:
    positions = np.asarray(meta.slice_positions, dtype=float)
    if positions.size < 2 or meta.slice_spacing_median <= 0:
        return []
    diffs = np.diff(positions)
    positive = diffs[diffs > 1e-4]
    if not positive.size:
        return []
    max_gap = float(positive.max())
    limit = tol.gap_factor * meta.slice_spacing_median
    if max_gap <= limit:
        return []
    return [
        Finding(
            code="SLICE_GAP",
            severity=Severity.FAIL,
            message=(
                f"largest slice gap {max_gap:.3f} mm exceeds "
                f"{tol.gap_factor:.2f}x median ({limit:.3f} mm)"
            ),
            observed=round(max_gap, 4),
        )
    ]


def check_consistency(meta: SeriesMeta) -> list[Finding]:
    if not meta.inconsistent_geometry:
        return []
    return [
        Finding(
            code="INCONSISTENT_GEOMETRY",
            severity=Severity.FAIL,
            message="Rows/Columns, PixelSpacing, or orientation differ across slices",
            observed=True,
        )
    ]


def check_corrupt(meta: SeriesMeta) -> list[Finding]:
    if meta.unreadable_count <= 0:
        return []
    return [
        Finding(
            code="CORRUPT_FILES",
            severity=Severity.FAIL,
            message=f"{meta.unreadable_count} file(s) could not be decoded",
            observed=meta.unreadable_count,
        )
    ]


def validate_geometry(
    meta: SeriesMeta,
    *,
    spacing_cv_tolerance: float = 0.05,
    gap_factor: float = 1.5,
    axial_tolerance_deg: float = 5.0,
    min_slices: int = 3,
) -> list[Finding]:
    """Return structured geometry findings; never raise for a defect."""
    tol = GeometryTolerances(
        min_slices=min_slices,
        spacing_cv_tolerance=spacing_cv_tolerance,
        gap_factor=gap_factor,
        axial_tolerance_deg=axial_tolerance_deg,
    )
    findings: list[Finding] = []
    findings.extend(check_slice_count(meta, tol))
    findings.extend(check_orientation(meta, tol))
    findings.extend(check_spacing(meta, tol))
    findings.extend(check_slice_gap(meta, tol))
    findings.extend(check_consistency(meta))
    findings.extend(check_corrupt(meta))
    return findings
