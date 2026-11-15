"""Intensity windowing, isotropic resampling, and mask alignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from dicomvolumeqc.config import WINDOW_PRESETS, PreprocessConfig, validate_config
from dicomvolumeqc.errors import ConfigError, GeometryError
from dicomvolumeqc.ingest import load_volume
from dicomvolumeqc.models import SeriesMeta, SeriesRef, Volume

_SITK_INTERPOLATORS: dict[str, int] = {
    "linear": sitk.sitkLinear,
    "nearest": sitk.sitkNearestNeighbor,
    "bspline": sitk.sitkBSpline,
}


@dataclass(frozen=True)
class MaskAlignment:
    """Result of aligning a binary mask to an image grid."""

    aligned: Volume
    dice: float
    roundtrip_spacing_mm: float


@dataclass
class PreparedVolume:
    """A windowed, isotropic volume plus its aligned mask."""

    series_ref: SeriesRef
    image: Volume
    mask: Volume | None
    meta: SeriesMeta
    mask_dice: float | None
    target_spacing_mm: float
    window_preset: str

    @property
    def patient_id(self) -> str:
        return self.meta.patient_id

    @property
    def series_uid(self) -> str:
        return self.meta.series_uid


def _to_sitk(volume: Volume) -> sitk.Image:
    image = sitk.GetImageFromArray(volume.array)
    image.SetSpacing(tuple(float(v) for v in volume.spacing))
    image.SetOrigin(tuple(float(v) for v in volume.origin))
    image.SetDirection(tuple(float(v) for v in volume.direction))
    return image


def _meta_for_grid(
    meta: SeriesMeta, array: np.ndarray, spacing: tuple[float, float, float]
) -> SeriesMeta:
    shape = tuple(int(v) for v in array.shape)
    return replace(
        meta,
        n_slices=shape[0],
        volume_shape=shape,
        in_plane_spacing=(float(spacing[0]), float(spacing[1])),
        slice_spacing_median=float(spacing[2]),
        slice_spacing_std=0.0,
        intensity_min=float(np.min(array)),
        intensity_max=float(np.max(array)),
    )


def _interpolator(name: str) -> int:
    try:
        return _SITK_INTERPOLATORS[name]
    except KeyError as exc:
        raise ConfigError(f"unknown interpolator '{name}'") from exc


def _resolve_window(
    center: float | None, width: float | None, preset: str | None
) -> tuple[float, float]:
    if preset is not None:
        if preset not in WINDOW_PRESETS:
            raise ConfigError(
                f"unknown window preset '{preset}'; choose from {sorted(WINDOW_PRESETS)}"
            )
        return WINDOW_PRESETS[preset]
    if center is None or width is None:
        raise ConfigError("apply_window needs a preset or both center and width")
    if width <= 0:
        raise ConfigError("window width must be greater than 0")
    return float(center), float(width)


def apply_window(
    vol: Volume,
    center: float | None = None,
    width: float | None = None,
    *,
    preset: str | None = None,
) -> Volume:
    """Clip to a CT window and min-max normalize float32 output into [0, 1]."""
    center, width = _resolve_window(center, width, preset)
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(vol.array.astype(np.float32), low, high)
    normalized = ((clipped - low) / width).astype(np.float32)
    normalized = np.clip(normalized, 0.0, 1.0)
    return Volume(
        array=normalized,
        spacing=vol.spacing,
        origin=vol.origin,
        direction=vol.direction,
        meta=_meta_for_grid(vol.meta, normalized, vol.spacing),
    )


def resample_isotropic(
    vol: Volume, target_spacing_mm: float, interpolator: str = "linear"
) -> Volume:
    """Resample to equal spacing in all axes, preserving origin and direction."""
    target = float(target_spacing_mm)
    if target <= 0:
        raise ConfigError("target spacing must be greater than 0")
    image = _to_sitk(vol)
    size = tuple(
        max(1, int(round(dim * spacing / target)))
        for dim, spacing in zip(image.GetSize(), image.GetSpacing(), strict=True)
    )
    output = sitk.Resample(
        image,
        size,
        sitk.Transform(),
        _interpolator(interpolator),
        image.GetOrigin(),
        (target, target, target),
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )
    array = sitk.GetArrayFromImage(output)
    spacing = tuple(float(v) for v in output.GetSpacing())
    dtype = vol.array.dtype
    array = array.astype(dtype, copy=False)
    return Volume(
        array=array,
        spacing=spacing,
        origin=tuple(float(v) for v in output.GetOrigin()),
        direction=tuple(float(v) for v in output.GetDirection()),
        meta=_meta_for_grid(vol.meta, array, spacing),
    )


def load_mask(path: str | Path, reference_meta: SeriesMeta) -> Volume:
    """Read a binary NIfTI mask, retaining its own native physical grid."""
    try:
        image = sitk.ReadImage(str(path))
    except RuntimeError as exc:
        raise GeometryError(f"could not read mask {path}: {exc}") from exc
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3:
        raise GeometryError(f"mask {path} is not a 3D volume")
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    spacing = tuple(float(v) for v in image.GetSpacing())
    return Volume(
        array=np.ascontiguousarray(array),
        spacing=spacing,
        origin=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
        meta=_meta_for_grid(reference_meta, array, spacing),
    )


def grids_match(first: Volume, second: Volume, atol: float = 1e-4) -> bool:
    """True when two volumes occupy exactly the same voxel grid."""
    return (
        first.shape == second.shape
        and np.allclose(first.spacing, second.spacing, atol=atol, rtol=0)
        and np.allclose(first.origin, second.origin, atol=atol, rtol=0)
        and np.allclose(first.direction, second.direction, atol=atol, rtol=0)
    )


def resample_mask_like(mask: Volume, reference_volume: Volume) -> Volume:
    """Nearest-neighbour resample a mask onto the reference image grid."""
    image = _to_sitk(mask)
    size = (
        int(reference_volume.shape[2]),
        int(reference_volume.shape[1]),
        int(reference_volume.shape[0]),
    )
    output = sitk.Resample(
        image,
        size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        tuple(float(v) for v in reference_volume.origin),
        tuple(float(v) for v in reference_volume.spacing),
        tuple(float(v) for v in reference_volume.direction),
        0.0,
        image.GetPixelID(),
    )
    output_size = tuple(int(v) for v in output.GetSize())
    expected_size = (
        reference_volume.shape[2],
        reference_volume.shape[1],
        reference_volume.shape[0],
    )
    if (
        output_size != expected_size
        or not np.allclose(output.GetSpacing(), reference_volume.spacing, atol=1e-6, rtol=0)
        or not np.allclose(output.GetOrigin(), reference_volume.origin, atol=1e-6, rtol=0)
        or not np.allclose(output.GetDirection(), reference_volume.direction, atol=1e-6, rtol=0)
    ):
        raise GeometryError("mask resampling did not produce the reference image grid")

    array = sitk.GetArrayFromImage(output)
    source_labels = np.unique(mask.array)
    output_labels = np.unique(array)
    if not set(int(v) for v in output_labels) <= set(int(v) for v in source_labels):
        raise GeometryError("nearest-neighbour mask resampling introduced new labels")
    spacing = tuple(float(v) for v in reference_volume.spacing)
    array = array.astype(mask.array.dtype, copy=False)
    return Volume(
        array=array,
        spacing=spacing,
        origin=tuple(float(v) for v in reference_volume.origin),
        direction=tuple(float(v) for v in reference_volume.direction),
        meta=_meta_for_grid(reference_volume.meta, array, spacing),
    )


def dice_coefficient(first: np.ndarray, second: np.ndarray) -> float:
    """Dice overlap of two binary arrays; empty/empty is defined as 1.0."""
    a = first > 0
    b = second > 0
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denominator)


def align_mask(volume: Volume, mask: Volume) -> MaskAlignment:
    """Verify grid agreement and measure a coarse round-trip Dice.

    A mask already on the image grid is resampled to a 1.5x coarser grid and
    back; the Dice of that round trip quantifies how well small structures
    survive resampling.
    """
    if not grids_match(volume, mask):
        raise GeometryError(
            "mask grid does not match image grid "
            f"(mask shape={mask.shape}, spacing={mask.spacing}, origin={mask.origin}; "
            f"image shape={volume.shape}, spacing={volume.spacing}, origin={volume.origin})"
        )
    aligned = resample_mask_like(mask, volume)
    coarse_target = 1.5 * float(np.mean(volume.spacing))
    coarse = resample_isotropic(aligned, coarse_target, interpolator="nearest")
    back = resample_mask_like(coarse, volume)
    dice = dice_coefficient(aligned.array, back.array)
    return MaskAlignment(aligned=aligned, dice=dice, roundtrip_spacing_mm=coarse_target)


def preprocess_series(series_ref: SeriesRef, config: PreprocessConfig) -> PreparedVolume:
    """Load, resample to isotropic spacing, window, and align the mask."""
    validate_config(config)
    raw = load_volume(series_ref)
    isotropic = resample_isotropic(raw, config.target_spacing_mm, config.interpolator)
    if config.window_center is not None and config.window_width is not None:
        windowed = apply_window(isotropic, config.window_center, config.window_width)
    else:
        windowed = apply_window(isotropic, preset=config.window_preset)

    mask: Volume | None = None
    mask_dice: float | None = None
    if series_ref.mask_path is not None:
        native_mask = load_mask(series_ref.mask_path, raw.meta)
        aligned = resample_mask_like(native_mask, isotropic)
        alignment = align_mask(isotropic, aligned)
        mask = alignment.aligned
        mask_dice = alignment.dice

    return PreparedVolume(
        series_ref=series_ref,
        image=windowed,
        mask=mask,
        meta=isotropic.meta,
        mask_dice=mask_dice,
        target_spacing_mm=float(config.target_spacing_mm),
        window_preset=config.window_preset,
    )


def describe_volume(volume: Volume, label: str = "volume") -> dict[str, Any]:
    """Compact serializable description used by CLI summaries and tests."""
    return {
        "label": label,
        "shape": list(volume.shape),
        "spacing_mm": [round(float(v), 6) for v in volume.spacing],
        "origin_mm": [round(float(v), 6) for v in volume.origin],
        "dtype": str(volume.array.dtype),
        "min": round(float(volume.array.min()), 6),
        "max": round(float(volume.array.max()), 6),
    }
