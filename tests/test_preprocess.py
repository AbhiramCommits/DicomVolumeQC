"""Tests for windowing, isotropic resampling, and mask alignment."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dicomvolumeqc.config import PreprocessConfig, load_config
from dicomvolumeqc.errors import ConfigError, GeometryError
from dicomvolumeqc.fixtures import build_fixture_set
from dicomvolumeqc.ingest import discover_series, load_volume
from dicomvolumeqc.models import SeriesRef
from dicomvolumeqc.preprocess import (
    align_mask,
    apply_window,
    describe_volume,
    load_mask,
    preprocess_series,
    resample_isotropic,
    resample_mask_like,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("preprocess_corpus")
    build_fixture_set(root)
    return root


@pytest.fixture(scope="module")
def refs(corpus: Path) -> list[SeriesRef]:
    return discover_series(corpus / "raw")


@pytest.fixture(scope="module")
def config() -> PreprocessConfig:
    return load_config()


def _ref(refs: list[SeriesRef], name: str) -> SeriesRef:
    return next(ref for ref in refs if ref.directory.name == name)


def test_default_config_values(config: PreprocessConfig) -> None:
    assert config.target_spacing_mm == 1.0
    assert config.window_preset == "soft_tissue"
    assert config.interpolator == "linear"
    assert config.split_ratio_map == {"train": 0.6, "val": 0.2, "test": 0.2}
    assert config.seed == 20240101
    assert len(config.content_hash()) == 64


def test_config_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    first = load_config()
    second = load_config()
    assert first.content_hash() == second.content_hash()
    override = {
        "target_spacing_mm": 1.5,
        "window_preset": "lung",
        "interpolator": "nearest",
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "seed": 7,
    }
    path = tmp_path / "override.json"
    path.write_text(json.dumps(override))
    changed = load_config(path)
    assert changed.target_spacing_mm == 1.5
    assert changed.content_hash() != first.content_hash()


def test_config_rejects_bad_spacing(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"target_spacing_mm": 0}))
    with pytest.raises(ConfigError, match="target_spacing_mm"):
        load_config(path)


def test_config_rejects_ratios_not_summing_to_one(tmp_path: Path) -> None:
    path = tmp_path / "ratios.yaml"
    path.write_text(yaml.safe_dump({"split": {"train": 0.6, "val": 0.2, "test": 0.1}}))
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("target_spacing_mm: 1\ninterpolator: cubic\n", "interpolator"),
        ("target_spacing_mm: 1\nwindow_preset: nope\n", "window preset"),
        ("target_spacing_mm: 1\nwindow:\n  center: 40\n", "together"),
        ("target_spacing_mm: 1\nqc:\n  hu_min: 10\n  hu_max: -10\n", "hu_min"),
        ("target_spacing_mm: 1\nqc:\n  gap_factor: 1.0\n", "gap_factor"),
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_config_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")
    path = tmp_path / "broken.yaml"
    path.write_text("target_spacing_mm: [unclosed")
    with pytest.raises(ConfigError, match="parse"):
        load_config(path)


def test_window_is_normalized_and_bounded(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "clean_axial_080"))
    windowed = apply_window(volume, preset="soft_tissue")
    assert windowed.array.dtype == np.float32
    assert windowed.array.min() >= 0.0
    assert windowed.array.max() <= 1.0
    assert windowed.shape == volume.shape

    wide = dataclasses.replace(
        volume, array=np.array([[[-1000.0, 1000.0]]], dtype=np.float32)
    )
    stretched = apply_window(wide, preset="soft_tissue")
    assert float(stretched.array.min()) == pytest.approx(0.0)
    assert float(stretched.array.max()) == pytest.approx(1.0)


def test_window_named_presets_use_expected_center_and_width(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "clean_axial_080"))
    bone = apply_window(volume, preset="bone")
    explicit = apply_window(volume, center=300.0, width=1500.0)
    assert np.allclose(bone.array, explicit.array)


def test_window_rejects_bad_arguments(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "clean_axial_080"))
    with pytest.raises(ConfigError, match="preset or both"):
        apply_window(volume)
    with pytest.raises(ConfigError, match="width"):
        apply_window(volume, center=40.0, width=0.0)
    with pytest.raises(ConfigError, match="preset"):
        apply_window(volume, preset="unknown")


def test_resample_makes_spacing_isotropic_and_preserves_extent(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "clean_axial_080"))
    resampled = resample_isotropic(volume, 1.0)
    assert resampled.spacing == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)
    assert resampled.array.dtype == volume.array.dtype
    assert resampled.shape == (30, 51, 51)
    for old_extent, new_extent, voxel in zip(
        volume.extent, resampled.extent, volume.spacing, strict=True
    ):
        assert abs(old_extent - new_extent) <= voxel + 1e-6
    assert resampled.origin == pytest.approx(volume.origin, abs=1e-6)
    assert resampled.direction == pytest.approx(volume.direction, abs=1e-6)


def test_resample_rejects_bad_spacing_and_interpolator(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "clean_axial_080"))
    with pytest.raises(ConfigError, match="greater than 0"):
        resample_isotropic(volume, 0.0)
    with pytest.raises(ConfigError, match="interpolator"):
        resample_isotropic(volume, 1.0, interpolator="cubic")


def test_mask_resampling_stays_binary_and_matches_grid(refs: list[SeriesRef]) -> None:
    ref = _ref(refs, "clean_axial_080")
    raw = load_volume(ref)
    isotropic = resample_isotropic(raw, 1.0)
    native = load_mask(ref.mask_path, raw.meta)
    aligned = resample_mask_like(native, isotropic)
    assert set(int(v) for v in np.unique(aligned.array)) <= {0, 1}
    assert aligned.shape == isotropic.shape
    assert aligned.spacing == pytest.approx(isotropic.spacing, abs=1e-6)
    assert aligned.origin == pytest.approx(isotropic.origin, abs=1e-6)
    assert aligned.direction == pytest.approx(isotropic.direction, abs=1e-6)


def test_align_mask_roundtrip_dice(refs: list[SeriesRef]) -> None:
    ref = _ref(refs, "clean_axial_080")
    raw = load_volume(ref)
    isotropic = resample_isotropic(raw, 1.0)
    aligned = resample_mask_like(load_mask(ref.mask_path, raw.meta), isotropic)
    result = align_mask(isotropic, aligned)
    assert 0.95 < result.dice <= 1.0
    assert result.roundtrip_spacing_mm == pytest.approx(1.5)
    assert result.aligned.shape == isotropic.shape


def test_align_mask_rejects_grid_mismatch(refs: list[SeriesRef]) -> None:
    ref = _ref(refs, "clean_axial_080")
    raw = load_volume(ref)
    isotropic = resample_isotropic(raw, 1.0)
    native_mask = load_mask(ref.mask_path, raw.meta)
    with pytest.raises(GeometryError, match="does not match"):
        align_mask(isotropic, native_mask)


def test_align_mask_empty_mask_is_defined(refs: list[SeriesRef]) -> None:
    ref = _ref(refs, "clean_axial_080")
    raw = load_volume(ref)
    empty = dataclasses.replace(
        load_mask(ref.mask_path, raw.meta), array=np.zeros(raw.shape, dtype=np.uint8)
    )
    assert align_mask(raw, empty).dice == 1.0


def test_preprocess_series_end_to_end(refs: list[SeriesRef], config: PreprocessConfig) -> None:
    prepared = preprocess_series(_ref(refs, "clean_axial_080"), config)
    assert prepared.image.shape == (30, 51, 51)
    assert prepared.image.spacing == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)
    assert prepared.image.array.min() >= 0.0
    assert prepared.image.array.max() <= 1.0
    assert prepared.mask is not None
    assert set(int(v) for v in np.unique(prepared.mask.array)) == {0, 1}
    assert prepared.mask.shape == prepared.image.shape
    assert prepared.mask_dice is not None
    assert prepared.mask_dice > 0.95
    assert prepared.patient_id == "MRN-PAT002"
    assert prepared.window_preset == "soft_tissue"


def test_preprocess_series_without_mask(refs: list[SeriesRef], config: PreprocessConfig) -> None:
    ref = dataclasses.replace(_ref(refs, "clean_axial_070"), mask_path=None)
    prepared = preprocess_series(ref, config)
    assert prepared.mask is None
    assert prepared.mask_dice is None


def test_preprocess_respects_overridden_window(refs: list[SeriesRef]) -> None:
    config = dataclasses.replace(load_config(), window_preset="bone", target_spacing_mm=1.5)
    prepared = preprocess_series(_ref(refs, "clean_axial_070"), config)
    assert prepared.image.spacing == pytest.approx((1.5, 1.5, 1.5), abs=1e-6)
    assert prepared.window_preset == "bone"


def test_describe_volume_is_serializable(refs: list[SeriesRef]) -> None:
    description = describe_volume(load_volume(_ref(refs, "clean_axial_080")), "before")
    assert description["shape"] == [12, 64, 64]
    assert len(description["spacing_mm"]) == 3
    assert json.dumps(description)
