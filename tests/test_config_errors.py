"""Missing-branch tests for configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dicomvolumeqc.config import load_config
from dicomvolumeqc.errors import ConfigError


def _load(tmp_path: Path, payload: object):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    return load_config(path)


def test_window_section_must_be_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="section 'window'"):
        _load(tmp_path, {"window": "wide"})


def test_ratios_accept_a_three_item_sequence(tmp_path: Path) -> None:
    config = _load(tmp_path, {"split": [0.5, 0.25, 0.25]})
    assert config.split_ratios == (0.5, 0.25, 0.25)


def test_ratios_reject_wrong_type(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping or a 3-item sequence"):
        _load(tmp_path, {"split": "most"})


def test_seed_must_be_integer(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="seed"):
        _load(tmp_path, {"seed": True})


def test_target_spacing_upper_bound(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="above 10 mm"):
        _load(tmp_path, {"target_spacing_mm": 11})


def test_axial_tolerance_bounds(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="axial_tolerance_deg"):
        _load(tmp_path, {"qc": {"axial_tolerance_deg": 90}})


def test_empty_salt_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="salt"):
        _load(tmp_path, {"deid": {"salt": ""}})


def test_empty_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_non_numeric_target_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="numeric"):
        _load(tmp_path, {"target_spacing_mm": "wide"})


def test_explicit_window_pair_is_accepted(tmp_path: Path) -> None:
    config = _load(tmp_path, {"window": {"center": 40, "width": 400}})
    assert config.window_center == 40.0
    assert config.window_width == 400.0


def test_min_slices_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="min_slices"):
        _load(tmp_path, {"qc": {"min_slices": 0}})


def test_zero_width_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="width"):
        _load(tmp_path, {"window": {"center": 40, "width": 0}})
