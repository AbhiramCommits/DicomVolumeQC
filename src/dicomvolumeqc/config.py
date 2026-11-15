"""Pipeline configuration loaded from YAML or JSON."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from dicomvolumeqc.errors import ConfigError

WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    "soft_tissue": (40.0, 400.0),
    "lung": (-600.0, 1500.0),
    "bone": (300.0, 1500.0),
}

INTERPOLATORS = ("linear", "nearest", "bspline")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class PreprocessConfig:
    """Validated, immutable pipeline configuration."""

    target_spacing_mm: float = 1.0
    window_preset: str = "soft_tissue"
    window_center: float | None = None
    window_width: float | None = None
    interpolator: str = "linear"
    min_slices: int = 3
    spacing_cv_tolerance: float = 0.05
    gap_factor: float = 1.5
    axial_tolerance_deg: float = 5.0
    hu_min: float = -1024.0
    hu_max: float = 3071.0
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    seed: int = 20240101
    deid_salt: str = "dvqc-local-development-salt"

    @property
    def split_ratio_map(self) -> dict[str, float]:
        train, val, test = self.split_ratios
        return {"train": train, "val": val, "test": test}

    def content_hash(self) -> str:
        """Stable SHA-256 of all configuration values, used in the manifest."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{key}' must be a mapping")
    return value


def _float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config value '{name}' must be numeric") from exc


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"config value '{name}' must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config value '{name}' must be an integer") from exc


def _ratios(raw: dict[str, Any]) -> tuple[float, float, float]:
    split = raw.get("split", raw.get("split_ratios", {}))
    if isinstance(split, dict):
        values = (
            _float(split.get("train", 0.6), "split.train"),
            _float(split.get("val", 0.2), "split.val"),
            _float(split.get("test", 0.2), "split.test"),
        )
    elif isinstance(split, (list, tuple)) and len(split) == 3:
        values = tuple(_float(v, f"split[{i}]") for i, v in enumerate(split))  # type: ignore[assignment]
    else:
        raise ConfigError("split ratios must be a mapping or a 3-item sequence")
    if any(v < 0 for v in values):
        raise ConfigError("split ratios must be non-negative")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ConfigError(f"split ratios must sum to 1.0, got {sum(values):.6f}")
    return values  # type: ignore[return-value]


def _build(raw: dict[str, Any]) -> PreprocessConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    window = _section(raw, "window")
    qc = _section(raw, "qc")
    deid = _section(raw, "deid")

    center_raw = raw.get("window_center", window.get("center"))
    width_raw = raw.get("window_width", window.get("width"))
    if (center_raw is None) != (width_raw is None):
        raise ConfigError("window center and width must be provided together")
    center = None if center_raw is None else _float(center_raw, "window.center")
    width = None if width_raw is None else _float(width_raw, "window.width")

    config = PreprocessConfig(
        target_spacing_mm=_float(raw.get("target_spacing_mm", 1.0), "target_spacing_mm"),
        window_preset=str(raw.get("window_preset", window.get("preset", "soft_tissue"))),
        window_center=center,
        window_width=width,
        interpolator=str(raw.get("interpolator", "linear")).lower(),
        min_slices=_int(qc.get("min_slices", 3), "qc.min_slices"),
        spacing_cv_tolerance=_float(
            qc.get("spacing_cv_tolerance", 0.05), "qc.spacing_cv_tolerance"
        ),
        gap_factor=_float(qc.get("gap_factor", 1.5), "qc.gap_factor"),
        axial_tolerance_deg=_float(
            qc.get("axial_tolerance_deg", 5.0), "qc.axial_tolerance_deg"
        ),
        hu_min=_float(qc.get("hu_min", -1024.0), "qc.hu_min"),
        hu_max=_float(qc.get("hu_max", 3071.0), "qc.hu_max"),
        split_ratios=_ratios(raw),
        seed=_int(raw.get("seed", 20240101), "seed"),
        deid_salt=str(deid.get("salt", "dvqc-local-development-salt")),
    )
    validate_config(config)
    return config


def validate_config(config: PreprocessConfig) -> None:
    """Raise ConfigError when a loaded config is outside safe ranges."""
    if config.target_spacing_mm <= 0:
        raise ConfigError("target_spacing_mm must be greater than 0")
    if config.target_spacing_mm > 10:
        raise ConfigError("target_spacing_mm above 10 mm is not supported")
    if config.window_preset not in WINDOW_PRESETS:
        raise ConfigError(
            f"unknown window preset '{config.window_preset}'; "
            f"choose from {sorted(WINDOW_PRESETS)}"
        )
    if config.window_width is not None and config.window_width <= 0:
        raise ConfigError("window.width must be greater than 0")
    if config.interpolator not in INTERPOLATORS:
        raise ConfigError(f"unknown interpolator '{config.interpolator}'")
    if config.min_slices < 1:
        raise ConfigError("qc.min_slices must be at least 1")
    if config.spacing_cv_tolerance <= 0 or config.gap_factor <= 1:
        raise ConfigError("QC spacing tolerances must be positive (gap_factor > 1)")
    if config.axial_tolerance_deg <= 0 or config.axial_tolerance_deg >= 90:
        raise ConfigError("qc.axial_tolerance_deg must be between 0 and 90")
    if config.hu_min >= config.hu_max:
        raise ConfigError("qc.hu_min must be less than qc.hu_max")
    if not config.deid_salt:
        raise ConfigError("deid.salt must not be empty")
    if abs(sum(config.split_ratios) - 1.0) > 1e-6:
        raise ConfigError("split ratios must sum to 1.0")


def load_config(path: str | Path | None = None) -> PreprocessConfig:
    """Load YAML/JSON configuration, raising ConfigError on any problem."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        text = config_path.read_text()
        if config_path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raw = yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not parse config {config_path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"config file is empty: {config_path}")
    return _build(raw)
