"""Dataset export with a SHA-256 checksummed, signed manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dicomvolumeqc.config import PreprocessConfig
from dicomvolumeqc.errors import DvqcError, QCFailure
from dicomvolumeqc.preprocess import PreparedVolume
from dicomvolumeqc.qc import QCReport, Verdict
from dicomvolumeqc.splits import SPLITS, SplitResult

SCHEMA_VERSION = "1.0"
MANIFEST_JSON = "manifest.json"
MANIFEST_CSV = "manifest.csv"


@dataclass(frozen=True)
class ExportResult:
    """Outcome of writing an export and its manifest."""

    export_dir: str
    manifest_json: str
    manifest_csv: str
    records: int
    files: int
    manifest_digest: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestVerification:
    """Result of recomputing every checksum in an export."""

    ok: bool
    files_checked: int
    digest_ok: bool
    problems: tuple[str, ...]
    manifest_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "files_checked": self.files_checked,
            "digest_ok": self.digest_ok,
            "problems": list(self.problems),
            "manifest_path": self.manifest_path,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _safe_component(value: str, label: str) -> str:
    if not value or any(char in value for char in ("/", "\\", "..")) or value in {".", ".."}:
        raise DvqcError(f"unsafe {label} for export path: {value!r}")
    return value


def export_dataset(
    prepared: list[PreparedVolume],
    split: SplitResult,
    qc_report: QCReport,
    config: PreprocessConfig,
    export_dir: str | Path,
) -> ExportResult:
    """Write windowed volumes/masks and a manifest with per-file checksums."""
    destination = Path(export_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    verdicts = {result.series_uid: result for result in qc_report.series}
    records: list[dict[str, Any]] = []
    files = 0
    for item in sorted(prepared, key=lambda p: (p.meta.patient_id, p.series_uid)):
        qc_result = verdicts.get(item.series_uid)
        if qc_result is None:
            raise DvqcError(f"no QC result for series {item.series_uid}")
        if qc_result.verdict == Verdict.QUARANTINE:
            raise QCFailure(f"refusing to export quarantined series {item.series_uid}")
        split_name = split.assignments.get(item.series_uid)
        if split_name not in SPLITS:
            raise DvqcError(f"series {item.series_uid} has no split assignment")

        patient = _safe_component(item.meta.patient_id, "patient pseudonym")
        series = _safe_component(item.series_uid, "series pseudonym")
        relative_dir = Path(split_name) / patient / series
        volume_path = relative_dir / "volume.npy"
        absolute_volume = destination / volume_path
        absolute_volume.parent.mkdir(parents=True, exist_ok=True)
        np.save(absolute_volume, item.image.array, allow_pickle=False)
        files += 1

        mask_relative: str | None = None
        mask_sha: str | None = None
        if item.mask is not None:
            mask_path = relative_dir / "mask.npy"
            absolute_mask = destination / mask_path
            np.save(absolute_mask, item.mask.array, allow_pickle=False)
            mask_relative = str(mask_path)
            mask_sha = sha256_file(absolute_mask)
            files += 1

        records.append(
            {
                "split": split_name,
                "patient_pseudonym": patient,
                "series_pseudonym": series,
                "volume_path": str(volume_path),
                "mask_path": mask_relative,
                "voxel_shape": [int(v) for v in item.image.shape],
                "spacing_mm": [round(float(v), 9) for v in item.image.spacing],
                "source_shape": [int(v) for v in item.source_shape],
                "source_spacing_mm": [round(float(v), 9) for v in item.source_spacing],
                "window_preset": item.window_preset,
                "qc_verdict": qc_result.verdict.value,
                "qc_findings": [finding.to_dict() for finding in qc_result.findings],
                "mask_dice": None if item.mask_dice is None else round(float(item.mask_dice), 9),
                "sha256": {
                    "volume": sha256_file(absolute_volume),
                    "mask": mask_sha,
                },
            }
        )

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    counts = {
        "series": len(records),
        "files": files,
        "splits": split.counts(),
    }
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_commit": _git_commit(),
        "config_hash": config.content_hash(),
        "configuration": asdict(config),
        "split_seed": config.seed,
        "counts": counts,
        "records": records,
    }
    manifest_digest = _canonical_digest(body)
    payload = {**body, "manifest_digest": manifest_digest}

    manifest_json = destination / MANIFEST_JSON
    manifest_json.write_text(json.dumps(payload, indent=2) + "\n")
    rows = []
    for record in records:
        rows.append(
            {
                "split": record["split"],
                "patient_pseudonym": record["patient_pseudonym"],
                "series_pseudonym": record["series_pseudonym"],
                "volume_path": record["volume_path"],
                "mask_path": record["mask_path"],
                "voxel_shape": "x".join(str(v) for v in record["voxel_shape"]),
                "spacing_mm": "x".join(str(v) for v in record["spacing_mm"]),
                "window_preset": record["window_preset"],
                "qc_verdict": record["qc_verdict"],
                "mask_dice": record["mask_dice"],
                "volume_sha256": record["sha256"]["volume"],
                "mask_sha256": record["sha256"]["mask"],
            }
        )
    manifest_csv = destination / MANIFEST_CSV
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)

    return ExportResult(
        export_dir=str(destination),
        manifest_json=str(manifest_json),
        manifest_csv=str(manifest_csv),
        records=len(records),
        files=files,
        manifest_digest=manifest_digest,
    )


def verify_manifest(export_dir: str | Path) -> ManifestVerification:
    """Recompute all file checksums and the top-level manifest digest."""
    root = Path(export_dir)
    manifest_path = root / MANIFEST_JSON
    if not manifest_path.is_file():
        raise DvqcError(f"manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise DvqcError(f"could not read manifest {manifest_path}: {exc}") from exc
    records = payload.get("records")
    if not isinstance(records, list):
        raise DvqcError("manifest has no records list")

    problems: list[str] = []
    files_checked = 0
    for record in records:
        for key in ("volume", "mask"):
            reference = record.get("sha256", {}).get(key)
            relative = record.get("volume_path") if key == "volume" else record.get("mask_path")
            if reference is None and relative is None:
                continue
            files_checked += 1
            if relative is None:
                problems.append(
                    f"missing path for {key} in series {record.get('series_pseudonym')}"
                )
                continue
            path = root / relative
            if not path.is_file():
                problems.append(f"missing file: {relative}")
                continue
            actual = sha256_file(path)
            if actual != reference:
                problems.append(f"checksum mismatch: {relative}")

    digest_ok = True
    stored_digest = payload.get("manifest_digest")
    if not isinstance(stored_digest, str):
        digest_ok = False
        problems.append("manifest digest is missing")
    else:
        body = {key: value for key, value in payload.items() if key != "manifest_digest"}
        if _canonical_digest(body) != stored_digest:
            digest_ok = False
            problems.append("manifest digest mismatch")

    return ManifestVerification(
        ok=not problems and digest_ok,
        files_checked=files_checked,
        digest_ok=digest_ok,
        problems=tuple(problems),
        manifest_path=str(manifest_path),
    )
