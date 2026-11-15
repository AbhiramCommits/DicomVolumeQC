"""End-to-end pipeline orchestration shared by the CLI and tests."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dicomvolumeqc.config import PreprocessConfig
from dicomvolumeqc.deid import (
    collect_patient_ids,
    deidentify_dataset,
    scan_corpus,
)
from dicomvolumeqc.errors import DeidError, DvqcError
from dicomvolumeqc.fixtures import build_fixture_set
from dicomvolumeqc.ingest import discover_series
from dicomvolumeqc.manifest import export_dataset, verify_manifest
from dicomvolumeqc.models import SeriesRef
from dicomvolumeqc.preprocess import PreparedVolume, preprocess_series
from dicomvolumeqc.qc import QCReport, Verdict, run_qc
from dicomvolumeqc.splits import patient_disjoint_split

Log = Callable[[str], None]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def prepare_passing(
    refs: list[SeriesRef], report: QCReport, config: PreprocessConfig
) -> list[PreparedVolume]:
    """Preprocess only series that were not quarantined."""
    by_uid = {ref.series_uid: ref for ref in refs}
    prepared: list[PreparedVolume] = []
    for result in report.series:
        if result.verdict == Verdict.QUARANTINE:
            continue
        ref = by_uid.get(result.series_uid)
        if ref is None:
            raise DvqcError(f"QC result has no matching series reference: {result.series_uid}")
        prepared.append(preprocess_series(ref, config))
    return prepared


def run_export(
    config: PreprocessConfig,
    input_dir: str | Path,
    reports_dir: str | Path,
    export_dir: str | Path,
    log: Log = lambda _message: None,
) -> dict[str, Any]:
    """Ingest, QC, prepare, split, export, and verify one de-identified tree."""
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    stage_seconds: dict[str, float] = {}

    started = time.perf_counter()
    refs = discover_series(input_dir)
    stage_seconds["ingest"] = time.perf_counter() - started
    log(f"ingest: discovered {len(refs)} series in {input_dir}")
    if not refs:
        raise DvqcError(f"no DICOM series found in {input_dir}")

    started = time.perf_counter()
    report = run_qc(refs, config)
    report.write(reports)
    stage_seconds["qc"] = time.perf_counter() - started
    log(f"qc: {report.counts}")

    started = time.perf_counter()
    prepared = prepare_passing(refs, report, config)
    stage_seconds["prepare"] = time.perf_counter() - started
    log(f"prepare: {len(prepared)} series resampled to {config.target_spacing_mm} mm")

    started = time.perf_counter()
    split = patient_disjoint_split(
        [item.meta for item in prepared], config.split_ratio_map, config.seed
    )
    stage_seconds["split"] = time.perf_counter() - started
    log(f"split: {split.counts()}")

    started = time.perf_counter()
    export = export_dataset(prepared, split, report, config, export_dir)
    stage_seconds["export"] = time.perf_counter() - started
    log(f"export: {export.records} records, {export.files} files")

    started = time.perf_counter()
    verification = verify_manifest(export_dir)
    stage_seconds["verify"] = time.perf_counter() - started
    log(
        f"verify: ok={verification.ok} files={verification.files_checked} "
        f"digest_ok={verification.digest_ok}"
    )
    if not verification.ok:
        raise DvqcError("manifest verification failed: " + "; ".join(verification.problems))

    geometry_examples = [
        {
            "series": item.series_uid,
            "source_shape": list(item.source_shape),
            "source_spacing_mm": [round(float(v), 6) for v in item.source_spacing],
            "prepared_shape": list(item.image.shape),
            "prepared_spacing_mm": [round(float(v), 6) for v in item.image.spacing],
            "mask_dice": None if item.mask_dice is None else round(float(item.mask_dice), 6),
        }
        for item in prepared[:3]
    ]

    return {
        "input_dir": str(input_dir),
        "series_discovered": len(refs),
        "qc": {
            "counts": report.counts,
            "failures_by_rule": report.failures_by_rule,
            "report_csv": str(reports / "qc_report.csv"),
            "report_json": str(reports / "qc_report.json"),
        },
        "prepared_series": len(prepared),
        "geometry_examples": geometry_examples,
        "split": {
            "seed": config.seed,
            "ratios": config.split_ratio_map,
            **split.summary(),
        },
        "export": export.to_dict(),
        "verification": verification.to_dict(),
        "stage_seconds": stage_seconds,
    }


def run_pipeline(
    config: PreprocessConfig,
    fixture_root: str | Path,
    reports_dir: str | Path,
    export_dir: str | Path,
    log: Log = lambda _message: None,
) -> dict[str, Any]:
    """Fixtures -> de-identification -> export/verify, returning measured results."""
    started = time.perf_counter()
    fixture_root = Path(fixture_root)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    stage = time.perf_counter()
    artifacts = build_fixture_set(fixture_root, config.seed)
    timings["fixtures"] = time.perf_counter() - stage
    log(
        f"fixtures: {len(artifacts)} series, "
        f"{len({a.patient_id for a in artifacts})} patients at {fixture_root}"
    )
    if not artifacts:
        raise DvqcError("fixture generation produced no series")

    raw_dir = fixture_root / "raw"
    deid_dir = fixture_root / "deid"
    crosswalk = reports / "deid_crosswalk.csv"
    stage = time.perf_counter()
    original_ids = collect_patient_ids(raw_dir)
    leaks_before = scan_corpus(raw_dir)
    deid_summary = deidentify_dataset(raw_dir, deid_dir, config.deid_salt, crosswalk)
    leaks_after = scan_corpus(deid_dir, original_ids=original_ids)
    timings["deid"] = time.perf_counter() - stage
    log(
        f"deid: {deid_summary.files_processed} files, "
        f"leaks {len(leaks_before)} -> {len(leaks_after)}"
    )
    if leaks_after:
        raise DeidError(f"{len(leaks_after)} PHI leaks survived de-identification")

    deid_report = {
        "summary": deid_summary.to_dict(),
        "original_patient_count": len(original_ids),
        "leaks_before": len(leaks_before),
        "leaks_after": len(leaks_after),
        "leak_reasons_before": dict(sorted(Counter(leak.reason for leak in leaks_before).items())),
    }
    _write_json(reports / "deid_report.json", deid_report)

    exported = run_export(config, deid_dir, reports, export_dir, log=log)
    timings.update(exported.pop("stage_seconds"))
    timings["total"] = time.perf_counter() - started

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fixtures": {
            "root": str(fixture_root),
            "series": len(artifacts),
            "patients": len({a.patient_id for a in artifacts}),
        },
        "deid": {**deid_report, "destination": str(deid_dir), "crosswalk": str(crosswalk)},
        **exported,
        "timings_seconds": {key: round(value, 6) for key, value in timings.items()},
        "runtime_seconds": round(timings["total"], 6),
    }
    _write_json(reports / "pipeline_summary.json", result)
    return result


def clone_config(config: PreprocessConfig, seed: int | None) -> PreprocessConfig:
    """Apply a CLI seed override without mutating the loaded config."""
    return config if seed is None else replace(config, seed=seed)
