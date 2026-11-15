"""Automated QC execution and report generation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dicomvolumeqc.config import PreprocessConfig
from dicomvolumeqc.errors import GeometryError
from dicomvolumeqc.ingest import load_volume
from dicomvolumeqc.models import Finding, SeriesMeta, SeriesRef, Severity
from dicomvolumeqc.preprocess import grids_match, load_mask
from dicomvolumeqc.qc_rules import Rule, RuleRegistry, Verdict

__all__ = ["QCReport", "Rule", "RuleRegistry", "SeriesQC", "Verdict", "run_qc"]

@dataclass
class SeriesQC:
    """QC result for one series."""

    series_uid: str
    patient_id: str
    directory: str
    verdict: Verdict
    findings: list[Finding]
    meta: SeriesMeta | None = None
    error: str | None = None

    @property
    def codes(self) -> list[str]:
        return sorted({finding.code for finding in self.findings})

    @property
    def fail_codes(self) -> list[str]:
        return sorted(
            {finding.code for finding in self.findings if finding.severity == Severity.FAIL}
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "series_uid": self.series_uid,
            "patient_id": self.patient_id,
            "directory": self.directory,
            "verdict": self.verdict.value,
            "codes": self.codes,
            "findings": [finding.to_dict() for finding in self.findings],
            "error": self.error,
        }
        if self.meta is not None:
            payload.update(
                {
                    "n_slices": self.meta.n_slices,
                    "volume_shape": list(self.meta.volume_shape),
                    "slice_spacing_median_mm": round(self.meta.slice_spacing_median, 6),
                }
            )
        return payload


@dataclass
class QCReport:
    """Aggregate QC result across all series."""

    series: list[SeriesQC]
    config_hash: str
    rule_catalog: tuple[dict[str, str], ...]
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def counts(self) -> dict[str, int]:
        counter = Counter(result.verdict.value.lower() for result in self.series)
        return {
            "total": len(self.series),
            "pass": counter.get("pass", 0),
            "warn": counter.get("warn", 0),
            "quarantine": counter.get("quarantine", 0),
        }

    @property
    def failures_by_rule(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for result in self.series:
            for code in result.fail_codes:
                counter[code] += 1
        return dict(sorted(counter.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "failures_by_rule": self.failures_by_rule,
            "generated_at": self.generated_at,
            "config_hash": self.config_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "generated_at": self.generated_at,
            "config_hash": self.config_hash,
            "counts": self.counts,
            "failures_by_rule": self.failures_by_rule,
            "rules": list(self.rule_catalog),
            "series": [result.to_dict() for result in self.series],
        }

    def write(self, reports_dir: str | Path) -> tuple[Path, Path]:
        directory = Path(reports_dir)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "qc_report.json"
        csv_path = directory / "qc_report.csv"
        json_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        rows = [
            {
                "series_uid": result.series_uid,
                "patient_id": result.patient_id,
                "directory": result.directory,
                "verdict": result.verdict.value,
                "n_slices": result.meta.n_slices if result.meta else 0,
                "slice_spacing_median_mm": (
                    round(result.meta.slice_spacing_median, 6) if result.meta else None
                ),
                "finding_codes": ";".join(result.codes),
                "fail_codes": ";".join(result.fail_codes),
                "error": result.error or "",
            }
            for result in self.series
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path, json_path


def run_qc(series_refs: list[SeriesRef], config: PreprocessConfig) -> QCReport:
    """Load each series, enrich mask facts, run every rule, and aggregate."""
    registry = RuleRegistry(config)
    results: list[SeriesQC] = []
    for ref in sorted(series_refs, key=lambda r: (r.patient_id, r.series_uid, str(r.directory))):
        try:
            volume = load_volume(ref)
        except GeometryError as exc:
            results.append(
                SeriesQC(
                    series_uid=ref.series_uid,
                    patient_id=ref.patient_id,
                    directory=str(ref.directory),
                    verdict=Verdict.QUARANTINE,
                    findings=[
                        Finding(
                            code="LOAD_FAILED",
                            severity=Severity.FAIL,
                            message=f"volume reconstruction failed: {exc}",
                            observed=False,
                        )
                    ],
                    error=str(exc),
                )
            )
            continue

        meta = volume.meta
        extra: list[Finding] = []
        if ref.mask_path is not None:
            try:
                mask = load_mask(ref.mask_path, meta)
                labels = {int(v) for v in np.unique(mask.array)}
                if not labels <= {0, 1}:
                    extra.append(
                        Finding(
                            code="MASK_NOT_BINARY",
                            severity=Severity.WARN,
                            message=f"mask contains non-binary labels {sorted(labels)}",
                            observed=str(sorted(labels)),
                        )
                    )
                meta = replace(
                    meta,
                    mask_path=str(ref.mask_path),
                    mask_label_count=int(np.count_nonzero(mask.array)),
                    mask_grid_matches=grids_match(volume, mask),
                )
            except GeometryError as exc:
                extra.append(
                    Finding(
                        code="MASK_UNREADABLE",
                        severity=Severity.FAIL,
                        message=f"mask could not be read: {exc}",
                        observed=False,
                    )
                )
        else:
            meta = replace(meta, mask_path=None, mask_grid_matches=None, mask_label_count=None)

        findings = registry.run(meta, volume) + extra
        findings = sorted(
            {(f.code, f.message): f for f in findings}.values(), key=lambda f: (f.code, f.message)
        )
        if any(f.severity == Severity.FAIL for f in findings):
            verdict = Verdict.QUARANTINE
        elif any(f.severity == Severity.WARN for f in findings):
            verdict = Verdict.WARN
        else:
            verdict = Verdict.PASS
        results.append(
            SeriesQC(
                series_uid=meta.series_uid,
                patient_id=meta.patient_id,
                directory=str(ref.directory),
                verdict=verdict,
                findings=findings,
                meta=meta,
            )
        )
    return QCReport(
        series=results,
        config_hash=config.content_hash(),
        rule_catalog=registry.catalog(),
    )
