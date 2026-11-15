#!/usr/bin/env python
"""Build reports/metrics.json from measured pipeline, coverage, and manifest artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text())


def _test_count(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"(\d+) tests? collected", completed.stdout)
    if not match:
        raise SystemExit("could not parse pytest test count")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--export", default="export")
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument(
        "--wall-clock",
        type=float,
        default=None,
        help="measured end-to-end seconds; defaults to reports/final_wall_clock.txt",
    )
    args = parser.parse_args()

    reports = Path(args.reports)
    export = Path(args.export)
    summary = _read_json(reports / "pipeline_summary.json")
    manifest = _read_json(export / "manifest.json")
    coverage = _read_json(reports / "coverage.json")
    count = _test_count(args.test_count)

    if args.wall_clock is not None:
        wall_clock = args.wall_clock
    elif (reports / "final_wall_clock.txt").is_file():
        wall_clock = float((reports / "final_wall_clock.txt").read_text().strip())
    else:
        wall_clock = float(summary["runtime_seconds"])

    dice = [record["mask_dice"] for record in manifest["records"] if record["mask_dice"]]
    metrics = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "measured_code_commit": manifest.get("git_commit"),
        "corpus": {
            "series": summary["fixtures"]["series"],
            "patients": summary["fixtures"]["patients"],
            "series_ingested": summary["series_discovered"],
        },
        "qc": {
            "pass": summary["qc"]["counts"]["pass"],
            "warn": summary["qc"]["counts"]["warn"],
            "quarantine": summary["qc"]["counts"]["quarantine"],
            "series_passing_qc": summary["prepared_series"],
            "quarantined_by_rule": summary["qc"]["failures_by_rule"],
        },
        "deidentification": {
            "files_processed": summary["deid"]["summary"]["files_processed"],
            "tags_blanked": summary["deid"]["summary"]["tags_blanked"],
            "tags_remapped": summary["deid"]["summary"]["tags_remapped"],
            "private_tags_removed": summary["deid"]["summary"]["private_tags_removed"],
            "leaks_before": summary["deid"]["leaks_before"],
            "leaks_after": summary["deid"]["leaks_after"],
        },
        "geometry": {
            "target_spacing_mm": manifest["configuration"]["target_spacing_mm"],
            "window_preset": manifest["configuration"]["window_preset"],
            "examples": summary["geometry_examples"],
            "mask_roundtrip_dice": {
                "min": round(min(dice), 6),
                "median": round(statistics.median(dice), 6),
                "max": round(max(dice), 6),
            },
        },
        "splits": summary["split"]["counts"],
        "export": {
            "records": manifest["counts"]["series"],
            "files": manifest["counts"]["files"],
            "manifest_verified": summary["verification"]["ok"],
            "digest_ok": summary["verification"]["digest_ok"],
            "manifest_digest": manifest["manifest_digest"],
            "config_hash": manifest["config_hash"],
        },
        "tests": {
            "count": count,
            "coverage_percent": round(float(coverage["totals"]["percent_covered"]), 4),
            "coverage_fail_under": 85,
        },
        "runtime": {
            "pipeline_internal_seconds": summary["runtime_seconds"],
            "end_to_end_wall_clock_seconds": round(wall_clock, 3),
        },
    }
    output = reports / "metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {output}")
    print(
        f"series={metrics['corpus']['series']} pass={metrics['qc']['pass']} "
        f"quarantine={metrics['qc']['quarantine']} tests={count} "
        f"coverage={metrics['tests']['coverage_percent']}% "
        f"runtime={metrics['runtime']['end_to_end_wall_clock_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
