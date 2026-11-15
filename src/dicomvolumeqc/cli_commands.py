"""Implementations of the dvqc subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dicomvolumeqc.config import PreprocessConfig
from dicomvolumeqc.console import Console
from dicomvolumeqc.deid import collect_patient_ids, deidentify_dataset, scan_corpus
from dicomvolumeqc.errors import DvqcError
from dicomvolumeqc.fixtures import build_fixture_set
from dicomvolumeqc.ingest import discover_series, load_volume
from dicomvolumeqc.manifest import verify_manifest
from dicomvolumeqc.pipeline import prepare_passing, run_export, run_pipeline
from dicomvolumeqc.qc import run_qc


def _cmd_fixtures(
    args: argparse.Namespace, config: PreprocessConfig, console: Console
) -> tuple[int, dict]:
    artifacts = build_fixture_set(args.out, config.seed)
    console.table(
        ["patient", "series", "files", "defects"],
        [
            [
                item.patient_id,
                item.directory.name,
                len(item.files),
                ",".join(item.defects) or "clean",
            ]
            for item in artifacts
        ],
    )
    console.line(
        f"fixtures: {len(artifacts)} series, {len({a.patient_id for a in artifacts})} patients"
    )
    return 0, {
        "series": len(artifacts),
        "patients": len({a.patient_id for a in artifacts}),
        "root": str(args.out),
    }


def _cmd_ingest(
    args: argparse.Namespace, config: PreprocessConfig, console: Console
) -> tuple[int, dict]:
    refs = discover_series(args.input)
    rows = []
    loaded = 0
    for ref in refs:
        try:
            volume = load_volume(ref)
            rows.append(
                [
                    volume.meta.patient_id or "-",
                    ref.series_uid[-12:],
                    volume.meta.n_slices,
                    "x".join(str(v) for v in volume.meta.volume_shape),
                    "x".join(f"{v:.2f}" for v in volume.spacing),
                    "yes" if volume.meta.is_axial else "no",
                ]
            )
            loaded += 1
        except DvqcError as exc:
            rows.append([ref.patient_id or "-", ref.series_uid[-12:], 0, "-", "-", str(exc)])
    console.table(["patient", "series_uid", "slices", "shape", "spacing_mm", "axial"], rows)
    console.line(f"ingest: {loaded}/{len(refs)} series reconstructed")
    return (0 if loaded else 1), {"discovered": len(refs), "loaded": loaded}


def _cmd_qc(
    args: argparse.Namespace, config: PreprocessConfig, console: Console
) -> tuple[int, dict]:
    refs = discover_series(args.input)
    if not refs:
        console.line(f"qc: no series found in {args.input}")
        return 1, {"error": "no series"}
    report = run_qc(refs, config)
    report.write(args.reports)
    console.table(
        ["series_uid", "patient", "verdict", "fail_codes"],
        [
            [
                result.series_uid[-12:],
                result.patient_id[:16],
                result.verdict.value,
                ",".join(result.fail_codes) or "-",
            ]
            for result in report.series
        ],
    )
    console.line(f"qc: {report.counts}")
    return 0, {**report.summary(), "series": [result.to_dict() for result in report.series]}


def _cmd_deid(
    args: argparse.Namespace, config: PreprocessConfig, console: Console
) -> tuple[int, dict]:
    original_ids = collect_patient_ids(args.input)
    before = scan_corpus(args.input)
    summary = deidentify_dataset(args.input, args.out, config.deid_salt, args.crosswalk)
    after = scan_corpus(args.out, original_ids=original_ids)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary.to_dict(),
        "leaks_before": len(before),
        "leaks_after": len(after),
    }
    (reports / "deid_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    console.table(
        ["files", "series", "patients", "blanked", "remapped", "private", "masks"],
        [
            [
                summary.files_processed,
                summary.series_count,
                summary.patient_count,
                summary.tags_blanked,
                summary.tags_remapped,
                summary.private_tags_removed,
                summary.masks_copied,
            ]
        ],
    )
    console.line(f"deid: PHI leaks {len(before)} -> {len(after)}")
    if after:
        console.line(f"deid: {len(after)} leaks remain; refusing to continue")
        return 1, payload
    return 0, payload


def _cmd_prepare(
    args: argparse.Namespace, config: PreprocessConfig, console: Console
) -> tuple[int, dict]:
    refs = discover_series(args.input)
    report = run_qc(refs, config) if refs else None
    prepared = prepare_passing(refs, report, config) if report else []
    console.table(
        ["series_uid", "source_shape", "prepared_shape", "spacing_mm", "mask_dice"],
        [
            [
                item.series_uid[-12:],
                "x".join(str(v) for v in item.source_shape),
                "x".join(str(v) for v in item.image.shape),
                "x".join(f"{v:.2f}" for v in item.image.spacing),
                "-" if item.mask_dice is None else f"{item.mask_dice:.4f}",
            ]
            for item in prepared
        ],
    )
    console.line(f"prepare: {len(prepared)} series prepared")
    return (0 if prepared else 1), {
        "prepared": len(prepared),
        "examples": [
            {
                "source_shape": list(item.source_shape),
                "prepared_shape": list(item.image.shape),
                "mask_dice": item.mask_dice,
            }
            for item in prepared
        ],
    }


def _cmd_export(
    args: argparse.Namespace, config: PreprocessConfig, console: Console
) -> tuple[int, dict]:
    result = run_export(config, args.input, args.reports, args.out, log=console.line)
    console.table(
        ["records", "files", "digest_ok", "verified"],
        [
            [
                result["export"]["records"],
                result["export"]["files"],
                result["verification"]["digest_ok"],
                result["verification"]["ok"],
            ]
        ],
    )
    return (0 if result["verification"]["ok"] else 1), result


def _cmd_verify(
    args: argparse.Namespace, config: PreprocessConfig, console: Console
) -> tuple[int, dict]:
    verification = verify_manifest(args.export)
    if verification.ok:
        console.line(
            f"verify: OK ({verification.files_checked} files, digest_ok={verification.digest_ok})"
        )
    else:
        for problem in verification.problems:
            console.line(f"verify: {problem}")
    return (0 if verification.ok else 1), verification.to_dict()


def _print_run_all(console: Console, result: dict[str, Any]) -> None:
    console.line("")
    fixtures = result["fixtures"]
    deid = result["deid"]
    console.table(
        ["stage", "result"],
        [
            [
                "fixtures",
                f"{fixtures['series']} series / {fixtures['patients']} patients",
            ],
            ["deid", f"{deid['leaks_before']} -> {deid['leaks_after']} leaks"],
            ["qc", json.dumps(result["qc"]["counts"])],
            ["prepared", str(result["prepared_series"])],
            [
                "export",
                f"{result['export']['records']} records / {result['export']['files']} files",
            ],
            [
                "verify",
                f"ok={result['verification']['ok']} "
                f"digest_ok={result['verification']['digest_ok']}",
            ],
            ["runtime", f"{result['runtime_seconds']:.2f} s"],
        ],
    )
    console.line("")
    console.table(
        ["split", "patients", "series"],
        [
            [split, counts["patients"], counts["series"]]
            for split, counts in result["split"]["counts"].items()
        ],
    )
    console.line("")
    console.table(
        ["rule", "quarantined series"],
        [[code, count] for code, count in result["qc"]["failures_by_rule"].items()],
    )


def dispatch(
    args: argparse.Namespace, config: PreprocessConfig, console: Console
) -> tuple[int, dict[str, Any]]:
    if args.command == "fixtures":
        return _cmd_fixtures(args, config, console)
    if args.command == "ingest":
        return _cmd_ingest(args, config, console)
    if args.command == "qc":
        return _cmd_qc(args, config, console)
    if args.command == "deid":
        return _cmd_deid(args, config, console)
    if args.command == "prepare":
        return _cmd_prepare(args, config, console)
    if args.command == "export":
        return _cmd_export(args, config, console)
    if args.command == "verify":
        return _cmd_verify(args, config, console)
    result = run_pipeline(config, args.out, args.reports, args.export, log=console.line)
    _print_run_all(console, result)
    return 0, result
