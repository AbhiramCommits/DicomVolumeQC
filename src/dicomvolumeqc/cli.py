"""Command-line interface for DicomVolumeQC."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dicomvolumeqc.cli_commands import dispatch
from dicomvolumeqc.config import load_config
from dicomvolumeqc.console import Console
from dicomvolumeqc.errors import DvqcError
from dicomvolumeqc.pipeline import clone_config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="YAML/JSON pipeline configuration")
    parser.add_argument("--seed", type=int, default=None, help="override the configured seed")
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")


def _add_out(parser: argparse.ArgumentParser, default: str, help_text: str) -> None:
    parser.add_argument("--out", default=default, help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dvqc", description="DICOM -> QC -> de-identified, ML-ready 3D dataset"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fixtures = sub.add_parser("fixtures", help="generate the synthetic DICOM corpus")
    _add_common(fixtures)
    _add_out(fixtures, "data/fixtures", "fixture root directory")

    ingest = sub.add_parser("ingest", help="discover and reconstruct series")
    _add_common(ingest)
    ingest.add_argument("--input", default="data/fixtures/raw")

    qc = sub.add_parser("qc", help="run QC rules and write reports")
    _add_common(qc)
    qc.add_argument("--input", default="data/fixtures/deid")
    qc.add_argument("--reports", default="reports")

    deid = sub.add_parser("deid", help="de-identify a DICOM tree and audit leaks")
    _add_common(deid)
    deid.add_argument("--input", default="data/fixtures/raw")
    _add_out(deid, "data/deid", "de-identified output directory")
    deid.add_argument("--crosswalk", default="reports/deid_crosswalk.csv")
    deid.add_argument("--reports", default="reports")

    prepare = sub.add_parser("prepare", help="window and resample passing series")
    _add_common(prepare)
    prepare.add_argument("--input", default="data/fixtures/deid")

    export = sub.add_parser("export", help="split, export volumes, and write the manifest")
    _add_common(export)
    export.add_argument("--input", default="data/fixtures/deid")
    _add_out(export, "export", "export directory")
    export.add_argument("--reports", default="reports")

    verify = sub.add_parser("verify", help="recompute manifest checksums")
    _add_common(verify)
    verify.add_argument("--export", default="export")

    run_all = sub.add_parser(
        "run-all", help="fixtures -> deid -> ingest -> qc -> prepare -> export"
    )
    _add_common(run_all)
    _add_out(run_all, "data/fixtures", "fixture root directory")
    run_all.add_argument("--reports", default="reports")
    run_all.add_argument("--export", default="export")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = None
    if args.command == "run-all":
        log_path = Path(args.reports) / "run_all.txt"
    console = Console(log_path, quiet=args.json)
    try:
        config = clone_config(load_config(args.config), args.seed)
        code, payload = dispatch(args, config, console)
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        return code
    except DvqcError as exc:
        console.line(f"error: {exc}")
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        console.close()


__all__ = ["Console", "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
