"""Console output helpers for the CLI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    cells = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in cells:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header = "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers))
    rule = "  ".join("-" * width for width in widths)
    lines = [header, rule]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in cells
    )
    return lines


class Console:
    """Print to stdout and optionally tee to a run log."""

    def __init__(self, log_path: Path | None = None, quiet: bool = False) -> None:
        self._handle = None
        self._quiet = quiet
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = log_path.open("w", encoding="utf-8")

    def line(self, text: str = "") -> None:
        if not self._quiet:
            print(text)
        if self._handle is not None:
            self._handle.write(text + "\n")
            self._handle.flush()

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        for line in _table(headers, rows):
            self.line(line)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
