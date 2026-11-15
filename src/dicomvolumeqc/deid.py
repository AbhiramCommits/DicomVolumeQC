"""PHI de-identification with deterministic pseudonyms and a crosswalk."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset

from dicomvolumeqc.errors import DeidError
from dicomvolumeqc.phi_audit import (
    PHI_TAGS,
    UID_REMAP_TAGS,
    Leak,
    collect_patient_ids,
    is_dicom_candidate,
    iter_files,
    patient_pseudonym,
    scan_corpus,
    scan_for_phi,
    uid_pseudonym,
)

__all__ = [
    "DeidSummary",
    "Leak",
    "PHI_TAGS",
    "UID_REMAP_TAGS",
    "collect_patient_ids",
    "deidentify_dataset",
    "patient_pseudonym",
    "scan_corpus",
    "scan_for_phi",
    "uid_pseudonym",
]


@dataclass
class DeidSummary:
    """Counts describing one de-identification run."""

    source: str
    destination: str
    crosswalk: str
    files_processed: int
    series_count: int
    patient_count: int
    tags_blanked: int
    tags_remapped: int
    private_tags_removed: int
    masks_copied: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "destination": self.destination,
            "crosswalk": self.crosswalk,
            "files_processed": self.files_processed,
            "series_count": self.series_count,
            "patient_count": self.patient_count,
            "tags_blanked": self.tags_blanked,
            "tags_remapped": self.tags_remapped,
            "private_tags_removed": self.private_tags_removed,
            "masks_copied": self.masks_copied,
        }


def _blank_phi(ds: Dataset) -> int:
    blanked = 0
    for keyword in PHI_TAGS:
        if keyword in UID_REMAP_TAGS or keyword == "PatientID":
            continue
        if keyword not in ds:
            continue
        element = ds.data_element(keyword)
        if element is not None and element.value not in ("", None):
            setattr(ds, keyword, "")
            blanked += 1
    for element in ds.iterall():
        if element.VR in ("DA", "DT") and element.value not in ("", None):
            element.value = ""
            blanked += 1
    return blanked


def _write_crosswalk(crosswalk_path: Path, mapping: dict[tuple[str, str], str]) -> None:
    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    with crosswalk_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entity", "original", "pseudonym"])
        for (entity, original), pseudonym in sorted(mapping.items()):
            if original:
                writer.writerow([entity, original, pseudonym])


def deidentify_dataset(
    src: str | Path,
    dst: str | Path,
    salt: str,
    crosswalk_path: str | Path,
) -> DeidSummary:
    """Copy and de-identify every DICOM file, preserving pixel bytes.

    The crosswalk maps originals to pseudonyms and is written outside the
    exported dataset; placing it inside ``dst`` is rejected.
    """
    if not salt:
        raise DeidError("de-identification salt must not be empty")
    source = Path(src)
    destination = Path(dst)
    crosswalk = Path(crosswalk_path)
    if not source.is_dir():
        raise DeidError(f"source directory does not exist: {source}")
    try:
        if crosswalk.resolve().is_relative_to(destination.resolve()):
            raise DeidError("crosswalk file must not be written inside the de-identified dataset")
    except ValueError as exc:  # pragma: no cover - different drives on Windows
        raise DeidError(f"could not validate crosswalk location: {exc}") from exc
    if source.resolve() == destination.resolve():
        raise DeidError("source and destination must be different directories")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    mapping: dict[tuple[str, str], str] = {}
    series: set[str] = set()
    patients: set[str] = set()
    files_processed = 0
    blanked = 0
    remapped = 0
    private_removed = 0
    masks_copied = 0

    for path in iter_files(source):
        relative = path.relative_to(source)
        if path.name.lower().endswith(".nii.gz"):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            masks_copied += 1
            continue
        if not is_dicom_candidate(path):
            continue
        try:
            ds = pydicom.dcmread(path)
        except Exception as exc:
            raise DeidError(f"cannot de-identify unreadable file {path}: {exc}") from exc

        original_patient = str(ds.get("PatientID", "") or "")
        if original_patient:
            pseudonym = patient_pseudonym(original_patient, salt)
            mapping[("PatientID", original_patient)] = pseudonym
            ds.PatientID = pseudonym
            patients.add(pseudonym)
        for keyword in UID_REMAP_TAGS:
            if keyword not in ds:
                continue
            original = str(ds.data_element(keyword).value)
            pseudonym = uid_pseudonym(original, salt, kind=keyword)
            mapping[(keyword, original)] = pseudonym
            setattr(ds, keyword, pseudonym)
            remapped += 1

        blanked += _blank_phi(ds)
        private_count = sum(1 for element in ds.iterall() if element.tag.is_private)
        if private_count:
            ds.remove_private_tags()
            private_removed += private_count

        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "dvqc-hmac-sha256-v1; PHI/private tags removed"
        if "SOPInstanceUID" in ds and getattr(ds, "file_meta", None):
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        series.add(str(ds.SeriesInstanceUID))

        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            ds.save_as(target, enforce_file_format=True)
        except Exception as exc:
            raise DeidError(f"could not write de-identified file {target}: {exc}") from exc
        files_processed += 1

    if files_processed == 0:
        raise DeidError(f"no DICOM files found under {source}")
    _write_crosswalk(crosswalk, mapping)
    return DeidSummary(
        source=str(source),
        destination=str(destination),
        crosswalk=str(crosswalk),
        files_processed=files_processed,
        series_count=len(series),
        patient_count=len(patients),
        tags_blanked=blanked,
        tags_remapped=remapped,
        private_tags_removed=private_removed,
        masks_copied=masks_copied,
    )
