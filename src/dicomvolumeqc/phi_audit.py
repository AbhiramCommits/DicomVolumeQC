"""Independent PHI leak auditor and deterministic pseudonym primitives."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pydicom

UID_REMAP_TAGS: tuple[str, ...] = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
)

PHI_TAGS: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "IssuerOfPatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientMotherBirthName",
    "MedicalRecordLocator",
    "PatientComments",
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "StudyTime",
    "SeriesDate",
    "SeriesTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "ContentDate",
    "ContentTime",
    "InstanceCreationDate",
    "InstanceCreationTime",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "DeviceSerialNumber",
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
)

_PATIENT_PSEUDONYM = re.compile(r"^PSN-[0-9a-f]{16}$")
_UID_PSEUDONYM = re.compile(r"^2\.25\.\d{1,40}$")
_FULL_DATE = re.compile(r"^\d{8}$")
_SKIP_SUFFIXES = {
    ".nii",
    ".json",
    ".txt",
    ".csv",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".yaml",
    ".yml",
    ".npy",
    ".npz",
}


@dataclass(frozen=True)
class Leak:
    """One surviving PHI signal found by the independent auditor."""

    path: str
    tag: str
    reason: str
    value: str = ""

    def __str__(self) -> str:
        return f"{self.reason}:{self.tag} in {self.path}"


def iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path


def is_dicom_candidate(path: Path) -> bool:
    return not (path.name.lower().endswith(".nii.gz") or path.suffix.lower() in _SKIP_SUFFIXES)


def _digest(salt: str, kind: str, value: str) -> str:
    return hmac.new(salt.encode(), f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()


def patient_pseudonym(value: str, salt: str) -> str:
    """Stable patient pseudonym; identical input always maps to the same value."""
    return "PSN-" + _digest(salt, "patient", value)[:16]


def uid_pseudonym(value: str, salt: str, kind: str = "uid") -> str:
    """Stable DICOM UID pseudonym (2.25 dotted-decimal, under 64 characters)."""
    return "2.25." + str(int(_digest(salt, kind, value)[:32], 16))


def is_pseudonym(keyword: str, value: str) -> bool:
    if keyword == "PatientID":
        return bool(_PATIENT_PSEUDONYM.fullmatch(value))
    if keyword in UID_REMAP_TAGS:
        return bool(_UID_PSEUDONYM.fullmatch(value))
    return False


def scan_for_phi(path: str | Path, original_ids: Iterable[str] = ()) -> list[Leak]:
    """Audit one file for surviving PHI without trusting the de-identifier."""
    file_path = Path(path)
    try:
        ds = pydicom.dcmread(file_path, force=True)
        raw = file_path.read_bytes()
    except Exception:
        return [Leak(str(file_path), "", "UNREADABLE_FILE")]
    if len(ds) == 0:
        return [Leak(str(file_path), "", "UNREADABLE_FILE")]
    leaks: list[Leak] = []

    for element in ds.iterall():
        keyword = element.keyword or ""
        value = "" if element.value is None else str(element.value)
        if element.tag.is_private:
            leaks.append(Leak(str(file_path), keyword or str(element.tag), "PRIVATE_TAG", value))
            continue
        if keyword in UID_REMAP_TAGS or keyword == "PatientID":
            if value and not is_pseudonym(keyword, value):
                leaks.append(Leak(str(file_path), keyword, "PHI_TAG", value))
            continue
        if keyword in PHI_TAGS:
            if value:
                leaks.append(Leak(str(file_path), keyword, "PHI_TAG", value))
            continue
        if element.VR == "DA" and _FULL_DATE.match(value):
            leaks.append(
                Leak(
                    str(file_path),
                    keyword or str(element.tag),
                    "FULL_PRECISION_DATE",
                    value,
                )
            )

    for original in original_ids:
        if original and original.encode("utf-8") in raw:
            leaks.append(Leak(str(file_path), "PatientID", "RESIDUAL_ORIGINAL_ID", original))
    return leaks


def scan_corpus(root: str | Path, original_ids: Iterable[str] = ()) -> list[Leak]:
    """Audit every DICOM file below ``root``."""
    base = Path(root)
    leaks: list[Leak] = []
    for path in iter_files(base):
        if is_dicom_candidate(path):
            leaks.extend(scan_for_phi(path, original_ids))
    return leaks


def collect_patient_ids(root: str | Path) -> list[str]:
    """Collect original PatientID values before de-identification."""
    base = Path(root)
    found: set[str] = set()
    for path in iter_files(base):
        if not is_dicom_candidate(path):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        value = str(ds.get("PatientID", "") or "")
        if value:
            found.add(value)
    return sorted(found)
