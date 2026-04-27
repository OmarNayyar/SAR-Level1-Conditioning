from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DEFAULT_LARGE_DOWNLOAD_THRESHOLD_BYTES = 5 * 1024**3
DEFAULT_MAX_FREE_SPACE_FRACTION = 0.65


class DatasetStatus(str, Enum):
    MISSING = "missing"
    METADATA_ONLY = "metadata-only"
    PARTIAL = "partial"
    COMPLETE = "complete"
    EXTERNAL_LINKED = "external-linked"


class DatasetError(RuntimeError):
    """Base class for dataset-related errors."""


class StorageGuardError(DatasetError):
    """Raised when a requested download would violate storage safeguards."""


@dataclass(slots=True)
class StorageEstimate:
    description: str
    size_bytes: int | None
    source: str | None = None


def repo_root_from_path(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start.resolve()


def ensure_data_layout(repo_root: Path) -> dict[str, Path | bool]:
    override_root = os.getenv("SAR_DATA_LAYOUT_ROOT")
    data_root = Path(override_root).expanduser().resolve() if override_root else repo_root / "data"

    preferred_external = data_root / "external"
    preferred_manifests = preferred_external / "manifests"
    preferred_catalogs = preferred_external / "catalogs"
    fallback_index_root = data_root / "metadata_indexes"

    catalogs_flat = False
    try:
        preferred_external.mkdir(parents=True, exist_ok=True)
        preferred_manifests.mkdir(parents=True, exist_ok=True)
        preferred_catalogs.mkdir(parents=True, exist_ok=True)
        data_external = preferred_external
        manifests = preferred_manifests
        catalogs = preferred_catalogs
    except PermissionError:
        data_external = fallback_index_root
        manifests = fallback_index_root
        catalogs = fallback_index_root
        catalogs_flat = True

    splits = data_root / "splits"
    try:
        splits.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        splits = data_root / "interim"

    layout = {
        "data_external": data_external,
        "manifests": manifests,
        "catalogs": catalogs,
        "catalogs_flat": catalogs_flat,
        "raw": data_root / "raw",
        "interim": data_root / "interim",
        "processed": data_root / "processed",
        "splits": splits,
        "logs": repo_root / "outputs" / "logs",
    }
    for key, path in layout.items():
        if key == "catalogs_flat":
            continue
        path.mkdir(parents=True, exist_ok=True)
    return layout


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value < 1024:
        return f"{value} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} EB"


def to_posix_path(path: Path | None) -> str:
    return "" if path is None else path.resolve().as_posix()


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_bbox(text: str | Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if text is None:
        return None
    if isinstance(text, Sequence) and not isinstance(text, str):
        values = [float(part) for part in text]
    else:
        values = [float(part.strip()) for part in str(text).split(",")]
    if len(values) != 4:
        raise ValueError("Bounding box must have four comma-separated numbers: min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = values
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Bounding box must satisfy min_lon < max_lon and min_lat < max_lat")
    return min_lon, min_lat, max_lon, max_lat


def bbox_to_polygon_wkt(bbox: tuple[float, float, float, float]) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        "POLYGON(("
        f"{min_lon} {min_lat},"
        f"{min_lon} {max_lat},"
        f"{max_lon} {max_lat},"
        f"{max_lon} {min_lat},"
        f"{min_lon} {min_lat}"
        "))"
    )


def polygon_text_to_wkt(polygon_text: str) -> str:
    cleaned = polygon_text.strip()
    if cleaned.upper().startswith("POLYGON(("):
        return cleaned
    points: list[str] = []
    for point in cleaned.split(";"):
        lon_str, lat_str = [item.strip() for item in point.split(",")]
        points.append(f"{float(lon_str)} {float(lat_str)}")
    if len(points) < 4:
        raise ValueError("Polygon must contain at least four points")
    if points[0] != points[-1]:
        points.append(points[0])
    return f"POLYGON(({','.join(points)}))"


def available_disk_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def ensure_storage_guard(
    estimates: Sequence[StorageEstimate],
    target_dir: Path,
    *,
    force: bool,
    large_download_threshold_bytes: int = DEFAULT_LARGE_DOWNLOAD_THRESHOLD_BYTES,
    max_free_space_fraction: float = DEFAULT_MAX_FREE_SPACE_FRACTION,
) -> list[str]:
    warnings: list[str] = []
    known_sizes = [estimate.size_bytes for estimate in estimates if estimate.size_bytes is not None]
    unknown_size_items = [estimate.description for estimate in estimates if estimate.size_bytes is None]
    total_known_size = sum(known_sizes)
    free_bytes = available_disk_bytes(target_dir)

    warnings.append(f"Free space at target: {human_bytes(free_bytes)}")
    if known_sizes:
        warnings.append(f"Projected download size: {human_bytes(total_known_size)}")
    if unknown_size_items:
        warnings.append(
            "Projected download size unavailable for: " + ", ".join(unknown_size_items)
        )

    if force:
        return warnings

    if unknown_size_items:
        raise StorageGuardError(
            "At least one download has unknown size. Re-run with --force or choose metadata-only / external-path mode."
        )

    if total_known_size > large_download_threshold_bytes:
        raise StorageGuardError(
            "Projected download exceeds the default safety threshold "
            f"({human_bytes(total_known_size)} > {human_bytes(large_download_threshold_bytes)}). "
            "Re-run with --force to allow large downloads."
        )

    if total_known_size > int(free_bytes * max_free_space_fraction):
        raise StorageGuardError(
            "Projected download would consume too much of the remaining free space "
            f"({human_bytes(total_known_size)} requested, {human_bytes(free_bytes)} free). "
            "Re-run with --force or use metadata-only / external storage."
        )

    return warnings


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        inferred: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    inferred.append(key)
                    seen.add(key)
        fieldnames = inferred
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_manifest_value(row.get(key)) for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_manifest_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if isinstance(value, Path):
        return to_posix_path(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def deserialize_json_field(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def list_matching_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted({path for path in files if path.is_file()})


def infer_split_from_parts(path: Path) -> str:
    lowered_parts = [part.lower() for part in path.parts]
    for token in lowered_parts:
        if token in {"train", "training"}:
            return "train"
        if token in {"val", "valid", "validation"}:
            return "val"
        if token in {"test", "testing"}:
            return "test"
    return "all"


def create_directory_link(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists():
        return
    try:
        link_path.symlink_to(target_path, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise
    command = ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DatasetError(
            f"Failed to create directory link from {link_path} to {target_path}: {result.stderr.strip() or result.stdout.strip()}"
        )


def resolve_external_dataset_path(base_path: Path, dataset_name: str, *, single_dataset: bool) -> Path | None:
    normalized_names = {
        dataset_name,
        dataset_name.lower(),
        dataset_name.replace("-", "_"),
        dataset_name.replace("_", "-"),
    }
    candidates = [base_path / name for name in normalized_names]
    if single_dataset:
        candidates.insert(0, base_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


class ManifestDataset(Sequence[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str | None = None,
        sample_limit: int | None = None,
        include_placeholders: bool = False,
    ) -> None:
        filtered: list[dict[str, Any]] = []
        normalized_split = split.lower() if split else None
        for record in records:
            row = dict(record)
            if not include_placeholders and row.get("record_type") == "placeholder":
                continue
            row_split = str(row.get("split", "all")).lower()
            if normalized_split and row_split not in {normalized_split, "all"}:
                continue
            filtered.append(row)
        if sample_limit is not None:
            filtered = filtered[:sample_limit]
        self._records = filtered

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._records)


def load_manifest_dataset(
    manifest_path: Path,
    *,
    split: str | None = None,
    sample_limit: int | None = None,
    include_placeholders: bool = False,
) -> ManifestDataset:
    rows = read_csv_rows(manifest_path)
    return ManifestDataset(
        rows,
        split=split,
        sample_limit=sample_limit,
        include_placeholders=include_placeholders,
    )


def placeholder_manifest_row(
    *,
    dataset: str,
    remote_source: str | None,
    notes: str,
    status: DatasetStatus,
) -> dict[str, Any]:
    return {
        "record_type": "placeholder",
        "dataset": dataset,
        "sample_id": f"{dataset}-placeholder",
        "split": "all",
        "image_path": "",
        "annotation_path": "",
        "remote_source": remote_source or "",
        "status": status.value,
        "notes": notes,
        "metadata_json": json.dumps({}, ensure_ascii=True),
    }


def prepend_repo_root_to_syspath(script_path: Path) -> Path:
    repo_root = repo_root_from_path(script_path)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root
