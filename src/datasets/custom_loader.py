from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .common import ManifestDataset, deserialize_json_field, infer_split_from_parts, list_matching_files, write_csv


DEFAULT_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
DEFAULT_ANNOTATION_PATTERNS = ("*.json", "*.xml", "*.txt", "*.csv", "*.png", "*.tif", "*.tiff")


def _sample_id_from_relative_path(root: Path, image_path: Path) -> str:
    relative_path = image_path.resolve().relative_to(root.resolve())
    return relative_path.with_suffix("").as_posix()


def _index_annotations(
    root: Path,
    annotation_patterns: Sequence[str],
    *,
    allow_image_suffixes: bool,
) -> dict[str, Path]:
    suffixes_to_skip = (
        {pattern.replace("*", "").lower() for pattern in DEFAULT_IMAGE_PATTERNS}
        if not allow_image_suffixes
        else set()
    )
    annotation_index: dict[str, Path] = {}
    for annotation_path in list_matching_files(root, annotation_patterns):
        if annotation_path.suffix.lower() in suffixes_to_skip:
            continue
        annotation_index.setdefault(annotation_path.stem.lower(), annotation_path.resolve())
    return annotation_index


def build_custom_manifest(
    root: Path,
    manifest_path: Path,
    *,
    dataset_name: str,
    image_patterns: Sequence[str] | None = None,
    annotation_patterns: Sequence[str] | None = None,
    annotation_match_mode: str = "stem",
    pixel_domain: str = "unknown",
    complex_slc_available: bool = False,
    source_name: str = "",
    notes: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    image_patterns = tuple(image_patterns or DEFAULT_IMAGE_PATTERNS)
    annotation_patterns = tuple(annotation_patterns or DEFAULT_ANNOTATION_PATTERNS)
    match_mode = annotation_match_mode.lower().strip()
    annotation_index = (
        _index_annotations(root, annotation_patterns, allow_image_suffixes=False)
        if match_mode == "stem"
        else {}
    )

    rows: list[dict[str, Any]] = []
    for image_path in list_matching_files(root, image_patterns):
        annotation_path = annotation_index.get(image_path.stem.lower())
        relative_image = image_path.resolve().relative_to(root.resolve()).as_posix()
        row_metadata = {
            "complex_slc_available": complex_slc_available,
            "pixel_domain": pixel_domain,
            "source_name": source_name,
            "image_relative_path": relative_image,
            "annotation_match_mode": match_mode,
        }
        if extra_metadata:
            row_metadata.update(extra_metadata)
        rows.append(
            {
                "record_type": "sample",
                "dataset": dataset_name,
                "sample_id": _sample_id_from_relative_path(root, image_path),
                "split": infer_split_from_parts(image_path),
                "image_path": image_path.resolve().as_posix(),
                "annotation_path": annotation_path.as_posix() if annotation_path else "",
                "remote_source": "",
                "status": "partial",
                "notes": notes if annotation_path else (notes or "No paired annotation file found during custom scan."),
                "width": "",
                "height": "",
                "annotation_count": "",
                "metadata_json": row_metadata,
            }
        )

    write_csv(manifest_path, rows)
    return rows


class CustomDataset(ManifestDataset):
    def __init__(self, records: list[dict[str, Any]], *, split: str | None = None, sample_limit: int | None = None) -> None:
        normalized_records: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            row["metadata"] = deserialize_json_field(row.get("metadata_json"))
            normalized_records.append(row)
        super().__init__(normalized_records, split=split, sample_limit=sample_limit)
