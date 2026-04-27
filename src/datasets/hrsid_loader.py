from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import ManifestDataset, deserialize_json_field, infer_split_from_parts, list_matching_files, write_csv


def _find_hrsid_annotations(root: Path) -> list[Path]:
    return list_matching_files(root, ["*.json"])


def _find_hrsid_images(root: Path) -> list[Path]:
    return list_matching_files(root, ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"])


def _infer_hrsid_split(annotation_path: Path) -> str:
    name = annotation_path.name.lower()
    if "train_test2017" in name:
        return "all"
    if "train2017" in name:
        return "train"
    if "test2017" in name:
        return "test"
    return infer_split_from_parts(annotation_path)


def _resolve_hrsid_image_path(root: Path, annotation_path: Path, file_name: str) -> Path:
    candidates = [
        root / file_name,
        root / "JPEGImages" / file_name,
        annotation_path.parent / file_name,
        annotation_path.parent.parent / file_name,
        annotation_path.parent.parent / "JPEGImages" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[1].resolve()


def index_hrsid_dataset(root: Path) -> list[dict[str, Any]]:
    annotation_files = _find_hrsid_annotations(root)
    annotation_names = {path.name.lower() for path in annotation_files}
    rows: list[dict[str, Any]] = []
    image_rows_by_path: dict[str, dict[str, Any]] = {}

    for annotation_path in annotation_files:
        if annotation_path.name.lower() == "train_test2017.json" and {
            "train2017.json",
            "test2017.json",
        }.issubset(annotation_names):
            continue
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not {"images", "annotations"}.issubset(payload.keys()):
            continue

        images_by_id = {int(image["id"]): image for image in payload.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            image_id = int(annotation.get("image_id", -1))
            annotations_by_image[image_id].append(annotation)
        categories = {
            int(category["id"]): category.get("name", "unknown")
            for category in payload.get("categories", [])
        }

        for image_id, image_payload in images_by_id.items():
            file_name = str(image_payload.get("file_name", ""))
            image_path = _resolve_hrsid_image_path(root, annotation_path, file_name)
            split = _infer_hrsid_split(annotation_path)
            annotations = annotations_by_image.get(image_id, [])
            sample_stem = Path(file_name).stem or f"hrsid-{image_id}"
            row = {
                "record_type": "sample",
                "dataset": "hrsid",
                "sample_id": sample_stem,
                "split": split,
                "image_path": image_path.as_posix(),
                "annotation_path": annotation_path.resolve().as_posix(),
                "remote_source": "",
                "status": "partial",
                "notes": "",
                "width": image_payload.get("width", ""),
                "height": image_payload.get("height", ""),
                "annotation_count": len(annotations),
                "metadata_json": {
                    "categories": sorted(
                        {
                            categories.get(int(annotation.get("category_id", -1)), "unknown")
                            for annotation in annotations
                        }
                    ),
                    "has_segmentation": any("segmentation" in annotation for annotation in annotations),
                    "complex_slc_available": False,
                    "pixel_domain": "unknown_detected_image_chip",
                    "domain_notes": (
                        "Public HRSID benchmark assets are distributed as image chips / annotations, "
                        "not as complex SLC."
                    ),
                },
            }
            rows.append(row)
            image_rows_by_path[row["image_path"]] = row

    for image_path in _find_hrsid_images(root):
        key = image_path.resolve().as_posix()
        if key in image_rows_by_path:
            continue
        rows.append(
            {
                "record_type": "sample",
                "dataset": "hrsid",
                "sample_id": image_path.stem,
                "split": infer_split_from_parts(image_path),
                "image_path": key,
                "annotation_path": "",
                "remote_source": "",
                "status": "partial",
                "notes": "Image discovered without parsed COCO annotations.",
                "width": "",
                "height": "",
                "annotation_count": 0,
                "metadata_json": {
                    "complex_slc_available": False,
                    "pixel_domain": "unknown_detected_image_chip",
                },
            }
        )

    return rows


def build_hrsid_manifest(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    rows = index_hrsid_dataset(root)
    write_csv(manifest_path, rows)
    return rows


class HRSIDDataset(ManifestDataset):
    def __init__(self, records: list[dict[str, Any]], *, split: str | None = None, sample_limit: int | None = None) -> None:
        normalized_records: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            row["metadata"] = deserialize_json_field(row.get("metadata_json"))
            normalized_records.append(row)
        super().__init__(normalized_records, split=split, sample_limit=sample_limit)
