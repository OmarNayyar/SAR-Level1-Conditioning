from __future__ import annotations

"""Validate future external ship-detection datasets before ingestion.

This is intentionally a compatibility layer, not a new dataset architecture.
Partner/internal data can arrive as COCO JSON, YOLO folders, or a simple CSV
of boxes. The validator checks that image/annotation files are resolvable and
returns a compact report before any Stage-1 or detector work is attempted.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_EXTERNAL_FORMATS = {"coco", "yolo", "bbox_csv", "image_annotation_map", "custom_map"}


@dataclass(slots=True)
class ExternalDetectionValidation:
    dataset_name: str
    dataset_format: str
    root: str
    status: str
    image_count: int
    annotation_count: int
    missing_paths: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _validate_coco(name: str, root: Path, config: dict[str, Any]) -> ExternalDetectionValidation:
    annotation_path = _resolve(root, str(config.get("annotation_path", "")))
    missing: list[str] = []
    notes: list[str] = []
    if annotation_path is None or not annotation_path.exists():
        return ExternalDetectionValidation(name, "coco", root.as_posix(), "failed", 0, 0, [str(annotation_path or "")], ["COCO annotation JSON is missing."])
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_dir = _resolve(root, str(config.get("image_dir", ""))) or root
    image_count = 0
    for image in payload.get("images", []):
        image_path = image_dir / str(image.get("file_name", ""))
        if image_path.exists():
            image_count += 1
        else:
            missing.append(image_path.as_posix())
            if len(missing) >= 20:
                notes.append("Only the first 20 missing image paths are reported.")
                break
    annotation_count = len(payload.get("annotations", []))
    status = "ready" if image_count > 0 and annotation_count > 0 and not missing else "partial"
    return ExternalDetectionValidation(name, "coco", root.as_posix(), status, image_count, annotation_count, missing, notes)


def _validate_yolo(name: str, root: Path, config: dict[str, Any]) -> ExternalDetectionValidation:
    dataset_yaml = _resolve(root, str(config.get("dataset_yaml", "dataset.yaml")))
    missing: list[str] = []
    notes: list[str] = []
    if dataset_yaml is None or not dataset_yaml.exists():
        return ExternalDetectionValidation(name, "yolo", root.as_posix(), "failed", 0, 0, [str(dataset_yaml or "")], ["YOLO dataset.yaml is missing."])
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    base = _resolve(dataset_yaml.parent, str(payload.get("path", ""))) or dataset_yaml.parent
    image_count = 0
    annotation_count = 0
    for split in ("train", "val", "test"):
        split_value = payload.get(split)
        split_path = _resolve(base, str(split_value)) if split_value else None
        if split_path is None:
            continue
        if split_path.is_file():
            image_paths = [Path(line.strip()) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            image_paths = list(split_path.glob("*.*")) if split_path.exists() else []
        for image_path in image_paths:
            resolved_image = image_path if image_path.is_absolute() else base / image_path
            if not resolved_image.exists():
                missing.append(resolved_image.as_posix())
                continue
            image_count += 1
            try:
                rel_image = resolved_image.relative_to(base / "images")
                label_path = (base / "labels" / rel_image).with_suffix(".txt")
            except ValueError:
                label_path = resolved_image.with_suffix(".txt")
            if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
                annotation_count += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    status = "ready" if image_count > 0 and annotation_count > 0 and not missing else "partial"
    return ExternalDetectionValidation(name, "yolo", root.as_posix(), status, image_count, annotation_count, missing[:20], notes)


def _validate_bbox_csv(name: str, root: Path, config: dict[str, Any]) -> ExternalDetectionValidation:
    csv_path = _resolve(root, str(config.get("annotation_path", "")))
    image_dir = _resolve(root, str(config.get("image_dir", ""))) or root
    if csv_path is None or not csv_path.exists():
        return ExternalDetectionValidation(name, "bbox_csv", root.as_posix(), "failed", 0, 0, [str(csv_path or "")], ["Bounding-box CSV is missing."])
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ExternalDetectionValidation(name, "bbox_csv", root.as_posix(), "failed", 0, 0, [], ["Bounding-box CSV is empty."])
    headers = [item.strip() for item in lines[0].split(",")]
    image_col = "image_path" if "image_path" in headers else "file_name" if "file_name" in headers else ""
    required = {"xmin", "ymin", "xmax", "ymax"}
    missing_cols = sorted(required.difference(headers))
    if not image_col or missing_cols:
        return ExternalDetectionValidation(name, "bbox_csv", root.as_posix(), "failed", 0, 0, [], [f"CSV requires image_path/file_name and xmin/ymin/xmax/ymax columns; missing {missing_cols}."])
    image_index = headers.index(image_col)
    seen_images: set[str] = set()
    missing_paths: list[str] = []
    annotation_count = 0
    for line in lines[1:]:
        parts = [item.strip() for item in line.split(",")]
        if len(parts) <= image_index:
            continue
        image_path = _resolve(image_dir, parts[image_index])
        if image_path is None or not image_path.exists():
            missing_paths.append(str(image_path or ""))
            continue
        seen_images.add(image_path.as_posix())
        annotation_count += 1
    status = "ready" if seen_images and annotation_count and not missing_paths else "partial"
    return ExternalDetectionValidation(name, "bbox_csv", root.as_posix(), status, len(seen_images), annotation_count, missing_paths[:20], [])


def _validate_image_annotation_map(name: str, root: Path, config: dict[str, Any]) -> ExternalDetectionValidation:
    """Validate a simple custom mapping CSV.

    This is useful when a downstream team receives a manifest
    with arbitrary image/annotation paths rather than COCO or YOLO structure.
    Required columns are `image_path` and `annotation_path`; optional columns
    such as `split`, `sample_id`, or `domain` are ignored by validation but can
    be used later by dataset-registration tooling.
    """

    manifest_path = _resolve(root, str(config.get("manifest_path", "")))
    if manifest_path is None or not manifest_path.exists():
        return ExternalDetectionValidation(
            name,
            "image_annotation_map",
            root.as_posix(),
            "failed",
            0,
            0,
            [str(manifest_path or "")],
            ["Custom image/annotation manifest is missing."],
        )
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ExternalDetectionValidation(name, "image_annotation_map", root.as_posix(), "failed", 0, 0, [], ["Manifest is empty."])
    headers = [item.strip() for item in lines[0].split(",")]
    missing_columns = [column for column in ("image_path", "annotation_path") if column not in headers]
    if missing_columns:
        return ExternalDetectionValidation(
            name,
            "image_annotation_map",
            root.as_posix(),
            "failed",
            0,
            0,
            [],
            [f"Manifest requires image_path and annotation_path columns; missing {missing_columns}."],
        )
    image_index = headers.index("image_path")
    annotation_index = headers.index("annotation_path")
    image_count = 0
    annotation_count = 0
    missing_paths: list[str] = []
    for line in lines[1:]:
        parts = [item.strip() for item in line.split(",")]
        if len(parts) <= max(image_index, annotation_index):
            continue
        image_path = _resolve(root, parts[image_index])
        annotation_path = _resolve(root, parts[annotation_index])
        if image_path is None or not image_path.exists():
            missing_paths.append(str(image_path or ""))
            continue
        if annotation_path is None or not annotation_path.exists():
            missing_paths.append(str(annotation_path or ""))
            continue
        image_count += 1
        annotation_count += 1
    status = "ready" if image_count > 0 and annotation_count > 0 and not missing_paths else "partial"
    notes = ["Custom map validation checks path existence, not annotation schema semantics."]
    return ExternalDetectionValidation(
        name,
        "image_annotation_map",
        root.as_posix(),
        status,
        image_count,
        annotation_count,
        missing_paths[:20],
        notes,
    )


def validate_external_detection_dataset(config_path: Path) -> ExternalDetectionValidation:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = config.get("dataset", {})
    name = str(dataset.get("name", config_path.stem))
    dataset_format = str(dataset.get("format", "")).strip().lower()
    if dataset_format not in SUPPORTED_EXTERNAL_FORMATS:
        return ExternalDetectionValidation(
            name,
            dataset_format or "unknown",
            "",
            "failed",
            0,
            0,
            [],
            [f"Unsupported format {dataset_format!r}; expected one of {sorted(SUPPORTED_EXTERNAL_FORMATS)}."],
        )
    root = Path(str(dataset.get("path", "."))).expanduser().resolve()
    if dataset_format == "coco":
        return _validate_coco(name, root, dataset)
    if dataset_format == "yolo":
        return _validate_yolo(name, root, dataset)
    if dataset_format in {"image_annotation_map", "custom_map"}:
        return _validate_image_annotation_map(name, root, dataset)
    return _validate_bbox_csv(name, root, dataset)
