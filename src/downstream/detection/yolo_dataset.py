from __future__ import annotations

"""Prepare public ship-detection datasets for YOLO baselines.

The input datasets here are detected image chips, not complex SLC. Labels are
converted to standard axis-aligned YOLO boxes. For conditioned variants, the
image chip is first passed through a Stage-1 bundle in the intensity domain and
saved as a normalized PNG; the original annotations are reused because image
geometry is unchanged.
"""

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image

from src.bundles.bundle_a_classical import process_bundle_a_sample
from src.bundles.bundle_b_noiseaware import process_bundle_b_sample
from src.bundles.bundle_d_inverse_problem import process_bundle_d_sample
from src.datasets.common import read_csv_rows, write_csv, write_json
from src.datasets.ssdd_loader import parse_voc_annotation
from src.stage1.pipeline import load_sample
from src.stage1.viz.side_by_side import prepare_display_image
from src.utils import payload_fingerprint, read_artifact_manifest, write_artifact_manifest


SUPPORTED_DATASETS = {"ssdd", "hrsid"}
SUPPORTED_VARIANTS = {"raw", "bundle_a", "bundle_a_conservative", "bundle_b", "bundle_d"}


@dataclass(slots=True)
class PreparedYoloDataset:
    """Summary for one prepared YOLO dataset variant."""

    dataset_name: str
    variant: str
    root: Path
    dataset_yaml: Path
    input_record_count: int
    split_counts: dict[str, int]
    image_count: int
    box_count: int
    skipped_count: int
    missing_image_count: int
    missing_annotation_count: int
    empty_label_count: int
    diagnostics: dict[str, Any]
    status: str
    warnings: list[str]


def prepared_yolo_artifact_identity(
    *,
    dataset_name: str,
    variant: str,
    manifest_path: Path,
    limit_per_split: int | None,
    val_fraction: float,
    bundle_a_config: dict[str, Any] | None = None,
    bundle_a_conservative_config: dict[str, Any] | None = None,
    bundle_b_config: dict[str, Any] | None = None,
    bundle_d_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_kind": "prepared_yolo_dataset",
        "dataset": dataset_name.lower(),
        "variant": variant.lower(),
        "manifest_path": manifest_path.resolve().as_posix(),
        "manifest_mtime": manifest_path.stat().st_mtime if manifest_path.exists() else None,
        "limit_per_split": limit_per_split if limit_per_split is not None else "all",
        "val_fraction": float(val_fraction),
        "bundle_a_config_hash": payload_fingerprint(bundle_a_config) if bundle_a_config is not None else "",
        "bundle_a_conservative_config_hash": (
            payload_fingerprint(bundle_a_conservative_config) if bundle_a_conservative_config is not None else ""
        ),
        "bundle_b_config_hash": payload_fingerprint(bundle_b_config) if bundle_b_config is not None else "",
        "bundle_d_config_hash": payload_fingerprint(bundle_d_config) if bundle_d_config is not None else "",
    }


def _slug(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in str(text))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "sample"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _record_width_height(record: dict[str, Any], fallback: tuple[int, int] | None = None) -> tuple[int, int]:
    width = int(_as_float(record.get("width"), 0.0))
    height = int(_as_float(record.get("height"), 0.0))
    if width > 0 and height > 0:
        return width, height
    if fallback is not None:
        return fallback
    image = Image.open(Path(str(record.get("image_path", ""))))
    return int(image.width), int(image.height)


def _clip_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x0 = max(0.0, min(float(width), x_min))
    y0 = max(0.0, min(float(height), y_min))
    x1 = max(0.0, min(float(width), x_max))
    y1 = max(0.0, min(float(height), y_max))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _to_yolo_box(box: tuple[float, float, float, float], *, width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    box_width = x1 - x0
    box_height = y1 - y0
    x_center = x0 + box_width / 2.0
    y_center = y0 + box_height / 2.0
    return (
        x_center / max(width, 1),
        y_center / max(height, 1),
        box_width / max(width, 1),
        box_height / max(height, 1),
    )


def _ssdd_boxes(record: dict[str, Any]) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    parsed = parse_voc_annotation(Path(str(record["annotation_path"])))
    width, height = _record_width_height(record)
    boxes: list[tuple[float, float, float, float]] = []
    for item in parsed.get("objects", []):
        bbox = item.get("bbox") or {}
        clipped = _clip_box(
            _as_float(bbox.get("xmin")),
            _as_float(bbox.get("ymin")),
            _as_float(bbox.get("xmax")),
            _as_float(bbox.get("ymax")),
            width=width,
            height=height,
        )
        if clipped is not None:
            boxes.append(clipped)
    return width, height, boxes


@lru_cache(maxsize=12)
def _load_coco_payload(annotation_path: str) -> dict[str, Any]:
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    images_by_stem: dict[str, dict[str, Any]] = {}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image in payload.get("images", []):
        stem = Path(str(image.get("file_name", ""))).stem
        images_by_stem[stem] = image
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation.get("image_id", -1))].append(annotation)
    return {"images_by_stem": images_by_stem, "annotations_by_image": annotations_by_image}


def _hrsid_boxes(record: dict[str, Any]) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    annotation_path = str(record.get("annotation_path", ""))
    payload = _load_coco_payload(annotation_path)
    image_payload = payload["images_by_stem"].get(str(record.get("sample_id", "")))
    width, height = _record_width_height(record)
    boxes: list[tuple[float, float, float, float]] = []
    if image_payload is None:
        return width, height, boxes
    for item in payload["annotations_by_image"].get(int(image_payload.get("id", -1)), []):
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x, y, w, h = [_as_float(value) for value in bbox[:4]]
        clipped = _clip_box(x, y, x + w, y + h, width=width, height=height)
        if clipped is not None:
            boxes.append(clipped)
    return width, height, boxes


def detection_boxes(record: dict[str, Any], dataset_name: str) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    """Return axis-aligned ship boxes for a manifest record."""

    if dataset_name == "ssdd":
        return _ssdd_boxes(record)
    if dataset_name == "hrsid":
        return _hrsid_boxes(record)
    raise ValueError(f"Unsupported detection dataset: {dataset_name}")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
        return
    except OSError:
        shutil.copy2(source, destination)


def _save_display_png(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    display = prepare_display_image(array)
    image = np.clip(display * 255.0, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 2:
        Image.fromarray(image, mode="L").save(destination)
    else:
        if image.shape[-1] == 1:
            Image.fromarray(image[..., 0], mode="L").save(destination)
        else:
            Image.fromarray(image[..., :3], mode="RGB").save(destination)


def _to_intensity_2d(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim == 3:
        image = np.mean(image[..., :3], axis=-1)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image.astype(np.float32)


def _load_record_intensity(record: dict[str, Any], dataset_name: str) -> np.ndarray:
    return _to_intensity_2d(load_sample(record, dataset_name).intensity_image)


def _write_variant_image(
    *,
    record: dict[str, Any],
    dataset_name: str,
    variant: str,
    destination: Path,
    bundle_a_config: dict[str, Any] | None,
    bundle_a_conservative_config: dict[str, Any] | None,
    bundle_b_config: dict[str, Any] | None,
    bundle_d_config: dict[str, Any] | None,
) -> np.ndarray:
    if variant == "raw":
        _link_or_copy(Path(str(record["image_path"])), destination)
        return _load_record_intensity(record, dataset_name)
    if variant in {"bundle_a", "bundle_a_conservative"}:
        if bundle_a_config is None:
            raise ValueError("Bundle A config is required for the bundle_a detection variant.")
        selected_config = bundle_a_conservative_config if variant == "bundle_a_conservative" else bundle_a_config
        if selected_config is None:
            raise ValueError("Bundle A conservative config is required for the bundle_a_conservative detection variant.")
        sample = load_sample(record, dataset_name)
        result = process_bundle_a_sample(sample, selected_config)
        _save_display_png(result.final_output, destination)
        return _to_intensity_2d(result.final_output)
    if variant == "bundle_b":
        if bundle_b_config is None:
            raise ValueError("Bundle B config is required for the bundle_b detection variant.")
        sample = load_sample(record, dataset_name)
        result = process_bundle_b_sample(sample, bundle_b_config)
        _save_display_png(result.final_output, destination)
        return _to_intensity_2d(result.final_output)
    if variant == "bundle_d":
        if bundle_d_config is None:
            raise ValueError("Bundle D config is required for the bundle_d detection variant.")
        sample = load_sample(record, dataset_name)
        result = process_bundle_d_sample(sample, bundle_d_config)
        _save_display_png(result.final_output, destination)
        return _to_intensity_2d(result.final_output)
    raise ValueError(f"Unsupported YOLO dataset variant: {variant}")


def _reset_variant_root(variant_root: Path, output_root: Path) -> None:
    """Clear one prepared YOLO variant before rewriting it.

    YOLO reads whole `images/<split>` and `labels/<split>` directories, not our
    manifest.  Without this reset, a smaller rerun can silently train on stale
    files left by a previous larger run.  The safety check keeps deletion scoped
    to the caller-provided prepared-output root.
    """

    resolved_variant = variant_root.resolve()
    resolved_output = output_root.resolve()
    try:
        resolved_variant.relative_to(resolved_output)
    except ValueError as exc:
        raise ValueError(f"Refusing to clear YOLO output outside prepared root: {resolved_variant}") from exc
    if resolved_variant.exists():
        shutil.rmtree(resolved_variant)
    resolved_variant.mkdir(parents=True, exist_ok=True)


def load_prepared_yolo_dataset(variant_root: Path) -> PreparedYoloDataset | None:
    manifest = read_artifact_manifest(variant_root)
    summary_path = variant_root / "prepared_summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    warnings = list(payload.get("warnings", [])) if isinstance(payload.get("warnings"), list) else []
    if manifest and str(manifest.get("artifact_kind", "")) == "prepared_yolo_dataset":
        warnings = [*warnings, *[str(note) for note in manifest.get("notes", []) if str(note)]]
    return PreparedYoloDataset(
        dataset_name=str(payload.get("dataset_name", "")),
        variant=str(payload.get("variant", "")),
        root=Path(str(payload.get("root", variant_root.resolve().as_posix()))),
        dataset_yaml=Path(str(payload.get("dataset_yaml", (variant_root / "dataset.yaml").resolve().as_posix()))),
        input_record_count=int(payload.get("input_record_count", 0)),
        split_counts=dict(payload.get("split_counts", {})),
        image_count=int(payload.get("image_count", 0)),
        box_count=int(payload.get("box_count", 0)),
        skipped_count=int(payload.get("skipped_count", 0)),
        missing_image_count=int(payload.get("missing_image_count", 0)),
        missing_annotation_count=int(payload.get("missing_annotation_count", 0)),
        empty_label_count=int(payload.get("empty_label_count", 0)),
        diagnostics=dict(payload.get("diagnostics", {})),
        status=str(payload.get("status", "prepared")),
        warnings=warnings,
    )


def _write_yolo_label(label_path: Path, boxes: Sequence[tuple[float, float, float, float]], *, width: int, height: int) -> int:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        x_center, y_center, box_width, box_height = _to_yolo_box(box, width=width, height=height)
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        lines.append(f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _target_diagnostics(
    image: np.ndarray,
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Measure target preservation around detection labels.

    These diagnostics are deliberately simple and detector-adjacent: if
    conditioning lowers target/background contrast, local target variance, or
    edge strength around labeled ships, it gives us a concrete explanation for
    why YOLO may lose recall even while proxy despeckling metrics improve.
    """

    intensity = _to_intensity_2d(image)
    if intensity.size == 0 or not boxes:
        return {
            "diagnostic_box_count": 0,
            "target_contrast": "",
            "target_local_variance": "",
            "target_edge_strength": "",
            "target_mean_intensity": "",
            "background_context_mean": "",
        }
    gy, gx = np.gradient(intensity)
    edge = np.sqrt(gx * gx + gy * gy)

    contrasts: list[float] = []
    variances: list[float] = []
    edge_strengths: list[float] = []
    target_means: list[float] = []
    context_means: list[float] = []
    for x0f, y0f, x1f, y1f in boxes:
        x0 = max(0, min(width - 1, int(np.floor(x0f))))
        y0 = max(0, min(height - 1, int(np.floor(y0f))))
        x1 = max(x0 + 1, min(width, int(np.ceil(x1f))))
        y1 = max(y0 + 1, min(height, int(np.ceil(y1f))))
        pad = int(max(4, round(max(x1 - x0, y1 - y0) * 0.75)))
        cx0 = max(0, x0 - pad)
        cy0 = max(0, y0 - pad)
        cx1 = min(width, x1 + pad)
        cy1 = min(height, y1 + pad)

        target = intensity[y0:y1, x0:x1]
        context = intensity[cy0:cy1, cx0:cx1]
        edge_target = edge[y0:y1, x0:x1]
        if target.size == 0 or context.size == 0:
            continue
        mask = np.ones(context.shape, dtype=bool)
        mask[(y0 - cy0) : (y1 - cy0), (x0 - cx0) : (x1 - cx0)] = False
        background = context[mask]
        if background.size == 0:
            background = context.reshape(-1)
        target_mean = float(np.mean(target))
        background_mean = float(np.mean(background))
        background_std = float(np.std(background) + 1e-6)
        contrasts.append((target_mean - background_mean) / background_std)
        variances.append(float(np.var(target)))
        edge_strengths.append(float(np.mean(edge_target)))
        target_means.append(target_mean)
        context_means.append(background_mean)

    if not contrasts:
        return {
            "diagnostic_box_count": 0,
            "target_contrast": "",
            "target_local_variance": "",
            "target_edge_strength": "",
            "target_mean_intensity": "",
            "background_context_mean": "",
        }
    return {
        "diagnostic_box_count": len(contrasts),
        "target_contrast": float(np.mean(contrasts)),
        "target_local_variance": float(np.mean(variances)),
        "target_edge_strength": float(np.mean(edge_strengths)),
        "target_mean_intensity": float(np.mean(target_means)),
        "background_context_mean": float(np.mean(context_means)),
    }


def _aggregate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len(rows)}
    for key in (
        "diagnostic_box_count",
        "target_contrast",
        "target_local_variance",
        "target_edge_strength",
        "target_mean_intensity",
        "background_context_mean",
    ):
        values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else ""
    return summary


def _split_records(records: list[dict[str, Any]], *, val_fraction: float) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    all_rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: str(row.get("sample_id", ""))):
        split = str(record.get("split", "all")).lower()
        if split in {"valid", "validation"}:
            split = "val"
        if split in grouped:
            grouped[split].append(record)
        else:
            all_rows.append(record)

    if all_rows and not grouped["train"]:
        cutoff = max(1, int(round(len(all_rows) * (1.0 - val_fraction))))
        grouped["train"] = all_rows[:cutoff]
        grouped["val"] = all_rows[cutoff:] or all_rows[-1:]
    if not grouped["val"] and len(grouped["train"]) > 1:
        val_count = max(1, int(round(len(grouped["train"]) * val_fraction)))
        grouped["val"] = grouped["train"][-val_count:]
        grouped["train"] = grouped["train"][:-val_count]
    return grouped


def _limit_split(grouped: dict[str, list[dict[str, Any]]], *, limit_per_split: int | None) -> dict[str, list[dict[str, Any]]]:
    if limit_per_split is None:
        return grouped
    return {split: rows[:limit_per_split] for split, rows in grouped.items()}


def _load_manifest_records(manifest_path: Path, dataset_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    missing_rows: list[dict[str, Any]] = []
    for row in read_csv_rows(manifest_path):
        if row.get("record_type") == "placeholder":
            continue
        if str(row.get("dataset", "")).lower() != dataset_name:
            continue
        image_path = Path(str(row.get("image_path", "")))
        annotation_path = Path(str(row.get("annotation_path", "")))
        image_missing = not image_path.exists()
        annotation_missing = not annotation_path.exists()
        if image_missing or annotation_missing:
            missing_rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "split": row.get("split", "all"),
                    "reason": "missing image" if image_missing else "missing annotation",
                    "image_path": image_path.as_posix(),
                    "annotation_path": annotation_path.as_posix(),
                }
            )
        else:
            rows.append(row)
    return rows, missing_rows


def prepare_yolo_dataset(
    *,
    dataset_name: str,
    manifest_path: Path,
    output_root: Path,
    variant: str,
    bundle_a_config: dict[str, Any] | None = None,
    bundle_a_conservative_config: dict[str, Any] | None = None,
    bundle_b_config: dict[str, Any] | None = None,
    bundle_d_config: dict[str, Any] | None = None,
    limit_per_split: int | None = 64,
    val_fraction: float = 0.2,
    reset_root: bool = True,
) -> PreparedYoloDataset:
    """Prepare one raw or conditioned ship-detection dataset variant."""

    dataset_name = dataset_name.lower()
    variant = variant.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset must be one of {sorted(SUPPORTED_DATASETS)}.")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Variant must be one of {sorted(SUPPORTED_VARIANTS)}.")

    variant_root = output_root / dataset_name / variant
    if reset_root:
        _reset_variant_root(variant_root, output_root)
    else:
        variant_root.mkdir(parents=True, exist_ok=True)
    records, missing_rows = _load_manifest_records(manifest_path, dataset_name)
    grouped = _limit_split(_split_records(records, val_fraction=val_fraction), limit_per_split=limit_per_split)

    prepared_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = list(missing_rows)
    split_counts: dict[str, int] = {}
    box_count = 0
    empty_label_count = 0
    for split, split_records in grouped.items():
        split_counts[split] = 0
        image_list_path = variant_root / f"{split}.txt"
        image_paths_for_split: list[str] = []
        for record in split_records:
            sample_name = _slug(f"{dataset_name}_{record.get('sample_id', '')}")
            suffix = ".png" if variant != "raw" else Path(str(record["image_path"])).suffix.lower() or ".jpg"
            image_destination = variant_root / "images" / split / f"{sample_name}{suffix}"
            label_destination = variant_root / "labels" / split / f"{sample_name}.txt"
            try:
                width, height, boxes = detection_boxes(record, dataset_name)
                written_boxes = _write_yolo_label(label_destination, boxes, width=width, height=height)
                variant_intensity = _write_variant_image(
                    record=record,
                    dataset_name=dataset_name,
                    variant=variant,
                    destination=image_destination,
                    bundle_a_config=bundle_a_config,
                    bundle_a_conservative_config=bundle_a_conservative_config,
                    bundle_b_config=bundle_b_config,
                    bundle_d_config=bundle_d_config,
                )
                diagnostics = _target_diagnostics(variant_intensity, boxes, width=width, height=height)
            except Exception as exc:
                skipped_rows.append(
                    {
                        "sample_id": record.get("sample_id", ""),
                        "split": split,
                        "reason": str(exc),
                    }
                )
                continue
            if written_boxes == 0:
                empty_label_count += 1
            box_count += written_boxes
            split_counts[split] += 1
            image_paths_for_split.append(image_destination.resolve().as_posix())
            prepared_rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant,
                    "split": split,
                    "sample_id": record.get("sample_id", ""),
                    "image_path": image_destination.resolve().as_posix(),
                    "label_path": label_destination.resolve().as_posix(),
                    "box_count": written_boxes,
                }
            )
            diagnostic_rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant,
                    "split": split,
                    "sample_id": record.get("sample_id", ""),
                    **diagnostics,
                }
            )
        image_list_path.parent.mkdir(parents=True, exist_ok=True)
        image_list_path.write_text("\n".join(image_paths_for_split) + ("\n" if image_paths_for_split else ""), encoding="utf-8")

    dataset_yaml = variant_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": variant_root.resolve().as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 1,
                "names": {0: "ship"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_csv(variant_root / "manifest.csv", prepared_rows)
    write_json(variant_root / "manifest.json", {"samples": prepared_rows})
    diagnostic_summary = _aggregate_diagnostics(diagnostic_rows)
    write_csv(variant_root / "diagnostics.csv", diagnostic_rows)
    write_json(variant_root / "diagnostics.json", {"samples": diagnostic_rows, "summary": diagnostic_summary})
    write_csv(variant_root / "skipped.csv", skipped_rows)

    missing_image_count = sum(str(row.get("reason", "")).startswith("missing image") for row in missing_rows)
    missing_annotation_count = sum(str(row.get("reason", "")).startswith("missing annotation") for row in missing_rows)

    warnings: list[str] = []
    if not records:
        warnings.append("No usable manifest records had both image and annotation files.")
    if not split_counts.get("train"):
        warnings.append("No training images were prepared.")
    if not split_counts.get("val"):
        warnings.append("No validation images were prepared.")
    if box_count <= 0:
        warnings.append("No detection boxes were converted. Check annotation parsing before training.")
    if empty_label_count:
        warnings.append(f"{empty_label_count} prepared image(s) had empty YOLO label files.")
    if missing_rows:
        warnings.append(f"{len(missing_rows)} manifest record(s) were missing image or annotation files.")
    if skipped_rows:
        warnings.append(f"{len(skipped_rows)} sample(s) were skipped during YOLO preparation.")
    if variant in {"bundle_a", "bundle_a_conservative"}:
        warnings.append("Bundle A images are normalized conditioned PNGs; geometry is unchanged, labels are reused.")
    if variant == "bundle_b":
        warnings.append("Bundle B images are normalized conditioned PNGs; geometry is unchanged, labels are reused.")
    if variant == "bundle_d":
        warnings.append("Bundle D images are normalized conditioned PNGs; geometry is unchanged, labels are reused.")

    prepared = PreparedYoloDataset(
        dataset_name=dataset_name,
        variant=variant,
        root=variant_root.resolve(),
        dataset_yaml=dataset_yaml.resolve(),
        input_record_count=len(records) + len(missing_rows),
        split_counts=split_counts,
        image_count=sum(split_counts.values()),
        box_count=box_count,
        skipped_count=len(skipped_rows),
        missing_image_count=missing_image_count,
        missing_annotation_count=missing_annotation_count,
        empty_label_count=empty_label_count,
        diagnostics=diagnostic_summary,
        status="prepared" if sum(split_counts.values()) else "empty",
        warnings=warnings,
    )
    summary_payload = {
        "dataset_name": prepared.dataset_name,
        "variant": prepared.variant,
        "root": prepared.root.as_posix(),
        "dataset_yaml": prepared.dataset_yaml.as_posix(),
        "input_record_count": prepared.input_record_count,
        "split_counts": prepared.split_counts,
        "image_count": prepared.image_count,
        "box_count": prepared.box_count,
        "skipped_count": prepared.skipped_count,
        "missing_image_count": prepared.missing_image_count,
        "missing_annotation_count": prepared.missing_annotation_count,
        "empty_label_count": prepared.empty_label_count,
        "diagnostics": prepared.diagnostics,
        "status": prepared.status,
        "warnings": prepared.warnings,
    }
    write_json(variant_root / "prepared_summary.json", summary_payload)
    write_artifact_manifest(
        variant_root,
        artifact_kind="prepared_yolo_dataset",
        identity=prepared_yolo_artifact_identity(
            dataset_name=dataset_name,
            variant=variant,
            manifest_path=manifest_path,
            limit_per_split=limit_per_split,
            val_fraction=val_fraction,
            bundle_a_config=bundle_a_config,
            bundle_a_conservative_config=bundle_a_conservative_config,
            bundle_b_config=bundle_b_config,
            bundle_d_config=bundle_d_config,
        ),
        status=prepared.status,
        files={
            "dataset_yaml": dataset_yaml.resolve().as_posix(),
            "manifest_csv": (variant_root / "manifest.csv").resolve().as_posix(),
            "manifest_json": (variant_root / "manifest.json").resolve().as_posix(),
            "prepared_summary": (variant_root / "prepared_summary.json").resolve().as_posix(),
            "diagnostics_json": (variant_root / "diagnostics.json").resolve().as_posix(),
            "skipped_csv": (variant_root / "skipped.csv").resolve().as_posix(),
        },
        metadata={
            "dataset": dataset_name,
            "variant": variant,
            "split_counts": split_counts,
            "image_count": prepared.image_count,
            "box_count": prepared.box_count,
        },
        notes=warnings,
    )
    return prepared
