from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as skio

from .common import read_csv_rows, write_csv, write_json
from .hrsid_loader import index_hrsid_dataset
from .registry import DatasetRegistry, default_registry_path
from .sen1floods11_loader import build_sen1floods11_manifest
from .ssdd_loader import parse_voc_annotation
from ..stage1.viz.side_by_side import prepare_display_image


PREVIEWABLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TRAIN_LIKE_SPLITS = {"train", "val", "test"}


@dataclass(slots=True)
class DatasetAuditSummary:
    dataset_name: str
    status: str
    local_path: str
    manifest_path: str
    manifest_exists: bool
    total_count: int
    split_names: list[str]
    split_counts: dict[str, int]
    missing_image_files: int
    missing_annotation_files: int
    broken_manifest_paths: int
    duplicate_sample_ids: int
    duplicate_image_files_across_splits: int
    duplicate_annotation_files_across_splits: int
    leakage_by_canonical_id: int
    leakage_by_file_stem: int
    preview_count: int
    preview_note: str
    issue_examples: dict[str, list[str]]


def _canonical_identifier(row: dict[str, str]) -> str:
    sample_id = row.get("sample_id", "").strip()
    if sample_id:
        return sample_id.lower()

    for key in ("image_path", "annotation_path"):
        value = row.get(key, "").strip()
        if value:
            stem = Path(value).stem.lower()
            for suffix in ("_s1hand", "_labelhand", "_qc", "_s1perm", "_s1weak"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            return stem
    return ""


def _safe_path(value: str) -> Path | None:
    text = (value or "").strip()
    if not text:
        return None
    return Path(text)


def _read_image_array(path: Path) -> np.ndarray | None:
    try:
        array = skio.imread(path)
    except Exception:
        return None
    return np.asarray(array)


def _render_plain_preview(image_path: Path, output_path: Path, title: str) -> bool:
    image = _read_image_array(image_path)
    if image is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(prepare_display_image(image), cmap="gray" if image.ndim == 2 else None)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return True


def _render_voc_preview(row: dict[str, str], output_path: Path) -> bool:
    image_path = _safe_path(row.get("image_path", ""))
    annotation_path = _safe_path(row.get("annotation_path", ""))
    if image_path is None or annotation_path is None or not image_path.exists() or not annotation_path.exists():
        return False

    image = _read_image_array(image_path)
    if image is None:
        return False
    parsed = parse_voc_annotation(annotation_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(prepare_display_image(image), cmap="gray" if image.ndim == 2 else None)
    axis.set_title(f"{row.get('dataset')} | {row.get('sample_id')}")
    for object_payload in parsed["objects"]:
        bbox = object_payload.get("bbox")
        if bbox:
            xmin = float(bbox.get("xmin", 0))
            ymin = float(bbox.get("ymin", 0))
            xmax = float(bbox.get("xmax", 0))
            ymax = float(bbox.get("ymax", 0))
            axis.add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    max(xmax - xmin, 1.0),
                    max(ymax - ymin, 1.0),
                    linewidth=1.5,
                    edgecolor="#ffcc00",
                    facecolor="none",
                )
            )
        polygon = object_payload.get("segmentation") or []
        polygon_points = []
        for point in polygon:
            if "x" in point and "y" in point:
                polygon_points.append((float(point["x"]), float(point["y"])))
        if len(polygon_points) >= 3:
            axis.add_patch(
                patches.Polygon(
                    polygon_points,
                    closed=True,
                    linewidth=1.0,
                    edgecolor="#00d4ff",
                    facecolor="none",
                )
            )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return True


def _render_sen1floods11_preview(row: dict[str, str], output_path: Path) -> bool:
    image_path = _safe_path(row.get("image_path", ""))
    annotation_path = _safe_path(row.get("annotation_path", ""))
    if image_path is None or annotation_path is None or not image_path.exists() or not annotation_path.exists():
        return False

    image = _read_image_array(image_path)
    mask = _read_image_array(annotation_path)
    if image is None or mask is None:
        return False

    display = prepare_display_image(image)
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = np.squeeze(mask_array)
    overlay = np.dstack([display, display, display]) if display.ndim == 2 else display.copy()
    water_mask = mask_array == 1
    nodata_mask = mask_array == -1
    overlay[water_mask] = 0.65 * overlay[water_mask] + 0.35 * np.array([1.0, 0.1, 0.1], dtype=np.float32)
    overlay[nodata_mask] = 0.65 * overlay[nodata_mask] + 0.35 * np.array([0.7, 0.7, 0.7], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(overlay)
    axis.set_title(f"sen1floods11 | {row.get('sample_id')}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return True


def _preview_rows(rows: list[dict[str, str]], preview_count: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_splits: set[str] = set()
    for row in rows:
        image_path = _safe_path(row.get("image_path", ""))
        if image_path is None or image_path.suffix.lower() not in PREVIEWABLE_SUFFIXES or not image_path.exists():
            continue
        split = row.get("split", "all")
        if split not in seen_splits:
            selected.append(row)
            seen_splits.add(split)
        if len(selected) >= preview_count:
            return selected
    for row in rows:
        if len(selected) >= preview_count:
            break
        image_path = _safe_path(row.get("image_path", ""))
        if image_path is None or image_path.suffix.lower() not in PREVIEWABLE_SUFFIXES or not image_path.exists():
            continue
        if row in selected:
            continue
        selected.append(row)
    return selected


def _generate_previews(dataset_name: str, rows: list[dict[str, str]], dataset_dir: Path, preview_count: int) -> tuple[int, str]:
    preview_rows = _preview_rows(rows, preview_count)
    if not preview_rows:
        return 0, "No previewable image files were available for this dataset."

    preview_dir = dataset_dir / "previews"
    created = 0
    for index, row in enumerate(preview_rows, start=1):
        image_path = _safe_path(row.get("image_path", ""))
        if image_path is None:
            continue
        output_path = preview_dir / f"{index:02d}_{row.get('sample_id', 'sample')}.png"
        if dataset_name in {"ssdd", "ls_ssdd"}:
            success = _render_voc_preview(row, output_path)
        elif dataset_name == "sen1floods11":
            success = _render_sen1floods11_preview(row, output_path)
        else:
            success = _render_plain_preview(image_path, output_path, f"{dataset_name} | {row.get('sample_id')}")
        if success:
            created += 1
    if created == 0:
        return 0, "Preview generation was attempted, but none of the candidate files could be rendered."
    return created, f"Saved {created} preview image(s) under {preview_dir.as_posix()}."


def _path_issues(rows: list[dict[str, str]], key: str) -> tuple[int, list[str]]:
    missing_paths: list[str] = []
    for row in rows:
        path = _safe_path(row.get(key, ""))
        if path is None:
            continue
        if not path.exists():
            missing_paths.append(f"{row.get('sample_id', '')}: {path.as_posix()}")
    return len(missing_paths), missing_paths


def _broken_manifest_paths(rows: list[dict[str, str]]) -> tuple[int, list[str]]:
    broken: list[str] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        sample_broken: list[str] = []
        for key in ("image_path", "annotation_path"):
            path = _safe_path(row.get(key, ""))
            if path is None:
                continue
            if not path.exists():
                sample_broken.append(f"{key}={path.as_posix()}")
        if sample_broken:
            broken.append(f"{sample_id}: " + "; ".join(sample_broken))
    return len(broken), broken


def _duplicates_across_splits(rows: list[dict[str, str]], key: str) -> tuple[int, list[str]]:
    path_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(key, "").strip()
        split = row.get("split", "all").lower()
        if not value or split not in TRAIN_LIKE_SPLITS:
            continue
        path_to_splits[value].add(split)
    leaks = [f"{path} -> {sorted(splits)}" for path, splits in path_to_splits.items() if len(splits) > 1]
    return len(leaks), leaks


def _canonical_leakage(rows: list[dict[str, str]]) -> tuple[int, list[str], int, list[str]]:
    identifier_to_splits: dict[str, set[str]] = defaultdict(set)
    stem_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split", "all").lower()
        if split not in TRAIN_LIKE_SPLITS:
            continue
        canonical_id = _canonical_identifier(row)
        if canonical_id:
            identifier_to_splits[canonical_id].add(split)
        image_path = row.get("image_path", "").strip()
        if image_path:
            stem_to_splits[Path(image_path).stem.lower()].add(split)
    id_leaks = [f"{identifier} -> {sorted(splits)}" for identifier, splits in identifier_to_splits.items() if len(splits) > 1]
    stem_leaks = [f"{stem} -> {sorted(splits)}" for stem, splits in stem_to_splits.items() if len(splits) > 1]
    return len(id_leaks), id_leaks, len(stem_leaks), stem_leaks


def _write_dataset_tables(dataset_dir: Path, rows: list[dict[str, str]], split_counts: dict[str, int], issues: dict[str, list[str]]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        dataset_dir / "split_counts.csv",
        [{"split": split_name, "count": count} for split_name, count in sorted(split_counts.items())],
    )
    issue_rows = []
    for category, examples in issues.items():
        issue_rows.append(
            {
                "category": category,
                "count": len(examples),
                "examples_json": examples[:25],
            }
        )
    write_csv(dataset_dir / "issues.csv", issue_rows)


def _placeholder_only(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(row.get("record_type") == "placeholder" for row in rows)


def _fallback_local_rows(dataset_name: str, registration_local_path: str, manifest_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], str | None]:
    local_root = Path(registration_local_path) if registration_local_path else Path()
    if not registration_local_path or not local_root.exists():
        return [], None
    try:
        has_files = any(local_root.iterdir())
    except OSError:
        return [], None
    if not has_files:
        return [], None

    if dataset_name == "hrsid" and (not manifest_rows or _placeholder_only(manifest_rows)):
        rows = index_hrsid_dataset(local_root)
        if rows:
            return rows, (
                "Audit used local HRSID files directly because the saved manifest/registry snapshot was still "
                "metadata-only. Re-run setup_public_datasets.py --dataset hrsid --no-download to refresh the "
                "tracked manifest and registry."
            )
    return [], None


def audit_registered_datasets(
    repo_root: Path,
    *,
    preview_count: int = 4,
    output_root: Path | None = None,
    docs_summary_path: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root or repo_root / "results" / "data_audit"
    docs_summary_path = docs_summary_path or repo_root / "docs" / "data_audit_summary.md"
    output_root.mkdir(parents=True, exist_ok=True)
    docs_summary_path.parent.mkdir(parents=True, exist_ok=True)

    registry = DatasetRegistry(default_registry_path(repo_root))
    dataset_payloads: dict[str, Any] = {}
    markdown_lines = [
        "# Data Audit Summary",
        "",
        "| Dataset | Status | Total | Splits | Missing Images | Missing Annotations | Duplicate IDs | Leakage | Preview |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    detail_sections: list[str] = []

    for dataset_name, registration in sorted(registry._records.items()):  # noqa: SLF001 - local audit view
        manifest_path = Path(registration.manifest_path) if registration.manifest_path else Path()
        dataset_dir = output_root / dataset_name
        manifest_exists = manifest_path.exists()
        rows = read_csv_rows(manifest_path) if manifest_exists else []
        fallback_rows, fallback_note = _fallback_local_rows(dataset_name, registration.local_path, rows)
        effective_rows = fallback_rows if fallback_rows else rows
        sample_rows = [row for row in effective_rows if row.get("record_type") != "placeholder"]
        split_counts = Counter(row.get("split", "all") for row in sample_rows)

        missing_images_count, missing_images = _path_issues(sample_rows, "image_path")
        missing_annotations_count, missing_annotations = _path_issues(sample_rows, "annotation_path")
        broken_manifest_paths, broken_paths = _broken_manifest_paths(sample_rows)

        duplicate_sample_ids_counter = Counter(row.get("sample_id", "").strip().lower() for row in sample_rows if row.get("sample_id"))
        duplicate_sample_ids = [sample_id for sample_id, count in duplicate_sample_ids_counter.items() if count > 1]
        duplicate_images_count, duplicate_images = _duplicates_across_splits(sample_rows, "image_path")
        duplicate_annotations_count, duplicate_annotations = _duplicates_across_splits(sample_rows, "annotation_path")
        leakage_id_count, leakage_by_id, leakage_stem_count, leakage_by_stem = _canonical_leakage(sample_rows)

        issue_examples = {
            "broken_manifest_paths": broken_paths,
            "missing_image_files": missing_images,
            "missing_annotation_files": missing_annotations,
            "duplicate_sample_ids": duplicate_sample_ids,
            "duplicate_image_files_across_splits": duplicate_images,
            "duplicate_annotation_files_across_splits": duplicate_annotations,
            "leakage_by_canonical_id": leakage_by_id,
            "leakage_by_file_stem": leakage_by_stem,
        }
        effective_status = registration.status
        if fallback_rows and effective_status == "metadata-only":
            effective_status = "partial"
            issue_examples["stale_manifest_registry"] = [fallback_note] if fallback_note else []
        _write_dataset_tables(dataset_dir, sample_rows, dict(split_counts), issue_examples)
        preview_total, preview_note = _generate_previews(dataset_name, sample_rows, dataset_dir, preview_count)

        summary = DatasetAuditSummary(
            dataset_name=dataset_name,
            status=effective_status,
            local_path=registration.local_path,
            manifest_path=registration.manifest_path,
            manifest_exists=manifest_exists,
            total_count=len(sample_rows),
            split_names=sorted(split_counts.keys()),
            split_counts=dict(sorted(split_counts.items())),
            missing_image_files=missing_images_count,
            missing_annotation_files=missing_annotations_count,
            broken_manifest_paths=broken_manifest_paths,
            duplicate_sample_ids=len(duplicate_sample_ids),
            duplicate_image_files_across_splits=duplicate_images_count,
            duplicate_annotation_files_across_splits=duplicate_annotations_count,
            leakage_by_canonical_id=leakage_id_count,
            leakage_by_file_stem=leakage_stem_count,
            preview_count=preview_total,
            preview_note=preview_note,
            issue_examples={key: values[:10] for key, values in issue_examples.items() if values},
        )
        dataset_payloads[dataset_name] = asdict(summary)
        split_text = ", ".join(f"{split}:{count}" for split, count in summary.split_counts.items()) or "-"
        markdown_lines.append(
            f"| {dataset_name} | {summary.status} | {summary.total_count} | {split_text} | "
            f"{summary.missing_image_files} | {summary.missing_annotation_files} | {summary.duplicate_sample_ids} | "
            f"{summary.leakage_by_canonical_id + summary.leakage_by_file_stem} | {summary.preview_count} |"
        )
        detail_sections.extend(
            [
                "",
                f"## {dataset_name}",
                "",
                f"- Local path: `{summary.local_path or '-'}`",
                f"- Manifest path: `{summary.manifest_path or '-'}`",
                f"- Status: `{summary.status}`",
                f"- Preview note: {summary.preview_note}",
            ]
        )
        if summary.issue_examples:
            detail_sections.append("- Issue examples:")
            for category, examples in summary.issue_examples.items():
                detail_sections.append(f"  - `{category}`:")
                for example in examples[:5]:
                    detail_sections.append(f"    - `{example}`")
        else:
            detail_sections.append("- Issue examples: none detected in the current partial/local snapshot.")
        detail_sections.append("")

    audit_summary = {
        "datasets": dataset_payloads,
    }
    markdown_lines.extend(detail_sections)
    write_json(output_root / "audit_summary.json", audit_summary)
    (output_root / "audit_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    docs_summary_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    return audit_summary


def rebuild_manifest_if_missing(repo_root: Path, dataset_name: str) -> None:
    """Small helper for ad-hoc audits when a manifest was removed.

    The audit script itself does not mutate manifests by default, but this hook
    is kept available for future troubleshooting.
    """

    registry = DatasetRegistry(default_registry_path(repo_root))
    registration = registry.get(dataset_name)
    if registration is None:
        raise KeyError(f"Dataset {dataset_name} is not registered.")

    manifest_path = Path(registration.manifest_path)
    if manifest_path.exists():
        return

    local_root = Path(registration.local_path)
    if dataset_name == "sen1floods11":
        build_sen1floods11_manifest(local_root, manifest_path)
        return
    raise NotImplementedError(f"Manifest rebuild is not implemented for {dataset_name}.")
