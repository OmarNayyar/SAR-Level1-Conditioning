from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ManifestDataset, deserialize_json_field, infer_split_from_parts, list_matching_files, write_csv
from .ssdd_loader import parse_voc_annotation


class LSSSDDDataset(ManifestDataset):
    def __init__(self, records: list[dict[str, Any]], *, split: str | None = None, sample_limit: int | None = None) -> None:
        normalized_records: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            row["metadata"] = deserialize_json_field(row.get("metadata_json"))
            normalized_records.append(row)
        super().__init__(normalized_records, split=split, sample_limit=sample_limit)


def resolve_ls_ssdd_root(root: Path) -> Path:
    root = root.resolve()
    direct_candidates = [root, root / "LS-SSDD-v1.0", root / "LS-SSDD-v1.0-OPEN"]
    direct_candidates.extend(candidate for candidate in root.iterdir() if candidate.is_dir())

    for candidate in direct_candidates:
        if (candidate / "Annotations_sub").exists() and (candidate / "JPEGImages_sub").exists():
            return candidate
    return root


def _collect_split_mapping(root: Path) -> dict[str, str]:
    split_mapping: dict[str, str] = {}
    split_dir_candidates = [
        root / "ImageSets" / "Main",
        root / "ImageSets",
    ]
    for split_dir in split_dir_candidates:
        if not split_dir.exists():
            continue
        for split_name in ("train", "val", "test", "trainval", "test_inshore", "test_offshore"):
            split_file = split_dir / f"{split_name}.txt"
            if not split_file.exists():
                continue
            normalized_split = "val" if split_name == "trainval" else ("test" if split_name.startswith("test") else split_name)
            for line in split_file.read_text(encoding="utf-8").splitlines():
                sample_name = line.strip()
                if sample_name:
                    split_mapping[sample_name] = normalized_split
    return split_mapping


def _collect_annotation_files(root: Path) -> list[Path]:
    annotation_dir_candidates = [
        root / "Annotations_sub",
        root / "Annotations",
    ]
    for annotation_dir in annotation_dir_candidates:
        if annotation_dir.exists():
            return sorted(annotation_dir.glob("*.xml"))
    return list_matching_files(root, ["*.xml"])


def _resolve_image_path(root: Path, image_file_name: str, sample_id: str) -> Path:
    candidates = [
        root / "JPEGImages_sub" / image_file_name,
        root / "JPEGImages_sub" / "JPEGImages_sub_train" / image_file_name,
        root / "JPEGImages_sub" / "JPEGImages_sub_test" / image_file_name,
        root / "JPEGImages_sub" / f"{sample_id}.jpg",
        root / "JPEGImages_sub" / f"{sample_id}.png",
        root / "JPEGImages" / image_file_name,
        root / "JPEGImages" / f"{sample_id}.jpg",
        root / "JPEGImages_VH" / image_file_name,
        root / f"{sample_id}.jpg",
        root / f"{sample_id}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def build_ls_ssdd_manifest(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    dataset_root = resolve_ls_ssdd_root(root)
    split_mapping = _collect_split_mapping(dataset_root)
    annotation_files = _collect_annotation_files(dataset_root)
    rows: list[dict[str, Any]] = []

    for annotation_path in annotation_files:
        parsed = parse_voc_annotation(annotation_path)
        sample_id = annotation_path.stem
        split = split_mapping.get(sample_id, infer_split_from_parts(annotation_path))
        image_path = _resolve_image_path(dataset_root, parsed["filename"], sample_id)
        object_names = sorted({obj["name"] for obj in parsed["objects"]})
        rows.append(
            {
                "record_type": "sample",
                "dataset": "ls_ssdd",
                "sample_id": sample_id,
                "split": split,
                "image_path": image_path.as_posix(),
                "annotation_path": annotation_path.resolve().as_posix(),
                "remote_source": "",
                "status": "partial",
                "notes": "",
                "width": parsed["width"],
                "height": parsed["height"],
                "annotation_count": len(parsed["objects"]),
                "metadata_json": {
                    "categories": object_names,
                    "has_rotated_boxes": any("robndbox" in obj for obj in parsed["objects"]),
                    "has_polygons": any("segmentation" in obj for obj in parsed["objects"]),
                    "complex_slc_available": False,
                    "pixel_domain": "detected_image_chip",
                    "annotation_root": (dataset_root / "Annotations_sub").resolve().as_posix()
                    if (dataset_root / "Annotations_sub").exists()
                    else dataset_root.resolve().as_posix(),
                    "source_sensor": "Sentinel-1",
                    "domain_notes": (
                        "LS-SSDD-v1.0 is distributed as detected image chips derived from large-scale Sentinel-1 "
                        "scenes. The public package does not include complex SLC."
                    ),
                },
            }
        )

    write_csv(manifest_path, rows)
    return rows
