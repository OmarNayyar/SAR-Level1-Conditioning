from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .common import ManifestDataset, deserialize_json_field, infer_split_from_parts, list_matching_files, write_csv


SSDD_LAYOUT_CANDIDATES = (
    Path("BBox_RBox_PSeg_SSDD") / "voc_style",
    Path("BBox_SSDD") / "voc_style",
    Path("RBox_SSDD") / "voc_style",
    Path("PSeg_SSDD") / "voc_style",
    Path("voc_style"),
)


def _collect_split_mapping(root: Path) -> dict[str, str]:
    split_mapping: dict[str, str] = {}
    for split_name in ("train", "val", "test", "trainval"):
        split_files = list_matching_files(root, [f"{split_name}.txt"])
        for split_file in split_files:
            for line in split_file.read_text(encoding="utf-8").splitlines():
                sample_name = line.strip()
                if sample_name:
                    split_mapping[sample_name] = "val" if split_name == "trainval" else split_name
    return split_mapping


def resolve_ssdd_root(root: Path) -> Path:
    if (root / "Annotations").exists() and (root / "JPEGImages").exists():
        return root

    for relative_path in SSDD_LAYOUT_CANDIDATES:
        candidate = root / relative_path
        if (candidate / "Annotations").exists() and (candidate / "JPEGImages").exists():
            return candidate

    return root


def _parse_rotated_box(rotated_box_node: ET.Element) -> dict[str, Any]:
    payload = {
        "cx": rotated_box_node.findtext("cx") or rotated_box_node.findtext("rotated_bbox_cx", default=""),
        "cy": rotated_box_node.findtext("cy") or rotated_box_node.findtext("rotated_bbox_cy", default=""),
        "w": rotated_box_node.findtext("w") or rotated_box_node.findtext("rotated_bbox_w", default=""),
        "h": rotated_box_node.findtext("h") or rotated_box_node.findtext("rotated_bbox_h", default=""),
        "angle": rotated_box_node.findtext("angle") or rotated_box_node.findtext("rotated_bbox_theta", default=""),
    }
    corners: list[dict[str, str]] = []
    for point_index in range(1, 5):
        x_value = rotated_box_node.findtext(f"x{point_index}", default="")
        y_value = rotated_box_node.findtext(f"y{point_index}", default="")
        if x_value or y_value:
            corners.append({"x": x_value, "y": y_value})
    if corners:
        payload["corners"] = corners
    return payload


def _parse_segmentation(segmentation_node: ET.Element) -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    for child in segmentation_node:
        raw_text = (child.text or "").strip()
        if not raw_text:
            continue
        if "," in raw_text:
            x_value, y_value = [token.strip() for token in raw_text.split(",", maxsplit=1)]
            points.append({"x": x_value, "y": y_value})
        else:
            points.append({"value": raw_text})
    return points


def parse_voc_annotation(annotation_path: Path) -> dict[str, Any]:
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    filename = root.findtext("filename", default=f"{annotation_path.stem}.jpg")
    size_node = root.find("size")
    width = size_node.findtext("width", default="") if size_node is not None else ""
    height = size_node.findtext("height", default="") if size_node is not None else ""
    objects: list[dict[str, Any]] = []
    for object_node in root.findall("object"):
        bbox_node = object_node.find("bndbox")
        rotated_box_node = object_node.find("robndbox") or object_node.find("rotated_bndbox")
        segmentation_node = object_node.find("segm")
        object_payload: dict[str, Any] = {
            "name": object_node.findtext("name", default="ship"),
            "difficult": object_node.findtext("difficult", default="0"),
        }
        if bbox_node is not None:
            object_payload["bbox"] = {
                "xmin": bbox_node.findtext("xmin", default=""),
                "ymin": bbox_node.findtext("ymin", default=""),
                "xmax": bbox_node.findtext("xmax", default=""),
                "ymax": bbox_node.findtext("ymax", default=""),
            }
            bbox_width = bbox_node.findtext("bbox_w", default="")
            bbox_height = bbox_node.findtext("bbox_h", default="")
            if bbox_width or bbox_height:
                object_payload["bbox"]["bbox_w"] = bbox_width
                object_payload["bbox"]["bbox_h"] = bbox_height
        if rotated_box_node is not None:
            object_payload["robndbox"] = _parse_rotated_box(rotated_box_node)
        if segmentation_node is not None:
            object_payload["segmentation"] = _parse_segmentation(segmentation_node)
        objects.append(object_payload)
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def _resolve_image_path(root: Path, image_file_name: str, sample_id: str) -> Path:
    candidates = [
        root / "JPEGImages" / image_file_name,
        root / "images" / image_file_name,
        root / "Images" / image_file_name,
        root / f"{sample_id}.jpg",
        root / f"{sample_id}.png",
        root / f"{sample_id}.bmp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _collect_annotation_files(root: Path) -> list[Path]:
    canonical_dirs = [
        root / "Annotations",
        root / "annotations",
    ]
    for annotation_dir in canonical_dirs:
        if annotation_dir.exists():
            return sorted(annotation_dir.glob("*.xml"))
    return list_matching_files(root, ["*.xml"])


def build_voc_manifest(root: Path, manifest_path: Path, *, dataset_name: str) -> list[dict[str, Any]]:
    root = root.resolve()
    split_mapping = _collect_split_mapping(root)
    annotation_files = _collect_annotation_files(root)
    rows: list[dict[str, Any]] = []

    for annotation_path in annotation_files:
        parsed = parse_voc_annotation(annotation_path)
        sample_id = annotation_path.stem
        split = split_mapping.get(sample_id, infer_split_from_parts(annotation_path))
        image_path = _resolve_image_path(root, parsed["filename"], sample_id)
        object_names = sorted({obj["name"] for obj in parsed["objects"]})
        rows.append(
            {
                "record_type": "sample",
                "dataset": dataset_name,
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
                    "pixel_domain": "unknown_detected_image_chip",
                    "annotation_root": root.as_posix(),
                    "domain_notes": (
                        "Public ship-detection benchmarks are distributed as detected image chips; "
                        "complex SLC is not present in the benchmark package."
                    ),
                },
            }
        )

    write_csv(manifest_path, rows)
    return rows


class SSDDDataset(ManifestDataset):
    def __init__(self, records: list[dict[str, Any]], *, split: str | None = None, sample_limit: int | None = None) -> None:
        normalized_records: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            row["metadata"] = deserialize_json_field(row.get("metadata_json"))
            normalized_records.append(row)
        super().__init__(normalized_records, split=split, sample_limit=sample_limit)


def build_ssdd_manifest(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    return build_voc_manifest(resolve_ssdd_root(root), manifest_path, dataset_name="ssdd")
