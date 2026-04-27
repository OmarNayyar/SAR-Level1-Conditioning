from __future__ import annotations

import json
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import ManifestDataset, deserialize_json_field, infer_split_from_parts, list_matching_files, write_csv


IMAGE_LAYER_PRIORITY = ("S1", "S1Hand", "S1Weak", "S1Perm")
ANNOTATION_LAYER_PRIORITY = ("QC", "LabelHand", "S1OtsuLabelHand", "S1OtsuLabelWeak", "JRCPerm")


def _load_event_metadata(root: Path) -> dict[str, dict[str, Any]]:
    metadata_path = root / "Sen1Floods11_Metadata.geojson"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    events: dict[str, dict[str, Any]] = {}
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        location = properties.get("location")
        if location:
            events[str(location)] = properties
    return events


def _sample_key_from_filename(file_name: str) -> str:
    stem_parts = Path(file_name).stem.split("_")
    if len(stem_parts) < 3:
        return Path(file_name).stem
    return "_".join(stem_parts[:-1])


def _load_split_mapping(root: Path) -> dict[str, str]:
    split_mapping: dict[str, str] = {}
    split_candidates = {
        "train": ["flood_train_data.csv", "train.csv"],
        "val": ["flood_valid_data.csv", "valid.csv", "val.csv"],
        "test": ["flood_test_data.csv", "test.csv"],
    }
    for split_name, file_names in split_candidates.items():
        for file_name in file_names:
            for split_file in list_matching_files(root, [file_name]):
                try:
                    with split_file.open("r", encoding="utf-8", newline="") as handle:
                        for row in csv.reader(handle):
                            if not row:
                                continue
                            sample_key = _sample_key_from_filename(row[0].strip())
                            if sample_key:
                                split_mapping[sample_key] = split_name
                except OSError:
                    continue
    return split_mapping


def _pick_first_existing(layers: dict[str, Path], priorities: tuple[str, ...]) -> str:
    for layer_name in priorities:
        if layer_name in layers:
            return layers[layer_name].as_posix()
    return ""


def build_sen1floods11_manifest(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    layer_files = list_matching_files(root, ["*.tif", "*.tiff"])
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for file_path in layer_files:
        stem_parts = file_path.stem.split("_")
        if len(stem_parts) < 3:
            continue
        layer = stem_parts[-1]
        sample_key = "_".join(stem_parts[:-1])
        groups[sample_key][layer] = file_path.resolve()

    events = _load_event_metadata(root)
    split_mapping = _load_split_mapping(root)
    rows: list[dict[str, Any]] = []
    for sample_key, layers in sorted(groups.items()):
        event_name = sample_key.split("_")[0]
        split = split_mapping.get(sample_key, infer_split_from_parts(next(iter(layers.values()))))
        rows.append(
            {
                "record_type": "sample",
                "dataset": "sen1floods11",
                "sample_id": sample_key,
                "split": split,
                "image_path": _pick_first_existing(layers, IMAGE_LAYER_PRIORITY),
                "annotation_path": _pick_first_existing(layers, ANNOTATION_LAYER_PRIORITY),
                "remote_source": "gs://sen1floods11",
                "status": "partial",
                "notes": "",
                "metadata_json": {
                    "available_layers": sorted(layers.keys()),
                    "event_metadata": events.get(event_name, {}),
                    "primary_image_layer": next((layer for layer in IMAGE_LAYER_PRIORITY if layer in layers), ""),
                    "primary_annotation_layer": next((layer for layer in ANNOTATION_LAYER_PRIORITY if layer in layers), ""),
                    "complex_slc_available": False,
                    "pixel_domain": "log_db",
                    "domain_notes": (
                        "The official Sen1Floods11 README describes the Sentinel-1 layers as GRD imagery in dB "
                        "with VV/VH bands. The public dataset is distributed as GeoTIFF chips, not complex SLC."
                    ),
                },
            }
        )
    write_csv(manifest_path, rows)
    return rows


class Sen1Floods11Dataset(ManifestDataset):
    def __init__(self, records: list[dict[str, Any]], *, split: str | None = None, sample_limit: int | None = None) -> None:
        normalized_records: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            row["metadata"] = deserialize_json_field(row.get("metadata_json"))
            normalized_records.append(row)
        super().__init__(normalized_records, split=split, sample_limit=sample_limit)
