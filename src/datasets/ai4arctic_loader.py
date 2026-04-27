from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ManifestDataset, deserialize_json_field, infer_split_from_parts, list_matching_files, write_csv


def build_ai4arctic_manifest(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    nc_files = list_matching_files(root, ["*.nc", "*.nc4", "*.netcdf"])
    rows: list[dict[str, Any]] = []
    for nc_path in nc_files:
        rows.append(
            {
                "record_type": "sample",
                "dataset": "ai4arctic",
                "sample_id": nc_path.stem,
                "split": infer_split_from_parts(nc_path),
                "image_path": nc_path.resolve().as_posix(),
                "annotation_path": nc_path.resolve().as_posix(),
                "remote_source": "",
                "status": "partial",
                "notes": "",
                "metadata_json": {
                    "embedded_annotations": True,
                    "complex_slc_available": False,
                    "pixel_domain": "ready_to_train_netcdf_or_raw_netcdf",
                    "domain_notes": (
                        "AI4Arctic challenge packages are distributed as netCDF scenes containing "
                        "Sentinel-1-derived inputs plus auxiliary variables and labels, not as complex SLC."
                    ),
                },
            }
        )
    write_csv(manifest_path, rows)
    return rows


class AI4ArcticDataset(ManifestDataset):
    def __init__(self, records: list[dict[str, Any]], *, split: str | None = None, sample_limit: int | None = None) -> None:
        normalized_records: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            row["metadata"] = deserialize_json_field(row.get("metadata_json"))
            normalized_records.append(row)
        super().__init__(normalized_records, split=split, sample_limit=sample_limit)
