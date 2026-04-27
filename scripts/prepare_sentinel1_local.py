from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.common import ensure_data_layout, read_csv_rows, write_csv
from src.datasets.registry import DatasetRegistration, DatasetRegistry, default_registry_path
from src.datasets.sentinel1_loader import prepare_sentinel1_record


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("prepare_sentinel1_local")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local Sentinel-1 SAFE content so Bundle A can consume measurement TIFFs."
    )
    parser.add_argument(
        "--manifest",
        help="Explicit Sentinel-1 manifest path. Defaults to data/external/manifests/sentinel1_manifest.csv.",
    )
    parser.add_argument("--product-family", choices=["GRD", "SLC"], help="Only prepare one Sentinel-1 family.")
    parser.add_argument("--product-id", action="append", help="Only prepare one or more specific product IDs.")
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Re-extract local SAFE support files even when the prepared cache already exists.",
    )
    return parser.parse_args()


def _manifest_path_from_layout(layout: dict[str, Path | bool]) -> Path:
    return Path(layout["manifests"]) / "sentinel1_manifest.csv"


def _row_matches(row: dict[str, str], *, product_family: str | None, requested_ids: set[str]) -> bool:
    if row.get("record_type") == "placeholder":
        return False
    if product_family and str(row.get("product_family", "")).upper() != product_family:
        return False
    if requested_ids and row.get("product_id") not in requested_ids:
        return False
    return True


def _merge_notes(existing: str, incoming: str) -> str:
    values: list[str] = []
    for candidate in (existing, incoming):
        value = candidate.strip() if candidate else ""
        if value and value not in values:
            values.append(value)
    return " ".join(values)


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    layout = ensure_data_layout(REPO_ROOT)
    manifest_path = Path(args.manifest).resolve() if args.manifest else _manifest_path_from_layout(layout)
    rows = read_csv_rows(manifest_path)
    requested_ids = set(args.product_id or [])

    prepared_count = 0
    failed_count = 0
    local_count = 0
    for row in rows:
        if not _row_matches(row, product_family=args.product_family, requested_ids=requested_ids):
            continue
        local_count += 1
        prepared = prepare_sentinel1_record(
            row,
            repo_root=REPO_ROOT,
            force_reextract=args.force_reextract,
        )
        updates = prepared.manifest_updates()
        row.update({key: str(value) if isinstance(value, Path) else value for key, value in updates.items()})
        row["notes"] = _merge_notes(row.get("notes", ""), prepared.notes)
        row["prepared_status"] = "ready" if prepared.usable else "failed"
        if prepared.usable and prepared.image_path is not None:
            row["image_path"] = prepared.image_path.resolve().as_posix()
            if row.get("status") != "complete":
                row["status"] = "partial"
            prepared_count += 1
            logger.info(
                "%s | ready | %s",
                row.get("product_id") or row.get("sample_id"),
                prepared.image_path.resolve().as_posix(),
            )
        else:
            failed_count += 1
            logger.warning(
                "%s | unusable | %s",
                row.get("product_id") or row.get("sample_id"),
                prepared.notes,
            )

    write_csv(manifest_path, rows)
    registry = DatasetRegistry(default_registry_path(REPO_ROOT))
    registry.upsert(
        DatasetRegistration(
            dataset_name="sentinel1",
            manifest_path=manifest_path.resolve().as_posix(),
            local_path=(Path(layout["raw"]) / "sentinel1").resolve().as_posix(),
            remote_source="https://dataspace.copernicus.eu/",
            notes=(
                "Local Sentinel-1 SAFE products can be prepared into data/interim/sentinel1/prepared "
                "to expose measurement TIFFs and calibration metadata for Bundle A."
            ),
            status="partial" if prepared_count > 0 else "metadata-only",
            sample_count=prepared_count,
        )
    )
    registry.save()
    logger.info(
        "Sentinel-1 local preparation complete. matched=%s ready=%s failed=%s manifest=%s",
        local_count,
        prepared_count,
        failed_count,
        manifest_path,
    )


if __name__ == "__main__":
    main()
