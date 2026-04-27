from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.common import ensure_data_layout, ensure_storage_guard, read_csv_rows, write_csv
from src.datasets.registry import DatasetRegistration, DatasetRegistry, default_registry_path
from src.datasets.sentinel1_catalog import (
    Sentinel1Query,
    merge_manifest_row,
    product_target_path,
    products_to_manifest_rows,
    query_from_mapping,
    save_search_outputs,
    search_sentinel1_products,
)
from src.datasets.sentinel1_fetch import CDSEAuth, download_sentinel1_product, product_download_estimates


def catalog_output_path(layout: dict[str, Path | bool], dataset_name: str, file_name: str) -> Path:
    catalogs_root = Path(layout["catalogs"])
    if bool(layout.get("catalogs_flat")):
        return catalogs_root / f"{dataset_name}__{file_name}"
    dataset_catalog_dir = catalogs_root / dataset_name
    try:
        dataset_catalog_dir.mkdir(parents=True, exist_ok=True)
        return dataset_catalog_dir / file_name
    except PermissionError:
        return catalogs_root / f"{dataset_name}__{file_name}"


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fetch_sentinel1_subset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("query", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search and selectively fetch tiny Sentinel-1 subsets from CDSE.")
    parser.add_argument("--config", help="YAML config file with query defaults.")
    parser.add_argument("--product-type", help="Broad product family (SLC/GRD) or exact CDSE productType value.")
    parser.add_argument("--mode", help="Acquisition mode such as IW or EW.")
    parser.add_argument("--start", help="Inclusive start date/time, e.g. 2024-01-01 or 2024-01-01T00:00:00Z.")
    parser.add_argument("--end", help="Inclusive end date/time, e.g. 2024-01-07 or 2024-01-07T23:59:59Z.")
    parser.add_argument("--bbox", help="Bounding box: min_lon,min_lat,max_lon,max_lat.")
    parser.add_argument("--polygon", help="Polygon in WKT or lon,lat;lon,lat;... form.")
    parser.add_argument("--max-results", type=int, help="Maximum number of catalog results to keep.")
    parser.add_argument("--metadata-only", action="store_true", help="Search and save manifests without downloading products.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve a download plan without downloading archives.")
    parser.add_argument("--download-count", type=int, default=0, help="Download the first N matched products.")
    parser.add_argument("--include-auxiliary", action="store_true", help="Include AUX / PREORB / OPOD / EOF products in the result set.")
    parser.add_argument("--product-id", action="append", help="Exact product IDs to download from the matched result set.")
    parser.add_argument("--force", action="store_true", help="Allow large or unknown-size downloads.")
    return parser.parse_args()


def merge_query(config_payload: dict[str, Any], args: argparse.Namespace) -> Sentinel1Query:
    requested_download = bool(args.product_id) or args.download_count > 0
    if args.metadata_only:
        metadata_only = True
    elif requested_download:
        metadata_only = False
    else:
        metadata_only = bool(config_payload.get("metadata_only", True))
    merged = {
        "product_type": args.product_type or config_payload.get("product_type"),
        "mode": args.mode or config_payload.get("mode"),
        "start": args.start or config_payload.get("start"),
        "end": args.end or config_payload.get("end"),
        "bbox": args.bbox or config_payload.get("bbox"),
        "polygon": args.polygon or config_payload.get("polygon"),
        "max_results": args.max_results or config_payload.get("max_results", 10),
        "metadata_only": metadata_only,
        "order_desc": bool(config_payload.get("order_desc", True)),
        "include_auxiliary": bool(args.include_auxiliary or config_payload.get("include_auxiliary", False)),
    }
    return query_from_mapping(merged)


def select_products(
    products,
    *,
    requested_ids: set[str],
    download_count: int,
):
    if requested_ids:
        return [product for product in products if product.product_id in requested_ids]
    if download_count > 0:
        return products[:download_count]
    return []


def log_selected_products(logger: logging.Logger, products, *, sentinel1_root: Path, dry_run: bool) -> None:
    if not products:
        logger.info("No Sentinel-1 products were selected after final filtering.")
        return
    logger.info("Final filtered Sentinel-1 product list%s:", " (dry-run)" if dry_run else "")
    for index, product in enumerate(products, start=1):
        logger.info(
            "[%s] %s | %s | %s | %s | %s",
            index,
            product.product_id,
            product.product_type,
            product.mode,
            product.content_length or "unknown-bytes",
            product_target_path(product, sentinel1_root),
        )


def main() -> None:
    args = parse_args()
    layout = ensure_data_layout(REPO_ROOT)
    logger = configure_logging(Path(layout["logs"]) / "fetch_sentinel1_subset.log")
    registry = DatasetRegistry(default_registry_path(REPO_ROOT))
    raw_root = Path(layout["raw"])
    manifests_root = Path(layout["manifests"])

    config_path = Path(args.config).resolve() if args.config else None
    config_payload = load_config(config_path)
    query = merge_query(config_payload, args)

    products = search_sentinel1_products(query)
    sentinel1_root = raw_root / "sentinel1"
    json_path = catalog_output_path(layout, "sentinel1", "search_results.json")
    manifest_path = manifests_root / "sentinel1_manifest.csv"
    save_search_outputs(products, json_path=json_path, manifest_path=manifest_path, sentinel1_root=sentinel1_root)
    logger.info("Saved %s catalog results to %s", len(products), json_path)

    requested_ids = set(args.product_id or [])
    selected_products = select_products(
        products,
        requested_ids=requested_ids,
        download_count=args.download_count,
    )
    log_selected_products(logger, selected_products, sentinel1_root=sentinel1_root, dry_run=args.dry_run)

    manifest_rows = products_to_manifest_rows(products, sentinel1_root=sentinel1_root)
    existing_manifest_rows = read_csv_rows(manifest_path) if manifest_path.exists() else []
    rows_by_product_id = {
        row["product_id"]: dict(row)
        for row in existing_manifest_rows
        if str(row.get("product_id", "")).strip()
    }
    for row in manifest_rows:
        product_id = row["product_id"]
        if product_id in rows_by_product_id:
            rows_by_product_id[product_id] = merge_manifest_row(rows_by_product_id[product_id], row)
        else:
            rows_by_product_id[product_id] = dict(row)

    if selected_products and not query.metadata_only:
        warnings = ensure_storage_guard(product_download_estimates(selected_products), raw_root, force=args.force)
        for warning in warnings:
            logger.info(warning)

        auth = CDSEAuth(
            username=os.getenv("CDSE_USERNAME"),
            password=os.getenv("CDSE_PASSWORD"),
            access_token=os.getenv("CDSE_ACCESS_TOKEN"),
        )
        for product in selected_products:
            row = rows_by_product_id[product.product_id]
            destination = product_target_path(product, sentinel1_root)
            row["local_target_path"] = destination.resolve().as_posix()
            try:
                downloaded_path = download_sentinel1_product(
                    product,
                    destination,
                    auth=auth,
                    dry_run=args.dry_run,
                    force=args.force,
                )
                row["image_path"] = downloaded_path.resolve().as_posix()
                row["status"] = "partial" if args.dry_run else "complete"
                row["download_status"] = "planned" if args.dry_run else "complete"
                row["notes"] = "dry-run only" if args.dry_run else ""
                logger.info("%s planned/downloaded to %s", product.product_id, downloaded_path)
            except Exception as exc:
                row["status"] = "failed"
                row["download_status"] = "failed"
                row["notes"] = str(exc)
                logger.error("Failed to fetch %s (%s): %s", product.product_id, product.name, exc)
        write_csv(manifest_path, list(rows_by_product_id.values()))

    registry.upsert(
        DatasetRegistration(
            dataset_name="sentinel1",
            manifest_path=manifest_path.resolve().as_posix(),
            local_path=(raw_root / "sentinel1").resolve().as_posix(),
            remote_source="https://dataspace.copernicus.eu/",
            notes=(
                "Sentinel-1 catalog metadata stored locally. "
                "SLC products are marked as complex_slc; GRD products are marked as detected_ground_range."
            ),
            status="partial" if selected_products else "metadata-only",
            sample_count=len(selected_products),
        )
    )
    registry.save()
    logger.info("Updated dataset registry at %s", default_registry_path(REPO_ROOT))


if __name__ == "__main__":
    main()
