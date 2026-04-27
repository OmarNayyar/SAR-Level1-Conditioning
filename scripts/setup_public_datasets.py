from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.ai4arctic_loader import build_ai4arctic_manifest
from src.datasets.common import (
    DatasetError,
    DatasetStatus,
    StorageEstimate,
    create_directory_link,
    ensure_data_layout,
    ensure_storage_guard,
    placeholder_manifest_row,
    resolve_external_dataset_path,
    to_posix_path,
    write_csv,
    write_json,
)
from src.datasets.hrsid_loader import build_hrsid_manifest
from src.datasets.ls_ssdd_loader import build_ls_ssdd_manifest
from src.datasets.registry import DatasetRegistration, DatasetRegistry, default_registry_path
from src.datasets.sen1floods11_loader import build_sen1floods11_manifest
from src.datasets.ssdd_loader import build_ssdd_manifest


ManifestBuilder = Callable[[Path, Path], list[dict[str, Any]]]

SEN1FLOODS11_HTTP_ROOT = "https://storage.googleapis.com/sen1floods11/v1.1"
SEN1FLOODS11_SPLIT_URLS = {
    "train": f"{SEN1FLOODS11_HTTP_ROOT}/splits/flood_handlabeled/flood_train_data.csv",
    "val": f"{SEN1FLOODS11_HTTP_ROOT}/splits/flood_handlabeled/flood_valid_data.csv",
    "test": f"{SEN1FLOODS11_HTTP_ROOT}/splits/flood_handlabeled/flood_test_data.csv",
}
SEN1FLOODS11_LAYER_URLS = {
    "S1Hand": f"{SEN1FLOODS11_HTTP_ROOT}/data/flood_events/HandLabeled/S1Hand",
    "LabelHand": f"{SEN1FLOODS11_HTTP_ROOT}/data/flood_events/HandLabeled/LabelHand",
}
DEFAULT_SEN1FLOODS11_SAMPLE_COUNT = 8


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


def serializable_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in spec.items()
        if not callable(value)
    }


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "hrsid": {
        "display_name": "HRSID",
        "local_dir": "hrsid",
        "manifest_builder": build_hrsid_manifest,
        "remote_source": "https://github.com/chaozhong2010/HRSID",
        "notes": (
            "Primary HRSID archives are served through Google Drive/Baidu and are not downloaded by default. "
            "Use --force to allow direct Google Drive fetch attempts or register an external local copy."
        ),
        "default_artifacts": [],
        "sample_artifacts": [
            {
                "name": "hrsid_background_testset",
                "type": "gdrive",
                "file_id": "1U0Sj1SHoq-2VjXXUKwpXae6rBI3YjyDP",
                "dest_name": "hrsid_background_testset.zip",
                "size_bytes": None,
                "source_url": "https://drive.google.com/file/d/1U0Sj1SHoq-2VjXXUKwpXae6rBI3YjyDP/view?usp=sharing",
            }
        ],
        "force_artifacts": [
            {
                "name": "hrsid_jpg_full",
                "type": "gdrive",
                "file_id": "1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms",
                "dest_name": "hrsid_jpg_full.zip",
                "size_bytes": None,
                "source_url": "https://drive.google.com/file/d/1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms/view?usp=sharing",
            }
        ],
    },
    "ssdd": {
        "display_name": "SSDD",
        "local_dir": "ssdd",
        "manifest_builder": build_ssdd_manifest,
        "remote_source": "https://github.com/TianwenZhang0825/Official-SSDD",
        "notes": (
            "The official SSDD release points to Google Drive/Baidu mirrors. Downloads are blocked by default "
            "because the archive size is not published; use --force to allow an attempted Google Drive fetch. "
            "If you manually extract the official package, the setup flow prefers BBox_RBox_PSeg_SSDD/voc_style "
            "as the canonical SSDD root because it keeps bbox, rotated-box, and polygon annotations in one copy."
        ),
        "default_artifacts": [],
        "sample_artifacts": [],
        "force_artifacts": [
            {
                "name": "ssdd_full",
                "type": "gdrive",
                "file_id": "1glNJUGotrbEyk43twwB9556AdngJsynZ",
                "dest_name": "ssdd_full.zip",
                "size_bytes": None,
                "source_url": "https://drive.google.com/file/d/1glNJUGotrbEyk43twwB9556AdngJsynZ/view?usp=sharing",
            }
        ],
    },
    "ls_ssdd": {
        "display_name": "LS-SSDD-v1.0",
        "local_dir": "ls_ssdd",
        "manifest_builder": build_ls_ssdd_manifest,
        "remote_source": "https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN",
        "notes": (
            "LS-SSDD-v1.0 is documented on GitHub but currently distributed through the radars.ac.cn portal. "
            "This setup script creates catalogs, manifests, and registration hooks, but leaves the actual data "
            "download as a manual step. After manual extraction under data/raw/ls_ssdd/, the loader prefers "
            "the official Annotations_sub and JPEGImages_sub chip layout."
        ),
        "default_artifacts": [],
        "sample_artifacts": [],
        "force_artifacts": [],
    },
    "sen1floods11": {
        "display_name": "Sen1Floods11",
        "local_dir": "sen1floods11",
        "manifest_builder": build_sen1floods11_manifest,
        "remote_source": "gs://sen1floods11",
        "notes": (
            "The full Sen1Floods11 bucket is about 14 GB. This script defaults to metadata-only mode, downloads "
            "the event metadata GeoJSON by default, supports a tiny exact hand-labeled subset in --sample-only mode, "
            "and requires --force before any large bucket sync."
        ),
        "default_artifacts": [
            {
                "name": "sen1floods11_metadata",
                "type": "http",
                "url": "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/Sen1Floods11_Metadata.geojson",
                "dest_name": "Sen1Floods11_Metadata.geojson",
                "size_bytes": 2 * 1024 * 1024,
                "source_url": "https://github.com/cloudtostreet/Sen1Floods11",
            }
        ],
        "sample_artifacts": [
            {
                "name": "sen1floods11_flood_train_split",
                "type": "http",
                "url": SEN1FLOODS11_SPLIT_URLS["train"],
                "dest_name": "v1.1/splits/flood_handlabeled/flood_train_data.csv",
                "size_bytes": 16 * 1024,
                "source_url": "https://github.com/cloudtostreet/Sen1Floods11",
            },
            {
                "name": "sen1floods11_flood_valid_split",
                "type": "http",
                "url": SEN1FLOODS11_SPLIT_URLS["val"],
                "dest_name": "v1.1/splits/flood_handlabeled/flood_valid_data.csv",
                "size_bytes": 8 * 1024,
                "source_url": "https://github.com/cloudtostreet/Sen1Floods11",
            },
            {
                "name": "sen1floods11_flood_test_split",
                "type": "http",
                "url": SEN1FLOODS11_SPLIT_URLS["test"],
                "dest_name": "v1.1/splits/flood_handlabeled/flood_test_data.csv",
                "size_bytes": 8 * 1024,
                "source_url": "https://github.com/cloudtostreet/Sen1Floods11",
            },
        ],
        "force_artifacts": [],
        "full_bucket_gsutil": {
            "bucket": "gs://sen1floods11",
            "approx_size_bytes": 14 * 1024**3,
        },
    },
    "ai4arctic": {
        "display_name": "AI4Arctic / ASIP Sea Ice Dataset",
        "local_dir": "ai4arctic",
        "manifest_builder": build_ai4arctic_manifest,
        "remote_source": "https://data.dtu.dk/collections/AI4Arctic_Sea_Ice_Challenge_Dataset/6244065",
        "notes": (
            "AI4Arctic is intentionally metadata-only by default. The official datasets are very large "
            "(~56.08 GB ready-to-train, ~243.04 GB raw) and hosted on the DTU data portal, so this script "
            "does not bulk-download them automatically."
        ),
        "default_artifacts": [],
        "sample_artifacts": [],
        "force_artifacts": [],
        "reference_sizes": {
            "ready_to_train_gb": 56.08,
            "raw_gb": 243.04,
            "ready_to_train_test_gb": 2.37,
        },
    },
}


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("setup_public_datasets")
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


def download_http_file(url: str, destination: Path, *, timeout_seconds: int = 120) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def download_google_drive_file(file_id: str, destination: Path, *, timeout_seconds: int = 120) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    response = session.get(base_url, params={"id": file_id}, stream=True, timeout=timeout_seconds)
    response.raise_for_status()

    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break
    if confirm_token:
        response.close()
        response = session.get(
            base_url,
            params={"id": file_id, "confirm": confirm_token},
            stream=True,
            timeout=timeout_seconds,
        )
        response.raise_for_status()

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    try:
        with destination.open("rb") as handle:
            head = handle.read(512)
        if head.lstrip().startswith(b"<!DOCTYPE html") or b"Google Drive" in head:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "Google Drive returned an HTML warning/landing page instead of the dataset archive. "
                "Please download this dataset manually in a browser, then register the extracted folder."
            )
    finally:
        response.close()
    return destination


def probe_http_content_length(url: str, *, timeout_seconds: int = 60) -> int | None:
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        header_value = response.headers.get("Content-Length")
        if header_value and header_value.isdigit():
            return int(header_value)
    return None


def attempt_gcs_sync(bucket: str, destination: Path, *, logger: logging.Logger) -> bool:
    destination.mkdir(parents=True, exist_ok=True)
    gcloud = shutil.which("gcloud")
    if gcloud:
        result = subprocess.run(
            [gcloud, "storage", "rsync", "--recursive", bucket, str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        logger.warning("gcloud storage rsync failed for %s: %s", bucket, result.stderr.strip() or result.stdout.strip())

    gsutil = shutil.which("gsutil")
    if gsutil:
        result = subprocess.run(
            [gsutil, "-m", "rsync", "-r", bucket, str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        logger.warning("gsutil rsync failed for %s: %s", bucket, result.stderr.strip() or result.stdout.strip())

    if not gcloud and not gsutil:
        logger.warning("Neither gcloud nor gsutil is installed; skipping bucket sync for %s", bucket)
    return False


def _read_sen1floods11_split_rows(split_csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with split_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            left = row[0].strip()
            right = row[1].strip()
            if left and right:
                rows.append((left, right))
    return rows


def ensure_sen1floods11_split_csvs(local_root: Path, *, logger: logging.Logger) -> dict[str, Path]:
    split_root = local_root / "v1.1" / "splits" / "flood_handlabeled"
    split_root.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    for split_name, url in SEN1FLOODS11_SPLIT_URLS.items():
        destination = split_root / Path(url).name
        if not destination.exists():
            download_http_file(url, destination)
            logger.info("sen1floods11 downloaded split file %s to %s", split_name, destination)
        split_paths[split_name] = destination
    return split_paths


def plan_sen1floods11_handlabeled_sample(
    local_root: Path,
    *,
    sample_count: int,
    logger: logging.Logger,
) -> tuple[list[StorageEstimate], list[tuple[str, str, Path, Path]]]:
    split_paths = ensure_sen1floods11_split_csvs(local_root, logger=logger)
    planned_files: list[tuple[str, str, Path, Path]] = []
    seen_sample_ids: set[str] = set()
    split_rows = {
        split_name: _read_sen1floods11_split_rows(split_paths[split_name])
        for split_name in ("train", "val", "test")
    }
    split_indices = {split_name: 0 for split_name in split_rows}

    while len(planned_files) < sample_count:
        added_this_round = False
        for split_name in ("train", "val", "test"):
            rows = split_rows[split_name]
            index = split_indices[split_name]
            while index < len(rows):
                s1_name, label_name = rows[index]
                index += 1
                sample_id = Path(s1_name).stem.rsplit("_", maxsplit=1)[0]
                if sample_id in seen_sample_ids:
                    continue
                seen_sample_ids.add(sample_id)
                s1_dest = local_root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand" / s1_name
                label_dest = local_root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand" / label_name
                planned_files.append((s1_name, label_name, s1_dest, label_dest))
                added_this_round = True
                break
            split_indices[split_name] = index
            if len(planned_files) >= sample_count:
                break
        if not added_this_round:
            break

    estimates: list[StorageEstimate] = []
    for s1_name, label_name, _, _ in planned_files:
        estimates.append(
            StorageEstimate(
                description=s1_name,
                size_bytes=probe_http_content_length(f"{SEN1FLOODS11_LAYER_URLS['S1Hand']}/{s1_name}"),
                source=f"{SEN1FLOODS11_LAYER_URLS['S1Hand']}/{s1_name}",
            )
        )
        estimates.append(
            StorageEstimate(
                description=label_name,
                size_bytes=probe_http_content_length(f"{SEN1FLOODS11_LAYER_URLS['LabelHand']}/{label_name}"),
                source=f"{SEN1FLOODS11_LAYER_URLS['LabelHand']}/{label_name}",
            )
        )
    logger.info("sen1floods11 planned %s hand-labeled sample pairs", len(planned_files))
    return estimates, planned_files


def download_sen1floods11_handlabeled_sample(
    local_root: Path,
    *,
    sample_count: int,
    force: bool,
    logger: logging.Logger,
) -> int:
    estimates, planned_files = plan_sen1floods11_handlabeled_sample(local_root, sample_count=sample_count, logger=logger)
    warnings = ensure_storage_guard(estimates, local_root.parent, force=force)
    for warning in warnings:
        logger.info("sen1floods11 | %s", warning)

    downloaded_pairs = 0
    for s1_name, label_name, s1_dest, label_dest in planned_files:
        if not s1_dest.exists():
            download_http_file(f"{SEN1FLOODS11_LAYER_URLS['S1Hand']}/{s1_name}", s1_dest)
            logger.info("sen1floods11 downloaded sample S1 chip %s", s1_name)
        if not label_dest.exists():
            download_http_file(f"{SEN1FLOODS11_LAYER_URLS['LabelHand']}/{label_name}", label_dest)
            logger.info("sen1floods11 downloaded sample label %s", label_name)
        downloaded_pairs += 1
    return downloaded_pairs


def choose_artifacts(spec: dict[str, Any], *, sample_only: bool, force: bool) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    artifacts.extend(spec.get("default_artifacts", []))
    if sample_only:
        artifacts.extend(spec.get("sample_artifacts", []))
    if force:
        artifacts.extend(spec.get("force_artifacts", []))
    deduped: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        deduped[artifact["name"]] = artifact
    return list(deduped.values())


def artifact_estimates(artifacts: list[dict[str, Any]]) -> list[StorageEstimate]:
    return [
        StorageEstimate(
            description=artifact["name"],
            size_bytes=artifact.get("size_bytes"),
            source=artifact.get("source_url", artifact.get("url", "")),
        )
        for artifact in artifacts
    ]


def build_or_placeholder(
    *,
    manifest_builder: ManifestBuilder,
    dataset_root: Path,
    manifest_path: Path,
    dataset_name: str,
    remote_source: str,
    notes: str,
    status: DatasetStatus,
) -> tuple[list[dict[str, Any]], DatasetStatus]:
    rows = manifest_builder(dataset_root, manifest_path) if dataset_root.exists() else []
    if rows:
        return rows, status
    placeholder = placeholder_manifest_row(
        dataset=dataset_name,
        remote_source=remote_source,
        notes=notes,
        status=status,
    )
    write_csv(manifest_path, [placeholder])
    return [placeholder], status


def register_dataset(
    registry: DatasetRegistry,
    *,
    dataset_name: str,
    manifest_path: Path,
    local_path: Path | None,
    external_path: Path | None,
    remote_source: str,
    notes: str,
    status: DatasetStatus,
    sample_count: int,
) -> None:
    registry.upsert(
        DatasetRegistration(
            dataset_name=dataset_name,
            manifest_path=to_posix_path(manifest_path),
            local_path=to_posix_path(local_path),
            external_path=to_posix_path(external_path),
            remote_source=remote_source,
            notes=notes,
            status=status.value,
            sample_count=sample_count,
        )
    )


def process_dataset(
    dataset_name: str,
    spec: dict[str, Any],
    *,
    args: argparse.Namespace,
    layout: dict[str, Path | bool],
    registry: DatasetRegistry,
    logger: logging.Logger,
    selected_count: int,
) -> None:
    raw_root = Path(layout["raw"])
    manifests_root = Path(layout["manifests"])
    local_root = raw_root / spec["local_dir"]
    manifest_path = manifests_root / f"{dataset_name}_manifest.csv"
    catalog_path = catalog_output_path(layout, dataset_name, "source_catalog.json")
    try:
        write_json(catalog_path, serializable_spec(spec))
    except PermissionError as exc:
        logger.warning("Could not write source catalog for %s at %s: %s", dataset_name, catalog_path, exc)

    external_root = Path(args.external_path).expanduser().resolve() if args.external_path else None
    symlink_root = Path(args.symlink_from).expanduser().resolve() if args.symlink_from else None
    resolved_external = None

    if external_root:
        resolved_external = resolve_external_dataset_path(external_root, dataset_name, single_dataset=selected_count == 1)
        if resolved_external:
            rows, status = build_or_placeholder(
                manifest_builder=spec["manifest_builder"],
                dataset_root=resolved_external,
                manifest_path=manifest_path,
                dataset_name=dataset_name,
                remote_source=spec["remote_source"],
                notes=f"Registered external dataset path: {resolved_external}",
                status=DatasetStatus.EXTERNAL_LINKED,
            )
            register_dataset(
                registry,
                dataset_name=dataset_name,
                manifest_path=manifest_path,
                local_path=None,
                external_path=resolved_external,
                remote_source=spec["remote_source"],
                notes=f"Registered via --external-path: {resolved_external}",
                status=DatasetStatus.EXTERNAL_LINKED,
                sample_count=len([row for row in rows if row.get("record_type") != "placeholder"]),
            )
            logger.info("%s registered from external path %s", dataset_name, resolved_external)
            return
        logger.warning("No external path match found for %s under %s", dataset_name, external_root)

    if symlink_root:
        resolved_external = resolve_external_dataset_path(symlink_root, dataset_name, single_dataset=selected_count == 1)
        if resolved_external:
            try:
                create_directory_link(local_root, resolved_external)
            except DatasetError as exc:
                logger.warning("Failed to create link for %s: %s", dataset_name, exc)
            rows, status = build_or_placeholder(
                manifest_builder=spec["manifest_builder"],
                dataset_root=local_root if local_root.exists() else resolved_external,
                manifest_path=manifest_path,
                dataset_name=dataset_name,
                remote_source=spec["remote_source"],
                notes=f"Linked to external dataset path: {resolved_external}",
                status=DatasetStatus.EXTERNAL_LINKED,
            )
            register_dataset(
                registry,
                dataset_name=dataset_name,
                manifest_path=manifest_path,
                local_path=local_root if local_root.exists() else None,
                external_path=resolved_external,
                remote_source=spec["remote_source"],
                notes=f"Linked via --symlink-from: {resolved_external}",
                status=DatasetStatus.EXTERNAL_LINKED,
                sample_count=len([row for row in rows if row.get("record_type") != "placeholder"]),
            )
            logger.info("%s linked to external path %s", dataset_name, resolved_external)
            return
        logger.warning("No symlink source match found for %s under %s", dataset_name, symlink_root)

    local_has_existing_files = local_root.exists() and any(local_root.iterdir())
    allow_sen1floods11_sample_expansion = dataset_name == "sen1floods11" and args.sample_only and not list(local_root.rglob("*.tif"))
    if local_has_existing_files and not allow_sen1floods11_sample_expansion:
        rows, _ = build_or_placeholder(
            manifest_builder=spec["manifest_builder"],
            dataset_root=local_root,
            manifest_path=manifest_path,
            dataset_name=dataset_name,
            remote_source=spec["remote_source"],
            notes="Local dataset root already contains files.",
            status=DatasetStatus.PARTIAL,
        )
        register_dataset(
            registry,
            dataset_name=dataset_name,
            manifest_path=manifest_path,
            local_path=local_root,
            external_path=None,
            remote_source=spec["remote_source"],
            notes="Indexed existing local dataset files.",
            status=DatasetStatus.PARTIAL,
            sample_count=len([row for row in rows if row.get("record_type") != "placeholder"]),
        )
        logger.info("%s indexed from existing local files", dataset_name)
        return

    if args.no_download:
        rows, _ = build_or_placeholder(
            manifest_builder=spec["manifest_builder"],
            dataset_root=local_root,
            manifest_path=manifest_path,
            dataset_name=dataset_name,
            remote_source=spec["remote_source"],
            notes=spec["notes"],
            status=DatasetStatus.METADATA_ONLY,
        )
        register_dataset(
            registry,
            dataset_name=dataset_name,
            manifest_path=manifest_path,
            local_path=local_root,
            external_path=None,
            remote_source=spec["remote_source"],
            notes=spec["notes"],
            status=DatasetStatus.METADATA_ONLY,
            sample_count=0,
        )
        logger.info("%s left in metadata-only mode (--no-download)", dataset_name)
        return

    artifacts = choose_artifacts(spec, sample_only=args.sample_only, force=args.force)
    if artifacts:
        try:
            warnings = ensure_storage_guard(artifact_estimates(artifacts), local_root.parent, force=args.force)
            for warning in warnings:
                logger.info("%s | %s", dataset_name, warning)
        except Exception as exc:
            logger.warning("%s | storage guard blocked downloads: %s", dataset_name, exc)
            artifacts = []

    for artifact in artifacts:
        destination = local_root / artifact["dest_name"]
        try:
            if artifact["type"] == "http":
                download_http_file(artifact["url"], destination)
            elif artifact["type"] == "gdrive":
                download_google_drive_file(artifact["file_id"], destination)
            logger.info("%s downloaded %s to %s", dataset_name, artifact["name"], destination)
        except Exception as exc:
            logger.warning("%s failed to download %s: %s", dataset_name, artifact["name"], exc)

    if dataset_name == "sen1floods11" and args.sample_only:
        try:
            downloaded_pairs = download_sen1floods11_handlabeled_sample(
                local_root,
                sample_count=args.sample_count,
                force=args.force,
                logger=logger,
            )
            logger.info("%s downloaded %s exact hand-labeled sample pairs", dataset_name, downloaded_pairs)
        except Exception as exc:
            logger.warning("%s sample subset download skipped: %s", dataset_name, exc)

    if dataset_name == "sen1floods11" and args.force and not args.sample_only:
        bucket_spec = spec.get("full_bucket_gsutil")
        if bucket_spec:
            estimate = StorageEstimate(
                description="sen1floods11-full-bucket",
                size_bytes=int(bucket_spec["approx_size_bytes"]),
                source=bucket_spec["bucket"],
            )
            try:
                warnings = ensure_storage_guard([estimate], local_root.parent, force=args.force)
                for warning in warnings:
                    logger.info("%s | %s", dataset_name, warning)
                if attempt_gcs_sync(bucket_spec["bucket"], local_root, logger=logger):
                    logger.info("%s full bucket sync completed", dataset_name)
            except Exception as exc:
                logger.warning("%s full bucket sync skipped: %s", dataset_name, exc)

    status = DatasetStatus.PARTIAL if local_root.exists() and any(local_root.iterdir()) else DatasetStatus.METADATA_ONLY
    rows, _ = build_or_placeholder(
        manifest_builder=spec["manifest_builder"],
        dataset_root=local_root,
        manifest_path=manifest_path,
        dataset_name=dataset_name,
        remote_source=spec["remote_source"],
        notes=spec["notes"],
        status=status,
    )
    register_dataset(
        registry,
        dataset_name=dataset_name,
        manifest_path=manifest_path,
        local_path=local_root,
        external_path=None,
        remote_source=spec["remote_source"],
        notes=spec["notes"],
        status=status,
        sample_count=len([row for row in rows if row.get("record_type") != "placeholder"]),
    )
    logger.info("%s finished with status=%s", dataset_name, status.value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up public benchmark datasets with storage-aware safeguards and manifest-first behavior."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_SPECS.keys()),
        help="Dataset(s) to process. Defaults to all supported public datasets.",
    )
    parser.add_argument("--sample-only", action="store_true", help="Prefer the smallest known sample artifacts when available.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SEN1FLOODS11_SAMPLE_COUNT,
        help="Number of exact sample items to download when a dataset supports sample-only subset fetches.",
    )
    parser.add_argument("--no-download", action="store_true", help="Create folders, catalogs, manifests, and registry entries without attempting downloads.")
    parser.add_argument("--symlink-from", help="External root directory to link datasets from when local storage should stay minimal.")
    parser.add_argument("--external-path", help="External root directory to register without creating links.")
    parser.add_argument("--force", action="store_true", help="Allow large or unknown-size downloads that are blocked by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = ensure_data_layout(REPO_ROOT)
    logger = configure_logging(Path(layout["logs"]) / "setup_public_datasets.log")
    registry = DatasetRegistry(default_registry_path(REPO_ROOT))

    selected = args.dataset or list(DATASET_SPECS.keys())
    logger.info("Selected datasets: %s", ", ".join(selected))
    for dataset_name in selected:
        process_dataset(
            dataset_name,
            DATASET_SPECS[dataset_name],
            args=args,
            layout=layout,
            registry=registry,
            logger=logger,
            selected_count=len(selected),
        )
    registry.save()
    logger.info("Dataset registry written to %s", default_registry_path(REPO_ROOT))


if __name__ == "__main__":
    main()
