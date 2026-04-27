from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.common import (
    DatasetStatus,
    create_directory_link,
    ensure_data_layout,
    placeholder_manifest_row,
    to_posix_path,
    write_csv,
    write_json,
)
from src.datasets.custom_loader import (
    DEFAULT_ANNOTATION_PATTERNS,
    DEFAULT_IMAGE_PATTERNS,
    build_custom_manifest,
)
from src.datasets.registry import DatasetRegistration, DatasetRegistry, default_registry_path


def _slugify_dataset_name(name: str) -> str:
    lowered = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if not slug:
        raise ValueError("Dataset name must contain at least one alphanumeric character.")
    return slug


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("register_local_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register any local or external dataset into the manifest/registry/audit flow."
    )
    parser.add_argument("--dataset-name", required=True, help="Human-facing dataset name; it will be normalized for registry keys.")
    parser.add_argument("--path", required=True, help="Root folder of the dataset you want to register.")
    parser.add_argument(
        "--remote-source",
        default="",
        help="Optional source URL or provenance string, e.g. a portal URL or 'partner restricted share'.",
    )
    parser.add_argument("--notes", default="", help="Optional notes that should travel with every manifest row and registry entry.")
    parser.add_argument("--owner", default="", help="Optional source owner, e.g. 'partner team'.")
    parser.add_argument(
        "--source-access",
        default="local-handoff",
        help="Optional provenance tag, e.g. public, internal, private-share, local-handoff.",
    )
    parser.add_argument(
        "--pixel-domain",
        default="unknown",
        help="Dataset pixel domain note, e.g. complex_slc, amplitude, intensity, log_intensity, detected_chip.",
    )
    parser.add_argument(
        "--annotation-match",
        choices=("stem", "none"),
        default="stem",
        help="How to pair annotations with images. 'stem' looks for same-stem annotation files anywhere under the root.",
    )
    parser.add_argument(
        "--image-pattern",
        action="append",
        dest="image_patterns",
        help="Optional image glob(s). Repeat to override defaults such as *.jpg, *.png, *.tif.",
    )
    parser.add_argument(
        "--annotation-pattern",
        action="append",
        dest="annotation_patterns",
        help="Optional annotation glob(s). Repeat to override defaults such as *.json, *.xml, *.txt, *.png.",
    )
    parser.add_argument("--complex-slc", action="store_true", help="Mark that this dataset includes complex SLC imagery.")
    parser.add_argument(
        "--status",
        choices=[status.value for status in DatasetStatus],
        default="",
        help="Optional explicit status override. Otherwise the script derives partial vs external-linked automatically.",
    )
    parser.add_argument(
        "--link-into-raw",
        action="store_true",
        help="Create a Windows-friendly junction under data/raw/<dataset_name>/ instead of registering an external path only.",
    )
    return parser.parse_args()


def _effective_status(
    *,
    explicit_status: str,
    row_count: int,
    using_external_path: bool,
) -> DatasetStatus:
    if explicit_status:
        return DatasetStatus(explicit_status)
    if row_count == 0:
        return DatasetStatus.METADATA_ONLY
    return DatasetStatus.EXTERNAL_LINKED if using_external_path else DatasetStatus.PARTIAL


def main() -> None:
    args = _parse_args()
    logger = _configure_logging()
    layout = ensure_data_layout(REPO_ROOT)
    registry = DatasetRegistry(default_registry_path(REPO_ROOT))

    dataset_key = _slugify_dataset_name(args.dataset_name)
    source_root = Path(args.path).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source_root}")

    manifests_root = Path(layout["manifests"])
    catalogs_root = Path(layout["catalogs"])
    raw_root = Path(layout["raw"])
    manifest_path = manifests_root / f"{dataset_key}_manifest.csv"
    catalog_path = catalogs_root / dataset_key / "source_catalog.json"

    registered_root = source_root
    local_path = ""
    external_path = ""
    using_external_path = False
    raw_target = raw_root / dataset_key

    if args.link_into_raw and source_root != raw_target:
        create_directory_link(raw_target, source_root)
        registered_root = raw_target
        local_path = to_posix_path(raw_target)
        external_path = to_posix_path(source_root)
        using_external_path = True
    elif source_root.is_relative_to(raw_root):
        local_path = to_posix_path(source_root)
    else:
        external_path = to_posix_path(source_root)
        using_external_path = True

    image_patterns = tuple(args.image_patterns or DEFAULT_IMAGE_PATTERNS)
    annotation_patterns = tuple(args.annotation_patterns or DEFAULT_ANNOTATION_PATTERNS)
    provenance_metadata = {
        "dataset_kind": "custom",
        "owner": args.owner,
        "source_access": args.source_access,
        "pixel_domain": args.pixel_domain,
        "complex_slc_available": args.complex_slc,
        "image_patterns": list(image_patterns),
        "annotation_patterns": list(annotation_patterns),
        "annotation_match": args.annotation_match,
        "registered_from": to_posix_path(source_root),
    }

    rows = build_custom_manifest(
        registered_root,
        manifest_path,
        dataset_name=dataset_key,
        image_patterns=image_patterns,
        annotation_patterns=annotation_patterns,
        annotation_match_mode=args.annotation_match,
        pixel_domain=args.pixel_domain,
        complex_slc_available=args.complex_slc,
        source_name=args.owner,
        notes=args.notes,
        extra_metadata=provenance_metadata,
    )
    effective_status = _effective_status(
        explicit_status=args.status,
        row_count=len(rows),
        using_external_path=using_external_path,
    )
    if not rows:
        write_csv(
            manifest_path,
            [
                placeholder_manifest_row(
                    dataset=dataset_key,
                    remote_source=args.remote_source,
                    notes=args.notes or "Custom dataset registered before sample files were indexed.",
                    status=effective_status,
                )
            ],
        )

    split_counts = Counter(row.get("split", "all") for row in rows)
    write_json(
        catalog_path,
        {
            "dataset_name": dataset_key,
            "display_name": args.dataset_name,
            "registered_root": to_posix_path(registered_root),
            "source_root": to_posix_path(source_root),
            "remote_source": args.remote_source,
            "notes": args.notes,
            "metadata": provenance_metadata,
        },
    )

    registry.upsert(
        DatasetRegistration(
            dataset_name=dataset_key,
            manifest_path=to_posix_path(manifest_path),
            local_path=local_path,
            external_path=external_path,
            remote_source=args.remote_source,
            split_info=dict(sorted(split_counts.items())),
            metadata=provenance_metadata,
            notes=args.notes or "Custom dataset registration. Re-run this command after adding new files to rebuild the manifest.",
            status=effective_status.value,
            sample_count=len(rows),
        )
    )
    registry.save()

    logger.info("Registered dataset %s", dataset_key)
    logger.info("Manifest path: %s", manifest_path)
    logger.info("Registered root: %s", registered_root)
    logger.info("Status: %s", effective_status.value)
    logger.info("Sample count: %s", len(rows))
    logger.info("Split counts: %s", dict(sorted(split_counts.items())) if split_counts else {})


if __name__ == "__main__":
    main()
