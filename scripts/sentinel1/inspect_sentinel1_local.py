from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stage1.pipeline import resolve_manifest_path
from src.stage1.sentinel1_batch import inspect_sentinel1_manifest, load_sentinel1_manifest_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect local Sentinel-1 manifest readiness and next fetch/prepare actions.")
    parser.add_argument("--manifest", help="Explicit Sentinel-1 manifest CSV path.")
    parser.add_argument("--status", nargs="+", default=["ready", "failed", "metadata-only"])
    parser.add_argument("--polarization", choices=["VV", "VH", "HH", "HV"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = resolve_manifest_path(REPO_ROOT, "sentinel1", args.manifest)
    records = load_sentinel1_manifest_records(manifest_path)
    payload = inspect_sentinel1_manifest(records, statuses=args.status, polarization=args.polarization)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
