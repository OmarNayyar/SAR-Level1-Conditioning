from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.sentinel1_evidence import expand_sentinel1_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a small set of additional Sentinel-1 GRD scenes, prepare them locally, and rerun Bundle A multi-scene comparison."
    )
    parser.add_argument("--config", default="configs/bundle_a.yaml", help="Bundle A config used for the batch rerun.")
    parser.add_argument("--manifest", help="Optional explicit Sentinel-1 manifest path.")
    parser.add_argument("--target-ready-scenes", type=int, default=12, help="Aim for this many locally ready GRD scenes.")
    parser.add_argument("--max-new-downloads", type=int, default=8, help="Cap extra new scene downloads beyond current manifest rows.")
    parser.add_argument("--output-root", help="Override the default Sentinel-1 batch output root.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the expansion without downloading archives.")
    parser.add_argument("--force", action="store_true", help="Allow large downloads if storage guards would normally stop them.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    summary = expand_sentinel1_evidence(
        repo_root=REPO_ROOT,
        batch_config_path=(REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config),
        manifest_path=Path(args.manifest).resolve() if args.manifest else None,
        target_ready_scenes=args.target_ready_scenes,
        max_new_downloads=args.max_new_downloads,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        batch_output_root=Path(args.output_root).resolve() if args.output_root else None,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
