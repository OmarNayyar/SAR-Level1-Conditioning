from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.sentinel1_evidence import plan_sentinel1_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan the next Sentinel-1 GRD evidence expansion round without downloading data.")
    parser.add_argument("--manifest", help="Optional explicit Sentinel-1 manifest path.")
    parser.add_argument("--batch-output-root", default="outputs/bundle_a_sentinel1_batch", help="Existing Sentinel-1 batch output root.")
    parser.add_argument("--target-scene-count", type=int, default=10, help="Scene count target for the next evidence round.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = plan_sentinel1_evidence(
        repo_root=REPO_ROOT,
        manifest_path=Path(args.manifest).resolve() if args.manifest else None,
        batch_output_root=(REPO_ROOT / args.batch_output_root).resolve()
        if not Path(args.batch_output_root).is_absolute()
        else Path(args.batch_output_root),
        target_scene_count=args.target_scene_count,
    )
    print(json.dumps(asdict(plan), indent=2))


if __name__ == "__main__":
    main()
