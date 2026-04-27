from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stage1.pipeline import configure_logging, resolve_manifest_path
from src.stage1.sentinel1_batch import default_batch_output_root, evaluate_bundle_a_sentinel1_batch
from src.utils import add_execution_policy_args, describe_policy, execution_policy_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Bundle A across local Sentinel-1 GRD scenes and write scene-level comparison artifacts.",
    )
    parser.add_argument("--config", default="configs/bundle_a.yaml", help="Bundle A configuration YAML.")
    parser.add_argument("--manifest", help="Explicit Sentinel-1 manifest CSV path.")
    parser.add_argument("--status", nargs="+", default=["ready", "failed", "metadata-only"])
    parser.add_argument("--max-scenes", type=int, help="Limit the number of ready scenes to evaluate.")
    parser.add_argument("--polarization", choices=["VV", "VH", "HH", "HV"])
    parser.add_argument("--additive-submethod", choices=["auto", "A0", "A1", "A2", "A3"])
    parser.add_argument("--compare-submethods", action="store_true")
    parser.add_argument("--output-root", help="Override the default batch output root.")
    add_execution_policy_args(parser, include_conditioning=True)
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    policy = execution_policy_from_args(args)
    if args.compare_submethods and args.additive_submethod and args.additive_submethod.lower() != "auto":
        raise ValueError("Use either --compare-submethods or a forced --additive-submethod, not both at once.")
    manifest_path = resolve_manifest_path(REPO_ROOT, "sentinel1", args.manifest)
    output_root = Path(args.output_root).resolve() if args.output_root else default_batch_output_root(REPO_ROOT)
    artifacts = evaluate_bundle_a_sentinel1_batch(
        repo_root=REPO_ROOT,
        config_path=(REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config),
        manifest_path=manifest_path,
        output_root=output_root,
        statuses=args.status,
        max_scenes=args.max_scenes,
        polarization=args.polarization,
        additive_submethod=args.additive_submethod,
        compare_submethods=bool(args.compare_submethods),
        policy=policy,
    )
    print(
        json.dumps(
            {
                "output_root": artifacts.output_root.as_posix(),
                "scene_count": len(artifacts.scene_summary_rows),
                "comparison_row_count": len(artifacts.comparison_rows),
                "topline_metrics": artifacts.topline_metrics,
                "execution_policy": describe_policy(policy),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
