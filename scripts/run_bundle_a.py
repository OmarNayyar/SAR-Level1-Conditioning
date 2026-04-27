from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bundles.bundle_a_classical import run_bundle_a
from src.datasets.common import read_json
from src.stage1.pipeline import (
    bundle_artifact_identity,
    build_bundle_arg_parser,
    configure_logging,
    load_manifest_records,
    load_yaml,
    resolve_bundle_output_root,
    resolve_manifest_path,
)
from src.utils import (
    add_execution_policy_args,
    decide_artifact_action,
    describe_policy,
    execution_policy_from_args,
    write_artifact_manifest,
)


def main() -> None:
    configure_logging()
    parser = build_bundle_arg_parser(
        description="Run Bundle A: additive submethod family plus Refined Lee.",
        default_config_path="configs/bundle_a.yaml",
    )
    parser.add_argument(
        "--additive-submethod",
        choices=["auto", "A0", "A1", "A2", "A3"],
        help="Override the Bundle A additive submethod selection. Use auto to let the router choose.",
    )
    add_execution_policy_args(parser, include_conditioning=True)
    args = parser.parse_args()
    policy = execution_policy_from_args(args)

    config = load_yaml(Path(args.config).resolve(), expected_kind="bundle")
    original_additive_cfg = config.get("processing", {}).get("additive", {})
    configured_submethod = str(original_additive_cfg.get("submethod", "auto"))
    if args.additive_submethod:
        config.setdefault("processing", {}).setdefault("additive", {})["submethod"] = args.additive_submethod
    dataset_cfg = dict(config.get("dataset", {}))
    dataset_name = args.dataset or dataset_cfg.get("name", "ssdd")
    split = args.split or dataset_cfg.get("split")
    sample_limit = args.sample_limit if args.sample_limit is not None else dataset_cfg.get("sample_limit")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else resolve_bundle_output_root(
            REPO_ROOT,
            config,
            bundle_name="bundle_a",
            dataset_name=dataset_name,
            split=split,
            sample_limit=int(sample_limit) if sample_limit is not None else None,
            extra_tokens=[args.additive_submethod] if args.additive_submethod and args.additive_submethod != configured_submethod else [],
        )
    )

    manifest_path = resolve_manifest_path(REPO_ROOT, dataset_name, args.manifest)
    identity = bundle_artifact_identity(
        bundle_name="bundle_a",
        dataset_name=dataset_name,
        split=split,
        sample_limit=int(sample_limit) if sample_limit is not None else None,
        manifest_path=manifest_path,
        config=config,
        extra_fields={"requested_additive_submethod": args.additive_submethod or configured_submethod},
    )
    decision = decide_artifact_action(
        artifact_kind="bundle_run",
        output_root=output_root,
        identity=identity,
        required_files=["metrics/run_summary.json", "metrics/topline_metrics.json", "tables/run_summary.md"],
        capability="conditioning",
        policy=policy,
        accept_existing_without_manifest=True,
    )
    if decision.action == "reuse":
        summary = read_json(output_root / "metrics" / "run_summary.json")
        write_artifact_manifest(
            output_root,
            artifact_kind="bundle_run",
            identity=identity,
            status="complete",
            files={
                "run_summary": (output_root / "metrics" / "run_summary.json").resolve().as_posix(),
                "topline_metrics": (output_root / "metrics" / "topline_metrics.json").resolve().as_posix(),
                "markdown_summary": (output_root / "tables" / "run_summary.md").resolve().as_posix(),
            },
            metadata={"bundle_name": "bundle_a", "dataset": dataset_name},
            notes=[decision.reason],
        )
        print(json.dumps(summary, indent=2))
        return
    if decision.action == "would_run":
        print(json.dumps({"status": "dry-run", "policy": describe_policy(policy), "artifact_decision": decision.as_dict()}, indent=2))
        return
    if decision.action == "blocked":
        raise RuntimeError(decision.reason)

    records = load_manifest_records(
        manifest_path,
        split=split,
        sample_limit=int(sample_limit) if sample_limit is not None else None,
        dataset_name=dataset_name,
        product_family=dataset_cfg.get("product_family"),
    )
    if not records:
        raise RuntimeError(
            f"No records were available for dataset={dataset_name!r}, split={split!r}, manifest={manifest_path.as_posix()}."
        )

    summary = run_bundle_a(records, dataset_name=dataset_name, config=config, output_root=output_root)
    write_artifact_manifest(
        output_root,
        artifact_kind="bundle_run",
        identity=identity,
        status="complete",
        files={
            "run_summary": (output_root / "metrics" / "run_summary.json").resolve().as_posix(),
            "topline_metrics": (output_root / "metrics" / "topline_metrics.json").resolve().as_posix(),
            "markdown_summary": (output_root / "tables" / "run_summary.md").resolve().as_posix(),
        },
        metadata={
            "bundle_name": "bundle_a",
            "dataset": dataset_name,
            "policy": describe_policy(policy),
        },
        notes=["Heavy conditioning work was run intentionally."],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
