from __future__ import annotations

"""Intentional end-of-project heavy sweep.

This is the single place where expensive downstream preparation/training/eval
should be launched on purpose. Everything else in the repo defaults to
reuse-first behavior.
"""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_detection_baseline import build_parser as build_detection_parser
from scripts.run_detection_baseline import run_detection_workflow
from src.reporting.demo_examples import write_demo_index
from src.reporting.handoff import write_surface_artifacts
from src.stage1.pipeline import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final intentional heavy evaluation sweep with dry-run and resume support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/final_sweep.yaml", help="Frozen final sweep config YAML.")
    parser.add_argument("--datasets", nargs="+", help="Override the configured datasets.")
    parser.add_argument("--variants", nargs="+", help="Override the configured variants.")
    parser.add_argument("--dry-run", action="store_true", help="Show the exact sweep plan without running heavy work.")
    parser.add_argument("--force", action="store_true", help="Force reruns even when matching artifacts already exist.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable cache reuse and resumable runs during the intentional heavy sweep.",
    )
    parser.add_argument("--skip-surface-packs", action="store_true", help="Skip public/private summary regeneration at the end.")
    parser.add_argument("--skip-demo-index", action="store_true", help="Skip demo index regeneration at the end.")
    return parser


def _summary_output_path(config: dict[str, object], *, dry_run: bool = False) -> Path:
    reports = config.get("reports", {}) if isinstance(config.get("reports", {}), dict) else {}
    configured = str(reports.get("summary_output", "outputs/final_sweep/final_sweep_summary.json"))
    path = (REPO_ROOT / configured).resolve()
    if dry_run:
        return path.with_name(f"{path.stem}_dry_run{path.suffix}")
    return path


def main() -> None:
    args = build_parser().parse_args()
    config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_yaml(config_path, expected_kind="final_sweep")
    detection_cfg = dict(config.get("detection", {}))
    reports_cfg = dict(config.get("reports", {}))
    datasets = args.datasets or list(detection_cfg.get("datasets", ["ssdd", "hrsid"]))
    variants = args.variants or list(detection_cfg.get("variants", ["raw", "bundle_a"]))
    output_root = str(detection_cfg.get("output_root", "outputs/final_sweep/downstream_detection"))
    resolved_detection_output_root = (
        Path(output_root).resolve() if Path(output_root).is_absolute() else (REPO_ROOT / output_root).resolve()
    )
    summary_path = _summary_output_path(config, dry_run=bool(args.dry_run))
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    plan_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    detection_parser = build_detection_parser()

    for dataset_name in datasets:
        detection_args = detection_parser.parse_args([])
        detection_args.config = str(detection_cfg.get("config", "configs/downstream/yolo_medium.yaml"))
        detection_args.dataset = dataset_name
        detection_args.manifest = None
        detection_args.variants = variants
        detection_args.limit_per_split = detection_cfg.get("limit_per_split")
        detection_args.mode = str(detection_cfg.get("mode", "all"))
        detection_args.epochs = detection_cfg.get("epochs")
        detection_args.imgsz = detection_cfg.get("imgsz")
        detection_args.batch = detection_cfg.get("batch")
        detection_args.workers = detection_cfg.get("workers")
        detection_args.model = detection_cfg.get("model")
        detection_args.device = detection_cfg.get("device")
        detection_args.output_root = output_root
        detection_args.bundle_a_config = str(detection_cfg.get("bundle_a_config", "configs/bundle_a.yaml"))
        detection_args.bundle_a_conservative_config = str(
            detection_cfg.get("bundle_a_conservative_config", "configs/bundle_a_conservative.yaml")
        )
        detection_args.bundle_b_config = str(detection_cfg.get("bundle_b_config", "configs/bundle_b.yaml"))
        detection_args.bundle_d_config = str(detection_cfg.get("bundle_d_config", "configs/bundle_d.yaml"))
        detection_args.reuse_only = False
        detection_args.allow_prepare = True
        detection_args.allow_train = True
        detection_args.allow_eval = True
        detection_args.dry_run = bool(args.dry_run)
        detection_args.force = bool(args.force)
        detection_args.resume = bool(args.resume)

        plan_rows.append(
            {
                "dataset": dataset_name,
                "variants": variants,
                "mode": detection_args.mode,
                "config": detection_args.config,
                "output_root": output_root,
                "dry_run": detection_args.dry_run,
                "force": detection_args.force,
                "resume": detection_args.resume,
            }
        )
        dataset_summaries.append(run_detection_workflow(detection_args))

    written_packs: dict[str, dict[str, str]] | None = None
    demo_index_path: str | None = None
    if not args.dry_run and not args.skip_surface_packs and bool(reports_cfg.get("generate_surface_packs", True)):
        written_packs = write_surface_artifacts(
            REPO_ROOT,
            surface=str(reports_cfg.get("surface", "all")),
            detection_output_root=resolved_detection_output_root,
        )
    if not args.dry_run and not args.skip_demo_index and bool(reports_cfg.get("generate_demo_index", True)):
        demo_index_path = write_demo_index(
            REPO_ROOT,
            max_examples=int(reports_cfg.get("demo_index_max_examples", 8)),
        ).resolve().as_posix()

    summary = {
        "config_path": config_path.as_posix(),
        "summary_output_path": summary_path.as_posix(),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "datasets": datasets,
        "variants": variants,
        "plan_rows": plan_rows,
        "dataset_summaries": dataset_summaries,
        "surface_packs": written_packs or {},
        "demo_index": demo_index_path or "",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
