from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.common import write_csv, write_json
from src.downstream.detection import (
    MissingDetectorDependency,
    detector_run_artifact_identity,
    load_detector_run_result,
    load_prepared_yolo_dataset,
    prepare_yolo_dataset,
    prepared_yolo_artifact_identity,
    run_ultralytics_detector,
)
from src.stage1.pipeline import load_yaml, resolve_manifest_path, save_config_snapshot
from src.utils import (
    add_execution_policy_args,
    decide_artifact_action,
    describe_policy,
    execution_policy_from_args,
    payload_fingerprint,
    write_artifact_index,
    write_artifact_manifest,
)


DETECTOR_METRIC_KEYS = ("map", "map50", "map75", "precision", "recall", "f1")
DETECTOR_DIAGNOSTIC_KEYS = ("target_contrast", "target_local_variance", "target_edge_strength")
DETECTION_VARIANTS = ("raw", "bundle_a", "bundle_a_conservative", "bundle_b", "bundle_d")


def _parse_limit_per_split(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "full", "all", "no_limit", "unlimited", "-1", "0"}:
        return None
    parsed = int(text)
    if parsed <= 0:
        return None
    return parsed


def _dataset_output_root(base_root: Path, dataset_name: str, explicit_output_root: bool) -> Path:
    """Keep detector runs per dataset while preserving explicit smoke roots.

    Default runs write to `outputs/downstream_detection/<dataset>/...` so SSDD
    and HRSID do not overwrite each other.  If the caller provides an explicit
    output root ending in the dataset name, use it directly.
    """

    if explicit_output_root and base_root.name.lower() == dataset_name.lower():
        return base_root
    return base_root / dataset_name


def _load_existing_rows(base_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(base_root.glob("*/metrics/downstream_comparison.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(dict(row) for row in payload.get("rows", []))
    return rows


def _load_existing_delta_rows(base_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(base_root.glob("*/metrics/variant_deltas.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(dict(row) for row in payload.get("rows", []))
    return rows


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_variant_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare conditioned variants against raw detector results.

    These are real detector-result deltas only when both rows have completed
    Ultralytics metrics.  If either side is still prepare-only, the delta row is
    kept with an explicit status so the app can explain the gap instead of
    looking broken or silently omitting the comparison.
    """

    by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row.get("dataset", "")), {})[str(row.get("variant", ""))] = row

    delta_rows: list[dict[str, Any]] = []
    for dataset_name, variants in sorted(by_dataset.items()):
        raw = variants.get("raw")
        if raw is None:
            continue
        for variant_name, conditioned in sorted(variants.items()):
            if variant_name == "raw":
                continue
            raw_metrics = raw.get("metrics", {}) if isinstance(raw.get("metrics"), dict) else {}
            conditioned_metrics = conditioned.get("metrics", {}) if isinstance(conditioned.get("metrics"), dict) else {}
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "baseline_variant": "raw",
                "comparison_variant": variant_name,
                "baseline_status": raw.get("status", ""),
                "comparison_status": conditioned.get("status", ""),
                "status": "completed"
                if raw.get("status") == "completed" and conditioned.get("status") == "completed"
                else "metrics_unavailable",
                "interpretation": "",
            }
            for key in DETECTOR_METRIC_KEYS:
                raw_value = _as_float(raw_metrics.get(key))
                conditioned_value = _as_float(conditioned_metrics.get(key))
                row[f"raw_{key}"] = raw_value if raw_value is not None else ""
                row[f"{variant_name}_{key}"] = conditioned_value if conditioned_value is not None else ""
                row[f"comparison_{key}"] = conditioned_value if conditioned_value is not None else ""
                row[f"delta_{key}"] = (
                    conditioned_value - raw_value if raw_value is not None and conditioned_value is not None else ""
                )
            raw_diagnostics = raw.get("diagnostics", {}) if isinstance(raw.get("diagnostics"), dict) else {}
            conditioned_diagnostics = conditioned.get("diagnostics", {}) if isinstance(conditioned.get("diagnostics"), dict) else {}
            diagnostic_notes: list[str] = []
            for key in DETECTOR_DIAGNOSTIC_KEYS:
                raw_value = _as_float(raw_diagnostics.get(f"mean_{key}"))
                conditioned_value = _as_float(conditioned_diagnostics.get(f"mean_{key}"))
                row[f"raw_{key}"] = raw_value if raw_value is not None else ""
                row[f"{variant_name}_{key}"] = conditioned_value if conditioned_value is not None else ""
                row[f"comparison_{key}"] = conditioned_value if conditioned_value is not None else ""
                row[f"delta_{key}"] = (
                    conditioned_value - raw_value if raw_value is not None and conditioned_value is not None else ""
                )
                if key == "target_edge_strength" and raw_value and conditioned_value is not None:
                    row["target_edge_retention_ratio"] = conditioned_value / raw_value
            contrast_delta = _as_float(row.get("delta_target_contrast"))
            edge_ratio = _as_float(row.get("target_edge_retention_ratio"))
            variance_delta = _as_float(row.get("delta_target_local_variance"))
            if contrast_delta is not None and contrast_delta < 0:
                diagnostic_notes.append("target/background contrast fell")
            if edge_ratio is not None and edge_ratio < 0.9:
                diagnostic_notes.append("target-edge strength was reduced")
            if variance_delta is not None and variance_delta < 0:
                diagnostic_notes.append("local target variance was suppressed")
            map_delta = _as_float(row.get("delta_map"))
            if row["status"] != "completed":
                row["interpretation"] = "Detector metrics are not available for both raw and conditioned variants yet."
            elif map_delta is None:
                row["interpretation"] = "Both variants completed, but mAP was not reported by the detector backend."
            elif map_delta > 0:
                row["interpretation"] = "Conditioning improved detector mAP on this detector run."
            elif map_delta < 0:
                row["interpretation"] = "Conditioning reduced detector mAP on this detector run."
            else:
                row["interpretation"] = "Conditioning matched raw detector mAP on this detector run."
            row["diagnostic_interpretation"] = (
                "Diagnostics suggest " + ", ".join(diagnostic_notes) + "."
                if diagnostic_notes
                else "Target-preservation diagnostics did not flag a clear contrast/edge-loss mechanism."
            )
            delta_rows.append(row)
    return delta_rows


def _build_diagnostic_summary(rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create compact detector-preservation summaries for app/docs consumption.

    The raw detector metrics answer *what happened*.  These rows summarize the
    likely image-level mechanism behind the result using target-preservation
    diagnostics computed around ground-truth boxes.  They remain heuristic: a
    high target/background contrast can still hurt YOLO if local target edges or
    target texture are suppressed.
    """

    by_dataset_variant = {
        (str(row.get("dataset", "")), str(row.get("variant", ""))): row
        for row in rows
    }
    summary_rows: list[dict[str, Any]] = []
    for delta in delta_rows:
        dataset_name = str(delta.get("dataset", ""))
        variant_name = str(delta.get("comparison_variant", ""))
        raw_row = by_dataset_variant.get((dataset_name, "raw"), {})
        conditioned_row = by_dataset_variant.get((dataset_name, variant_name), {})
        edge_ratio = _as_float(delta.get("target_edge_retention_ratio"))
        map_delta = _as_float(delta.get("delta_map"))
        f1_delta = _as_float(delta.get("delta_f1"))
        contrast_delta = _as_float(delta.get("delta_target_contrast"))
        variance_delta = _as_float(delta.get("delta_target_local_variance"))
        if map_delta is None:
            conclusion = "Detector metric unavailable."
        elif map_delta < 0:
            conclusion = "Detector performance decreased."
        elif map_delta > 0:
            conclusion = "Detector performance improved."
        else:
            conclusion = "Detector performance was unchanged."

        likely_reason: list[str] = []
        if contrast_delta is not None and contrast_delta > 0:
            likely_reason.append("target/background contrast increased")
        if edge_ratio is not None and edge_ratio < 0.9:
            likely_reason.append("target-edge strength fell")
        if variance_delta is not None and variance_delta < 0:
            likely_reason.append("local target variance fell")
        if map_delta is not None and map_delta < 0 and likely_reason:
            interpretation = (
                "Conditioning did not fail by simply lowering contrast; it likely removed detector-useful "
                f"edge/texture cues ({', '.join(likely_reason)})."
            )
        elif likely_reason:
            interpretation = f"Observed image changes: {', '.join(likely_reason)}."
        else:
            interpretation = "No clear target-preservation mechanism was flagged by the current diagnostics."

        summary_rows.append(
            {
                "dataset": dataset_name,
                "comparison": f"raw_vs_{variant_name}",
                "status": delta.get("status", ""),
                "raw_images": raw_row.get("image_count", ""),
                "conditioned_images": conditioned_row.get("image_count", ""),
                "raw_boxes": raw_row.get("box_count", ""),
                "conditioned_boxes": conditioned_row.get("box_count", ""),
                "delta_map": map_delta if map_delta is not None else "",
                "delta_f1": f1_delta if f1_delta is not None else "",
                "delta_target_contrast": contrast_delta if contrast_delta is not None else "",
                "delta_target_local_variance": variance_delta if variance_delta is not None else "",
                "target_edge_retention_ratio": edge_ratio if edge_ratio is not None else "",
                "conclusion": conclusion,
                "interpretation": interpretation,
            }
        )
    return summary_rows


def _write_diagnostic_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = ["# Downstream detector diagnostic summary", ""]
    lines.append(
        "This summary compares raw imagery against conditioned variants using real YOLO detector metrics plus target-local diagnostics."
    )
    lines.append(
        "The diagnostics are heuristic: they help explain detector behavior, but they are not a replacement for mAP / precision / recall."
    )
    lines.append("")
    lines.append(
        "| Dataset | Comparison | mAP delta | F1 delta | Contrast delta | Edge retention | Conclusion | Interpretation |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in summary_rows:
        lines.append(
            "| {dataset} | {comparison} | {delta_map} | {delta_f1} | {delta_target_contrast} | {target_edge_retention_ratio} | {conclusion} | {interpretation} |".format(
                dataset=row.get("dataset", ""),
                comparison=row.get("comparison", ""),
                delta_map=row.get("delta_map", ""),
                delta_f1=row.get("delta_f1", ""),
                delta_target_contrast=row.get("delta_target_contrast", ""),
                target_edge_retention_ratio=row.get("target_edge_retention_ratio", ""),
                conclusion=row.get("conclusion", ""),
                interpretation=row.get("interpretation", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_aggregate_index(base_root: Path) -> None:
    rows = _load_existing_rows(base_root)
    # Rebuild deltas from the current per-dataset rows instead of trusting any
    # older delta file on disk.  This keeps aggregate summaries consistent when
    # scoring/diagnostic logic evolves between runs.
    delta_rows = _build_variant_deltas(rows)
    diagnostic_rows = _build_diagnostic_summary(rows, delta_rows)
    metrics_root = base_root / "metrics"
    tables_root = base_root / "tables"
    metrics_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    write_json(metrics_root / "downstream_comparison.json", {"rows": rows})
    write_csv(
        metrics_root / "downstream_comparison.csv",
        [
            {
                "dataset": row.get("dataset", ""),
                "variant": row.get("variant", ""),
                "status": row.get("status", ""),
                "input_record_count": row.get("input_record_count", 0),
                "image_count": row.get("image_count", 0),
                "box_count": row.get("box_count", 0),
                "skipped_count": row.get("skipped_count", 0),
                "missing_image_count": row.get("missing_image_count", 0),
                "missing_annotation_count": row.get("missing_annotation_count", 0),
                "empty_label_count": row.get("empty_label_count", 0),
                "map": row.get("metrics", {}).get("map", "") if isinstance(row.get("metrics"), dict) else "",
                "precision": row.get("metrics", {}).get("precision", "") if isinstance(row.get("metrics"), dict) else "",
                "recall": row.get("metrics", {}).get("recall", "") if isinstance(row.get("metrics"), dict) else "",
                "f1": row.get("metrics", {}).get("f1", "") if isinstance(row.get("metrics"), dict) else "",
                "target_contrast": row.get("diagnostics", {}).get("mean_target_contrast", "") if isinstance(row.get("diagnostics"), dict) else "",
                "target_local_variance": row.get("diagnostics", {}).get("mean_target_local_variance", "") if isinstance(row.get("diagnostics"), dict) else "",
                "target_edge_strength": row.get("diagnostics", {}).get("mean_target_edge_strength", "") if isinstance(row.get("diagnostics"), dict) else "",
                "warnings": " ".join(row.get("warnings", [])),
            }
            for row in rows
        ],
    )
    write_json(metrics_root / "variant_deltas.json", {"rows": delta_rows})
    write_csv(metrics_root / "variant_deltas.csv", delta_rows)
    write_json(metrics_root / "diagnostic_summary.json", {"rows": diagnostic_rows})
    write_csv(metrics_root / "diagnostic_summary.csv", diagnostic_rows)
    _write_diagnostic_summary(tables_root / "diagnostic_summary.md", diagnostic_rows)
    write_json(
        metrics_root / "run_summary.json",
        {"mode": "aggregate", "rows": rows, "variant_deltas": delta_rows, "diagnostic_summary": diagnostic_rows},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally run a real YOLO ship-detection baseline on SSDD/HRSID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/downstream/yolo_smoke.yaml", help="Detection baseline config YAML.")
    parser.add_argument("--dataset", choices=["ssdd", "hrsid"], help="Override dataset from config.")
    parser.add_argument("--manifest", help="Explicit dataset manifest path.")
    parser.add_argument("--variants", nargs="+", choices=list(DETECTION_VARIANTS), help="Image variants to prepare/evaluate.")
    parser.add_argument(
        "--limit-per-split",
        help="Cap samples per split. Use `none`, `full`, or `all` to use all available records.",
    )
    parser.add_argument("--mode", choices=["prepare", "all"], default="prepare", help="`all` trains/evaluates if Ultralytics is installed.")
    parser.add_argument("--epochs", type=int, help="YOLO training epochs.")
    parser.add_argument("--imgsz", type=int, help="YOLO image size.")
    parser.add_argument("--batch", type=int, help="YOLO batch size.")
    parser.add_argument("--workers", type=int, help="Ultralytics dataloader workers. Use 0 on Windows for maximum robustness.")
    parser.add_argument("--model", help="Ultralytics model checkpoint/name, e.g. yolov8n.pt.")
    parser.add_argument("--device", help="Optional Ultralytics device string.")
    parser.add_argument(
        "--output-root",
        help="Override the detector output root. Default runs write per-dataset subdirectories under the configured root.",
    )
    parser.add_argument("--bundle-a-config", default="configs/bundle_a.yaml", help="Bundle A config used for `bundle_a`.")
    parser.add_argument(
        "--bundle-a-conservative-config",
        default="configs/bundle_a_conservative.yaml",
        help="Bundle A conservative config used for `bundle_a_conservative`.",
    )
    parser.add_argument("--bundle-b-config", default="configs/bundle_b.yaml", help="Bundle B config used for `bundle_b`.")
    parser.add_argument("--bundle-d-config", default="configs/bundle_d.yaml", help="Bundle D config used for `bundle_d`.")
    add_execution_policy_args(parser, include_prepare=True, include_train=True, include_eval=True)
    return parser


def _write_markdown_summary(
    path: Path,
    rows: list[dict[str, Any]],
    notes: list[str],
    delta_rows: list[dict[str, Any]],
) -> None:
    lines = ["# Downstream detection baseline", ""]
    lines.append("This is the first real downstream detector path. It is separate from proxy-only bundle screening.")
    lines.append("")
    lines.append("| Dataset | Variant | Status | Images | Boxes | Skipped | mAP | Precision | Recall | F1 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        lines.append(
            "| {dataset} | {variant} | {status} | {image_count} | {box_count} | {skipped_count} | {map} | {precision} | {recall} | {f1} |".format(
                dataset=row.get("dataset", ""),
                variant=row.get("variant", ""),
                status=row.get("status", ""),
                image_count=row.get("image_count", 0),
                box_count=row.get("box_count", 0),
                skipped_count=row.get("skipped_count", 0),
                map=metrics.get("map", "n/a"),
                precision=metrics.get("precision", "n/a"),
                recall=metrics.get("recall", "n/a"),
                f1=metrics.get("f1", "n/a"),
            )
        )
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])
    if delta_rows:
        lines.extend(["", "## Raw vs conditioned deltas", ""])
        lines.append("| Dataset | Comparison | Status | mAP delta | Precision delta | Recall delta | F1 delta | Interpretation |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
        for row in delta_rows:
            lines.append(
                "| {dataset} | {comparison} | {status} | {map_delta} | {precision_delta} | {recall_delta} | {f1_delta} | {interpretation} |".format(
                    dataset=row.get("dataset", ""),
                    comparison=f"{row.get('baseline_variant', 'raw')} -> {row.get('comparison_variant', '')}",
                    status=row.get("status", ""),
                    map_delta=row.get("delta_map", "n/a"),
                    precision_delta=row.get("delta_precision", "n/a"),
                    recall_delta=row.get("delta_recall", "n/a"),
                    f1_delta=row.get("delta_f1", "n/a"),
                    interpretation=" ".join(
                        part
                        for part in [str(row.get("interpretation", "")), str(row.get("diagnostic_interpretation", ""))]
                        if part
                    ),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _existing_row_map(output_root: Path) -> dict[str, dict[str, Any]]:
    payload_path = output_root / "metrics" / "downstream_comparison.json"
    if not payload_path.exists():
        return {}
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    return {
        str(row.get("variant", "")): dict(row)
        for row in rows
        if str(row.get("variant", "")).strip()
    }


def _row_from_prepared(prepared: Any, *, dataset_name: str, variant: str, existing_row: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "dataset": dataset_name,
        "variant": variant,
        "status": prepared.status,
        "dataset_yaml": prepared.dataset_yaml.as_posix(),
        "prepared_root": prepared.root.as_posix(),
        "input_record_count": prepared.input_record_count,
        "split_counts": prepared.split_counts,
        "image_count": prepared.image_count,
        "box_count": prepared.box_count,
        "skipped_count": prepared.skipped_count,
        "missing_image_count": prepared.missing_image_count,
        "missing_annotation_count": prepared.missing_annotation_count,
        "empty_label_count": prepared.empty_label_count,
        "diagnostics": prepared.diagnostics,
        "metrics": {},
        "warnings": list(prepared.warnings),
    }
    if existing_row:
        row["metrics"] = existing_row.get("metrics", {}) if isinstance(existing_row.get("metrics"), dict) else {}
        for key in ("model_path", "detector_run_dir", "error_traceback"):
            if existing_row.get(key):
                row[key] = existing_row.get(key)
        if existing_row.get("status") == "completed":
            row["status"] = "completed"
    return row


def _prepared_from_existing_row(existing_row: dict[str, Any]) -> Any | None:
    prepared_root = str(existing_row.get("prepared_root", "")).strip()
    dataset_yaml = str(existing_row.get("dataset_yaml", "")).strip()
    if not prepared_root or not dataset_yaml:
        return None
    return type("PreparedRow", (), {
        "dataset_name": str(existing_row.get("dataset", "")),
        "variant": str(existing_row.get("variant", "")),
        "root": Path(prepared_root),
        "dataset_yaml": Path(dataset_yaml),
        "input_record_count": int(existing_row.get("input_record_count", 0) or 0),
        "split_counts": dict(existing_row.get("split_counts", {})),
        "image_count": int(existing_row.get("image_count", 0) or 0),
        "box_count": int(existing_row.get("box_count", 0) or 0),
        "skipped_count": int(existing_row.get("skipped_count", 0) or 0),
        "missing_image_count": int(existing_row.get("missing_image_count", 0) or 0),
        "missing_annotation_count": int(existing_row.get("missing_annotation_count", 0) or 0),
        "empty_label_count": int(existing_row.get("empty_label_count", 0) or 0),
        "diagnostics": dict(existing_row.get("diagnostics", {})),
        "status": "prepared" if int(existing_row.get("image_count", 0) or 0) > 0 else str(existing_row.get("status", "prepared")),
        "warnings": list(existing_row.get("warnings", [])) if isinstance(existing_row.get("warnings"), list) else [],
    })()


def _planned_row_from_prepared(prepared: Any, *, dataset_name: str, variant: str, action: str, reason: str) -> dict[str, Any]:
    row = _row_from_prepared(prepared, dataset_name=dataset_name, variant=variant)
    row["status"] = action
    row["warnings"].append(reason)
    return row


def _write_variant_artifact_manifest(
    *,
    output_root: Path,
    dataset_name: str,
    variant: str,
    row: dict[str, Any],
    prepared_identity: dict[str, Any],
    detector_identity: dict[str, Any] | None = None,
) -> None:
    prepared_root = Path(str(row.get("prepared_root", "")))
    if prepared_root.exists():
        write_artifact_manifest(
            prepared_root,
            artifact_kind="prepared_yolo_dataset",
            identity=prepared_identity,
            status=str(row.get("status", "prepared")),
            files={
                "dataset_yaml": str(row.get("dataset_yaml", "")),
                "prepared_root": prepared_root.as_posix(),
            },
            metadata={
                "dataset": dataset_name,
                "variant": variant,
                "image_count": row.get("image_count", 0),
                "box_count": row.get("box_count", 0),
            },
            notes=row.get("warnings", []),
        )
    detector_run_dir = str(row.get("detector_run_dir", "")).strip()
    if detector_identity is not None and detector_run_dir:
        detector_root = Path(detector_run_dir)
        if detector_root.exists():
            write_artifact_manifest(
                detector_root,
                artifact_kind="detector_run",
                identity=detector_identity,
                status=str(row.get("status", "")),
                files={
                    "run_dir": detector_root.as_posix(),
                    "trained_model": str(row.get("model_path", "")),
                    "detector_run_result": (detector_root / "detector_run_result.json").as_posix(),
                },
                metadata={"dataset": dataset_name, "variant": variant, "metrics": row.get("metrics", {})},
                notes=row.get("warnings", []),
            )


def run_detection_workflow(args: argparse.Namespace) -> dict[str, Any]:
    policy = execution_policy_from_args(args)
    config = load_yaml(
        (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config),
        expected_kind="detection",
    )
    dataset_name = args.dataset or str(config.get("dataset", {}).get("name", "ssdd"))
    variants = args.variants or list(config.get("variants", ["raw", "bundle_a"]))
    limit_cfg = args.limit_per_split if args.limit_per_split is not None else config.get("dataset", {}).get("limit_per_split", 64)
    limit_per_split = _parse_limit_per_split(limit_cfg)
    base_output_root = Path(args.output_root).resolve() if args.output_root else (REPO_ROOT / config.get("outputs", {}).get("root", "outputs/downstream_detection")).resolve()
    output_root = _dataset_output_root(base_output_root, dataset_name, explicit_output_root=bool(args.output_root))
    manifest_path = resolve_manifest_path(REPO_ROOT, dataset_name, args.manifest)
    bundle_a_config_path = (REPO_ROOT / args.bundle_a_config).resolve() if not Path(args.bundle_a_config).is_absolute() else Path(args.bundle_a_config)
    bundle_a_conservative_config_path = (
        (REPO_ROOT / args.bundle_a_conservative_config).resolve()
        if not Path(args.bundle_a_conservative_config).is_absolute()
        else Path(args.bundle_a_conservative_config)
    )
    bundle_b_config_path = (REPO_ROOT / args.bundle_b_config).resolve() if not Path(args.bundle_b_config).is_absolute() else Path(args.bundle_b_config)
    bundle_d_config_path = (REPO_ROOT / args.bundle_d_config).resolve() if not Path(args.bundle_d_config).is_absolute() else Path(args.bundle_d_config)
    bundle_a_config = load_yaml(bundle_a_config_path, expected_kind="bundle")
    bundle_a_conservative_config = (
        load_yaml(bundle_a_conservative_config_path, expected_kind="bundle")
        if bundle_a_conservative_config_path.exists()
        else bundle_a_config
    )
    bundle_b_config = load_yaml(bundle_b_config_path, expected_kind="bundle")
    bundle_d_config = load_yaml(bundle_d_config_path, expected_kind="bundle")

    config_dir = output_root / "config"
    save_config_snapshot(config, config_dir)
    existing_rows = _existing_row_map(output_root)
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    artifact_index_rows: list[dict[str, Any]] = []
    detector_cfg = config.get("detector", {})
    prepared_identity_hashes: dict[str, str] = {}

    for variant in variants:
        existing_row = existing_rows.get(variant)
        prepared_root = output_root / "prepared" / dataset_name / variant
        prepared_identity = prepared_yolo_artifact_identity(
            dataset_name=dataset_name,
            variant=variant,
            manifest_path=manifest_path,
            limit_per_split=limit_per_split,
            val_fraction=float(config.get("dataset", {}).get("val_fraction", 0.2)),
            bundle_a_config=bundle_a_config if variant in {"bundle_a", "bundle_a_conservative"} else None,
            bundle_a_conservative_config=bundle_a_conservative_config if variant == "bundle_a_conservative" else None,
            bundle_b_config=bundle_b_config if variant == "bundle_b" else None,
            bundle_d_config=bundle_d_config if variant == "bundle_d" else None,
        )
        prepared_identity_hashes[variant] = payload_fingerprint(prepared_identity)
        prepared_decision = decide_artifact_action(
            artifact_kind="prepared_yolo_dataset",
            output_root=prepared_root,
            identity=prepared_identity,
            required_files=["dataset.yaml", "manifest.csv", "diagnostics.json"],
            capability="prepare",
            policy=policy,
            accept_existing_without_manifest=True,
        )

        if prepared_decision.action == "reuse":
            prepared = load_prepared_yolo_dataset(prepared_root)
            if prepared is None and existing_row is not None:
                prepared = _prepared_from_existing_row(existing_row)
            if prepared is None:
                raise RuntimeError(
                    f"Prepared artifact for {dataset_name}/{variant} looked reusable, but the cached summary could not be loaded."
                )
        elif prepared_decision.action == "would_run":
            planned_prepared = existing_row
            if planned_prepared is not None:
                row = dict(planned_prepared)
                row["status"] = "would_prepare"
                warnings = [
                    str(warning)
                    for warning in row.get("warnings", [])
                    if str(warning).strip()
                ] if isinstance(row.get("warnings"), list) else []
                row["warnings"] = list(dict.fromkeys(warnings))
                warnings = row["warnings"]
                if prepared_decision.reason not in warnings:
                    warnings.append(prepared_decision.reason)
            else:
                row = {
                    "dataset": dataset_name,
                    "variant": variant,
                    "status": "would_prepare",
                    "dataset_yaml": (prepared_root / "dataset.yaml").resolve().as_posix(),
                    "prepared_root": prepared_root.resolve().as_posix(),
                    "input_record_count": "",
                    "split_counts": {},
                    "image_count": 0,
                    "box_count": 0,
                    "skipped_count": 0,
                    "missing_image_count": 0,
                    "missing_annotation_count": 0,
                    "empty_label_count": 0,
                    "diagnostics": {},
                    "metrics": {},
                    "warnings": [prepared_decision.reason],
                }
            detector_followup_action = "skipped"
            detector_followup_reason = ""
            if args.mode == "all":
                detector_followup_action = "would_train_after_prepare"
                detector_followup_reason = (
                    "Dry-run: detector training/evaluation would be attempted after this prepared dataset is generated."
                )
                warnings = row.setdefault("warnings", [])
                if detector_followup_reason not in warnings:
                    warnings.append(detector_followup_reason)
            rows.append(row)
            artifact_index_rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant,
                    "prepared_action": prepared_decision.action,
                    "prepared_identity_hash": prepared_decision.identity_hash,
                    "prepared_root": prepared_root.resolve().as_posix(),
                    "prepared_reason": prepared_decision.reason,
                    "detector_action": detector_followup_action,
                    "detector_reason": detector_followup_reason,
                }
            )
            continue
        elif prepared_decision.action == "blocked":
            raise RuntimeError(prepared_decision.reason)
        else:
            prepared = prepare_yolo_dataset(
                dataset_name=dataset_name,
                manifest_path=manifest_path,
                output_root=output_root / "prepared",
                variant=variant,
                bundle_a_config=bundle_a_config if variant in {"bundle_a", "bundle_a_conservative"} else None,
                bundle_a_conservative_config=bundle_a_conservative_config if variant == "bundle_a_conservative" else None,
                bundle_b_config=bundle_b_config if variant == "bundle_b" else None,
                bundle_d_config=bundle_d_config if variant == "bundle_d" else None,
                limit_per_split=limit_per_split,
                val_fraction=float(config.get("dataset", {}).get("val_fraction", 0.2)),
                reset_root=True,
            )

        row = _row_from_prepared(prepared, dataset_name=dataset_name, variant=variant, existing_row=existing_row)
        if limit_per_split is None:
            row["warnings"].append("Full split mode was used; this can be slow and should be run deliberately.")
        elif prepared.image_count >= 768:
            row["warnings"].append("Large prepared split; keep batch size conservative on CPU/Windows.")

        detector_action_label = "skipped"
        if args.mode == "all" and prepared.status == "prepared":
            # Detector metrics belong to a specific training/eval request. Start
            # from the prepared variant and only restore completed metrics when
            # the detector artifact itself is verified as reusable.
            row["status"] = "prepared"
            row["metrics"] = {}
            row.pop("model_path", None)
            row.pop("detector_run_dir", None)
            detector_model = args.model or str(detector_cfg.get("model", "yolov8n.pt"))
            detector_epochs = int(args.epochs if args.epochs is not None else detector_cfg.get("epochs", 3))
            detector_imgsz = int(args.imgsz if args.imgsz is not None else detector_cfg.get("imgsz", 640))
            detector_batch = int(args.batch if args.batch is not None else detector_cfg.get("batch", 8))
            detector_workers = int(args.workers if args.workers is not None else detector_cfg.get("workers", 0))
            detector_device = args.device or detector_cfg.get("device") or None
            detector_eval_split = str(detector_cfg.get("eval_split", "test"))
            run_name = f"{dataset_name}_{variant}"
            detector_root = output_root / "ultralytics" / run_name
            detector_identity = detector_run_artifact_identity(
                dataset_yaml=prepared.dataset_yaml,
                variant_name=run_name,
                model=detector_model,
                epochs=detector_epochs,
                imgsz=detector_imgsz,
                batch=detector_batch,
                workers=detector_workers,
                device=detector_device,
                eval_split=detector_eval_split,
                prepared_identity_hash=prepared_identity_hashes[variant],
            )
            detector_decision = decide_artifact_action(
                artifact_kind="detector_run",
                output_root=detector_root,
                identity=detector_identity,
                required_files=["detector_run_result.json"],
                capability="train",
                policy=policy,
                accept_existing_without_manifest=True,
            )
            detector_action_label = detector_decision.action
            if detector_decision.action == "reuse":
                if existing_row is not None and str(existing_row.get("status", "")) == "completed":
                    row = dict(existing_row)
                    row.setdefault("warnings", []).append(detector_decision.reason)
                else:
                    result = load_detector_run_result(detector_root)
                    if result is None:
                        raise RuntimeError(
                            f"Detector artifact for {dataset_name}/{variant} looked reusable, but detector_run_result.json could not be loaded."
                        )
                    row.update(
                        {
                            "status": result.status,
                            "model_path": result.model,
                            "detector_run_dir": result.run_dir,
                            "metrics": result.metrics,
                        }
                    )
                    row["warnings"].extend(result.notes)
            elif detector_decision.action == "would_run":
                row["status"] = "would_train"
                row["warnings"].append(detector_decision.reason)
            elif detector_decision.action == "blocked":
                raise RuntimeError(detector_decision.reason)
            else:
                try:
                    result = run_ultralytics_detector(
                        dataset_yaml=prepared.dataset_yaml,
                        output_root=output_root,
                        variant_name=run_name,
                        model=detector_model,
                        epochs=detector_epochs,
                        imgsz=detector_imgsz,
                        batch=detector_batch,
                        workers=detector_workers,
                        device=detector_device,
                        eval_split=detector_eval_split,
                        prepared_identity_hash=prepared_identity_hashes[variant],
                    )
                    row.update(
                        {
                            "status": result.status,
                            "model_path": result.model,
                            "detector_run_dir": result.run_dir,
                            "metrics": result.metrics,
                        }
                    )
                    row["warnings"].extend(result.notes)
                except MissingDetectorDependency as exc:
                    row["status"] = "dependency_missing"
                    row["warnings"].append(str(exc))
                    notes.append(str(exc))
                except Exception as exc:
                    row["status"] = "failed"
                    error_summary = f"{type(exc).__name__}: {exc}"
                    row["warnings"].append(error_summary)
                    row["error_traceback"] = traceback.format_exc(limit=8)
                    notes.append(f"{variant} failed: {error_summary}")

            _write_variant_artifact_manifest(
                output_root=output_root,
                dataset_name=dataset_name,
                variant=variant,
                row=row,
                prepared_identity=prepared_identity,
                detector_identity=detector_identity,
            )
        else:
            _write_variant_artifact_manifest(
                output_root=output_root,
                dataset_name=dataset_name,
                variant=variant,
                row=row,
                prepared_identity=prepared_identity,
            )
        rows.append(row)
        artifact_index_rows.append(
            {
                "dataset": dataset_name,
                "variant": variant,
                "prepared_action": prepared_decision.action,
                "prepared_identity_hash": prepared_decision.identity_hash,
                "prepared_root": str(row.get("prepared_root", "")),
                "prepared_reason": prepared_decision.reason,
                "detector_action": detector_action_label,
                "detector_run_dir": str(row.get("detector_run_dir", "")),
                "status": row.get("status", ""),
            }
        )

    metrics_root = output_root / "metrics"
    tables_root = output_root / "tables"
    delta_rows = _build_variant_deltas(rows)
    write_json(
        metrics_root / "run_summary.json",
        {
            "dataset": dataset_name,
            "mode": args.mode,
            "variants": variants,
            "rows": rows,
            "variant_deltas": delta_rows,
            "notes": notes,
            "execution_policy": describe_policy(policy),
        },
    )
    write_json(metrics_root / "downstream_comparison.json", {"rows": rows})
    write_json(metrics_root / "variant_deltas.json", {"rows": delta_rows})
    write_csv(
        metrics_root / "downstream_comparison.csv",
        [
            {
                "dataset": row["dataset"],
                "variant": row["variant"],
                "status": row["status"],
                "input_record_count": row.get("input_record_count", 0),
                "image_count": row.get("image_count", 0),
                "box_count": row.get("box_count", 0),
                "skipped_count": row.get("skipped_count", 0),
                "missing_image_count": row.get("missing_image_count", 0),
                "missing_annotation_count": row.get("missing_annotation_count", 0),
                "empty_label_count": row.get("empty_label_count", 0),
                "map": row.get("metrics", {}).get("map", ""),
                "precision": row.get("metrics", {}).get("precision", ""),
                "recall": row.get("metrics", {}).get("recall", ""),
                "f1": row.get("metrics", {}).get("f1", ""),
                "target_contrast": row.get("diagnostics", {}).get("mean_target_contrast", ""),
                "target_local_variance": row.get("diagnostics", {}).get("mean_target_local_variance", ""),
                "target_edge_strength": row.get("diagnostics", {}).get("mean_target_edge_strength", ""),
                "warnings": " ".join(row.get("warnings", [])),
            }
            for row in rows
        ],
    )
    write_csv(metrics_root / "variant_deltas.csv", delta_rows)
    diagnostic_rows = _build_diagnostic_summary(rows, delta_rows)
    write_json(metrics_root / "diagnostic_summary.json", {"rows": diagnostic_rows})
    write_csv(metrics_root / "diagnostic_summary.csv", diagnostic_rows)
    _write_markdown_summary(tables_root / "run_summary.md", rows, notes, delta_rows)
    _write_diagnostic_summary(tables_root / "diagnostic_summary.md", diagnostic_rows)
    write_artifact_index(metrics_root / "artifact_index.json", artifact_index_rows)
    _write_aggregate_index(base_output_root)
    return {
        "output_root": output_root.as_posix(),
        "aggregate_output_root": base_output_root.as_posix(),
        "dataset": dataset_name,
        "mode": args.mode,
        "rows": rows,
        "notes": notes,
        "execution_policy": describe_policy(policy),
        "artifact_index": artifact_index_rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = run_detection_workflow(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
