from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.datasets.common import write_csv, write_json
from src.reporting.evaluation import evidence_grade_for_run, score_proxy_tradeoff
from src.stage1.downstream import evaluate_proxy_outputs
from src.stage1.metrics.detection_map import compute_detection_proxy_map
from src.stage1.metrics.edge_sharpness import compute_edge_sharpness
from src.stage1.metrics.proxy_enl import compute_proxy_enl
from src.stage1.pipeline import (
    BundleProcessResult,
    LoadedSample,
    aggregate_numeric_rows,
    load_sample,
    prepare_output_dirs,
    save_config_snapshot,
    save_map_figure,
)
from src.stage1.viz.failure_case_gallery import save_failure_case_gallery
from src.stage1.viz.side_by_side import save_side_by_side


BundleProcessor = Callable[[LoadedSample, dict[str, Any]], BundleProcessResult]

_BUNDLE_MATURITY_NOTES = {
    "bundle_a": "Most interpretable conditioning family; useful for proxy screening, but detector validation currently favors raw over default A.",
    "bundle_b": "Strongest current paired denoising candidate on Mendeley validation; detector-facing use still requires proof against raw.",
    "bundle_c": "Feasibility-only until stronger complex SLC coverage exists.",
    "bundle_d": "Metadata-poor inverse/self-supervised candidate with useful structure preservation; detector-facing use still trails raw in the current YOLO setup.",
}

_BUNDLE_DEFAULT_RECOMMENDATIONS = {
    "bundle_a": "Use Bundle A as the explainable conditioning family, but validate every detector-facing setting against raw before handoff.",
    "bundle_b": "Prioritize Bundle B for paired intensity-domain denoising checks, but validate detector-facing use against raw before adoption.",
    "bundle_c": "Keep Bundle C in feasibility mode until better complex SLC coverage is available.",
    "bundle_d": "Use Bundle D as the metadata-poor inverse/self-supervised comparator; prioritize conservative detector-preserving tuning.",
}


def _python_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_markdown_table(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_columns(rows: list[dict[str, Any]], preferred_columns: list[str]) -> list[str]:
    present = {key for row in rows for key in row.keys()}
    return [column for column in preferred_columns if column in present]


def _build_sample_summary_rows(metrics_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    preferred_columns = [
        "dataset",
        "sample_id",
        "split",
        "pixel_domain",
        "additive_submethod_code",
        "additive_submethod_name",
        "additive_metadata_available",
        "additive_applied",
        "additive_fallback_used",
        "additive_confidence_level",
        "proxy_enl_before",
        "proxy_enl_after",
        "edge_sharpness_before",
        "edge_sharpness_after",
        "distribution_separability_before",
        "distribution_separability_after",
        "threshold_f1_before",
        "threshold_f1_after",
        "bundle_quality_score",
        "decision_score",
        "decision_confidence",
        "decision_evidence_grade",
        "decision_flags",
        "decision_metric_bias_warning",
        "source_note",
    ]
    columns = _select_columns(metrics_rows, preferred_columns)
    rows = [{column: row.get(column, "") for column in columns} for row in metrics_rows]
    return rows, columns


def _safe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _metric_delta(aggregate_metrics: dict[str, Any], before_key: str, after_key: str) -> float | None:
    before = _safe_float(aggregate_metrics.get(before_key))
    after = _safe_float(aggregate_metrics.get(after_key))
    if before is None or after is None:
        return None
    return after - before


def _dominant_additive_submethod(summary: dict[str, Any]) -> str:
    counts = summary.get("additive_submethod_counts", {})
    if not counts:
        return ""
    return max(sorted(counts), key=lambda code: int(counts.get(code, 0)))


def _submethod_display(summary: dict[str, Any]) -> str:
    counts = summary.get("additive_submethod_counts", {})
    if not counts:
        return ""
    if len(counts) == 1:
        return next(iter(counts))
    dominant = _dominant_additive_submethod(summary)
    return f"mixed (dominated by {dominant})" if dominant else "mixed"


def _build_interpretation(summary: dict[str, Any]) -> str:
    aggregate_metrics = summary.get("aggregate_metrics", {})
    bundle_name = str(summary.get("bundle_name", "bundle"))
    submethod_text = _submethod_display(summary)
    lead = f"This run used {submethod_text} additive routing" if submethod_text else f"This {bundle_name} run"

    clauses: list[str] = []
    proxy_enl_gain = _safe_float(aggregate_metrics.get("proxy_enl_gain"))
    if proxy_enl_gain is not None:
        if proxy_enl_gain > 0.1:
            clauses.append(f"increased proxy ENL by {proxy_enl_gain:.3f}")
        elif proxy_enl_gain < -0.1:
            clauses.append(f"reduced proxy ENL by {abs(proxy_enl_gain):.3f}")

    separability_delta = _metric_delta(
        aggregate_metrics,
        "distribution_separability_before",
        "distribution_separability_after",
    )
    if separability_delta is not None:
        if separability_delta > 0.005:
            clauses.append(f"improved target/background separability by {separability_delta:.3f}")
        elif separability_delta < -0.005:
            clauses.append(f"reduced target/background separability by {abs(separability_delta):.3f}")

    threshold_f1_delta = _metric_delta(aggregate_metrics, "threshold_f1_before", "threshold_f1_after")
    if threshold_f1_delta is not None:
        if threshold_f1_delta > 0.005:
            clauses.append(f"improved threshold F1 by {threshold_f1_delta:.3f}")
        elif threshold_f1_delta < -0.005:
            clauses.append(f"reduced threshold F1 by {abs(threshold_f1_delta):.3f}")

    edge_delta = _safe_float(aggregate_metrics.get("edge_sharpness_delta"))
    if edge_delta is not None:
        if edge_delta < -1.0:
            clauses.append(f"reduced edge sharpness by {abs(edge_delta):.3f}")
        elif edge_delta > 1.0:
            clauses.append(f"increased edge sharpness by {edge_delta:.3f}")

    if not clauses:
        return lead + "."
    if len(clauses) == 1:
        return lead + " and " + clauses[0] + "."
    return lead + ", " + ", ".join(clauses[:-1]) + ", and " + clauses[-1] + "."


def _build_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    bundle_name = str(summary.get("bundle_name", ""))
    dataset_name = str(summary.get("dataset", ""))
    skipped_count = int(summary.get("skipped_count", 0))
    aggregate_metrics = summary.get("aggregate_metrics", {})

    if str(summary.get("downstream_status", "")) == "proxy-only":
        warnings.append("Downstream results are proxy-only and should not be interpreted as trained detector or segmenter claims.")
    if bundle_name == "bundle_c":
        warnings.append("Bundle C remains feasibility-only until stronger complex SLC coverage is available.")
    if dataset_name == "sentinel1":
        warnings.append("Current Sentinel-1 runs are overview/proxy-scale checks, not full-scene production-grade filtering.")
    if skipped_count > 0:
        warnings.append(f"{skipped_count} sample(s) were skipped during this run.")
    if bundle_name == "bundle_a" and (_safe_float(aggregate_metrics.get("additive_metadata_available")) or 0.0) <= 0.0:
        warnings.append("No additive product metadata was available in this run, so A1 could not be used.")
    if str(summary.get("evidence_grade", "")).startswith("proxy-only"):
        warnings.append("Decision scores are heuristics over proxy metrics, not calibrated downstream task performance.")
    return warnings


def _build_current_recommendation(summary: dict[str, Any]) -> str:
    bundle_name = str(summary.get("bundle_name", ""))
    dataset_name = str(summary.get("dataset", ""))
    dominant = _dominant_additive_submethod(summary)
    aggregate_metrics = summary.get("aggregate_metrics", {})
    metadata_available = _safe_float(aggregate_metrics.get("additive_metadata_available")) or 0.0

    if bundle_name == "bundle_a":
        if dataset_name == "sentinel1":
            return "Use A1 when real product metadata is present, but add 5-10 more maritime GRD scenes before drawing stronger real-product conclusions."
        if dominant == "A3":
            return "Treat A3 as a structured-artifact specialist. Keep A2 as the default fallback on metadata-poor public chip data and compare both against A0."
        if dominant == "A1" or metadata_available > 0.0:
            return "Trust A1 when product noise metadata is present, and compare it against A0 and A2 to show the value of metadata-driven correction."
        return "Use A2 as the practical default on metadata-poor public chip data, keep A0 as the control, and reserve A3 for clear structured-artifact cases."
    return _BUNDLE_DEFAULT_RECOMMENDATIONS.get(bundle_name, "Review this run alongside Bundle A before spending more tuning effort here.")


def _build_topline_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate_metrics = summary.get("aggregate_metrics", {})
    return {
        "bundle_name": summary.get("bundle_name", ""),
        "dataset": summary.get("dataset", ""),
        "processed_count": summary.get("processed_count", 0),
        "skipped_count": summary.get("skipped_count", 0),
        "downstream_status": summary.get("downstream_status", ""),
        "dominant_additive_submethod": _dominant_additive_submethod(summary),
        "proxy_enl_gain": aggregate_metrics.get("proxy_enl_gain"),
        "edge_sharpness_delta": aggregate_metrics.get("edge_sharpness_delta"),
        "distribution_separability_delta": _metric_delta(
            aggregate_metrics,
            "distribution_separability_before",
            "distribution_separability_after",
        ),
        "threshold_f1_delta": _metric_delta(aggregate_metrics, "threshold_f1_before", "threshold_f1_after"),
        "maturity_note": summary.get("maturity_note", ""),
        "evidence_grade": summary.get("evidence_grade", ""),
        "decision_basis": summary.get("decision_basis", ""),
        "current_recommendation": summary.get("current_recommendation", ""),
        "warnings": summary.get("warnings", []),
        "interpretation": summary.get("interpretation", ""),
    }


def _build_run_overview_row(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate_metrics = summary.get("aggregate_metrics", {})
    return {
        "bundle_name": summary.get("bundle_name", ""),
        "dataset": summary.get("dataset", ""),
        "processed_count": summary.get("processed_count", 0),
        "skipped_count": summary.get("skipped_count", 0),
        "input_record_count": summary.get("input_record_count", 0),
        "downstream_status": summary.get("downstream_status", ""),
        "dominant_additive_submethod": _dominant_additive_submethod(summary),
        "mean_proxy_enl_after": aggregate_metrics.get("proxy_enl_after"),
        "mean_proxy_enl_gain": aggregate_metrics.get("proxy_enl_gain"),
        "mean_edge_sharpness_after": aggregate_metrics.get("edge_sharpness_after"),
        "mean_edge_sharpness_delta": aggregate_metrics.get("edge_sharpness_delta"),
        "mean_distribution_separability_after": aggregate_metrics.get("distribution_separability_after"),
        "mean_distribution_separability_delta": _metric_delta(
            aggregate_metrics,
            "distribution_separability_before",
            "distribution_separability_after",
        ),
        "mean_threshold_f1_after": aggregate_metrics.get("threshold_f1_after"),
        "mean_threshold_f1_delta": _metric_delta(aggregate_metrics, "threshold_f1_before", "threshold_f1_after"),
        "interpretation": summary.get("interpretation", ""),
        "current_recommendation": summary.get("current_recommendation", ""),
        "evidence_grade": summary.get("evidence_grade", ""),
        "decision_basis": summary.get("decision_basis", ""),
    }


def write_summary_artifacts(output_root: Path, summary: dict[str, Any]) -> None:
    metrics_root = output_root / "metrics"
    tables_root = output_root / "tables"
    metrics_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    run_overview_row = _build_run_overview_row(summary)
    topline_metrics = _build_topline_metrics(summary)

    write_json(metrics_root / "run_summary.json", summary)
    write_json(metrics_root / "topline_metrics.json", topline_metrics)
    write_csv(tables_root / "run_overview.csv", [run_overview_row])
    write_json(tables_root / "run_overview.json", run_overview_row)
    overview_headers = list(run_overview_row.keys())
    _write_markdown_table(tables_root / "run_overview.md", overview_headers, [run_overview_row])

    warning_lines = summary.get("warnings", [])
    markdown_lines = [
        f"# {summary.get('bundle_name', 'bundle')} run summary",
        "",
        f"- Dataset: `{summary.get('dataset', '')}`",
        f"- Processed samples: `{summary.get('processed_count', 0)}`",
        f"- Skipped samples: `{summary.get('skipped_count', 0)}`",
        f"- Downstream status: `{summary.get('downstream_status', '')}`",
    ]
    dominant_submethod = _dominant_additive_submethod(summary)
    if dominant_submethod:
        markdown_lines.append(f"- Dominant additive submethod: `{dominant_submethod}`")
    if summary.get("maturity_note"):
        markdown_lines.append(f"- Maturity note: {summary['maturity_note']}")
    if summary.get("interpretation"):
        markdown_lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    if summary.get("current_recommendation"):
        markdown_lines.extend(["", "## Current Recommendation", "", summary["current_recommendation"]])
    if warning_lines:
        markdown_lines.extend(["", "## Warnings", ""])
        markdown_lines.extend([f"- {warning}" for warning in warning_lines])
    (tables_root / "run_summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def run_stage1_bundle(
    *,
    records: list[dict[str, str]],
    dataset_name: str,
    config: dict[str, Any],
    output_root: Path,
    processor: BundleProcessor,
    bundle_name: str,
    after_title: str,
    sample_analyzer: Any | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    outputs = prepare_output_dirs(output_root)
    save_config_snapshot(config, outputs["config"])

    patch_size = int(config.get("metrics", {}).get("proxy_enl", {}).get("patch_size", 32))
    edge_quantile = float(config.get("metrics", {}).get("edge_sharpness", {}).get("top_quantile", 0.9))
    threshold_quantile = float(config.get("metrics", {}).get("detection_proxy", {}).get("threshold_quantile", 0.98))
    outputs_cfg = config.get("outputs", {})
    max_visual_samples = int(outputs_cfg.get("max_visual_samples", 6))
    save_intermediate_arrays = bool(outputs_cfg.get("save_intermediate_arrays", False))

    metrics_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    success_cases: list[dict[str, object]] = []
    failure_cases: list[dict[str, object]] = []
    visual_count = 0

    for record in records:
        sample_id = record.get("sample_id", "sample")
        try:
            sample = load_sample(record, dataset_name)
            process_result = processor(sample, config)
        except Exception as exc:
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "split": record.get("split", "all"),
                    "reason": str(exc),
                }
            )
            continue

        detection_map = compute_detection_proxy_map(process_result.final_output)
        proxy_eval = evaluate_proxy_outputs(
            dataset_name=sample.dataset_name,
            sample_id=sample.sample_id,
            split=sample.split,
            annotation=sample.annotation,
            annotation_count=sample.annotation_count,
            downstream_target=sample.downstream_target,
            detection_map=detection_map,
            threshold_quantile=threshold_quantile,
        )

        enl_before = compute_proxy_enl(sample.intensity_image, patch_size=patch_size)
        enl_after = compute_proxy_enl(process_result.final_output, patch_size=patch_size)
        edge_before = compute_edge_sharpness(sample.intensity_image, top_quantile=edge_quantile)
        edge_after = compute_edge_sharpness(process_result.final_output, top_quantile=edge_quantile)

        side_by_side_path: Path | None = None
        detection_map_path: Path | None = None
        before_panel_path: Path | None = None
        after_panel_path: Path | None = None
        difference_panel_path: Path | None = None
        additive_panel_path: Path | None = None
        arrays_path: Path | None = None
        difference_map = np.maximum(sample.intensity_image - process_result.final_output, 0.0)
        if visual_count < max_visual_samples:
            side_by_side_path = outputs["side_by_side"] / f"{sample.sample_id}.png"
            detection_map_path = outputs["diagnostic_maps"] / f"{sample.sample_id}_detection.png"
            before_panel_path = outputs["panels"] / f"{sample.sample_id}_before.png"
            after_panel_path = outputs["panels"] / f"{sample.sample_id}_after.png"
            difference_panel_path = outputs["panels"] / f"{sample.sample_id}_difference.png"
            save_side_by_side(
                side_by_side_path,
                before=sample.display_image,
                after=process_result.display_output if process_result.display_output is not None else process_result.final_output,
                before_title="Input",
                after_title=after_title,
                difference=difference_map,
                caption=f"{sample.dataset_name} | {sample.sample_id}",
            )
            save_map_figure(before_panel_path, sample.display_image, title=f"{bundle_name} input | {sample.sample_id}", cmap="gray")
            save_map_figure(
                after_panel_path,
                process_result.display_output if process_result.display_output is not None else process_result.final_output,
                title=f"{bundle_name} output | {sample.sample_id}",
                cmap="gray",
            )
            save_map_figure(
                difference_panel_path,
                difference_map,
                title=f"{bundle_name} difference | {sample.sample_id}",
                cmap="magma",
            )
            save_map_figure(detection_map_path, detection_map, title=f"{bundle_name} detection proxy | {sample.sample_id}")
            if process_result.estimated_additive_component is not None:
                additive_panel_path = outputs["diagnostic_maps"] / f"{sample.sample_id}_additive.png"
                save_map_figure(
                    additive_panel_path,
                    process_result.estimated_additive_component,
                    title=f"{bundle_name} additive estimate | {sample.sample_id}",
                )
            visual_count += 1

        if save_intermediate_arrays:
            arrays_path = outputs["arrays"] / f"{sample.sample_id}.npz"
            np.savez_compressed(
                arrays_path,
                input_intensity=sample.intensity_image.astype(np.float32),
                additive_output=process_result.additive_output.astype(np.float32),
                final_output=process_result.final_output.astype(np.float32),
                predicted_mask=proxy_eval.predicted_mask.astype(np.uint8),
                target_mask=proxy_eval.target_mask.astype(np.uint8)
                if proxy_eval.target_mask is not None
                else np.zeros((1, 1), dtype=np.uint8),
                detection_proxy=detection_map.astype(np.float32),
                **{key: np.asarray(value) for key, value in process_result.extra_arrays.items()},
            )

        enl_gain = (
            float(enl_after.score - enl_before.score)
            if np.isfinite(enl_after.score) and np.isfinite(enl_before.score)
            else float("nan")
        )
        edge_delta = float(edge_after.score - edge_before.score)
        notes = [sample.source_note, process_result.additive_notes, process_result.multiplicative_notes, *process_result.notes]
        metrics_row = {
            "dataset": sample.dataset_name,
            "sample_id": sample.sample_id,
            "split": sample.split,
            "pixel_domain": sample.pixel_domain,
            "source_note": " ".join(note for note in notes if note).strip(),
            "annotation_count": int(sample.annotation_count),
            "additive_applied": bool(process_result.additive_applied),
            "additive_mode": process_result.additive_mode,
            "multiplicative_mode": process_result.multiplicative_mode,
            "proxy_enl_before": float(enl_before.score),
            "proxy_enl_after": float(enl_after.score),
            "proxy_enl_gain": enl_gain,
            "proxy_enl_patches_before": int(enl_before.patch_count),
            "proxy_enl_patches_after": int(enl_after.patch_count),
            "edge_sharpness_before": float(edge_before.score),
            "edge_sharpness_after": float(edge_after.score),
            "edge_sharpness_delta": edge_delta,
            "detection_proxy_mean": float(np.mean(detection_map)),
            "detection_proxy_p95": float(np.quantile(detection_map, 0.95)),
            "detection_proxy_max": float(np.max(detection_map)),
            "side_by_side_path": side_by_side_path.resolve().as_posix() if side_by_side_path is not None else "",
            "before_panel_path": before_panel_path.resolve().as_posix() if before_panel_path is not None else "",
            "after_panel_path": after_panel_path.resolve().as_posix() if after_panel_path is not None else "",
            "difference_panel_path": difference_panel_path.resolve().as_posix() if difference_panel_path is not None else "",
            "detection_map_path": detection_map_path.resolve().as_posix() if detection_map_path is not None else "",
            "additive_panel_path": additive_panel_path.resolve().as_posix() if additive_panel_path is not None else "",
            "arrays_path": arrays_path.resolve().as_posix() if arrays_path is not None else "",
            **{key: _python_number(value) for key, value in process_result.result_metadata.items()},
            **{key: _python_number(value) for key, value in proxy_eval.metrics.items()},
        }
        downstream_row = {
            **proxy_eval.downstream_row,
            "detection_map_path": detection_map_path.resolve().as_posix() if detection_map_path is not None else "",
            "side_by_side_path": side_by_side_path.resolve().as_posix() if side_by_side_path is not None else "",
        }
        if sample_analyzer is not None:
            analysis_payload = sample_analyzer.process_sample(
                sample=sample,
                process_result=process_result,
                proxy_evaluation=proxy_eval,
                metrics_row=metrics_row,
            )
            if analysis_payload:
                metrics_row.update(analysis_payload.get("metrics", {}))
                downstream_row.update(analysis_payload.get("downstream", {}))

        # This score is only a decision heuristic. It caps ENL reward and
        # penalizes edge loss so smoother images do not win by default.
        decision_score = score_proxy_tradeoff(
            {
                **metrics_row,
                "distribution_separability_delta": _metric_delta(
                    metrics_row,
                    "distribution_separability_before",
                    "distribution_separability_after",
                ),
                "threshold_f1_delta": _metric_delta(metrics_row, "threshold_f1_before", "threshold_f1_after"),
            }
        )
        bundle_quality_score = float(decision_score.score)
        metrics_row.update(decision_score.as_fields())
        metrics_row["bundle_quality_score"] = bundle_quality_score
        metrics_rows.append(metrics_row)
        downstream_rows.append(downstream_row)

        case_subtitle = (
            f"decision={bundle_quality_score:.3f} | enl_gain={enl_gain:.3f} | "
            f"edge_delta={edge_delta:.3f}"
        )
        success_cases.append(
            {
                "image": process_result.final_output,
                "title": sample.sample_id,
                "subtitle": case_subtitle,
                "score": float(bundle_quality_score),
            }
        )
        failure_cases.append(
            {
                "image": process_result.final_output,
                "title": sample.sample_id,
                "subtitle": case_subtitle,
                "score": float(bundle_quality_score),
            }
        )

    aggregate_rows = aggregate_numeric_rows(metrics_rows)
    metrics_root = outputs["metrics"]
    tables_root = outputs["tables"]
    logs_root = outputs["logs"]
    write_csv(metrics_root / "per_sample_metrics.csv", metrics_rows)
    write_json(metrics_root / "per_sample_metrics.json", {"samples": metrics_rows})
    write_csv(metrics_root / "aggregate_metrics.csv", aggregate_rows)
    write_json(metrics_root / "aggregate_metrics.json", {"metrics": aggregate_rows})
    write_csv(metrics_root / "downstream_eval_hooks.csv", downstream_rows)
    write_json(metrics_root / "downstream_eval_hooks.json", {"samples": downstream_rows})
    write_csv(logs_root / "skipped_samples.csv", skipped_rows)

    sample_summary_rows, sample_summary_columns = _build_sample_summary_rows(metrics_rows)
    write_csv(tables_root / "sample_summary.csv", sample_summary_rows, fieldnames=sample_summary_columns)
    write_json(tables_root / "sample_summary.json", {"samples": sample_summary_rows})
    if sample_summary_rows:
        _write_markdown_table(
            tables_root / "sample_summary.md",
            sample_summary_columns,
            sample_summary_rows[: min(len(sample_summary_rows), 20)],
        )

    success_cases_sorted = sorted(success_cases, key=lambda item: float(item["score"]), reverse=True)[:4]
    failure_cases_sorted = sorted(failure_cases, key=lambda item: float(item["score"]))[:4]
    save_failure_case_gallery(
        outputs["galleries"] / "success_gallery.png",
        cases=success_cases_sorted,
        title=f"{bundle_name} - strongest proxy cases",
        columns=2,
    )
    save_failure_case_gallery(
        outputs["galleries"] / "failure_gallery.png",
        cases=failure_cases_sorted,
        title=f"{bundle_name} - weakest proxy cases",
        columns=2,
    )

    summary = {
        "bundle_name": bundle_name,
        "dataset": dataset_name,
        "processed_count": int(len(metrics_rows)),
        "skipped_count": int(len(skipped_rows)),
        "input_record_count": int(len(records)),
        "output_root": output_root.as_posix(),
        "downstream_status": "proxy-only",
        "notes": [
            "Results are proxy-style Stage-1 screening outputs, not trained downstream benchmark claims.",
            "Per-sample metrics, aggregate metrics, galleries, and config snapshots follow a shared layout across bundles.",
        ],
        "aggregate_metrics": {row["metric"]: row["mean"] for row in aggregate_rows},
        "result_layout": {
            "config": outputs["config"].as_posix(),
            "metrics": outputs["metrics"].as_posix(),
            "plots": outputs["plots"].as_posix(),
            "panels": outputs["panels"].as_posix(),
            "galleries": outputs["galleries"].as_posix(),
            "statistics": outputs["statistics"].as_posix(),
            "tables": outputs["tables"].as_posix(),
            "logs": outputs["logs"].as_posix(),
        },
        "maturity_note": _BUNDLE_MATURITY_NOTES.get(bundle_name, ""),
        "evidence_grade": evidence_grade_for_run(
            downstream_status="proxy-only",
            dataset_name=dataset_name,
            bundle_name=bundle_name,
            sample_count=len(metrics_rows),
            overview_only=dataset_name == "sentinel1",
        ),
        "decision_basis": (
            "Balanced proxy decision heuristic: separability and threshold behavior are prioritized, "
            "ENL gain is capped, and edge loss is penalized. This is not detector mAP."
        ),
    }
    if any("additive_submethod_code" in row for row in metrics_rows):
        additive_counts: dict[str, int] = {}
        for row in metrics_rows:
            code = str(row.get("additive_submethod_code", "")).strip()
            if not code:
                continue
            additive_counts[code] = additive_counts.get(code, 0) + 1
        summary["additive_submethod_counts"] = additive_counts
    if sample_analyzer is not None:
        analysis_summary = sample_analyzer.finalize()
        if analysis_summary:
            summary["statistics_analysis"] = analysis_summary
    summary["interpretation"] = _build_interpretation(summary)
    summary["current_recommendation"] = _build_current_recommendation(summary)
    summary["warnings"] = _build_warnings(summary)
    write_summary_artifacts(output_root, summary)
    return summary
