from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.datasets.common import read_json, write_csv, write_json
from src.bundles.common import run_stage1_bundle, write_summary_artifacts
from src.stage1.additive.bundle_a_submethods import run_bundle_a_additive_submethod
from src.stage1.multiplicative.refined_lee import refined_lee_filter
from src.stage1.pipeline import BundleProcessResult, LoadedSample
from src.stage1.statistics import IntensityStatisticsAnalyzer


def process_bundle_a_sample(sample: LoadedSample, config: dict[str, Any]) -> BundleProcessResult:
    additive_cfg = config.get("processing", {}).get("additive", {})
    multiplicative_cfg = config.get("processing", {}).get("multiplicative", {})
    window_size = int(multiplicative_cfg.get("window_size", 7))
    lee_strength = max(0.0, min(1.0, float(multiplicative_cfg.get("strength", 1.0))))

    additive_result = run_bundle_a_additive_submethod(
        sample.intensity_image,
        sample.metadata,
        additive_cfg,
    )
    lee_filtered = refined_lee_filter(additive_result.corrected_intensity, window_size=window_size)
    # Blend strength lets detector-oriented runs preserve more chip-level target
    # texture while still testing the same Bundle A multiplicative path.
    filtered = (
        additive_result.corrected_intensity * (1.0 - lee_strength)
        + lee_filtered * lee_strength
    ).astype("float32")
    return BundleProcessResult(
        additive_output=additive_result.corrected_intensity.astype("float32"),
        final_output=filtered.astype("float32"),
        additive_applied=additive_result.additive_applied,
        additive_mode=additive_result.additive_mode,
        additive_notes=additive_result.additive_notes,
        multiplicative_mode="refined_lee",
        multiplicative_notes=f"Applied Refined Lee in the intensity domain with blend strength {lee_strength:.2f}.",
        estimated_additive_component=additive_result.estimated_additive_component.astype("float32"),
        extra_arrays=additive_result.extra_arrays,
        notes=[warning for warning in [additive_result.warning] if warning],
        result_metadata=additive_result.to_metadata_fields(),
    )


def _bundle_a_sample_summary_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metrics_rows:
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "dataset": row.get("dataset", ""),
                "split": row.get("split", ""),
                "additive_submethod_code": row.get("additive_submethod_code", ""),
                "additive_submethod_name": row.get("additive_submethod_name", ""),
                "metadata_available": row.get("additive_metadata_available", ""),
                "metadata_fields_present": row.get("additive_metadata_fields_present", ""),
                "additive_correction_applied": row.get("additive_applied", ""),
                "confidence_level": row.get("additive_confidence_level", ""),
                "fallback_used": row.get("additive_fallback_used", ""),
                "proxy_enl_before": row.get("proxy_enl_before", ""),
                "proxy_enl_after": row.get("proxy_enl_after", ""),
                "edge_sharpness_before": row.get("edge_sharpness_before", ""),
                "edge_sharpness_after": row.get("edge_sharpness_after", ""),
                "separability_before": row.get("distribution_separability_before", ""),
                "separability_after": row.get("distribution_separability_after", ""),
                "threshold_f1_before": row.get("threshold_f1_before", ""),
                "threshold_f1_after": row.get("threshold_f1_after", ""),
                "notes_warnings": " ".join(
                    part
                    for part in [
                        str(row.get("additive_selection_reason", "")).strip(),
                        str(row.get("additive_warning", "")).strip(),
                    ]
                    if part
                ),
            }
        )
    return rows


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _bundle_a_aggregate_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_rows:
        grouped[str(row.get("additive_submethod_code", "unknown"))].append(row)

    aggregates: list[dict[str, Any]] = []
    for code, rows in sorted(grouped.items()):
        numeric_keys = [
            "proxy_enl_before",
            "proxy_enl_after",
            "edge_sharpness_before",
            "edge_sharpness_after",
            "distribution_separability_before",
            "distribution_separability_after",
            "threshold_f1_before",
            "threshold_f1_after",
        ]
        aggregate_row: dict[str, Any] = {
            "additive_submethod_code": code,
            "additive_submethod_name": rows[0].get("additive_submethod_name", ""),
            "description": rows[0].get("additive_submethod_description", ""),
            "required_inputs": rows[0].get("additive_submethod_required_inputs", ""),
            "confidence_level": rows[0].get("additive_confidence_level", ""),
            "sample_count": len(rows),
            "metadata_available_count": sum(bool(row.get("additive_metadata_available")) for row in rows),
            "additive_applied_count": sum(bool(row.get("additive_applied")) for row in rows),
            "warning_count": sum(bool(str(row.get("additive_warning", "")).strip()) for row in rows),
        }
        aggregate_row["metadata_available_rate"] = aggregate_row["metadata_available_count"] / max(aggregate_row["sample_count"], 1)
        aggregate_row["additive_applied_rate"] = aggregate_row["additive_applied_count"] / max(aggregate_row["sample_count"], 1)
        for key in numeric_keys:
            values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
            aggregate_row[f"mean_{key}"] = _mean(values)
        aggregates.append(aggregate_row)
    return aggregates


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows were available.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bundle_a_submethod_tables(output_root: Path) -> dict[str, Any]:
    metrics_path = output_root / "metrics" / "per_sample_metrics.json"
    if not metrics_path.exists():
        return {}
    payload = read_json(metrics_path)
    metrics_rows = payload.get("samples", [])
    sample_rows = _bundle_a_sample_summary_rows(metrics_rows)
    aggregate_rows = _bundle_a_aggregate_rows(metrics_rows)
    tables_root = output_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    write_csv(tables_root / "submethod_summary.csv", sample_rows)
    write_json(tables_root / "submethod_summary.json", {"samples": sample_rows})
    _write_markdown_table(tables_root / "submethod_summary.md", sample_rows[: min(len(sample_rows), 20)])
    write_csv(tables_root / "submethod_aggregate.csv", aggregate_rows)
    write_json(tables_root / "submethod_aggregate.json", {"groups": aggregate_rows})
    _write_markdown_table(tables_root / "submethod_aggregate.md", aggregate_rows)
    return {
        "sample_table_path": (tables_root / "submethod_summary.csv").resolve().as_posix(),
        "aggregate_table_path": (tables_root / "submethod_aggregate.csv").resolve().as_posix(),
        "groups": aggregate_rows,
    }


def run_bundle_a(
    records: list[dict[str, str]],
    *,
    dataset_name: str,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    statistics_cfg = config.get("statistics", {})
    analyzer = None
    if bool(statistics_cfg.get("enabled", True)):
        analyzer = IntensityStatisticsAnalyzer(output_root / "statistics", statistics_cfg)
    summary = run_stage1_bundle(
        records=records,
        dataset_name=dataset_name,
        config=config,
        output_root=output_root,
        processor=process_bundle_a_sample,
        bundle_name="bundle_a",
        after_title="Bundle A output",
        sample_analyzer=analyzer,
    )
    submethod_summary = write_bundle_a_submethod_tables(output_root)
    if submethod_summary:
        summary["bundle_a_submethod_tables"] = submethod_summary
        write_summary_artifacts(output_root, summary)
    return summary
