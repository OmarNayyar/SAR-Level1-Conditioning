from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.bundles.bundle_a_classical import run_bundle_a
from src.datasets.common import deserialize_json_field, read_csv_rows, read_json, write_csv, write_json
from src.datasets.sentinel1_loader import prepare_sentinel1_record
from src.reporting.evaluation import (
    evidence_confidence_from_counts,
    score_proxy_tradeoff,
    summarize_winner_counts,
)
from src.stage1.pipeline import load_yaml, save_config_snapshot
from src.utils import (
    ExecutionPolicy,
    decide_artifact_action,
    describe_policy,
    payload_fingerprint,
    write_artifact_index,
    write_artifact_manifest,
)


DEFAULT_BATCH_ROOT_NAME = "bundle_a_sentinel1_batch"
DEFAULT_SCENE_STATUSES = ("ready", "failed", "metadata-only")
COMPARE_SUBMETHOD_ORDER = ("A0", "A1", "A2", "A3")
ARTIFACT_SCORE_THRESHOLD = 0.10


@dataclass(slots=True)
class Sentinel1BatchArtifacts:
    output_root: Path
    scene_summary_rows: list[dict[str, Any]]
    comparison_rows: list[dict[str, Any]]
    aggregate_rows: list[dict[str, Any]]
    topline_metrics: dict[str, Any]
    warnings: list[str]


def _safe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(text: str) -> str:
    collapsed = "".join(character.lower() if character.isalnum() else "_" for character in text)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_")


def _metric_delta(row: dict[str, Any], before_key: str, after_key: str) -> float | None:
    before = _safe_float(row.get(before_key))
    after = _safe_float(row.get(after_key))
    if before is None or after is None:
        return None
    return after - before


def _normalize_statuses(statuses: Iterable[str] | None) -> set[str]:
    if not statuses:
        return {value.lower() for value in DEFAULT_SCENE_STATUSES}
    return {str(value).strip().lower() for value in statuses if str(value).strip()}


def default_batch_output_root(repo_root: Path) -> Path:
    return (repo_root / "outputs" / DEFAULT_BATCH_ROOT_NAME).resolve()


def _batch_layout(output_root: Path) -> dict[str, Path]:
    layout = {
        "root": output_root.resolve(),
        "config": output_root / "config",
        "metrics": output_root / "metrics",
        "tables": output_root / "tables",
        "logs": output_root / "logs",
        "batch_runs": output_root / "batch_runs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def load_sentinel1_manifest_records(manifest_path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(manifest_path)
    return [
        dict(row)
        for row in rows
        if str(row.get("record_type", "")).strip().lower() != "placeholder"
        and str(row.get("dataset", "")).strip().lower() == "sentinel1"
        and str(row.get("product_family") or row.get("product_type") or "").upper().startswith("GRD")
    ]


def planned_submethods_for_scene(
    *,
    metadata_available: bool,
    compare_submethods: bool,
    additive_submethod: str | None,
) -> list[str]:
    if additive_submethod and additive_submethod.lower() != "auto":
        return [additive_submethod.upper()]
    if compare_submethods:
        methods = ["A0", "A2", "A3"]
        if metadata_available:
            methods.insert(1, "A1")
        return methods
    return ["auto"]


def manifest_scene_status(record: dict[str, Any]) -> str:
    prepared_status = str(record.get("prepared_status", "")).strip().lower()
    if prepared_status:
        return prepared_status
    status = str(record.get("status", "")).strip().lower()
    return status or "unknown"


def scene_presence_flags(record: dict[str, Any]) -> dict[str, bool]:
    metadata = deserialize_json_field(record.get("metadata_json")) or {}
    return {
        "noise_xml_present": bool(str(record.get("noise_xml_path", "")).strip()),
        "calibration_xml_present": bool(str(record.get("calibration_xml_path", "")).strip()),
        "annotation_xml_present": bool(str(record.get("annotation_xml_path", "")).strip()),
        "manifest_safe_present": bool(str(record.get("manifest_safe_path", "")).strip()),
        "noise_vector_cache_present": bool(str(record.get("noise_vector_path", "")).strip()),
        "geofootprint_present": bool(bool(metadata.get("geofootprint"))),
    }


def scene_metadata_ready(record: dict[str, Any]) -> bool:
    flags = scene_presence_flags(record)
    return flags["noise_xml_present"] or flags["noise_vector_cache_present"]


def _requested_config(base_config: dict[str, Any], *, requested_submethod: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("dataset", {})["name"] = "sentinel1"
    config["dataset"]["split"] = "all"
    config["dataset"]["sample_limit"] = 1
    config["dataset"]["product_family"] = "GRD"
    config.setdefault("outputs", {})["max_visual_samples"] = 1
    config["outputs"]["save_intermediate_arrays"] = False
    config.setdefault("processing", {}).setdefault("additive", {})["submethod"] = requested_submethod
    return config


def _scene_run_identity(
    *,
    record: dict[str, Any],
    prepared: Any,
    requested_submethod: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    local_target_path = Path(str(record.get("local_target_path", "")).strip()) if str(record.get("local_target_path", "")).strip() else None
    prepared_image_path = Path(str(record.get("prepared_image_path") or record.get("image_path", "")).strip()) if str(record.get("prepared_image_path") or record.get("image_path", "")).strip() else None
    return {
        "artifact_kind": "sentinel1_scene_bundle_run",
        "scene_id": str(record.get("product_id") or record.get("sample_id") or ""),
        "product_name": str(record.get("product_name", "")),
        "requested_additive_submethod": requested_submethod,
        "config_hash": payload_fingerprint(config),
        "local_target_path": local_target_path.resolve().as_posix() if local_target_path and local_target_path.exists() else "",
        "local_target_mtime": local_target_path.stat().st_mtime if local_target_path and local_target_path.exists() else None,
        "prepared_image_path": prepared_image_path.resolve().as_posix() if prepared_image_path and prepared_image_path.exists() else "",
        "prepared_image_mtime": prepared_image_path.stat().st_mtime if prepared_image_path and prepared_image_path.exists() else None,
        "measurement_count": len(getattr(prepared, "measurement_paths", [])),
    }


def _load_single_metrics_row(run_output_root: Path) -> dict[str, Any]:
    payload = read_json(run_output_root / "metrics" / "per_sample_metrics.json")
    rows = payload.get("samples", [])
    if not rows:
        raise RuntimeError(f"No per-sample metrics were written under {run_output_root.as_posix()}.")
    return dict(rows[0])


def _scale_fallback_used(source_note: str) -> bool:
    lowered = source_note.lower()
    return "overview page" in lowered or "memory-mapped" in lowered or "stride-" in lowered


def _build_comparison_row(
    *,
    record: dict[str, Any],
    metrics_row: dict[str, Any],
    requested_submethod: str,
    run_output_root: Path,
) -> dict[str, Any]:
    flags = scene_presence_flags(record)
    scene_id = str(record.get("product_id") or record.get("sample_id") or "")
    source_note = str(metrics_row.get("source_note", "")).strip()
    scale_fallback_used = _scale_fallback_used(source_note)
    warning_parts = [
        str(metrics_row.get("additive_warning", "")).strip(),
        source_note if scale_fallback_used else "",
    ]
    notes_parts = [
        str(metrics_row.get("additive_selection_reason", "")).strip(),
        str(metrics_row.get("statistics_note", "")).strip(),
    ]
    return {
        "scene_id": scene_id,
        "product_name": record.get("product_name", ""),
        "requested_additive_submethod": requested_submethod,
        "additive_submethod_used": metrics_row.get("additive_submethod_code", ""),
        "additive_submethod_name": metrics_row.get("additive_submethod_name", ""),
        "metadata_available": bool(metrics_row.get("additive_metadata_available")),
        "metadata_fields_present": metrics_row.get("additive_metadata_fields_present", ""),
        "additive_applied": bool(metrics_row.get("additive_applied")),
        "primary_polarization": record.get("primary_polarization", ""),
        "overview_fallback_used": scale_fallback_used,
        "noise_xml_present": flags["noise_xml_present"],
        "calibration_xml_present": flags["calibration_xml_present"],
        "annotation_xml_present": flags["annotation_xml_present"],
        "manifest_safe_present": flags["manifest_safe_present"],
        "measurement_count": record.get("measurement_count", ""),
        "proxy_enl_before": metrics_row.get("proxy_enl_before"),
        "proxy_enl_after": metrics_row.get("proxy_enl_after"),
        "proxy_enl_gain": metrics_row.get("proxy_enl_gain"),
        "edge_sharpness_before": metrics_row.get("edge_sharpness_before"),
        "edge_sharpness_after": metrics_row.get("edge_sharpness_after"),
        "edge_sharpness_delta": metrics_row.get("edge_sharpness_delta"),
        "distribution_separability_before": metrics_row.get("distribution_separability_before"),
        "distribution_separability_after": metrics_row.get("distribution_separability_after"),
        "distribution_separability_delta": _metric_delta(
            metrics_row,
            "distribution_separability_before",
            "distribution_separability_after",
        ),
        "threshold_f1_before": metrics_row.get("threshold_f1_before"),
        "threshold_f1_after": metrics_row.get("threshold_f1_after"),
        "threshold_f1_delta": _metric_delta(metrics_row, "threshold_f1_before", "threshold_f1_after"),
        "artifact_score": metrics_row.get("additive_artifact_score"),
        "background_pixel_count": metrics_row.get("background_pixel_count"),
        "background_exp_scale_before": metrics_row.get("background_exp_scale_before"),
        "proxy_enl_patches_before": metrics_row.get("proxy_enl_patches_before"),
        "warnings": " ".join(part for part in warning_parts if part),
        "notes": " ".join(part for part in notes_parts if part),
        "run_output_root": run_output_root.resolve().as_posix(),
        "before_panel_path": metrics_row.get("before_panel_path", ""),
        "after_panel_path": metrics_row.get("after_panel_path", ""),
        "difference_panel_path": metrics_row.get("difference_panel_path", ""),
        "side_by_side_path": metrics_row.get("side_by_side_path", ""),
        "detection_map_path": metrics_row.get("detection_map_path", ""),
        "additive_panel_path": metrics_row.get("additive_panel_path", ""),
        "source_note": source_note,
    }


def _scene_backscatter_rank(scene_rows: list[dict[str, Any]]) -> dict[str, float]:
    reference_by_scene: dict[str, float] = {}
    for row in scene_rows:
        value = _safe_float(row.get("background_exp_scale_before"))
        if value is not None:
            scene_id = str(row.get("scene_id", ""))
            reference_by_scene.setdefault(scene_id, value)
    reference_values = list(reference_by_scene.items())
    if len(reference_values) < 3:
        return {}
    ordered = sorted(reference_values, key=lambda item: item[1])
    denominator = max(len(ordered) - 1, 1)
    return {
        scene_id: index / denominator
        for index, (scene_id, _) in enumerate(ordered)
    }


def derive_scene_regime(
    scene_rows: list[dict[str, Any]],
    *,
    scene_backscatter_rank: float | None = None,
) -> dict[str, Any]:
    if not scene_rows:
        return {
            "regime_label": "no-evidence",
            "regime_flags": [],
            "metadata_regime": "unknown",
            "overview_only_evaluation": False,
            "structured_artifact_likely": False,
            "quiet_background_available": False,
            "likely_low_backscatter_open_ocean": False,
        }

    metadata_available = any(bool(row.get("metadata_available")) for row in scene_rows)
    partial_metadata_only = not metadata_available and any(
        bool(row.get("calibration_xml_present")) or bool(row.get("manifest_safe_present")) for row in scene_rows
    )
    overview_only = all(bool(row.get("overview_fallback_used")) for row in scene_rows)
    structured_artifact = any((_safe_float(row.get("artifact_score")) or 0.0) >= ARTIFACT_SCORE_THRESHOLD for row in scene_rows)
    quiet_background = any(
        (_safe_float(row.get("background_pixel_count")) or 0.0) >= 20_000
        and (_safe_float(row.get("proxy_enl_patches_before")) or 0.0) >= 50
        for row in scene_rows
    )
    low_backscatter = scene_backscatter_rank is not None and scene_backscatter_rank <= 0.34

    flags: list[str] = []
    flags.append("metadata-rich" if metadata_available else "metadata-poor")
    if partial_metadata_only:
        flags.append("partial metadata only")
    if low_backscatter:
        flags.append("likely low-backscatter/open-ocean")
    if structured_artifact:
        flags.append("likely structured-artifact present")
    if quiet_background:
        flags.append("likely quiet background available")
    if overview_only:
        flags.append("overview-only evaluation")

    return {
        "regime_label": " | ".join(flags) if flags else "unclassified",
        "regime_flags": flags,
        "metadata_regime": "metadata-rich" if metadata_available else "metadata-poor",
        "overview_only_evaluation": overview_only,
        "structured_artifact_likely": structured_artifact,
        "quiet_background_available": quiet_background,
        "likely_low_backscatter_open_ocean": low_backscatter,
    }


def recommend_scene_submethod(scene_rows: list[dict[str, Any]], regime: dict[str, Any]) -> dict[str, Any]:
    """Recommend A0/A1/A2/A3 using balanced proxy decision heuristics.

    This is deliberately not a claim-grade metric. It ranks practical tradeoffs
    for screening and records the score components so the app can explain when
    a method wins because of real separation gains versus because of a heuristic
    regime alignment.
    """

    if not scene_rows:
        return {
            "best_submethod": "",
            "runner_up_submethod": "",
            "why": "No usable Bundle A Sentinel-1 submethod runs were available.",
            "caveats": "Fetch or prepare local GRD content first.",
            "summary": "No within-scene evidence is available yet.",
            "decision_confidence": "none",
            "decision_score_margin": None,
            "decision_basis": "No comparison rows were available.",
        }

    scored_rows: list[tuple[dict[str, Any], Any]] = []
    for row in scene_rows:
        decision = score_proxy_tradeoff(row, regime=regime)
        row.update(decision.as_fields())
        scored_rows.append((row, decision))

    scored = sorted(
        scored_rows,
        key=lambda item: item[1].score,
        reverse=True,
    )
    best_row = scored[0][0]
    runner_up_row = scored[1][0] if len(scored) > 1 else None
    best_code = str(best_row.get("additive_submethod_used", "")).upper()
    runner_code = str(runner_up_row.get("additive_submethod_used", "")).upper() if runner_up_row else ""
    best_decision = scored[0][1]
    runner_score = scored[1][1].score if len(scored) > 1 else None
    score_margin = best_decision.score - runner_score if runner_score is not None else None

    if best_code == "A1":
        why = "Metadata was available and A1 gave the strongest practical balance across separability, thresholding, and additive cleanup."
    elif best_code == "A2":
        if regime.get("metadata_regime") == "metadata-rich":
            why = (
                "A2 ranked highest on the current proxy tradeoff, but metadata exists; inspect it against A1 before "
                "treating the image-derived fallback as preferable."
            )
        else:
            why = "A2 was the strongest realistic fallback under limited metadata while keeping the practical metric balance strongest."
    elif best_code == "A3":
        why = "Structured-artifact cues were present and A3 gave the strongest practical trade-off for this scene."
    else:
        why = "The clean control baseline held up best on this scene, so extra additive cleanup did not justify itself."

    caveats: list[str] = ["proxy-only evidence"]
    if regime.get("overview_only_evaluation"):
        caveats.append("overview-only evaluation")
        caveats.append("metadata-driven A1 may be under-credited at this scale")
    if regime.get("metadata_regime") == "metadata-poor":
        caveats.append("metadata missing for A1")
    if score_margin is not None and score_margin < 0.025:
        caveats.append("close ranking")
    if not runner_up_row:
        caveats.append("only one submethod was evaluated")
    if best_decision.metric_bias_warning:
        caveats.append(best_decision.metric_bias_warning)

    summary = f"Best: {best_code or 'n/a'}"
    if runner_code:
        summary += f"; runner-up: {runner_code}"
    summary += f". {why}"

    return {
        "best_submethod": best_code,
        "runner_up_submethod": runner_code,
        "why": why,
        "caveats": "; ".join(caveats),
        "summary": summary,
        "decision_score": best_decision.score,
        "decision_score_margin": score_margin,
        "decision_confidence": best_decision.confidence,
        "decision_evidence_grade": best_decision.evidence_grade,
        "decision_basis": "Balanced proxy heuristic with capped ENL reward, edge-loss penalty, and regime-aware caveats.",
        "decision_rationale": best_decision.rationale,
        "decision_flags": best_decision.flags,
        "decision_metric_bias_warning": best_decision.metric_bias_warning,
    }


def aggregate_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        code = str(row.get("additive_submethod_used", "")).upper() or "UNKNOWN"
        grouped.setdefault(code, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for code, grouped_rows in sorted(grouped.items()):
        aggregate_row: dict[str, Any] = {
            "additive_submethod_used": code,
            "sample_count": len(grouped_rows),
            "scene_count": len({str(row.get("scene_id", "")) for row in grouped_rows}),
            "metadata_available_rate": sum(bool(row.get("metadata_available")) for row in grouped_rows) / max(len(grouped_rows), 1),
            "overview_only_rate": sum(bool(row.get("overview_fallback_used")) for row in grouped_rows) / max(len(grouped_rows), 1),
        }
        for key in (
            "proxy_enl_before",
            "proxy_enl_after",
            "proxy_enl_gain",
            "edge_sharpness_before",
            "edge_sharpness_after",
            "edge_sharpness_delta",
            "distribution_separability_before",
            "distribution_separability_after",
            "distribution_separability_delta",
            "threshold_f1_before",
            "threshold_f1_after",
            "threshold_f1_delta",
            "decision_score",
        ):
            values = [_safe_float(row.get(key)) for row in grouped_rows]
            finite_values = [value for value in values if value is not None]
            aggregate_row[f"mean_{key}"] = sum(finite_values) / len(finite_values) if finite_values else None
        aggregates.append(aggregate_row)
    return aggregates


def build_batch_topline(
    *,
    all_scene_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    scene_summary_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    total_manifest_rows = len(all_scene_rows)
    ready_rows = [row for row in all_scene_rows if str(row.get("scene_status", "")).lower() == "ready"]
    evaluated_scenes = [row for row in scene_summary_rows if bool(row.get("scene_evaluated"))]
    warnings: list[str] = []
    if len(evaluated_scenes) <= 1:
        warnings.append("Only one Sentinel-1 scene is currently evaluated, so real-product evidence is still thin.")
    warnings.append("Current Sentinel-1 evidence is still proxy-only and overview-scale for large local COG products.")

    top_submethods = [str(row.get("best_submethod", "")).upper() for row in evaluated_scenes if row.get("best_submethod")]
    recommended_submethod = ""
    if top_submethods:
        recommended_submethod = max(sorted(set(top_submethods)), key=top_submethods.count)

    winner_counts = summarize_winner_counts(evaluated_scenes)
    evaluated_count = len(evaluated_scenes)
    overview_count = sum(bool(row.get("overview_only_evaluation")) for row in evaluated_scenes)
    evidence_confidence = evidence_confidence_from_counts(evaluated_count, overview_only_count=overview_count)

    current_recommendation = (
        "Use A0 as the control, prefer A1 when trustworthy noise metadata exists and the scene appears additive-noise affected, "
        "use A2 as the metadata-poor fallback, and reserve A3 for structured artifacts. Current rankings are proxy-only."
    )

    topline = {
        "dataset": "sentinel1",
        "bundle_name": "bundle_a_sentinel1_batch",
        "total_manifest_rows": total_manifest_rows,
        "ready_scene_count": len(ready_rows),
        "evaluated_scene_count": len(evaluated_scenes),
        "comparison_row_count": len(comparison_rows),
        "metadata_rich_scene_count": sum(str(row.get("metadata_regime", "")) == "metadata-rich" for row in evaluated_scenes),
        "a1_eligible_scene_count": sum(bool(row.get("metadata_ready_for_a1")) for row in evaluated_scenes),
        "overview_only_scene_count": sum(bool(row.get("overview_only_evaluation")) for row in evaluated_scenes),
        "recommended_submethod_hint": recommended_submethod,
        "winner_counts": winner_counts,
        "a0_win_count": winner_counts["A0"],
        "a1_win_count": winner_counts["A1"],
        "a2_win_count": winner_counts["A2"],
        "a3_win_count": winner_counts["A3"],
        "evidence_confidence_level": evidence_confidence,
        "evidence_grade": "proxy-only / overview-scale" if overview_count else "screening-grade / proxy-only",
        "decision_basis": "Balanced proxy heuristic: capped ENL reward, edge-loss penalty, regime alignment, and explicit low-confidence caveats.",
        "current_recommendation": current_recommendation,
        "warnings": warnings,
    }
    return topline, warnings


def render_scene_recommendations_markdown(scene_summary_rows: list[dict[str, Any]]) -> str:
    if not scene_summary_rows:
        return "# Sentinel-1 scene recommendations\n\nNo scene-level results were available.\n"
    lines = ["# Sentinel-1 scene recommendations", ""]
    for row in scene_summary_rows:
        lines.append(f"## {row.get('scene_id', 'scene')} | {row.get('product_name', '')}")
        lines.append("")
        lines.append(f"- Status: `{row.get('scene_status', '')}`")
        lines.append(f"- Regime: {row.get('regime_label', 'n/a')}")
        lines.append(f"- Best submethod: `{row.get('best_submethod', '') or 'n/a'}`")
        lines.append(f"- Runner-up: `{row.get('runner_up_submethod', '') or 'n/a'}`")
        lines.append(f"- Why: {row.get('recommendation_why', 'n/a')}")
        lines.append(f"- Caveats: {row.get('recommendation_caveats', 'n/a')}")
        lines.append(f"- Confidence: {row.get('decision_confidence', row.get('decision_evidence_grade', 'n/a'))}")
        lines.append(f"- Decision basis: {row.get('decision_basis', 'balanced proxy heuristic')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def refresh_sentinel1_batch_decisions(output_root: Path) -> Sentinel1BatchArtifacts:
    """Recompute scene recommendations from existing comparison tables.

    This is useful when only the decision heuristic changes. It avoids rerunning
    the expensive per-scene Bundle A processors while keeping the app-facing
    `scene_summary`, `submethod_comparison`, and `topline_metrics` artifacts in
    sync with the current recommendation logic.
    """

    layout = _batch_layout(output_root)
    scene_payload = read_json(layout["tables"] / "scene_summary.json")
    comparison_payload = read_json(layout["tables"] / "submethod_comparison.json")
    scene_summary_rows = [dict(row) for row in scene_payload.get("scenes", [])]
    comparison_rows = [dict(row) for row in comparison_payload.get("rows", [])]

    backscatter_ranks = _scene_backscatter_rank(comparison_rows)
    for scene_row in scene_summary_rows:
        if not scene_row.get("scene_evaluated"):
            continue
        scene_id = str(scene_row.get("scene_id", ""))
        scene_rows = [row for row in comparison_rows if str(row.get("scene_id", "")) == scene_id]
        regime = derive_scene_regime(scene_rows, scene_backscatter_rank=backscatter_ranks.get(scene_id))
        recommendation = recommend_scene_submethod(scene_rows, regime)
        scene_row.update(regime)
        scene_row.update(
            {
                "best_submethod": recommendation["best_submethod"],
                "runner_up_submethod": recommendation["runner_up_submethod"],
                "recommendation_why": recommendation["why"],
                "recommendation_caveats": recommendation["caveats"],
                "recommendation_summary": recommendation["summary"],
                "decision_score": recommendation["decision_score"],
                "decision_score_margin": recommendation["decision_score_margin"],
                "decision_confidence": recommendation["decision_confidence"],
                "decision_evidence_grade": recommendation["decision_evidence_grade"],
                "decision_basis": recommendation["decision_basis"],
                "decision_rationale": recommendation["decision_rationale"],
                "decision_flags": recommendation["decision_flags"],
                "decision_metric_bias_warning": recommendation["decision_metric_bias_warning"],
            }
        )

    for row in comparison_rows:
        matching_scene = next(
            (scene for scene in scene_summary_rows if str(scene.get("scene_id", "")) == str(row.get("scene_id", ""))),
            None,
        )
        if matching_scene is not None:
            row["regime_label"] = matching_scene.get("regime_label", "")
            row["best_submethod_for_scene"] = matching_scene.get("best_submethod", "")

    aggregate_rows = aggregate_comparison_rows(comparison_rows)
    topline_metrics, warnings = build_batch_topline(
        all_scene_rows=scene_summary_rows,
        comparison_rows=comparison_rows,
        scene_summary_rows=scene_summary_rows,
    )
    write_csv(layout["tables"] / "scene_summary.csv", scene_summary_rows)
    write_json(layout["tables"] / "scene_summary.json", {"scenes": scene_summary_rows})
    write_csv(layout["tables"] / "submethod_comparison.csv", comparison_rows)
    write_json(layout["tables"] / "submethod_comparison.json", {"rows": comparison_rows})
    write_csv(layout["tables"] / "submethod_aggregate.csv", aggregate_rows)
    write_json(layout["tables"] / "submethod_aggregate.json", {"groups": aggregate_rows})
    write_csv(layout["tables"] / "evidence_maturity.csv", [topline_metrics])
    (layout["tables"] / "scene_recommendations.md").write_text(
        render_scene_recommendations_markdown(scene_summary_rows),
        encoding="utf-8",
    )
    write_json(layout["metrics"] / "topline_metrics.json", topline_metrics)
    return Sentinel1BatchArtifacts(
        output_root=layout["root"],
        scene_summary_rows=scene_summary_rows,
        comparison_rows=comparison_rows,
        aggregate_rows=aggregate_rows,
        topline_metrics=topline_metrics,
        warnings=warnings,
    )


def inspect_sentinel1_manifest(
    records: list[dict[str, Any]],
    *,
    statuses: Iterable[str] | None = None,
    polarization: str | None = None,
) -> dict[str, Any]:
    normalized_statuses = _normalize_statuses(statuses)
    selected_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    ready_rows: list[dict[str, Any]] = []
    for record in records:
        status = manifest_scene_status(record)
        primary_pol = str(record.get("primary_polarization", "")).upper()
        if polarization and primary_pol and primary_pol != polarization.upper():
            continue
        scene_row = {
            "scene_id": str(record.get("product_id") or record.get("sample_id") or ""),
            "product_name": record.get("product_name", ""),
            "scene_status": status,
            "primary_polarization": primary_pol,
            "local_target_path": record.get("local_target_path", ""),
            "prepared_image_path": record.get("prepared_image_path", ""),
            "notes": record.get("notes", ""),
        }
        if status in normalized_statuses:
            selected_rows.append(scene_row)
        if status == "ready":
            ready_rows.append(scene_row)
        if status != "ready":
            missing_rows.append(scene_row)
    return {
        "selected_rows": selected_rows,
        "ready_rows": ready_rows,
        "missing_rows": missing_rows,
        "recommendation": (
            "All selected GRD products are locally ready. Rerun the Bundle A Sentinel-1 batch comparison."
            if not missing_rows and ready_rows
            else "Fetch or prepare the missing GRD products first, then rerun the Bundle A Sentinel-1 batch comparison."
        ),
    }


def evaluate_bundle_a_sentinel1_batch(
    *,
    repo_root: Path,
    config_path: Path,
    manifest_path: Path,
    output_root: Path,
    statuses: Iterable[str] | None = None,
    max_scenes: int | None = None,
    polarization: str | None = None,
    additive_submethod: str | None = None,
    compare_submethods: bool = False,
    policy: ExecutionPolicy | None = None,
) -> Sentinel1BatchArtifacts:
    policy = policy or ExecutionPolicy(reuse_only=True)
    base_config = load_yaml(config_path.resolve(), expected_kind="bundle")
    manifest_rows = load_sentinel1_manifest_records(manifest_path.resolve())
    layout = _batch_layout(output_root.resolve())
    save_config_snapshot(base_config, layout["config"])
    write_json(
        layout["config"] / "batch_request.json",
        {
            "config_path": config_path.resolve().as_posix(),
            "manifest_path": manifest_path.resolve().as_posix(),
            "statuses": sorted(_normalize_statuses(statuses)),
            "max_scenes": max_scenes,
            "polarization": polarization or "",
            "additive_submethod": additive_submethod or "auto",
            "compare_submethods": compare_submethods,
        },
    )

    normalized_statuses = _normalize_statuses(statuses)
    all_scene_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    scene_summary_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    artifact_index_rows: list[dict[str, Any]] = []

    ready_counter = 0
    for record in manifest_rows:
        prepared = prepare_sentinel1_record(record, repo_root=repo_root)
        scene_id = str(record.get("product_id") or record.get("sample_id") or "")
        scene_status = "ready" if prepared.usable else manifest_scene_status(record)
        primary_pol = str(record.get("primary_polarization") or prepared.primary_polarization or "").upper()
        if polarization and primary_pol and primary_pol != polarization.upper():
            continue

        flags = scene_presence_flags(record)
        metadata_ready = scene_metadata_ready(record)
        scene_row = {
            "scene_id": scene_id,
            "product_name": record.get("product_name", ""),
            "scene_status": scene_status,
            "scene_evaluated": False,
            "primary_polarization": primary_pol,
            "measurement_count": int(record.get("measurement_count") or len(prepared.measurement_paths) or 0),
            "noise_xml_present": flags["noise_xml_present"],
            "calibration_xml_present": flags["calibration_xml_present"],
            "annotation_xml_present": flags["annotation_xml_present"],
            "manifest_safe_present": flags["manifest_safe_present"],
            "metadata_ready_for_a1": metadata_ready,
            "local_path": record.get("local_target_path", ""),
            "prepared_image_path": record.get("prepared_image_path") or record.get("image_path", ""),
            "notes": prepared.notes or record.get("notes", ""),
        }
        all_scene_rows.append(scene_row)

        if scene_status not in normalized_statuses:
            scene_row["notes"] = f"{scene_row['notes']} Filtered out by status selection.".strip()
            scene_summary_rows.append(scene_row)
            continue
        if not prepared.usable:
            scene_row["recommendation_why"] = "Local SAFE content is not usable yet."
            scene_row["recommendation_caveats"] = prepared.notes or record.get("notes", "")
            scene_summary_rows.append(scene_row)
            skipped_rows.append(
                {
                    "scene_id": scene_id,
                    "product_name": record.get("product_name", ""),
                    "reason": prepared.notes or record.get("notes", ""),
                }
            )
            continue
        if max_scenes is not None and ready_counter >= max_scenes:
            scene_row["notes"] = f"{scene_row['notes']} Not evaluated because max_scenes was reached.".strip()
            scene_summary_rows.append(scene_row)
            continue

        ready_counter += 1
        submethods = planned_submethods_for_scene(
            metadata_available=metadata_ready,
            compare_submethods=compare_submethods,
            additive_submethod=additive_submethod,
        )
        scene_comparison_rows: list[dict[str, Any]] = []
        for requested_submethod in submethods:
            run_dir_name = f"{_slug(scene_id or record.get('product_name', 'scene'))}__{_slug(requested_submethod)}"
            run_output_root = layout["batch_runs"] / run_dir_name
            config = _requested_config(base_config, requested_submethod=requested_submethod)
            identity = _scene_run_identity(
                record=record,
                prepared=prepared,
                requested_submethod=requested_submethod,
                config=config,
            )
            decision = decide_artifact_action(
                artifact_kind="sentinel1_scene_bundle_run",
                output_root=run_output_root,
                identity=identity,
                required_files=["metrics/run_summary.json", "metrics/per_sample_metrics.json", "tables/run_summary.md"],
                capability="conditioning",
                policy=policy,
                accept_existing_without_manifest=True,
            )
            try:
                if decision.action == "run":
                    run_bundle_a([record], dataset_name="sentinel1", config=config, output_root=run_output_root)
                elif decision.action == "would_run":
                    skipped_rows.append(
                        {
                            "scene_id": scene_id,
                            "product_name": record.get("product_name", ""),
                            "reason": f"{requested_submethod}: {decision.reason}",
                        }
                    )
                    artifact_index_rows.append(
                        {
                            "scene_id": scene_id,
                            "requested_submethod": requested_submethod,
                            "action": decision.action,
                            "run_output_root": run_output_root.resolve().as_posix(),
                            "reason": decision.reason,
                        }
                    )
                    continue
                elif decision.action == "blocked":
                    skipped_rows.append(
                        {
                            "scene_id": scene_id,
                            "product_name": record.get("product_name", ""),
                            "reason": f"{requested_submethod}: {decision.reason}",
                        }
                    )
                    artifact_index_rows.append(
                        {
                            "scene_id": scene_id,
                            "requested_submethod": requested_submethod,
                            "action": decision.action,
                            "run_output_root": run_output_root.resolve().as_posix(),
                            "reason": decision.reason,
                        }
                    )
                    continue
                metrics_row = _load_single_metrics_row(run_output_root)
                comparison_row = _build_comparison_row(
                    record=record,
                    metrics_row=metrics_row,
                    requested_submethod=requested_submethod,
                    run_output_root=run_output_root,
                )
                comparison_rows.append(comparison_row)
                scene_comparison_rows.append(comparison_row)
                write_artifact_manifest(
                    run_output_root,
                    artifact_kind="sentinel1_scene_bundle_run",
                    identity=identity,
                    status="complete",
                    files={
                        "run_summary": (run_output_root / "metrics" / "run_summary.json").resolve().as_posix(),
                        "per_sample_metrics": (run_output_root / "metrics" / "per_sample_metrics.json").resolve().as_posix(),
                        "markdown_summary": (run_output_root / "tables" / "run_summary.md").resolve().as_posix(),
                    },
                    metadata={
                        "scene_id": scene_id,
                        "requested_submethod": requested_submethod,
                        "decision_policy": describe_policy(policy),
                    },
                    notes=[decision.reason],
                )
                artifact_index_rows.append(
                    {
                        "scene_id": scene_id,
                        "requested_submethod": requested_submethod,
                        "action": decision.action,
                        "run_output_root": run_output_root.resolve().as_posix(),
                        "reason": decision.reason,
                    }
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "scene_id": scene_id,
                        "product_name": record.get("product_name", ""),
                        "reason": f"{requested_submethod}: {exc}",
                    }
                )

        regime = derive_scene_regime(scene_comparison_rows)
        recommendation = recommend_scene_submethod(scene_comparison_rows, regime)
        scene_row.update(regime)
        scene_row.update(
            {
                "scene_evaluated": bool(scene_comparison_rows),
                "submethods_ran": ", ".join(str(row.get("additive_submethod_used", "")) for row in scene_comparison_rows),
                "comparison_row_count": len(scene_comparison_rows),
                "best_submethod": recommendation["best_submethod"],
                "runner_up_submethod": recommendation["runner_up_submethod"],
                "recommendation_why": recommendation["why"],
                "recommendation_caveats": recommendation["caveats"],
                "recommendation_summary": recommendation["summary"],
                "decision_score": recommendation["decision_score"],
                "decision_score_margin": recommendation["decision_score_margin"],
                "decision_confidence": recommendation["decision_confidence"],
                "decision_evidence_grade": recommendation["decision_evidence_grade"],
                "decision_basis": recommendation["decision_basis"],
                "decision_rationale": recommendation["decision_rationale"],
                "decision_flags": recommendation["decision_flags"],
                "decision_metric_bias_warning": recommendation["decision_metric_bias_warning"],
                "overview_only_evaluation": regime["overview_only_evaluation"],
                "structured_artifact_likely": regime["structured_artifact_likely"],
                "quiet_background_available": regime["quiet_background_available"],
                "likely_low_backscatter_open_ocean": regime["likely_low_backscatter_open_ocean"],
            }
        )
        scene_summary_rows.append(scene_row)

    backscatter_ranks = _scene_backscatter_rank(comparison_rows)
    if backscatter_ranks:
        for row in scene_summary_rows:
            if not row.get("scene_evaluated"):
                continue
            scene_id = str(row.get("scene_id", ""))
            scene_rows = [comparison_row for comparison_row in comparison_rows if str(comparison_row.get("scene_id", "")) == scene_id]
            if scene_id in backscatter_ranks:
                updated_regime = derive_scene_regime(scene_rows, scene_backscatter_rank=backscatter_ranks[scene_id])
                recommendation = recommend_scene_submethod(scene_rows, updated_regime)
                row.update(updated_regime)
                row["best_submethod"] = recommendation["best_submethod"]
                row["runner_up_submethod"] = recommendation["runner_up_submethod"]
                row["recommendation_why"] = recommendation["why"]
                row["recommendation_caveats"] = recommendation["caveats"]
                row["recommendation_summary"] = recommendation["summary"]
                row["decision_score"] = recommendation["decision_score"]
                row["decision_score_margin"] = recommendation["decision_score_margin"]
                row["decision_confidence"] = recommendation["decision_confidence"]
                row["decision_evidence_grade"] = recommendation["decision_evidence_grade"]
                row["decision_basis"] = recommendation["decision_basis"]
                row["decision_rationale"] = recommendation["decision_rationale"]
                row["decision_flags"] = recommendation["decision_flags"]
                row["decision_metric_bias_warning"] = recommendation["decision_metric_bias_warning"]

    for row in comparison_rows:
        scene_id = str(row.get("scene_id", ""))
        matching_scene = next((scene for scene in scene_summary_rows if str(scene.get("scene_id", "")) == scene_id), None)
        if matching_scene is not None:
            row["regime_label"] = matching_scene.get("regime_label", "")
            row["best_submethod_for_scene"] = matching_scene.get("best_submethod", "")

    aggregate_rows = aggregate_comparison_rows(comparison_rows)
    topline_metrics, warnings = build_batch_topline(
        all_scene_rows=all_scene_rows,
        comparison_rows=comparison_rows,
        scene_summary_rows=scene_summary_rows,
    )

    write_csv(layout["tables"] / "scene_summary.csv", scene_summary_rows)
    write_json(layout["tables"] / "scene_summary.json", {"scenes": scene_summary_rows})
    write_csv(layout["tables"] / "submethod_comparison.csv", comparison_rows)
    write_json(layout["tables"] / "submethod_comparison.json", {"rows": comparison_rows})
    write_csv(layout["tables"] / "submethod_aggregate.csv", aggregate_rows)
    write_json(layout["tables"] / "submethod_aggregate.json", {"groups": aggregate_rows})
    write_csv(layout["tables"] / "evidence_maturity.csv", [topline_metrics])
    write_artifact_index(layout["metrics"] / "artifact_index.json", artifact_index_rows)
    (layout["tables"] / "scene_recommendations.md").write_text(
        render_scene_recommendations_markdown(scene_summary_rows),
        encoding="utf-8",
    )
    write_csv(layout["logs"] / "skipped_scenes.csv", skipped_rows)
    write_json(layout["metrics"] / "topline_metrics.json", topline_metrics)
    write_artifact_manifest(
        layout["root"],
        artifact_kind="sentinel1_batch",
        identity={
            "config_path": config_path.resolve().as_posix(),
            "manifest_path": manifest_path.resolve().as_posix(),
            "statuses": sorted(_normalize_statuses(statuses)),
            "max_scenes": max_scenes,
            "polarization": polarization or "",
            "requested_submethod": additive_submethod or "auto",
            "compare_submethods": compare_submethods,
        },
        status="complete",
        files={
            "scene_summary_json": (layout["tables"] / "scene_summary.json").resolve().as_posix(),
            "submethod_comparison_json": (layout["tables"] / "submethod_comparison.json").resolve().as_posix(),
            "topline_metrics_json": (layout["metrics"] / "topline_metrics.json").resolve().as_posix(),
            "artifact_index_json": (layout["metrics"] / "artifact_index.json").resolve().as_posix(),
        },
        metadata={"execution_policy": describe_policy(policy), "warning_count": len(warnings)},
        notes=warnings,
    )

    return Sentinel1BatchArtifacts(
        output_root=layout["root"],
        scene_summary_rows=scene_summary_rows,
        comparison_rows=comparison_rows,
        aggregate_rows=aggregate_rows,
        topline_metrics=topline_metrics,
        warnings=warnings,
    )
