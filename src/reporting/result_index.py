from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.datasets.common import read_csv_rows, read_json
from src.datasets.registry import DatasetRegistry, default_registry_path


@dataclass(slots=True)
class BundleRunIndex:
    bundle_name: str
    dataset: str
    processed_count: int
    skipped_count: int
    output_root: str
    run_summary_path: str
    modified_timestamp: float


def resolve_bundle_output_root(repo_root: Path, bundle_name: str) -> Path:
    config_path = repo_root / "configs" / f"{bundle_name}.yaml"
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        configured_root = payload.get("outputs", {}).get("root")
        if configured_root:
            return (repo_root / configured_root).resolve()
    return (repo_root / "results" / bundle_name).resolve()


def bundle_layout(bundle_root: Path) -> dict[str, Path]:
    root = bundle_root.resolve()
    plots_root = root / "plots" if (root / "plots").exists() else root
    return {
        "root": root,
        "config": root / "config" if (root / "config").exists() else root,
        "metrics": root / "metrics" if (root / "metrics").exists() else root,
        "plots": plots_root,
        "panels": root / "plots" / "panels" if (root / "plots" / "panels").exists() else root / "plots",
        "galleries": root / "galleries" if (root / "galleries").exists() else root,
        "statistics": root / "statistics" if (root / "statistics").exists() else root,
        "tables": root / "tables" if (root / "tables").exists() else root,
        "logs": root / "logs" if (root / "logs").exists() else root,
        "side_by_side": root / "plots" / "side_by_side" if (root / "plots" / "side_by_side").exists() else root / "side_by_side",
        "diagnostic_maps": root / "plots" / "diagnostic_maps" if (root / "plots" / "diagnostic_maps").exists() else root / "diagnostic_maps",
    }


def _load_json_if_exists(path: Path, fallback: Any) -> Any:
    return read_json(path) if path.exists() else fallback


def _load_text_if_exists(path: Path, fallback: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def load_bundle_run(bundle_root: Path) -> dict[str, Any]:
    layout = bundle_layout(bundle_root)
    summary = _load_json_if_exists(layout["metrics"] / "run_summary.json", {})
    return {
        "layout": layout,
        "summary": summary,
        "aggregate_metrics": _load_json_if_exists(layout["metrics"] / "aggregate_metrics.json", {"metrics": []}),
        "per_sample_metrics": _load_json_if_exists(layout["metrics"] / "per_sample_metrics.json", {"samples": []}),
        "downstream_eval_hooks": _load_json_if_exists(layout["metrics"] / "downstream_eval_hooks.json", {"samples": []}),
        "topline_metrics": _load_json_if_exists(layout["metrics"] / "topline_metrics.json", {}),
        "sample_summary": _load_json_if_exists(layout["tables"] / "sample_summary.json", {"samples": []}),
        "submethod_summary": _load_json_if_exists(layout["tables"] / "submethod_summary.json", {"samples": []}),
        "submethod_aggregate": _load_json_if_exists(layout["tables"] / "submethod_aggregate.json", {"groups": []}),
        "statistics_summary": _load_json_if_exists(layout["statistics"] / "summary.json", summary.get("statistics_analysis", {})),
        "run_summary_markdown": _load_text_if_exists(layout["tables"] / "run_summary.md", ""),
    }


def load_sentinel1_batch_snapshot(repo_root: Path, output_name: str = "bundle_a_sentinel1_batch") -> dict[str, Any]:
    root = (repo_root / "outputs" / output_name).resolve()
    return {
        "root": root,
        "topline_metrics": _load_json_if_exists(root / "metrics" / "topline_metrics.json", {}),
        "scene_summary": _load_json_if_exists(root / "tables" / "scene_summary.json", {"scenes": []}),
        "submethod_comparison": _load_json_if_exists(root / "tables" / "submethod_comparison.json", {"rows": []}),
        "submethod_aggregate": _load_json_if_exists(root / "tables" / "submethod_aggregate.json", {"groups": []}),
        "evidence_plan": _load_json_if_exists(root / "tables" / "evidence_plan.json", {}),
        "scene_recommendations_markdown": _load_text_if_exists(root / "tables" / "scene_recommendations.md", ""),
    }


def load_detection_baseline_snapshot(repo_root: Path, output_name: str = "downstream_detection_validation_trained") -> dict[str, Any]:
    """Load the optional downstream detector baseline artifacts.

    These artifacts live under `outputs/` by default because detector datasets
    and model runs can become large.  Only the compact JSON/CSV summaries should
    be used by the app.
    """

    root = (repo_root / "outputs" / output_name).resolve()
    if not root.exists() and output_name == "downstream_detection_validation_trained":
        root = (repo_root / "outputs" / "downstream_detection").resolve()
    rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    for comparison_path in sorted(root.glob("*/metrics/downstream_comparison.json")):
        payload = _load_json_if_exists(comparison_path, {"rows": []})
        rows.extend(dict(row) for row in payload.get("rows", []))
        deltas_path = comparison_path.parent / "variant_deltas.json"
        deltas = _load_json_if_exists(deltas_path, {"rows": []})
        delta_rows.extend(dict(row) for row in deltas.get("rows", []))
        summary_path = comparison_path.parent / "run_summary.json"
        if summary_path.exists():
            dataset_summaries.append(_load_json_if_exists(summary_path, {}))
    aggregate_payload = _load_json_if_exists(root / "metrics" / "downstream_comparison.json", {"rows": []})
    aggregate_deltas = _load_json_if_exists(root / "metrics" / "variant_deltas.json", {"rows": []})
    if not rows:
        rows = [dict(row) for row in aggregate_payload.get("rows", [])]
    if not delta_rows:
        delta_rows = [dict(row) for row in aggregate_deltas.get("rows", [])]
    return {
        "root": root,
        "run_summary": _load_json_if_exists(root / "metrics" / "run_summary.json", {}),
        "dataset_summaries": dataset_summaries,
        "downstream_comparison": {"rows": rows},
        "variant_deltas": {"rows": delta_rows},
        "run_summary_markdown": _load_text_if_exists(root / "tables" / "run_summary.md", ""),
        "diagnostic_summary": _load_json_if_exists(root / "metrics" / "diagnostic_summary.json", {"rows": []}),
        "diagnostic_summary_markdown": _load_text_if_exists(root / "tables" / "diagnostic_summary.md", ""),
    }


def load_handoff_snapshot(repo_root: Path) -> dict[str, Any]:
    root = (repo_root / "results" / "handoff").resolve()
    return {
        "root": root,
        "recommendations": _load_json_if_exists(root / "project_recommendations.json", {}),
        "recommendations_markdown": _load_text_if_exists(root / "project_recommendations.md", ""),
        "bundle_matrix": read_csv_rows(root / "bundle_matrix.csv") if (root / "bundle_matrix.csv").exists() else [],
        "tuning_profiles": read_csv_rows(root / "tuning_profiles.csv") if (root / "tuning_profiles.csv").exists() else [],
        "data_strategy": read_csv_rows(root / "data_strategy.csv") if (root / "data_strategy.csv").exists() else [],
    }


def load_surface_pack_snapshot(repo_root: Path, surface: str = "public") -> dict[str, Any]:
    normalized = surface.strip().lower()
    if normalized in {"private", "handoff"}:
        return load_handoff_snapshot(repo_root)
    root = (repo_root / "results" / "public").resolve()
    return {
        "root": root,
        "recommendations": _load_json_if_exists(root / "project_summary.json", {}),
        "recommendations_markdown": _load_text_if_exists(root / "project_summary.md", ""),
        "bundle_matrix": read_csv_rows(root / "bundle_matrix.csv") if (root / "bundle_matrix.csv").exists() else [],
        "tuning_profiles": read_csv_rows(root / "tuning_profiles.csv") if (root / "tuning_profiles.csv").exists() else [],
        "data_strategy": read_csv_rows(root / "data_strategy.csv") if (root / "data_strategy.csv").exists() else [],
    }


def discover_bundle_runs(repo_root: Path) -> list[BundleRunIndex]:
    seen: set[Path] = set()
    runs: list[BundleRunIndex] = []
    for search_root in (repo_root / "results", repo_root / "outputs"):
        if not search_root.exists():
            continue
        for summary_path in search_root.rglob("run_summary.json"):
            if any(part in {"data_audit", "comparison_tables", "batch_runs"} for part in summary_path.parts):
                continue
            bundle_root = summary_path.parent.parent if summary_path.parent.name == "metrics" else summary_path.parent
            bundle_root = bundle_root.resolve()
            if bundle_root in seen:
                continue
            payload = _load_json_if_exists(summary_path, {})
            bundle_name = str(payload.get("bundle_name", "")).strip()
            if not bundle_name:
                continue
            runs.append(
                BundleRunIndex(
                    bundle_name=bundle_name,
                    dataset=str(payload.get("dataset", "")),
                    processed_count=int(payload.get("processed_count", 0)),
                    skipped_count=int(payload.get("skipped_count", 0)),
                    output_root=bundle_root.as_posix(),
                    run_summary_path=summary_path.resolve().as_posix(),
                    modified_timestamp=summary_path.stat().st_mtime,
                )
            )
            seen.add(bundle_root)
    runs.sort(key=lambda item: item.modified_timestamp, reverse=True)
    return runs


def available_bundles(repo_root: Path) -> list[str]:
    config_dir = repo_root / "configs"
    names = []
    for config_path in sorted(config_dir.glob("bundle_*.yaml")):
        names.append(config_path.stem)
    return names


def load_dataset_audit_snapshot(repo_root: Path) -> dict[str, Any]:
    audit_path = repo_root / "results" / "data_audit" / "audit_summary.json"
    return _load_json_if_exists(audit_path, {"datasets": {}})


def load_dataset_registry_snapshot(repo_root: Path) -> dict[str, Any]:
    registry = DatasetRegistry(default_registry_path(repo_root))
    return {
        dataset_name: registration
        for dataset_name, registration in sorted(registry._records.items(), key=lambda item: item[0])
    }


def load_sentinel1_manifest_rows(repo_root: Path) -> list[dict[str, str]]:
    registry = DatasetRegistry(default_registry_path(repo_root))
    registration = registry.get("sentinel1")
    if registration is not None and registration.manifest_path and Path(registration.manifest_path).exists():
        return read_csv_rows(Path(registration.manifest_path))
    manifest_path = repo_root / "data" / "external" / "manifests" / "sentinel1_manifest.csv"
    if not manifest_path.exists():
        return []
    return read_csv_rows(manifest_path)
