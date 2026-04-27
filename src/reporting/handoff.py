from __future__ import annotations

"""Surface-specific recommendation synthesis.

This module deliberately separates recommendations from raw metrics.  Detector
metrics, proxy metrics, and Sentinel-1 routing summaries each answer a different
question; the generated reports turn them into practical guidance without
pretending the evidence is stronger than it is. Public reports avoid private
operational framing, while private reports include handoff guidance.
"""

from pathlib import Path
from typing import Any

from src.datasets.common import read_csv_rows, read_json, write_csv, write_json
from src.utils import write_artifact_manifest


BUNDLE_MATRIX_ROWS: list[dict[str, str]] = [
    {
        "bundle": "A",
        "candidate_role": "interpretable conditioning family",
        "intended_domain": "intensity / power",
        "intended_regime": "metadata-rich products via A1; metadata-poor chips via A2; structured artifacts via A3",
        "additive_method": "A0 no-op, A1 metadata thermal/noise-vector subtraction, A2 image floor, A3 structured artifact cleanup",
        "multiplicative_method": "Refined Lee",
        "maturity": "screening-grade candidate",
        "current_evidence": "strong proxy interpretability, but detector-tested A variants reduce YOLO performance versus raw",
        "expected_use_case": "ablation, explainable routing, and metadata-availability studies",
    },
    {
        "bundle": "B",
        "candidate_role": "structured/additive artifact specialist",
        "intended_domain": "log-intensity",
        "intended_regime": "stripe-like, column/row-profile artifacts, harder additive contamination",
        "additive_method": "low-rank/sparse-inspired destriping",
        "multiplicative_method": "MuLoG-style log-domain denoising with BM3D/wavelet/gaussian fallback",
        "maturity": "secondary candidate",
        "current_evidence": "strongest paired Mendeley PSNR/SSIM/MSE result; current YOLO detector still prefers raw",
        "expected_use_case": "artifact stress tests and scenes where structured additive contamination is visible",
    },
    {
        "bundle": "C",
        "candidate_role": "complex/SLC future path",
        "intended_domain": "complex SLC preferred; demo fallback on intensity chips",
        "intended_regime": "true complex-valued data where Re/Im denoising and self-supervision are meaningful",
        "additive_method": "starlet shrinkage on complex components",
        "multiplicative_method": "MERLIN wrapper / documented fallback",
        "maturity": "feasibility-grade",
        "current_evidence": "not claim-grade without better complex SLC coverage",
        "expected_use_case": "future SLC experiments, not current downstream detector recommendation",
    },
    {
        "bundle": "D",
        "candidate_role": "metadata-poor learned/inverse candidate",
        "intended_domain": "intensity or log-intensity",
        "intended_regime": "metadata-poor scenes where inverse/self-supervised cleanup is worth testing",
        "additive_method": "PnP-ADMM additive cleanup",
        "multiplicative_method": "Speckle2Void-style blind-spot wrapper / fallback",
        "maturity": "secondary candidate",
        "current_evidence": "structure-preserving candidate; improves paired SSIM/edge preservation versus raw but trails raw detector mAP",
        "expected_use_case": "second-line detector comparison and metadata-poor robustness experiments",
    },
]


TUNING_PROFILE_ROWS: list[dict[str, str]] = [
    {
        "bundle": "A",
        "profile": "conservative",
        "config": "configs/bundles/profiles/bundle_a_conservative.yaml",
        "purpose": "preserve detector texture and edges while testing mild additive/speckle cleanup",
        "main_knobs": "Lee strength, image-floor quantile, artifact threshold/correction strength",
    },
    {
        "bundle": "A",
        "profile": "balanced",
        "config": "configs/bundles/profiles/bundle_a_balanced.yaml",
        "purpose": "current proxy-screening default",
        "main_knobs": "A2 floor quantile, A3 threshold, Refined Lee strength",
    },
    {
        "bundle": "A",
        "profile": "aggressive",
        "config": "configs/bundles/profiles/bundle_a_aggressive.yaml",
        "purpose": "stress-test cleanup when speckle/artifacts are visibly strong",
        "main_knobs": "higher floor/correction strength and full Refined Lee",
    },
    {
        "bundle": "B",
        "profile": "conservative/balanced/aggressive",
        "config": "configs/bundles/profiles/bundle_b_*.yaml",
        "purpose": "test structured artifact cleanup strength without hand-editing code",
        "main_knobs": "destripe correction strength, profile/background sigma, MuLoG blend strength",
    },
    {
        "bundle": "D",
        "profile": "conservative/balanced/aggressive",
        "config": "configs/bundles/profiles/bundle_d_*.yaml",
        "purpose": "test inverse/self-supervised cleanup strength against detector edge retention",
        "main_knobs": "PnP iterations/rho/strength, Speckle2Void fallback sigma/blend strength",
    },
]


DATA_STRATEGY_ROWS: list[dict[str, str]] = [
    {
        "dataset": "SSDD",
        "role": "public ship detector validation",
        "current_use": "final sweep completed for raw/A/A-conservative/B/D",
        "next_action": "inspect failures and only retune if a detector-adoption claim is needed",
    },
    {
        "dataset": "HRSID",
        "role": "public ship detector validation",
        "current_use": "final sweep completed for raw/A/A-conservative/B/D",
        "next_action": "inspect small-target failure cases and only retune if a detector-adoption claim is needed",
    },
    {
        "dataset": "Sentinel-1 GRD",
        "role": "real-product routing and metadata availability evidence",
        "current_use": "proxy-only, overview-scale Bundle A submethod routing",
        "next_action": "add targeted maritime GRD scenes and validate full-resolution subsets where feasible",
    },
    {
        "dataset": "Private/internal future data",
        "role": "handoff / deployment-relevant validation",
        "current_use": "not available yet; adapter scaffold exists",
        "next_action": "validate COCO/YOLO/bbox-CSV compatibility, then run raw-vs-best-candidates before deployment",
    },
]

PUBLIC_DATA_STRATEGY_ROWS = [
    row
    for row in DATA_STRATEGY_ROWS
    if not row["dataset"].lower().startswith("private/internal")
]


def _candidate_detection_roots(repo_root: Path, detection_output_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if detection_output_root is not None:
        candidates.append(detection_output_root.resolve())
    candidates.extend(
        [
            (repo_root / "outputs" / "downstream_detection_validation_trained").resolve(),
            (repo_root / "outputs" / "downstream_detection_baseline").resolve(),
            (repo_root / "outputs" / "downstream_detection").resolve(),
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _read_detection_rows(
    repo_root: Path,
    *,
    detection_output_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for root in _candidate_detection_roots(repo_root, detection_output_root):
        metrics_root = root / "metrics"
        rows_path = metrics_root / "downstream_comparison.json"
        deltas_path = metrics_root / "variant_deltas.json"
        if not rows_path.exists() and not deltas_path.exists():
            continue
        rows = [dict(row) for row in read_json(rows_path).get("rows", [])] if rows_path.exists() else []
        delta_rows = [dict(row) for row in read_json(deltas_path).get("rows", [])] if deltas_path.exists() else []
        return rows, delta_rows
    return [], []


def _read_sentinel_topline(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "outputs" / "bundle_a_sentinel1_batch" / "metrics" / "topline_metrics.json"
    return read_json(path) if path.exists() else {}


def _variant_winner_summary(delta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for row in delta_rows:
        variant = str(row.get("comparison_variant", ""))
        try:
            delta = float(row.get("delta_map"))
        except (TypeError, ValueError):
            continue
        if delta > 0:
            wins[variant] = wins.get(variant, 0) + 1
        elif delta < 0:
            losses[variant] = losses.get(variant, 0) + 1
    return {"mAP_wins_vs_raw": wins, "mAP_losses_vs_raw": losses}


def build_project_recommendations(
    repo_root: Path,
    *,
    detection_output_root: Path | None = None,
) -> dict[str, Any]:
    detector_rows, delta_rows = _read_detection_rows(repo_root, detection_output_root=detection_output_root)
    sentinel_topline = _read_sentinel_topline(repo_root)
    winner_summary = _variant_winner_summary(delta_rows)
    raw_currently_best = bool(delta_rows) and not winner_summary["mAP_wins_vs_raw"]
    current_recommendation = (
        "For ship detection handoff, start with raw imagery as the operational baseline and treat Stage-1 conditioning as an opt-in candidate that must beat raw on the target detector."
        if raw_currently_best
        else "At least one conditioned route beat raw in the current detector artifacts; validate that result on larger splits before recommending it."
    )
    return {
        "status": "handoff-draft",
        "current_recommendation": current_recommendation,
        "team_can_use_immediately": [
            "YOLO-format SSDD/HRSID preparation for raw, Bundle A, Bundle B, Bundle D, and conservative Bundle A variants.",
            "Downstream detector comparison tables with mAP / precision / recall / F1.",
            "Target-local diagnostics showing contrast, variance, and edge-retention tradeoffs.",
            "External dataset validation scaffold for COCO, YOLO, and bbox CSV-style restricted local handoffs.",
        ],
        "do_not_overclaim": [
            "Current detector validation is a lightweight compatibility run, not tuned SOTA.",
            "Sentinel-1 scene evidence is proxy-only / overview-scale.",
            "Paired denoising gains and detector gains are different evidence tracks.",
        ],
        "next_validation_steps": [
            "Validate Bundle B and Bundle D on representative operational raw/noisy products.",
            "Inspect detector failure cases before deciding whether retuning is worthwhile.",
            "Add targeted Sentinel-1 maritime GRD scenes for routing evidence.",
            "Validate any private/internal dataset with the external adapter before bundle processing.",
        ],
        "bundle_matrix": BUNDLE_MATRIX_ROWS,
        "tuning_profiles": TUNING_PROFILE_ROWS,
        "data_strategy": DATA_STRATEGY_ROWS,
        "detector_row_count": len(detector_rows),
        "detector_delta_summary": winner_summary,
        "sentinel1_topline": sentinel_topline,
    }


def build_public_project_summary(
    repo_root: Path,
    *,
    detection_output_root: Path | None = None,
) -> dict[str, Any]:
    """Build a public-safe summary from the same evidence sources."""

    detector_rows, delta_rows = _read_detection_rows(repo_root, detection_output_root=detection_output_root)
    winner_summary = _variant_winner_summary(delta_rows)
    raw_currently_best = bool(delta_rows) and not winner_summary["mAP_wins_vs_raw"]
    current_recommendation = (
        "For public ship-detection experiments, raw imagery remains the current baseline; conditioning routes should be evaluated as optional candidates against raw."
        if raw_currently_best
        else "At least one conditioned route beat raw in the current artifacts; validate that result on larger splits before presenting it as robust."
    )
    return {
        "status": "public-summary",
        "public_recommendation": current_recommendation,
        "project_positioning": [
            "A modular SAR Level-1 conditioning framework, not a universal denoiser.",
            "Separates additive-noise handling from multiplicative speckle handling.",
            "Reports both proxy-screening metrics and downstream detector validation when available.",
        ],
        "public_findings": [
            "Bundle B improves paired denoising metrics versus raw noisy input on the Mendeley validation split.",
            "Raw imagery remains strongest for the current SSDD/HRSID YOLO detector compatibility run.",
            "Paired denoising and detector compatibility answer different questions.",
        ],
        "public_caveats": [
            "Detector runs are lightweight compatibility evidence and should not be presented as tuned SOTA.",
            "Sentinel-1 evidence is proxy-only / overview-scale.",
            "Bundle C remains feasibility-only without stronger complex SLC coverage.",
        ],
        "bundle_matrix": BUNDLE_MATRIX_ROWS,
        "tuning_profiles": TUNING_PROFILE_ROWS,
        "data_strategy": PUBLIC_DATA_STRATEGY_ROWS,
        "detector_row_count": len(detector_rows),
        "detector_delta_summary": winner_summary,
    }


def write_handoff_artifacts(
    repo_root: Path,
    output_root: Path | None = None,
    *,
    detection_output_root: Path | None = None,
) -> dict[str, str]:
    output_root = output_root or repo_root / "results" / "handoff"
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_project_recommendations(repo_root, detection_output_root=detection_output_root)
    write_json(output_root / "project_recommendations.json", payload)
    write_csv(output_root / "bundle_matrix.csv", payload["bundle_matrix"])
    write_csv(output_root / "tuning_profiles.csv", payload["tuning_profiles"])
    write_csv(output_root / "data_strategy.csv", payload["data_strategy"])
    (output_root / "project_recommendations.md").write_text(_render_markdown(payload), encoding="utf-8")
    write_artifact_manifest(
        output_root,
        artifact_kind="surface_pack",
        identity={"surface": "private", "payload_status": payload.get("status", "")},
        status="complete",
        files={
            "summary_json": (output_root / "project_recommendations.json").resolve().as_posix(),
            "summary_markdown": (output_root / "project_recommendations.md").resolve().as_posix(),
            "bundle_matrix_csv": (output_root / "bundle_matrix.csv").resolve().as_posix(),
        },
        metadata={"surface": "private"},
        notes=["Generated from cached results only; no heavy compute was triggered."],
    )
    return {
        "project_recommendations_json": (output_root / "project_recommendations.json").resolve().as_posix(),
        "project_recommendations_md": (output_root / "project_recommendations.md").resolve().as_posix(),
        "bundle_matrix_csv": (output_root / "bundle_matrix.csv").resolve().as_posix(),
        "tuning_profiles_csv": (output_root / "tuning_profiles.csv").resolve().as_posix(),
        "data_strategy_csv": (output_root / "data_strategy.csv").resolve().as_posix(),
    }


def write_public_artifacts(
    repo_root: Path,
    output_root: Path | None = None,
    *,
    detection_output_root: Path | None = None,
) -> dict[str, str]:
    output_root = output_root or repo_root / "results" / "public"
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_public_project_summary(repo_root, detection_output_root=detection_output_root)
    write_json(output_root / "project_summary.json", payload)
    write_csv(output_root / "bundle_matrix.csv", payload["bundle_matrix"])
    write_csv(output_root / "tuning_profiles.csv", payload["tuning_profiles"])
    write_csv(output_root / "data_strategy.csv", payload["data_strategy"])
    (output_root / "project_summary.md").write_text(_render_public_markdown(payload), encoding="utf-8")
    write_artifact_manifest(
        output_root,
        artifact_kind="surface_pack",
        identity={"surface": "public", "payload_status": payload.get("status", "")},
        status="complete",
        files={
            "summary_json": (output_root / "project_summary.json").resolve().as_posix(),
            "summary_markdown": (output_root / "project_summary.md").resolve().as_posix(),
            "bundle_matrix_csv": (output_root / "bundle_matrix.csv").resolve().as_posix(),
        },
        metadata={"surface": "public"},
        notes=["Generated from cached results only; no heavy compute was triggered."],
    )
    return {
        "project_summary_json": (output_root / "project_summary.json").resolve().as_posix(),
        "project_summary_md": (output_root / "project_summary.md").resolve().as_posix(),
        "bundle_matrix_csv": (output_root / "bundle_matrix.csv").resolve().as_posix(),
        "tuning_profiles_csv": (output_root / "tuning_profiles.csv").resolve().as_posix(),
        "data_strategy_csv": (output_root / "data_strategy.csv").resolve().as_posix(),
    }


def write_surface_artifacts(
    repo_root: Path,
    *,
    surface: str = "all",
    detection_output_root: Path | None = None,
) -> dict[str, dict[str, str]]:
    normalized = surface.strip().lower()
    written: dict[str, dict[str, str]] = {}
    if normalized in {"all", "public"}:
        written["public"] = write_public_artifacts(
            repo_root,
            detection_output_root=detection_output_root,
        )
    if normalized in {"all", "private", "handoff"}:
        written["private"] = write_handoff_artifacts(
            repo_root,
            detection_output_root=detection_output_root,
        )
    if not written:
        raise ValueError("surface must be one of: public, private, all")
    return written


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# SAR Stage-1 Conditioning Handoff Recommendation", ""]
    lines.append(f"**Current recommendation:** {payload['current_recommendation']}")
    lines.extend(["", "## Team Can Use Immediately", ""])
    lines.extend(f"- {item}" for item in payload["team_can_use_immediately"])
    lines.extend(["", "## Do Not Overclaim", ""])
    lines.extend(f"- {item}" for item in payload["do_not_overclaim"])
    lines.extend(["", "## Next Validation Steps", ""])
    lines.extend(f"- {item}" for item in payload["next_validation_steps"])
    lines.extend(["", "## Bundle Matrix", ""])
    lines.append("| Bundle | Role | Domain | Maturity | Current Evidence |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["bundle_matrix"]:
        lines.append(
            f"| {row['bundle']} | {row['candidate_role']} | {row['intended_domain']} | {row['maturity']} | {row['current_evidence']} |"
        )
    lines.extend(["", "## Data Strategy", ""])
    lines.append("| Dataset | Role | Current Use | Next Action |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["data_strategy"]:
        lines.append(f"| {row['dataset']} | {row['role']} | {row['current_use']} | {row['next_action']} |")
    lines.append("")
    return "\n".join(lines)


def _render_public_markdown(payload: dict[str, Any]) -> str:
    lines = ["# SAR Stage-1 Conditioning Public Summary", ""]
    lines.append(f"**Current public-safe recommendation:** {payload['public_recommendation']}")
    lines.extend(["", "## Project Positioning", ""])
    lines.extend(f"- {item}" for item in payload["project_positioning"])
    lines.extend(["", "## Current Public Findings", ""])
    lines.extend(f"- {item}" for item in payload["public_findings"])
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in payload["public_caveats"])
    lines.extend(["", "## Bundle Matrix", ""])
    lines.append("| Bundle | Role | Domain | Maturity | Current Evidence |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["bundle_matrix"]:
        lines.append(
            f"| {row['bundle']} | {row['candidate_role']} | {row['intended_domain']} | {row['maturity']} | {row['current_evidence']} |"
        )
    lines.extend(["", "## Public Data Strategy", ""])
    lines.append("| Dataset | Role | Current Use | Next Action |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["data_strategy"]:
        lines.append(f"| {row['dataset']} | {row['role']} | {row['current_use']} | {row['next_action']} |")
    lines.append("")
    return "\n".join(lines)
