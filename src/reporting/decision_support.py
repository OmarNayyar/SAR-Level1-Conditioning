from __future__ import annotations

"""Small app-facing helpers for turning result artifacts into readable decisions."""

from typing import Any


BUNDLE_EVIDENCE_ROWS = [
    {
        "Bundle": "A",
        "Evidence status": "Screening-grade / detector-negative so far",
        "Interpretation": "Most interpretable conditioning family, but current YOLO evidence says default A hurts detection versus raw.",
    },
    {
        "Bundle": "B",
        "Evidence status": "Secondary / artifact specialist",
        "Interpretation": "Useful comparison route for harder additive artifacts; detector prep is supported and needs a real run.",
    },
    {
        "Bundle": "C",
        "Evidence status": "Feasibility-grade",
        "Interpretation": "Do not treat as claim-grade until better complex SLC coverage exists.",
    },
    {
        "Bundle": "D",
        "Evidence status": "Secondary / detector comparator",
        "Interpretation": "More promising than default A in current detector deltas, but still trails raw and needs tuning.",
    },
]

BUNDLE_A_SUBMETHOD_DETAILS = {
    "A0": {
        "label": "A0 - No additive correction",
        "requires": "Intensity image only",
        "trust": "baseline-only",
        "note": "Use as the control to show what the additive step is buying you.",
    },
    "A1": {
        "label": "A1 - Metadata thermal / noise-vector subtraction",
        "requires": "Noise vector, noise power, or NESZ-style metadata",
        "trust": "metadata-driven",
        "note": "Best when real product metadata exists and should be trusted on Sentinel-1 style products.",
    },
    "A2": {
        "label": "A2 - Image-derived additive floor estimate",
        "requires": "Intensity image only",
        "trust": "image-derived",
        "note": "Practical fallback when metadata is missing. This is the current recommendation on metadata-poor public chip data.",
    },
    "A3": {
        "label": "A3 - Structured additive artifact correction",
        "requires": "Intensity image with a detectable structured artifact pattern",
        "trust": "artifact-based",
        "note": "Use when stripe-like or structured additive contamination is actually visible.",
    },
}

GLOSSARY_ROWS = [
    {
        "term": "Level 1 conditioning",
        "plain meaning": "A practical cleanup layer after acquisition/calibration choices and before downstream AI.",
    },
    {
        "term": "Additive correction",
        "plain meaning": "Attempts to remove additive floors, thermal noise, or structured artifacts before speckle handling.",
    },
    {
        "term": "Speckle",
        "plain meaning": "Multiplicative granular SAR variation; smoothing it too much can erase edges and small targets.",
    },
    {
        "term": "ENL",
        "plain meaning": "A homogeneity proxy. Higher often means smoother background, but too much gain can indicate oversmoothing.",
    },
    {
        "term": "Separability",
        "plain meaning": "How distinct target-like pixels look from background pixels under the simple statistical baseline.",
    },
    {
        "term": "Threshold F1",
        "plain meaning": "A simple thresholding sanity check, not a trained detector score.",
    },
    {
        "term": "Proxy-only",
        "plain meaning": "Useful screening evidence, but not final mAP/IoU from a trained downstream model.",
    },
    {
        "term": "Overview-scale",
        "plain meaning": "The scene was evaluated on an internal overview or decimated sample to stay memory-safe.",
    },
    {
        "term": "Metadata-rich",
        "plain meaning": "Noise/calibration metadata exists, so A1 can be meaningfully attempted.",
    },
]

CONFIDENCE_BADGE_COLORS = {
    "baseline-only": "#6b7280",
    "metadata-driven": "#2563eb",
    "image-derived": "#0f766e",
    "artifact-based": "#b45309",
}

DATASET_USAGE_RECOMMENDATIONS = {
    "ssdd": "Ready for ship screening",
    "hrsid": "Ready for ship screening",
    "sentinel1": "Real-product proxy experiments",
    "sen1floods11": "Smoke segmentation",
    "ai4arctic": "Not locally ready",
    "ls_ssdd": "Unavailable",
}


def safe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_delta(summary: dict[str, Any], before_key: str, after_key: str) -> float | None:
    aggregate = summary.get("aggregate_metrics", {})
    before = safe_float(aggregate.get(before_key))
    after = safe_float(aggregate.get(after_key))
    if before is None or after is None:
        return None
    return after - before


def shorten_path(path_text: str, *, max_chars: int = 70) -> str:
    if len(path_text) <= max_chars:
        return path_text
    head = max_chars // 2 - 2
    tail = max_chars - head - 3
    return path_text[:head] + "..." + path_text[-tail:]


def dominant_additive_submethod(summary: dict[str, Any]) -> str:
    counts = summary.get("additive_submethod_counts", {})
    if not counts:
        return ""
    return max(sorted(counts), key=lambda code: int(counts.get(code, 0)))


def additive_submethod_display(summary: dict[str, Any]) -> str:
    counts = summary.get("additive_submethod_counts", {})
    if not counts:
        return "Not explicit"
    if len(counts) == 1:
        return next(iter(counts))
    dominant = dominant_additive_submethod(summary)
    return f"Mixed (dominated by {dominant})" if dominant else "Mixed"


def run_snapshot(run_payload: dict[str, Any]) -> dict[str, Any]:
    summary = run_payload.get("summary", {})
    topline = dict(run_payload.get("topline_metrics", {}) or {})
    if not topline:
        topline = {
            "proxy_enl_gain": summary.get("aggregate_metrics", {}).get("proxy_enl_gain"),
            "edge_sharpness_delta": summary.get("aggregate_metrics", {}).get("edge_sharpness_delta"),
            "distribution_separability_delta": metric_delta(
                summary,
                "distribution_separability_before",
                "distribution_separability_after",
            ),
            "threshold_f1_delta": metric_delta(summary, "threshold_f1_before", "threshold_f1_after"),
        }
    topline.setdefault("dominant_additive_submethod", dominant_additive_submethod(summary))
    topline.setdefault("maturity_note", summary.get("maturity_note", ""))
    topline.setdefault("evidence_grade", summary.get("evidence_grade", ""))
    topline.setdefault("decision_basis", summary.get("decision_basis", ""))
    topline.setdefault("current_recommendation", summary.get("current_recommendation", ""))
    topline.setdefault("warnings", summary.get("warnings", []))
    topline.setdefault("interpretation", summary.get("interpretation", ""))
    topline.setdefault("bundle_name", summary.get("bundle_name", ""))
    topline.setdefault("dataset", summary.get("dataset", ""))
    topline.setdefault("processed_count", summary.get("processed_count", 0))
    topline.setdefault("skipped_count", summary.get("skipped_count", 0))
    topline.setdefault("downstream_status", summary.get("downstream_status", ""))
    topline.setdefault("additive_submethod_label", additive_submethod_display(summary))
    return topline


def strongest_bundle_name() -> str:
    return "Bundle A for proxy screening and interpretation; raw imagery for current detector handoff"


def strongest_public_chip_submethod() -> str:
    return "A2 for metadata-poor proxy screening; raw remains detector baseline"


def sentinel1_readiness_text(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    ready = sum(str(row.get("prepared_status", "")).lower() == "ready" for row in rows)
    return f"{ready}/{total} manifest rows locally runnable" if total else "No local Sentinel-1 rows"


def main_blocker_text(rows: list[dict[str, Any]]) -> str:
    ready = sum(str(row.get("prepared_status", "")).lower() == "ready" for row in rows)
    if ready < 5:
        return "More Sentinel-1 maritime GRD scenes are needed for stronger real-product evidence."
    return "Detector validation is wired, but stronger medium/full runs are needed before handoff claims."


def next_action_text() -> str:
    return (
        "Use raw as the detector baseline, compare A-conservative/B/D on medium YOLO runs, add targeted Sentinel-1 GRD scenes, "
        "and validate any private/local handoff data before conditioning."
    )


def dataset_status_help_rows() -> list[dict[str, str]]:
    return [
        {"status": "full / complete", "meaning": "Enough local files are present for the intended workflow."},
        {"status": "partial", "meaning": "Usable locally, but not the full upstream dataset."},
        {"status": "metadata-only", "meaning": "Catalogs and manifests exist, but no real local samples are ready."},
        {"status": "external-linked", "meaning": "Data lives outside the repo and is referenced through the registry."},
    ]
