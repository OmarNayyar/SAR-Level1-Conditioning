from __future__ import annotations

from src.reporting.decision_support import (
    additive_submethod_display,
    run_snapshot,
    sentinel1_readiness_text,
)


def test_run_snapshot_uses_summary_fallbacks() -> None:
    payload = {
        "summary": {
            "bundle_name": "bundle_a",
            "dataset": "ssdd",
            "processed_count": 8,
            "skipped_count": 0,
            "downstream_status": "proxy-only",
            "maturity_note": "Current lead",
            "interpretation": "Useful summary.",
            "current_recommendation": "Use A2.",
            "warnings": ["Proxy-only"],
            "additive_submethod_counts": {"A2": 8},
            "aggregate_metrics": {
                "proxy_enl_gain": 3.2,
                "edge_sharpness_delta": -1.4,
                "distribution_separability_before": 0.5,
                "distribution_separability_after": 0.7,
                "threshold_f1_before": 0.2,
                "threshold_f1_after": 0.4,
            },
        },
        "topline_metrics": {},
    }
    snapshot = run_snapshot(payload)
    assert snapshot["dominant_additive_submethod"] == "A2"
    assert snapshot["distribution_separability_delta"] == 0.19999999999999996
    assert snapshot["additive_submethod_label"] == "A2"


def test_sentinel1_readiness_text_counts_ready_rows() -> None:
    rows = [
        {"prepared_status": "ready"},
        {"prepared_status": "failed"},
        {"prepared_status": "ready"},
    ]
    assert sentinel1_readiness_text(rows) == "2/3 manifest rows locally runnable"


def test_additive_submethod_display_handles_mixed_counts() -> None:
    summary = {"additive_submethod_counts": {"A2": 2, "A3": 5}}
    assert additive_submethod_display(summary) == "Mixed (dominated by A3)"
