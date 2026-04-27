from __future__ import annotations

from src.stage1.sentinel1_batch import (
    build_batch_topline,
    derive_scene_regime,
    planned_submethods_for_scene,
    recommend_scene_submethod,
)


def test_planned_submethods_compare_mode_only_adds_a1_when_metadata_exists() -> None:
    assert planned_submethods_for_scene(metadata_available=False, compare_submethods=True, additive_submethod=None) == [
        "A0",
        "A2",
        "A3",
    ]
    assert planned_submethods_for_scene(metadata_available=True, compare_submethods=True, additive_submethod=None) == [
        "A0",
        "A1",
        "A2",
        "A3",
    ]


def test_derive_scene_regime_marks_metadata_and_overview_flags() -> None:
    regime = derive_scene_regime(
        [
            {
                "metadata_available": True,
                "overview_fallback_used": True,
                "artifact_score": 0.12,
                "background_pixel_count": 50000,
                "proxy_enl_patches_before": 80,
                "calibration_xml_present": True,
                "manifest_safe_present": True,
            }
        ]
    )
    assert regime["metadata_regime"] == "metadata-rich"
    assert regime["overview_only_evaluation"] is True
    assert regime["structured_artifact_likely"] is True
    assert regime["quiet_background_available"] is True


def test_recommend_scene_submethod_prefers_a1_when_metadata_rich_and_metrics_help() -> None:
    regime = {
        "metadata_regime": "metadata-rich",
        "overview_only_evaluation": True,
        "structured_artifact_likely": False,
    }
    rows = [
        {
            "additive_submethod_used": "A1",
            "distribution_separability_delta": 0.04,
            "threshold_f1_delta": 0.03,
            "proxy_enl_before": 40.0,
            "proxy_enl_gain": 1.0,
            "edge_sharpness_before": 100.0,
            "edge_sharpness_delta": -2.0,
        },
        {
            "additive_submethod_used": "A2",
            "distribution_separability_delta": 0.02,
            "threshold_f1_delta": 0.01,
            "proxy_enl_before": 40.0,
            "proxy_enl_gain": 0.5,
            "edge_sharpness_before": 100.0,
            "edge_sharpness_delta": -1.0,
        },
    ]
    recommendation = recommend_scene_submethod(rows, regime)
    assert recommendation["best_submethod"] == "A1"
    assert "Metadata was available" in recommendation["why"]


def test_build_batch_topline_includes_winner_counts_and_confidence() -> None:
    topline, warnings = build_batch_topline(
        all_scene_rows=[{"scene_status": "ready"}, {"scene_status": "ready"}],
        comparison_rows=[{"scene_id": "scene-1"}],
        scene_summary_rows=[
            {"scene_evaluated": True, "best_submethod": "A1", "metadata_regime": "metadata-rich", "metadata_ready_for_a1": True, "overview_only_evaluation": True},
            {"scene_evaluated": True, "best_submethod": "A2", "metadata_regime": "metadata-poor", "metadata_ready_for_a1": False, "overview_only_evaluation": False},
        ],
    )
    assert topline["a1_win_count"] == 1
    assert topline["a2_win_count"] == 1
    assert topline["evidence_confidence_level"] == "thin"
    assert warnings
