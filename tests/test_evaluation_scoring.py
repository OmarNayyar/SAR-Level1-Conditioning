from __future__ import annotations

from src.reporting.evaluation import evidence_confidence_from_counts, score_proxy_tradeoff


def test_score_penalizes_oversmoothing_even_when_enl_gain_is_high() -> None:
    score = score_proxy_tradeoff(
        {
            "additive_submethod_used": "A3",
            "proxy_enl_before": 10.0,
            "proxy_enl_gain": 30.0,
            "edge_sharpness_before": 100.0,
            "edge_sharpness_delta": -60.0,
            "distribution_separability_delta": 0.0,
            "threshold_f1_delta": 0.0,
        },
        regime={"structured_artifact_likely": False},
    )
    assert score.score < 0.0
    assert any("oversmoothing" in flag for flag in score.flags)


def test_score_flags_possible_a1_under_crediting_on_overview_metadata_scene() -> None:
    score = score_proxy_tradeoff(
        {
            "additive_submethod_used": "A1",
            "metadata_available": True,
            "overview_fallback_used": True,
            "proxy_enl_before": 40.0,
            "proxy_enl_gain": 0.0,
            "edge_sharpness_before": 100.0,
            "edge_sharpness_delta": -5.0,
            "distribution_separability_delta": 0.02,
            "threshold_f1_delta": 0.01,
        },
        regime={"metadata_regime": "metadata-rich", "overview_only_evaluation": True},
    )
    assert score.score > 0.0
    assert "under-credited" in " ".join(score.flags)


def test_evidence_confidence_marks_overview_limited_developing_set() -> None:
    assert evidence_confidence_from_counts(8, overview_only_count=8) == "developing / overview-limited"

