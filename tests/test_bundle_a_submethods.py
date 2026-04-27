from __future__ import annotations

import numpy as np

from src.stage1.additive.bundle_a_submethods import run_bundle_a_additive_submethod


def test_bundle_a_auto_prefers_metadata_submethod_when_noise_vector_exists() -> None:
    image = np.full((32, 32), 10.0, dtype=np.float32)
    metadata = {"noise_vector": np.full((32,), 0.5, dtype=np.float32)}
    outcome = run_bundle_a_additive_submethod(image, metadata, {"submethod": "auto"})
    assert outcome.spec.code == "A1"
    assert outcome.metadata_available is True
    assert outcome.fallback_used is None
    assert float(outcome.corrected_intensity.mean()) < float(image.mean())


def test_bundle_a_forced_metadata_submethod_falls_back_when_metadata_is_missing() -> None:
    image = np.full((32, 32), 6.0, dtype=np.float32)
    outcome = run_bundle_a_additive_submethod(
        image,
        {},
        {
            "submethod": "A1",
            "fallback_submethod": "A2",
        },
    )
    assert outcome.spec.code == "A2"
    assert outcome.fallback_used == "A2"
    assert "falling back" in outcome.warning.lower()


def test_bundle_a_auto_uses_image_floor_without_metadata_or_artifact() -> None:
    rng = np.random.default_rng(7)
    image = np.abs(rng.normal(loc=4.0, scale=0.2, size=(48, 48))).astype(np.float32)
    outcome = run_bundle_a_additive_submethod(image, {}, {"submethod": "auto", "artifact_auto_threshold": 0.2})
    assert outcome.spec.code == "A2"
    assert outcome.metadata_available is False


def test_bundle_a_auto_uses_structured_artifact_submethod_when_striping_is_strong() -> None:
    base = np.full((64, 64), 12.0, dtype=np.float32)
    stripes = np.tile(np.array([0.0, 4.0, 0.0, -4.0], dtype=np.float32), 16)
    image = base + stripes[np.newaxis, :]
    outcome = run_bundle_a_additive_submethod(
        image,
        {},
        {"submethod": "auto", "artifact_auto_threshold": 0.05},
    )
    assert outcome.spec.code == "A3"
    assert outcome.artifact_score is not None
    assert outcome.artifact_score >= 0.05
