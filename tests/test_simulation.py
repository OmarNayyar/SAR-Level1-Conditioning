from __future__ import annotations

import numpy as np

from src.stage1.additive.thermal_noise_subtract import thermal_noise_subtract_intensity
from src.stage1.metrics.detection_map import compute_detection_proxy_map
from src.stage1.metrics.edge_sharpness import compute_edge_sharpness
from src.stage1.metrics.proxy_enl import compute_proxy_enl
from src.stage1.multiplicative.refined_lee import refined_lee_filter


def test_thermal_noise_subtraction_skips_when_metadata_missing() -> None:
    image = np.full((16, 16), 10.0, dtype=np.float32)
    result = thermal_noise_subtract_intensity(image, metadata={})
    assert not result.applied
    assert np.allclose(result.corrected_intensity, image)


def test_thermal_noise_subtraction_applies_noise_vector() -> None:
    image = np.full((8, 8), 10.0, dtype=np.float32)
    result = thermal_noise_subtract_intensity(image, metadata={"noise_vector": np.full(8, 2.0, dtype=np.float32)})
    assert result.applied
    assert np.allclose(result.corrected_intensity, 8.0)


def test_refined_lee_returns_nonnegative_image_and_metrics_smoke() -> None:
    rng = np.random.default_rng(42)
    base = np.ones((64, 64), dtype=np.float32)
    noisy = base * rng.gamma(shape=4.0, scale=0.25, size=(64, 64)).astype(np.float32)
    filtered = refined_lee_filter(noisy)
    assert filtered.shape == noisy.shape
    assert np.all(filtered >= 0.0)

    enl_before = compute_proxy_enl(noisy)
    enl_after = compute_proxy_enl(filtered)
    edge = compute_edge_sharpness(filtered)
    detection = compute_detection_proxy_map(filtered)

    assert detection.shape == noisy.shape
    assert enl_before.patch_count >= 0
    assert enl_after.patch_count >= 0
    assert edge.score >= 0.0
