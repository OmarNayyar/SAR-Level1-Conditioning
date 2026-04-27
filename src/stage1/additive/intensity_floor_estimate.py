from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class IntensityFloorEstimateResult:
    corrected_intensity: np.ndarray
    estimated_noise_power: np.ndarray
    applied: bool
    mode: str
    notes: str
    estimated_floor_value: float


def estimate_intensity_floor(
    intensity_image: np.ndarray,
    *,
    floor_quantile: float = 0.02,
    quiet_upper_quantile: float = 0.25,
) -> IntensityFloorEstimateResult:
    """Estimate a simple additive floor directly from the intensity image.

    This is intentionally lightweight and honest: it uses a lower-tail
    background estimate in intensity space and applies a constant floor
    subtraction across the frame.
    """

    intensity = np.asarray(intensity_image, dtype=np.float32)
    if intensity.ndim != 2:
        raise ValueError("Intensity floor estimation expects a 2D intensity image.")

    positive = intensity[np.isfinite(intensity) & (intensity > 0.0)]
    if positive.size < 32:
        return IntensityFloorEstimateResult(
            corrected_intensity=intensity.copy(),
            estimated_noise_power=np.zeros_like(intensity, dtype=np.float32),
            applied=False,
            mode="insufficient_support",
            notes="Skipped image-derived additive floor estimation because too few positive intensity pixels were available.",
            estimated_floor_value=0.0,
        )

    quiet_upper = float(np.quantile(positive, quiet_upper_quantile))
    quiet_values = positive[positive <= quiet_upper]
    if quiet_values.size < 16:
        quiet_values = positive

    floor_value = float(np.quantile(quiet_values, floor_quantile))
    estimated_noise = np.full_like(intensity, floor_value, dtype=np.float32)
    corrected = np.maximum(intensity - estimated_noise, 0.0, dtype=np.float32)

    return IntensityFloorEstimateResult(
        corrected_intensity=corrected,
        estimated_noise_power=estimated_noise,
        applied=floor_value > 0.0,
        mode="global_floor_quantile",
        notes=(
            "Applied an image-derived additive floor estimate from the lower-tail quiet-intensity subset. "
            "This is a practical baseline that does not require product noise metadata."
        ),
        estimated_floor_value=floor_value,
    )
