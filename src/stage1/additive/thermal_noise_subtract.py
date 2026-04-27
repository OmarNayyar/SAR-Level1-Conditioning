from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ThermalNoiseResult:
    corrected_intensity: np.ndarray
    estimated_noise_power: np.ndarray
    applied: bool
    mode: str
    notes: str


def _broadcast_noise_profile(noise_profile: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    if noise_profile.ndim == 0:
        return np.full((height, width), float(noise_profile), dtype=np.float32)
    flattened = noise_profile.astype(np.float32).reshape(-1)
    if flattened.size == width:
        return np.broadcast_to(flattened[np.newaxis, :], (height, width)).astype(np.float32)
    if flattened.size == height:
        return np.broadcast_to(flattened[:, np.newaxis], (height, width)).astype(np.float32)
    if flattened.size == height * width:
        return flattened.reshape(height, width).astype(np.float32)
    raise ValueError(
        "Noise profile length does not match image width, height, or full image size."
    )


def _noise_from_nesz(metadata: dict[str, Any], image_shape: tuple[int, int], reference_power: float) -> np.ndarray | None:
    nesz_db = metadata.get("nesz_db")
    if nesz_db is None:
        return None
    nesz_array = np.asarray(nesz_db, dtype=np.float32)
    nesz_linear = np.power(10.0, nesz_array / 10.0, dtype=np.float32)
    return _broadcast_noise_profile(nesz_linear * reference_power, image_shape)


def thermal_noise_subtract_intensity(
    intensity_image: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> ThermalNoiseResult:
    """Apply metadata-driven thermal-noise subtraction in the intensity domain.

    The function is intentionally conservative: when no metadata-derived noise
    estimate is available, it returns the input unchanged and clearly marks the
    additive step as skipped.
    """

    intensity = np.asarray(intensity_image, dtype=np.float32)
    if intensity.ndim != 2:
        raise ValueError("Thermal-noise subtraction expects a 2D intensity image.")

    metadata = metadata or {}
    image_shape = intensity.shape
    reference_power = float(max(np.nanmedian(intensity), 1e-6))
    noise_power: np.ndarray | None = None
    mode = "skipped"
    notes = "No metadata-derived thermal-noise estimate was available."

    if "noise_power" in metadata:
        noise_power = _broadcast_noise_profile(np.asarray(metadata["noise_power"], dtype=np.float32), image_shape)
        mode = "noise_power"
        notes = "Applied thermal-noise subtraction using explicit noise_power metadata."
    elif "noise_vector" in metadata:
        noise_power = _broadcast_noise_profile(np.asarray(metadata["noise_vector"], dtype=np.float32), image_shape)
        mode = "noise_vector"
        notes = "Applied thermal-noise subtraction using the provided noise-vector metadata."
    else:
        noise_power = _noise_from_nesz(metadata, image_shape, reference_power)
        if noise_power is not None:
            mode = "nesz_db"
            notes = "Applied approximate thermal-noise subtraction using NESZ metadata in the intensity domain."

    if noise_power is None:
        return ThermalNoiseResult(
            corrected_intensity=intensity.copy(),
            estimated_noise_power=np.zeros_like(intensity, dtype=np.float32),
            applied=False,
            mode=mode,
            notes=notes,
        )

    corrected = np.maximum(intensity - noise_power, 0.0, dtype=np.float32)
    return ThermalNoiseResult(
        corrected_intensity=corrected,
        estimated_noise_power=noise_power.astype(np.float32),
        applied=True,
        mode=mode,
        notes=notes,
    )
