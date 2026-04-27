from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve1d


@dataclass(slots=True)
class StarletComplexResult:
    denoised_complex: np.ndarray
    surrogate_intensity: np.ndarray
    applied: bool
    levels: int
    notes: str


def _atrous_kernel(scale: int) -> np.ndarray:
    base = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if scale == 1:
        return base
    step = 2 ** (scale - 1)
    kernel = np.zeros((len(base) - 1) * step + 1, dtype=np.float32)
    kernel[::step] = base
    return kernel


def _smooth_b3(image: np.ndarray, scale: int) -> np.ndarray:
    kernel = _atrous_kernel(scale)
    smoothed = convolve1d(image, kernel, axis=0, mode="mirror")
    smoothed = convolve1d(smoothed, kernel, axis=1, mode="mirror")
    return smoothed.astype(np.float32)


def _soft_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    magnitude = np.abs(coefficients)
    shrunk = np.maximum(magnitude - threshold, 0.0)
    return np.sign(coefficients) * shrunk


def _starlet_shrink(image: np.ndarray, *, levels: int, threshold_scale: float) -> np.ndarray:
    current = np.asarray(image, dtype=np.float32)
    details: list[np.ndarray] = []
    for scale in range(1, levels + 1):
        smooth = _smooth_b3(current, scale)
        detail = current - smooth
        sigma = float(np.median(np.abs(detail)) / 0.6745) if np.any(detail) else 0.0
        threshold = float(threshold_scale) * sigma
        details.append(_soft_threshold(detail, threshold).astype(np.float32))
        current = smooth
    reconstruction = current
    for detail in details:
        reconstruction = reconstruction + detail
    return reconstruction.astype(np.float32)


def _to_complex(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return np.asarray(array, dtype=np.complex64)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[..., 0].astype(np.float32) + 1j * array[..., 1].astype(np.float32)
    if array.ndim == 3 and array.shape[0] == 2 and array.shape[0] < array.shape[-1]:
        moved = np.moveaxis(array, 0, -1)
        return moved[..., 0].astype(np.float32) + 1j * moved[..., 1].astype(np.float32)
    raise ValueError("Starlet complex denoising expects a complex array or a 2-channel real/imag array.")


def starlet_complex_denoise(
    complex_image: np.ndarray,
    *,
    levels: int = 4,
    threshold_scale: float = 3.0,
) -> StarletComplexResult:
    complex_array = _to_complex(np.asarray(complex_image))
    real_part = _starlet_shrink(np.real(complex_array), levels=levels, threshold_scale=threshold_scale)
    imag_part = _starlet_shrink(np.imag(complex_array), levels=levels, threshold_scale=threshold_scale)
    denoised_complex = real_part.astype(np.float32) + 1j * imag_part.astype(np.float32)
    surrogate_intensity = np.square(np.abs(denoised_complex), dtype=np.float32)
    notes = (
        "Applied starlet-style shrinkage independently to the real and imaginary components. "
        "This is the practical complex-domain additive cleanup used for Bundle C."
    )
    return StarletComplexResult(
        denoised_complex=denoised_complex.astype(np.complex64),
        surrogate_intensity=surrogate_intensity.astype(np.float32),
        applied=True,
        levels=int(levels),
        notes=notes,
    )
