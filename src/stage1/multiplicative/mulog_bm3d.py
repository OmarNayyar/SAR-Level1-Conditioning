from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(slots=True)
class MuLoGResult:
    filtered_image: np.ndarray
    log_input: np.ndarray
    log_filtered: np.ndarray
    backend: str
    noise_sigma: float
    notes: str


def _estimate_log_sigma(log_image: np.ndarray) -> float:
    gradients = np.diff(log_image, axis=1)
    if gradients.size == 0:
        return 1e-3
    sigma = float(np.median(np.abs(gradients)) / 0.6745)
    return max(sigma, 1e-3)


def _run_log_denoiser(log_image: np.ndarray, *, backend_preference: str, sigma: float) -> tuple[np.ndarray, str]:
    normalized_backend = backend_preference.strip().lower()
    if normalized_backend == "bm3d":
        try:
            import bm3d as bm3d_module

            filtered = bm3d_module.bm3d(log_image, sigma_psd=sigma)
            return np.asarray(filtered, dtype=np.float32), "bm3d"
        except Exception:
            pass

    if normalized_backend == "gaussian":
        return gaussian_filter(log_image, sigma=max(sigma, 0.5), mode="reflect").astype(np.float32), "gaussian"

    try:
        from skimage.restoration import denoise_wavelet

        filtered = denoise_wavelet(
            log_image,
            sigma=sigma,
            mode="soft",
            wavelet="db2",
            channel_axis=None,
            rescale_sigma=True,
        )
        return np.asarray(filtered, dtype=np.float32), "wavelet"
    except Exception:
        return gaussian_filter(log_image, sigma=max(sigma, 0.75), mode="reflect").astype(np.float32), "gaussian"


def mulog_bm3d(
    intensity_image: np.ndarray,
    *,
    backend_preference: str = "bm3d",
) -> MuLoGResult:
    """Apply a single-channel MuLoG-style log-domain despeckling path.

    The implementation is intentionally lightweight: it log-linearizes the
    intensity image, estimates the noise level in the log domain, and then
    applies either BM3D (if installed) or a documented fallback denoiser.
    """

    image = np.asarray(intensity_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("MuLoG-BM3D expects a 2D intensity image.")

    eps = 1e-6
    log_input = np.log(np.maximum(image, eps)).astype(np.float32)
    sigma = _estimate_log_sigma(log_input)
    log_filtered, backend = _run_log_denoiser(log_input, backend_preference=backend_preference, sigma=sigma)
    filtered_image = np.exp(log_filtered).astype(np.float32)
    notes = (
        "Applied a practical single-channel MuLoG-style log-domain despeckling path. "
        "If BM3D is unavailable, the implementation falls back to a documented Gaussian/wavelet denoiser."
    )
    return MuLoGResult(
        filtered_image=filtered_image,
        log_input=log_input,
        log_filtered=log_filtered.astype(np.float32),
        backend=backend,
        noise_sigma=float(sigma),
        notes=notes,
    )
