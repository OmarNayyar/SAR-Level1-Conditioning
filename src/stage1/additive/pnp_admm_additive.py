from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(slots=True)
class PnPAdmmResult:
    corrected_image: np.ndarray
    residual_image: np.ndarray
    iterations: int
    backend: str
    notes: str


def _default_denoiser(image: np.ndarray, *, sigma: float, backend: str) -> np.ndarray:
    if backend == "gaussian":
        return gaussian_filter(image, sigma=max(float(sigma), 0.5), mode="reflect").astype(np.float32)
    try:
        from skimage.restoration import denoise_wavelet

        denoised = denoise_wavelet(
            image,
            sigma=float(max(sigma, 1e-3)),
            mode="soft",
            wavelet="db2",
            channel_axis=None,
            rescale_sigma=True,
        )
        return np.asarray(denoised, dtype=np.float32)
    except Exception:
        return gaussian_filter(image, sigma=max(float(sigma), 0.75), mode="reflect").astype(np.float32)


def pnp_admm_additive(
    image: np.ndarray,
    *,
    iterations: int = 6,
    rho: float = 1.0,
    denoiser_backend: str = "wavelet",
    domain: str = "intensity",
) -> PnPAdmmResult:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("PnP-ADMM additive cleanup expects a 2D image.")

    eps = 1e-6
    normalized_domain = domain.strip().lower()
    work = np.log(np.maximum(array, eps)) if normalized_domain in {"log", "log_intensity", "log-intensity"} else array.copy()
    sigma = float(max(np.median(np.abs(work - np.median(work))) / 0.6745, 1e-3))

    x = work.copy()
    z = work.copy()
    u = np.zeros_like(work, dtype=np.float32)
    for _ in range(int(iterations)):
        x = (work + float(rho) * (z - u)) / (1.0 + float(rho))
        z = _default_denoiser(x + u, sigma=sigma, backend=denoiser_backend)
        u = u + x - z

    corrected_work = z.astype(np.float32)
    if normalized_domain in {"log", "log_intensity", "log-intensity"}:
        corrected_image = np.exp(corrected_work).astype(np.float32)
    else:
        corrected_image = np.clip(corrected_work, 0.0, None).astype(np.float32)
    residual = np.clip(array - corrected_image, 0.0, None).astype(np.float32)
    notes = (
        "Applied plug-and-play ADMM for additive cleanup using a lightweight denoiser backend. "
        "This is the practical Bundle D additive path."
    )
    return PnPAdmmResult(
        corrected_image=corrected_image,
        residual_image=residual,
        iterations=int(iterations),
        backend=denoiser_backend,
        notes=notes,
    )
