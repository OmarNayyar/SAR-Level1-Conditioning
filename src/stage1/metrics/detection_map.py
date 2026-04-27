from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_detection_proxy_map(intensity_image: np.ndarray, *, background_sigma: float = 7.0) -> np.ndarray:
    """Compute a simple local-contrast ship-detection proxy map.

    This is intentionally a proxy hook, not a detector. It highlights compact
    bright structures above a smoothed local background so Bundle A can save a
    downstream-ready diagnostic without pretending to produce real mAP.
    """

    image = np.asarray(intensity_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Detection proxy map expects a 2D image.")

    clipped = np.clip(image, 0.0, None)
    background = gaussian_filter(clipped, sigma=background_sigma, mode="reflect")
    residual = np.maximum(clipped - background, 0.0)
    scale = np.std(residual) + 1e-6
    return (residual / scale).astype(np.float32)
