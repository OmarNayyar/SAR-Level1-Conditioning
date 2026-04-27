from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import sobel


@dataclass(slots=True)
class EdgeSharpnessResult:
    score: float
    mean_gradient: float
    top_quantile_mean: float


def compute_edge_sharpness(intensity_image: np.ndarray, *, top_quantile: float = 0.9) -> EdgeSharpnessResult:
    image = np.asarray(intensity_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Edge sharpness expects a 2D image.")

    grad_x = sobel(image, axis=0, mode="reflect")
    grad_y = sobel(image, axis=1, mode="reflect")
    magnitude = np.hypot(grad_x, grad_y)
    mean_gradient = float(np.mean(magnitude))
    threshold = float(np.quantile(magnitude, top_quantile))
    top_quantile_mean = float(np.mean(magnitude[magnitude >= threshold])) if np.any(magnitude >= threshold) else 0.0
    return EdgeSharpnessResult(
        score=top_quantile_mean,
        mean_gradient=mean_gradient,
        top_quantile_mean=top_quantile_mean,
    )
