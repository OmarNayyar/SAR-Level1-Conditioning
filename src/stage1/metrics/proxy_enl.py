from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import sobel


@dataclass(slots=True)
class ProxyEnlResult:
    score: float
    patch_count: int
    patch_size: int
    patch_scores: list[float]


def _candidate_mask(image: np.ndarray) -> np.ndarray:
    grad_x = sobel(image, axis=0, mode="reflect")
    grad_y = sobel(image, axis=1, mode="reflect")
    magnitude = np.hypot(grad_x, grad_y)
    threshold = float(np.quantile(magnitude, 0.25))
    return magnitude <= threshold


def compute_proxy_enl(
    intensity_image: np.ndarray,
    *,
    patch_size: int = 32,
    valid_mask: np.ndarray | None = None,
) -> ProxyEnlResult:
    image = np.asarray(intensity_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Proxy ENL expects a 2D intensity image.")

    mask = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else _candidate_mask(image)
    patch_scores: list[float] = []
    fallback_scores: list[float] = []
    for row in range(0, image.shape[0] - patch_size + 1, patch_size):
        for col in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch = image[row : row + patch_size, col : col + patch_size]
            patch_mask = mask[row : row + patch_size, col : col + patch_size]
            valid_patch = patch[patch_mask] if patch_mask.mean() >= 0.5 else patch.reshape(-1)
            mean_value = float(np.mean(valid_patch))
            variance = float(np.var(valid_patch))
            if mean_value <= 0.0 or variance <= 1e-9:
                continue
            score = float((mean_value * mean_value) / variance)
            fallback_scores.append(score)
            if patch_mask.mean() >= 0.5:
                patch_scores.append(score)

    if not patch_scores and fallback_scores:
        keep_count = max(1, int(np.ceil(len(fallback_scores) * 0.25)))
        patch_scores = sorted(fallback_scores, reverse=True)[:keep_count]

    score = float(np.median(patch_scores)) if patch_scores else float("nan")
    return ProxyEnlResult(
        score=score,
        patch_count=len(patch_scores),
        patch_size=patch_size,
        patch_scores=patch_scores,
    )
