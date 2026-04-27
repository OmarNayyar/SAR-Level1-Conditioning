from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from skimage import color
from skimage.filters import sobel
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize


@dataclass(frozen=True, slots=True)
class DenoisingMetrics:
    mse: float
    nrmse: float
    psnr: float
    ssim: float
    edge_preservation_index: float | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: ("" if value is None else value) for key, value in payload.items()}


def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] in {3, 4}:
            array = color.rgb2gray(array[..., :3])
        elif array.shape[0] in {3, 4} and array.shape[0] < array.shape[-1]:
            array = color.rgb2gray(np.moveaxis(array[:3], 0, -1))
        else:
            array = np.mean(array, axis=-1)
    array = np.asarray(array, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_scale_limits(noisy: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    stacked = np.concatenate([noisy.reshape(-1), reference.reshape(-1)])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return low, low + 1.0
    return float(low), float(high)


def _normalize_with_limits(image: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0).astype(np.float32)


def normalize_paired_images(noisy: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    noisy_gray = to_grayscale_float(noisy)
    reference_gray = to_grayscale_float(reference)
    if noisy_gray.shape != reference_gray.shape:
        noisy_gray = resize(noisy_gray, reference_gray.shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    low, high = _safe_scale_limits(noisy_gray, reference_gray)
    return _normalize_with_limits(noisy_gray, low, high), _normalize_with_limits(reference_gray, low, high)


def match_reference_shape(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    array = to_grayscale_float(image)
    if array.shape != reference.shape:
        array = resize(array, reference.shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def _edge_preservation_index(candidate: np.ndarray, reference: np.ndarray) -> float | None:
    candidate_edges = sobel(candidate).reshape(-1)
    reference_edges = sobel(reference).reshape(-1)
    if float(np.std(candidate_edges)) == 0.0 or float(np.std(reference_edges)) == 0.0:
        return None
    return float(np.corrcoef(candidate_edges, reference_edges)[0, 1])


def compute_denoising_metrics(candidate: np.ndarray, reference: np.ndarray) -> DenoisingMetrics:
    candidate_matched = match_reference_shape(candidate, reference)
    reference_matched = match_reference_shape(reference, reference)
    mse = float(mean_squared_error(reference_matched, candidate_matched))
    return DenoisingMetrics(
        mse=mse,
        nrmse=float(normalized_root_mse(reference_matched, candidate_matched)),
        psnr=float(peak_signal_noise_ratio(reference_matched, candidate_matched, data_range=1.0)),
        ssim=float(structural_similarity(reference_matched, candidate_matched, data_range=1.0)),
        edge_preservation_index=_edge_preservation_index(candidate_matched, reference_matched),
    )

