from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


@dataclass(slots=True)
class DestripeResult:
    corrected_image: np.ndarray
    stripe_component: np.ndarray
    lowrank_component: np.ndarray
    applied: bool
    mode: str
    notes: str


def destripe_lowrank_sparse(
    image: np.ndarray,
    *,
    domain: str = "intensity",
    orientation: str = "columns",
    background_sigma: float = 9.0,
    profile_sigma: float = 3.0,
    correction_strength: float = 1.0,
) -> DestripeResult:
    """Remove structured stripe-like additive artifacts with a light low-rank/sparse proxy.

    This is a practical low-dependency approximation, not a full RPCA solver:
    a smooth low-rank background is estimated, the residual stripe profile is
    reduced along the requested orientation, and the correction is mapped back
    to the original domain.
    """

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Destriping expects a 2D image.")

    eps = 1e-6
    normalized_domain = domain.strip().lower()
    work = np.log(np.maximum(array, eps)) if normalized_domain in {"log", "log_intensity", "log-intensity"} else array.copy()

    lowrank_component = gaussian_filter(work, sigma=float(background_sigma), mode="reflect")
    residual = work - lowrank_component

    normalized_orientation = orientation.strip().lower()
    if normalized_orientation.startswith("row"):
        profile = np.median(residual, axis=1)
        profile = gaussian_filter1d(profile, sigma=float(profile_sigma), mode="reflect")
        stripe_component = np.broadcast_to(profile[:, np.newaxis], work.shape).astype(np.float32)
        mode = "row_profile"
    else:
        profile = np.median(residual, axis=0)
        profile = gaussian_filter1d(profile, sigma=float(profile_sigma), mode="reflect")
        stripe_component = np.broadcast_to(profile[np.newaxis, :], work.shape).astype(np.float32)
        mode = "column_profile"

    corrected_work = work - float(correction_strength) * stripe_component
    if normalized_domain in {"log", "log_intensity", "log-intensity"}:
        corrected_image = np.exp(corrected_work).astype(np.float32)
    else:
        corrected_image = np.clip(corrected_work, 0.0, None).astype(np.float32)

    notes = (
        "Applied a practical low-rank-plus-sparse-inspired destriping correction. "
        "This is an artifact-aware additive cleanup heuristic, not a full low-rank/sparse decomposition."
    )
    return DestripeResult(
        corrected_image=corrected_image,
        stripe_component=stripe_component.astype(np.float32),
        lowrank_component=lowrank_component.astype(np.float32),
        applied=True,
        mode=mode,
        notes=notes,
    )
