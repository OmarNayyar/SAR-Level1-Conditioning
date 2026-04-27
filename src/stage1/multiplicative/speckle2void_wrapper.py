from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from src.stage1.external import run_external_array_command


@dataclass(slots=True)
class Speckle2VoidResult:
    filtered_image: np.ndarray
    backend: str
    fallback_used: bool
    notes: str


def _fallback_blind_spot(image: np.ndarray, *, sigma: float) -> np.ndarray:
    try:
        from skimage.restoration import denoise_invariant

        denoised = denoise_invariant(
            image,
            denoise_function=lambda patch: gaussian_filter(patch, sigma=max(float(sigma), 0.75), mode="reflect"),
        )
        return np.asarray(denoised, dtype=np.float32)
    except Exception:
        return gaussian_filter(image, sigma=max(float(sigma), 0.75), mode="reflect").astype(np.float32)


def run_speckle2void_wrapper(
    intensity_image: np.ndarray,
    *,
    external: dict[str, Any] | None = None,
    fallback_sigma: float = 1.0,
) -> Speckle2VoidResult:
    image = np.asarray(intensity_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Speckle2Void wrapper expects a 2D image.")

    external = external or {}
    command_template = external.get("command")
    cwd = external.get("cwd")
    if isinstance(command_template, list) and command_template:
        external_output = run_external_array_command(
            input_array=image,
            command_template=[str(token) for token in command_template],
            cwd=str(cwd) if cwd else None,
        )
        filtered = np.asarray(external_output, dtype=np.float32)
        if filtered.shape != image.shape:
            raise RuntimeError("Speckle2Void external backend returned an unexpected output shape.")
        return Speckle2VoidResult(
            filtered_image=filtered,
            backend="external_command",
            fallback_used=False,
            notes="Ran the configured external Speckle2Void-style backend via a command adapter.",
        )

    filtered = _fallback_blind_spot(image, sigma=float(fallback_sigma))
    return Speckle2VoidResult(
        filtered_image=np.clip(filtered, 0.0, None).astype(np.float32),
        backend="j_invariant_fallback",
        fallback_used=True,
        notes=(
            "Speckle2Void was not configured as an external backend, so Bundle D used a J-invariant blind-spot-style fallback. "
            "This is an honest self-supervised surrogate, not a claim of upstream Speckle2Void parity."
        ),
    )
